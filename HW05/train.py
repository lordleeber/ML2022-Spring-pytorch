import sys
import pdb
import pprint
import logging
import os
import random
import shutil
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
import wandb
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

from fairseq.tasks.translation import TranslationConfig, TranslationTask
from hw5_config import *
from labelsmooth_cross_entropy_criterion import LabelSmoothedCrossEntropyCriterion
from noam_opt import NoamOpt
from rnn_encoder import RNNEncoder
from rnn_decoder import RNNDecoder
from seq_2_seq import Seq2Seq
import matplotlib.pyplot as plt


"""# Training Procedure

## Training
"""


def build_model(args, task):
    """ build a model instance based on hyperparameters """
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)

    # initialization for seq2seq model is important, requires extra handling
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    # weight initialization
    model.apply(init_params)
    return model


def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)  # gradient accumulation: update every accum_steps samples

    stats = {"loss": []}
    scaler = GradScaler()  # automatic mixed precision (amp)

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))  # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(),
                                         config.clip_norm)  # grad norm clipping prevents gradient exploding

        scaler.step(optimizer)
        scaler.update()

        # logging
        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats


"""## Validation & Inference
To prevent overfitting, validation is required every epoch to validate the performance on unseen data.
- the procedure is essensially same as training, with the addition of inference step
- after validation we can save the model weights

Validation loss alone cannot describe the actual performance of the model
- Directly produce translation hypotheses based on current model, then calculate BLEU with the reference translation
- We can also manually examine the hypotheses' quality
- We use fairseq's sequence generator for beam search to generate translation hypotheses
"""


def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"


def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"],  # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)

            # do inference
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs

    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)

    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])

    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats


"""# Save and Load Model Weights

"""


def validate_and_save(model, task, criterion, optimizer, epoch, save=True):
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir / f"checkpoint{epoch}.pt")
        shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        # save epoch samples
        with open(savedir / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir / f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats


def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        # seed=seed,
        seed=73,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator


"""# Main
## Training loop
"""
if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

    """# CUDA Environments"""

    cuda_env = utils.CudaEnvironment()
    utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = "cuda"

    ## setup task
    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)

    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1, combine=True)  # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    sample = task.dataset("valid")[1]
    pprint.pprint(sample)
    pprint.pprint(
        "Source: " + \
        task.source_dictionary.string(
            sample['source'],
            config.post_process,
        )
    )
    pprint.pprint(
        "Target: " + \
        task.target_dictionary.string(
            sample['target'],
            config.post_process,
        )
    )

    if config.use_wandb:
        wandb.config.update(vars(arch_args))

    model = build_model(arch_args, task)
    logger.info(model)

    # generally, 0.1 is good enough
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    """## Scheduling Visualized"""

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim,
        factor=config.lr_factor,
        warmup=config.lr_warmup,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
    plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
    plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    # fairseq's beam search generator
    # given model and input seqeunce, produce translation hypotheses by beam search
    sequence_generator = task.build_generator([model], config)

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    try_load_checkpoint(model, optimizer, name=config.resume)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
