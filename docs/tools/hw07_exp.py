# HW07 experiment tool. With default arguments it is HW07/train.py, step for step
# (verified bit-identical under docs/tools/hw07_det.py). Options change one thing at a time.
# usage (from a working dir with hw7_*.json, HW07/ on PYTHONPATH):
#   CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw07_det.py docs/tools/hw07_exp.py --tag base_s0 [options]
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from legacy_adamw import AdamW
from dataset import read_data, QA_Dataset
from postprocess import evaluate

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--model', default='bert-base-chinese')
p.add_argument('--tokenizer', default=None, help='default: same as --model')
p.add_argument('--epochs', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--accum', type=int, default=1, help='gradient accumulation steps (effective batch = batch * accum)')
p.add_argument('--decay', choices=['none', 'linear'], default='none')
p.add_argument('--warmup', type=float, default=0.0, help='warmup fraction of total optimizer steps (linear decay only)')
p.add_argument('--optim', choices=['legacy', 'torch'], default='legacy', help='legacy = transformers 4.x AdamW; torch = torch.optim.AdamW defaults')
p.add_argument('--stride', type=int, default=150, help='doc_stride of the dev windows')
p.add_argument('--window', choices=['center', 'random'], default='center', help='where the answer sits in the training window')
p.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='none')
p.add_argument('--zero_shot', action='store_true', help='skip training, only evaluate the pretrained model')
p.add_argument('--save', default=None, help='directory for save_pretrained (default: do not save)')
p.add_argument('--jsonl', default=None, help='append a result line here')
args = p.parse_args()

device = "cuda"


class Exp_Dataset(QA_Dataset):
    # QA_Dataset with a configurable dev stride and training window position
    def __init__(self, *a, stride=150, window='center'):
        super().__init__(*a)
        self.doc_stride = stride
        self.window = window

    def __getitem__(self, idx):
        if self.split != "train" or self.window == 'center':
            return super().__getitem__(idx)
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]
        answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
        answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
        # any window start that keeps the whole answer inside the window (and inside the paragraph)
        lo = max(0, answer_end_token - self.max_paragraph_len + 1)
        hi = max(lo, min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len))
        paragraph_start = random.randint(lo, hi)
        paragraph_end = paragraph_start + self.max_paragraph_len
        input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
        input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]
        answer_start_token += len(input_ids_question) - paragraph_start
        answer_end_token += len(input_ids_question) - paragraph_start
        input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token


# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(args.seed)

t0 = time.time()
model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer or args.model)
n_params = sum(p.numel() for p in model.parameters())

train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)

train_set = Exp_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized, window=args.window)
dev_set = Exp_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized, stride=args.stride)

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)

logging_step = 100
if args.optim == 'legacy':
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
total_updates = (len(train_loader) * args.epochs) // args.accum
scheduler = None
if args.decay == 'linear':
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_updates), total_updates)
amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.amp)
scaler = torch.amp.GradScaler('cuda') if args.amp == 'fp16' else None

def dev_eval(epoch):
    # same as the validation block of train.py
    print("Evaluating Dev Set ...")
    t2 = time.time()
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dev_loader, disable=True)):
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
            dev_acc += evaluate(data, output, tokenizer) == dev_questions[i]["answer_text"]
        print(f"Validation | Epoch {epoch} | acc = {dev_acc / len(dev_loader):.3f}")
    model.train()
    return int(dev_acc), time.time() - t2


model.train()

print("Start Training ...")
history = []
dev_by_epoch = []
t_train = t_dev = 0.0
if args.zero_shot:
    dev_correct, t_dev = dev_eval(0)
    dev_by_epoch.append(round(dev_correct / len(dev_loader), 5))
for epoch in range(0 if args.zero_shot else args.epochs):
    step = 1
    train_loss = train_acc = 0
    t1 = time.time()

    for data in tqdm(train_loader, disable=True):
        data = [i.to(device) for i in data]

        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)

        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss

        loss = output.loss / args.accum if args.accum > 1 else output.loss
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        step += 1

        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            history.append([epoch + 1, step, round(train_loss.item() / logging_step, 4), round(float(train_acc) / logging_step, 4)])
            train_loss = train_acc = 0
    # with accumulation, a last incomplete group of batches is dropped
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_train += time.time() - t1

    dev_correct, t = dev_eval(epoch + 1)
    t_dev += t
    dev_by_epoch.append(round(dev_correct / len(dev_loader), 5))

if args.save:
    print("Saving Model ...")
    model.save_pretrained(args.save)

rec = dict(tag=args.tag, seed=args.seed, model=args.model, epochs=args.epochs, lr=args.lr, batch=args.batch, accum=args.accum,
           decay=args.decay, warmup=args.warmup, optim=args.optim, stride=args.stride, window=args.window, amp=args.amp,
           zero_shot=args.zero_shot, params=n_params, dev_em=dev_by_epoch[-1], dev_correct=dev_correct, dev_by_epoch=dev_by_epoch,
           train_s=round(t_train, 1), dev_s=round(t_dev, 1), total_s=round(time.time() - t0, 1),
           max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), history=history,
           deterministic=torch.are_deterministic_algorithms_enabled())
print('RESULT ' + json.dumps({k: v for k, v in rec.items() if k != 'history'}))
if args.jsonl:
    with open(args.jsonl, 'a') as f:
        f.write(json.dumps(rec) + '\n')
sys.stdout.flush()
