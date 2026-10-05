"""Replicates HW04 train.py with switchable variants.

Run from HW04/ with PYTHONPATH=. so dataset/classifier/train import from the repo:
    cd HW04 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw04_exp.py --name orig
Prints the same "Step N, best model saved" lines as train.py, one line per validation, and one
JSON line at the end. Nothing is written to disk unless --save / --save_live / --dump is given,
so HW04/model.ckpt and output.csv are left alone.

`import train` runs train.py's module level, i.e. set_seed(87), and gives the original
get_dataloader, collate_batch, get_cosine_schedule_with_warmup and model_fn, so the defaults are
train.py itself: Classifier (d_model 80, 1 encoder layer, nhead 2, ffn 256), batch 32, 8 workers,
AdamW lr 1e-3, warmup 1000, 70000 steps, validation every 2000 steps. The RNG order follows
train.py main(): set_seed -> random_split -> iter(train_loader) (base seed + sampler permutation,
drawn at once because the workers prefetch) -> model init -> loop; every validation draws one base
seed for its worker iterator and every exhausted train iterator is rebuilt. The tqdm bars of
train.py are plain `from tqdm import tqdm` and never wrap a loader, so leaving them out changes no
RNG draw. The extra bookkeeping below draws nothing from any RNG during training.

What train.py keeps as "best" is `model.state_dict()` without a copy (train.py:297): references to
the live parameters, so every save writes the weights of that moment. This tool keeps both:
  live  = what train.py saves at the last save step (bit-identical to its model.ckpt)
  best  = copy.deepcopy at the best validation (what the comment "keep the best model" means)

Variants (all default to train.py):
  --d_model --nhead --ffn --layers --dropout   transformer hyper-parameters (layers>1 uses
                                               nn.TransformerEncoder like the commented line 41)
  --norm_first 1                               pre-norm encoder layers (LayerNorm before attention/FFN)
  --arch conformer                             Conformer blocks (FFN/2, MHSA, conv module, FFN/2, LN)
  --kernel 31                                  depthwise conv kernel of the conformer conv module
  --pool sap                                   self-attention pooling instead of mean pooling
  --loss amsm --m 0.2 --s 30                   additive margin softmax (model outputs s*cos)
  --steps --warmup --lr --bs --seg             schedule, batch size, segment length
  --snap_dir DIR                               save the weights at every validation (no RNG draws)
"""
import argparse, copy, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import train  # runs set_seed(87) exactly like `python train.py`
from classifier import Classifier
from dataset import myDataset

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--arch', default='transformer')   # transformer (classifier.py) | conformer
ap.add_argument('--d_model', type=int, default=80)
ap.add_argument('--nhead', type=int, default=2)
ap.add_argument('--ffn', type=int, default=256)
ap.add_argument('--layers', type=int, default=1)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--norm_first', type=int, default=0)  # 1: pre-norm TransformerEncoderLayer
ap.add_argument('--kernel', type=int, default=31)
ap.add_argument('--pool', default='mean')          # mean (classifier.py) | sap
ap.add_argument('--loss', default='ce')            # ce (train.py) | amsm
ap.add_argument('--m', type=float, default=0.2)
ap.add_argument('--s', type=float, default=30.0)
ap.add_argument('--steps', type=int, default=70000)
ap.add_argument('--warmup', type=int, default=1000)
ap.add_argument('--valid_steps', type=int, default=2000)
ap.add_argument('--save_steps', type=int, default=10000)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seg', type=int, default=128)
ap.add_argument('--save', default='')        # deep-copied best state_dict
ap.add_argument('--save_live', default='')   # what train.py writes to model.ckpt
ap.add_argument('--dump', default='')        # .npz: full-length logits of best model on valid + test
ap.add_argument('--snap_dir', default='')    # save a copy of the weights at every validation (<dir>/step_<N>.pt)
a = ap.parse_args()
dev = 'cuda'
t0 = time.time()
DATA = './Dataset'
is_orig_model = (a.arch == 'transformer' and a.pool == 'mean' and a.loss == 'ce' and a.layers == 1 and not a.norm_first
                 and (a.d_model, a.nhead, a.ffn, a.dropout) == (80, 2, 256, 0.1))


# ---------------------------------------------------------------- models
class SAP(nn.Module):
    """Self-attention pooling: one score per frame, softmax over time, weighted sum."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):                       # x: (B, T, d)
        att = torch.softmax(self.w(x).squeeze(-1), dim=1)   # (B, T)
        return (att.unsqueeze(-1) * x).sum(dim=1)            # (B, d)


class AMHead(nn.Module):
    """Cosine classifier for additive margin softmax: outputs s * cos(theta); margin is in the loss."""
    def __init__(self, d, n, s):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, d) * 0.01)
        self.s = s

    def forward(self, x):
        return self.s * F.linear(F.normalize(x, dim=1), F.normalize(self.W, dim=1))


class AMSoftmaxLoss(nn.Module):
    def __init__(self, s, m):
        super().__init__()
        self.s, self.m = s, m

    def forward(self, logits, labels):          # logits = s*cos; subtract s*m at the target class
        return F.cross_entropy(logits - self.s * self.m * F.one_hot(labels, logits.size(1)), labels)


class ConvModule(nn.Module):
    """Conformer convolution module (slides p.15): LN, pointwise, GLU, depthwise, BN, Swish, pointwise, dropout."""
    def __init__(self, d, k, p):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):                       # (B, T, d)
        y = self.ln(x).transpose(1, 2)          # (B, d, T)
        y = F.glu(self.pw1(y), dim=1)
        y = F.silu(self.bn(self.dw(y)))
        y = self.drop(self.pw2(y)).transpose(1, 2)
        return y


class FFModule(nn.Module):
    def __init__(self, d, ff, p):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(p),
                                 nn.Linear(ff, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, d, h, ff, k, p):
        super().__init__()
        self.ff1 = FFModule(d, ff, p)
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.conv = ConvModule(d, k, p)
        self.ff2 = FFModule(d, ff, p)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        y = self.ln_att(x)
        x = x + self.drop(self.att(y, y, y, need_weights=False)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class Net(nn.Module):
    """Same skeleton as classifier.py (prenet -> encoder -> pooling -> pred_layer), with switches."""
    def __init__(self, n_spks):
        super().__init__()
        d = a.d_model
        self.prenet = nn.Linear(40, d)
        if a.arch == 'conformer':
            self.blocks = nn.ModuleList([ConformerBlock(d, a.nhead, a.ffn, a.kernel, a.dropout)
                                         for _ in range(a.layers)])
        else:
            layer = nn.TransformerEncoderLayer(d_model=d, dim_feedforward=a.ffn, nhead=a.nhead,
                                               dropout=a.dropout, norm_first=bool(a.norm_first))
            self.encoder = layer if a.layers == 1 else nn.TransformerEncoder(layer, num_layers=a.layers)
        self.pool = SAP(d) if a.pool == 'sap' else None
        if a.loss == 'amsm':
            self.pred_layer = nn.Sequential(nn.Linear(d, d), nn.ReLU(), AMHead(d, n_spks, a.s))
        else:
            self.pred_layer = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_spks))

    def forward(self, mels):
        out = self.prenet(mels)
        if a.arch == 'conformer':
            for b in self.blocks:
                out = b(out)
        else:
            out = self.encoder(out.permute(1, 0, 2)).transpose(0, 1)
        stats = self.pool(out) if self.pool is not None else out.mean(dim=1)
        return self.pred_layer(stats)


# ---------------------------------------------------------------- data (train.py:247-249)
def get_dataloader(data_dir, batch_size, n_workers):
    if a.seg == 128:
        return train.get_dataloader(data_dir, batch_size, n_workers)
    # same as train.get_dataloader, only the segment length differs
    dataset = myDataset(data_dir, segment_len=a.seg)
    trainlen = int(0.9 * len(dataset))
    trainset, validset = torch.utils.data.random_split(dataset, [trainlen, len(dataset) - trainlen])
    kw = dict(batch_size=batch_size, num_workers=n_workers, drop_last=True, pin_memory=True,
              collate_fn=train.collate_batch)
    return (DataLoader(trainset, shuffle=True, **kw), DataLoader(validset, **kw),
            dataset.get_speaker_number())


train_loader, valid_loader, speaker_num = get_dataloader(DATA, a.bs, a.workers)
train_iterator = iter(train_loader)

model = (Classifier(n_spks=speaker_num) if is_orig_model else Net(speaker_num)).to(dev)
nparams = sum(p.numel() for p in model.parameters())
criterion = AMSoftmaxLoss(a.s, a.m) if a.loss == 'amsm' else nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=a.lr)
scheduler = train.get_cosine_schedule_with_warmup(optimizer, a.warmup, a.steps)
print(f"[Info]: {a.name}: {nparams} parameters", flush=True)


def valid_like_train(dataloader):
    """train.py valid(): mean of per-batch accuracies; plus the exact count over the same batches."""
    model.eval()
    accs, losses, n_ok, n = 0.0, 0.0, 0, 0
    for batch in dataloader:
        with torch.no_grad():
            mels, labels = batch[0].to(dev), batch[1].to(dev)
            outs = model(mels)
            accs += (outs.argmax(1) == labels).float().mean().item()
            losses += criterion(outs, labels).item()
            n_ok += (outs.argmax(1) == labels).sum().item()
            n += labels.numel()
    model.train()
    return accs / len(dataloader), losses / len(dataloader), n_ok / n


best_accuracy, best_step, best_sd = -1.0, 0, None
live_sd_saved, live_step_saved, live_best_print = None, 0, None
hist, run_loss, run_acc = [], 0.0, 0.0
te = time.time()
for step in range(a.steps):
    try:
        batch = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

    loss, accuracy = train.model_fn(batch, model, criterion, dev)
    batch_loss = loss.item()
    batch_accuracy = accuracy.item()
    run_loss += batch_loss
    run_acc += batch_accuracy

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if (step + 1) % a.valid_steps == 0:
        va, vl, va_exact = valid_like_train(valid_loader)
        row = dict(step=step + 1, valid_acc=round(va, 5), valid_loss=round(vl, 5), valid_exact=round(va_exact, 5),
                   train_loss=round(run_loss / a.valid_steps, 5), train_acc=round(run_acc / a.valid_steps, 5),
                   last_batch_loss=round(batch_loss, 4), last_batch_acc=round(batch_accuracy, 4),
                   lr=scheduler.get_last_lr()[0], secs=round(time.time() - te, 1))
        hist.append(row)
        run_loss, run_acc = 0.0, 0.0
        print(f"valid step {step + 1}: acc={va:.4f} loss={vl:.4f} exact={va_exact:.5f} "
              f"train_loss={row['train_loss']:.4f} lr={row['lr']:.3e}", flush=True)
        if va > best_accuracy:
            best_accuracy, best_step = va, step + 1
            best_sd = copy.deepcopy(model.state_dict())
        if a.snap_dir:
            os.makedirs(a.snap_dir, exist_ok=True)
            torch.save(copy.deepcopy(model.state_dict()), os.path.join(a.snap_dir, f'step_{step + 1}.pt'))

    if (step + 1) % a.save_steps == 0 and best_sd is not None:
        # train.py:303 saves model.state_dict() as it is right now (the references it kept)
        live_sd_saved = copy.deepcopy(model.state_dict())
        live_step_saved, live_best_print = step + 1, best_accuracy
        print(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})", flush=True)
train_secs = time.time() - te


# ---------------------------------------------------------------- evaluation after training
def eval_state(sd):
    """Accuracy of a state on the 5,667 validation utterances: full length (like test.py, batch 1),
    and a fixed 128-frame crop (main-process python RNG with its own seed, no workers)."""
    model.load_state_dict(sd)
    model.eval()
    vs = valid_loader.dataset
    full_ok, crop_ok, ce_full = 0, 0, 0.0
    rng = random.Random(0)
    with torch.no_grad():
        for i in range(len(vs)):
            feat_path, spk = vs.dataset.data[vs.indices[i]]
            mel = torch.load(os.path.join(DATA, feat_path))
            y = torch.tensor([spk], device=dev)
            out = model(mel.unsqueeze(0).to(dev))
            full_ok += int(out.argmax(1).item() == spk)
            ce_full += F.cross_entropy(out, y).item() if a.loss == 'ce' else 0.0
            if len(mel) > a.seg:
                st = rng.randint(0, len(mel) - a.seg)
                mel = mel[st:st + a.seg]
            crop_ok += int(model(mel.unsqueeze(0).to(dev)).argmax(1).item() == spk)
    n = len(vs)
    return dict(full=round(full_ok / n, 5), crop=round(crop_ok / n, 5),
                ce_full=round(ce_full / n, 5) if a.loss == 'ce' else None)


res = dict(name=a.name, args=vars(a), nparams=nparams, train_secs=round(train_secs, 1),
           best_printed=round(best_accuracy, 5), best_step=best_step,
           live_step=live_step_saved, live_printed=round(live_best_print, 5) if live_best_print else None)
res['best_eval'] = eval_state(best_sd)
if live_sd_saved is not None:
    res['live_eval'] = eval_state(live_sd_saved)
if a.save:
    torch.save(best_sd, a.save)
if a.save_live and live_sd_saved is not None:
    torch.save(live_sd_saved, a.save_live)
if a.dump:
    import json as _j
    model.load_state_dict(best_sd)
    model.eval()
    vs = valid_loader.dataset
    test = _j.load(open(os.path.join(DATA, 'testdata.json')))['utterances']
    with torch.no_grad():
        lv = np.stack([model(torch.load(os.path.join(DATA, vs.dataset.data[vs.indices[i]][0])).unsqueeze(0).to(dev))
                       .cpu().numpy()[0] for i in range(len(vs))])
        yv = np.array([vs.dataset.data[vs.indices[i]][1] for i in range(len(vs))])
        lt = np.stack([model(torch.load(os.path.join(DATA, u['feature_path'])).unsqueeze(0).to(dev)).cpu().numpy()[0]
                       for u in test])
    np.savez_compressed(a.dump, valid_logits=lv, valid_y=yv, test_logits=lt,
                        test_paths=np.array([u['feature_path'] for u in test]))
res['hist'] = hist
res['total_secs'] = round(time.time() - t0, 1)
print(json.dumps(res), flush=True)
