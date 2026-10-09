"""Replicates HW13 train.py with switchable variants.

Run from HW13/ (imports config/dataset/model/kd from the repo):
    cd HW13 && ../.venv/bin/python ../docs/tools/hw13_exp.py --name ref
Prints the same Train/Valid lines as train.py and appends one JSON line to --jsonl at the end.
With the defaults (CE, sample student, sample augmentation, 10 epochs, --nw 0) the run is
bit-identical to train.py: same seeds, then datasets/loaders -> student -> torchsummary ->
teacher (its weight init draws from the global RNG) -> optimizer -> epochs. Both loops are wrapped
in tqdm (disabled) because `from tqdm.auto import tqdm` calls iter(loader) once more per loop
(one extra RNG draw), as in train.py. Extra bookkeeping (CE on the validation set, timing) adds
no RNG draws.

Variants:
  --loss KD --alpha A --T T   loss_fn_kd from kd.py (the teacher runs on every batch)
  --student NAME              a student from hw13_students.py (default: sample = model.py)
  --aug hw03                  stronger train_tfm (HW03's augmentation A at 224, plus normalize)
  --sched cos                 cosine learning-rate decay over all epochs (default: constant)
  --nw N                      DataLoader workers (changes the RNG stream: compare runs with equal N)
"""
import argparse, io, json, os, random, sys, time
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm.auto import tqdm

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import FoodDataset, train_tfm, test_tfm, normalize
from model import get_teacher_model
from kd import loss_fn_kd
from hw13_students import STUDENTS

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--loss', default='CE', choices=['CE', 'KD'])
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--T', type=float, default=1.0)
ap.add_argument('--student', default='sample')
ap.add_argument('--aug', default='sample', choices=['sample', 'hw03'])
ap.add_argument('--sched', default='none', choices=['none', 'cos'])
ap.add_argument('--epochs', type=int, default=cfg['n_epochs'])
ap.add_argument('--lr', type=float, default=cfg['lr'])
ap.add_argument('--seed', type=int, default=cfg['seed'])
ap.add_argument('--nw', type=int, default=0)
ap.add_argument('--save', default='')   # directory for best.ckpt and best_valid_logits.pt
ap.add_argument('--jsonl', default='')
a = ap.parse_args()
device = 'cuda'
t_start = time.time()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(a.seed)
torch.manual_seed(a.seed)
random.seed(a.seed)
torch.cuda.manual_seed_all(a.seed)

AUG = {
    'sample': train_tfm,
    'hw03': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        normalize,
    ]),
}

with redirect_stdout(io.StringIO()):
    train_set = FoodDataset(os.path.join(cfg['dataset_root'], "training"), tfm=AUG[a.aug])
    valid_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
pw = a.nw > 0
train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=a.nw, pin_memory=True, persistent_workers=pw)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=a.nw, pin_memory=True, persistent_workers=pw)

student = STUDENTS[a.student]()
with redirect_stdout(io.StringIO()) as buf:
    summary(student, (3, 224, 224), device='cpu')
total_params = [l for l in buf.getvalue().splitlines() if l.startswith('Total params')][0]
n_params = sum(p.numel() for p in student.parameters())
assert n_params <= 100_000, n_params
teacher = get_teacher_model(cfg['dataset_root'])

use_kd = a.loss == 'KD'
if use_kd:
    loss_fn = lambda s, y, t: loss_fn_kd(s, y, t, alpha=a.alpha, temperature=a.T)
else:
    loss_fn = nn.CrossEntropyLoss()

student.to(device)
if use_kd:
    teacher.to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=a.lr, weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs) if a.sched == 'cos' else None
if use_kd:
    teacher.eval()

print(f"{a.name}: {total_params}", flush=True)
best_acc, best_epoch, curve = 0.0, 0, []
for epoch in range(a.epochs):
    t0 = time.time()
    student.train()
    train_loss, train_accs, train_lens = [], [], []
    for imgs, labels in tqdm(train_loader, disable=True):
        imgs, labels = imgs.to(device), labels.to(device)
        if use_kd:
            with torch.no_grad():
                teacher_logits = teacher(imgs)
        logits = student(imgs)
        loss = loss_fn(logits, labels, teacher_logits) if use_kd else loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=cfg['grad_norm_max'])
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().sum()
        train_loss.append(loss.item() * len(imgs))
        train_accs.append(acc)
        train_lens.append(len(imgs))
    if scheduler is not None:
        scheduler.step()
    train_loss = sum(train_loss) / sum(train_lens)
    train_acc = sum(train_accs) / sum(train_lens)
    t1 = time.time()

    student.eval()
    valid_loss, valid_accs, valid_lens, ce_sum = [], [], [], 0.0
    logits_all = []
    for imgs, labels in tqdm(valid_loader, disable=True):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = student(imgs)
            if use_kd:
                teacher_logits = teacher(imgs)
        loss = loss_fn(logits, labels, teacher_logits) if use_kd else loss_fn(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().sum()
        valid_loss.append(loss.item() * len(imgs))
        valid_accs.append(acc)
        valid_lens.append(len(imgs))
        ce_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        logits_all.append(logits.cpu())
    valid_loss = sum(valid_loss) / sum(valid_lens)
    valid_acc = sum(valid_accs) / sum(valid_lens)
    n_valid = sum(valid_lens)

    print(f"[ Train | {epoch + 1:03d}/{a.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    best = valid_acc > best_acc
    print(f"[ Valid | {epoch + 1:03d}/{a.epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}" + (" -> best" if best else ""), flush=True)
    curve.append(dict(epoch=epoch + 1, train_loss=round(train_loss, 5), train_acc=round(train_acc.item(), 5),
                      valid_loss=round(valid_loss, 5), valid_acc=round(valid_acc.item(), 5),
                      valid_ce=round(ce_sum / n_valid, 5), train_secs=round(t1 - t0, 1), valid_secs=round(time.time() - t1, 1)))
    if best:
        best_acc, best_epoch = valid_acc, epoch + 1
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            torch.save(student.state_dict(), os.path.join(a.save, 'best.ckpt'))
            torch.save(torch.cat(logits_all), os.path.join(a.save, 'best_valid_logits.pt'))

rec = dict(name=a.name, loss=a.loss, alpha=a.alpha, T=a.T, student=a.student, aug=a.aug, sched=a.sched,
           epochs=a.epochs, lr=a.lr, seed=a.seed, nw=a.nw, params=n_params, best_acc=round(best_acc.item(), 5),
           best_epoch=best_epoch, total_secs=round(time.time() - t_start, 1), curve=curve)
print(f"{a.name}: best valid acc {best_acc:.5f} at epoch {best_epoch}, {rec['total_secs']} s")
if a.jsonl:
    with open(a.jsonl, 'a') as f:
        f.write(json.dumps(rec) + '\n')
