# Train a pytorchcv CIFAR-10 architecture from scratch on the CIFAR-10 training set and save checkpoints
# along the way, to test paper B (Query-Free Adversarial Transfer via Undertrained Surrogates):
# an early checkpoint can be a better attack surrogate than the fully trained model.
# Recipe: SGD lr 0.1, momentum 0.9, wd 5e-4, batch 128, random crop 4 + flip, step decay x0.1 at 50% and 75%.
# Inputs use the same Normalize(cifar_10_mean, cifar_10_std) as HW10, so the checkpoints plug into hw10_exp.py.
# The 200 HW10 images are removed from the training set first (see --exclude).
# usage (from HW10/): python ../docs/tools/hw10_train_surrogate.py --arch resnet20_cifar10 --epochs 60 --out_dir D --save 1,2,5,...
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, cifar_10_mean, cifar_10_std

p = argparse.ArgumentParser()
p.add_argument('--arch', default='resnet20_cifar10')
p.add_argument('--epochs', type=int, default=60)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--save', default='1,2,3,5,10,15,20,30,40,45,60')
p.add_argument('--exclude', default='../docs/tools/hw10_overlap.json')
p.add_argument('--out_dir', required=True)
p.add_argument('--jsonl', required=True)
opt = p.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
norm = transforms.Normalize(cifar_10_mean, cifar_10_std)
train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), norm])
test_tf = transforms.Compose([transforms.ToTensor(), norm])
train = torchvision.datasets.CIFAR10('cifar10', train=True, transform=train_tf)
test = torchvision.datasets.CIFAR10('cifar10', train=False, transform=test_tf)
excl = set(json.load(open(opt.exclude)).get('train', [])) if os.path.exists(opt.exclude) else set()
train = Subset(train, [i for i in range(len(train)) if i not in excl])
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
test_loader = DataLoader(test, batch_size=500, shuffle=False, num_workers=4)

model = ptcv_get_model(opt.arch, pretrained=False).to(device)
optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(optim, [opt.epochs // 2, opt.epochs * 3 // 4], 0.1)
loss_fn = nn.CrossEntropyLoss()
save = {int(e) for e in opt.save.split(',')}
os.makedirs(opt.out_dir, exist_ok=True)
t0 = time.time()
for epoch in range(1, opt.epochs + 1):
  model.train()
  tl, tc, n = 0.0, 0, 0
  for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = loss_fn(out, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
  sched.step()
  if epoch in save:
    model.eval()
    c = 0
    with torch.no_grad():
      for x, y in test_loader:
        c += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    path = os.path.join(opt.out_dir, f'{opt.arch}_s{opt.seed}_e{epoch}.pth')
    torch.save(model.state_dict(), path)
    rec = {'arch': opt.arch, 'seed': opt.seed, 'epoch': epoch, 'train_loss': tl / n, 'train_acc': tc / n,
           'test_acc': c / len(test), 'path': path, 'sec': round(time.time() - t0, 1)}
    print(json.dumps(rec), flush=True)
    with open(opt.jsonl, 'a') as f:
      f.write(json.dumps(rec) + '\n')
sys.stdout.flush()
os._exit(0)
