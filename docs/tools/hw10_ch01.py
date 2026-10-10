# ch01 facts: where FGSM's |adv - benign| < 8 comes from (zero gradient vs clamping at 0/255),
# printed (float) vs saved-PNG accuracy over all grid runs, and what epsilon = 8/255 (without /std) would be.
# usage (from HW10/, deterministic): CUBLAS_WORKSPACE_CONFIG=:4096:8 python ../docs/tools/hw10_det.py ../docs/tools/hw10_ch01.py
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root, mean, std, epsilon
from dataset import AdvDataset, transform

m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device).eval()
loss_fn = nn.CrossEntropyLoss()
ds = AdvDataset(root, transform=transform)
loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
zero_grad = sat = partial = full = 0
for x, y in loader:
  x, y = x.to(device), y.to(device)
  xa = x.detach().clone(); xa.requires_grad = True
  loss_fn(m(xa), y).backward()
  sg = xa.grad.sign()
  orig = ((x * std + mean) * 255).round()
  new = (orig + 8 * sg).clamp(0, 255)
  d = (new - orig).abs()
  zero_grad += (sg == 0).sum().item()
  sat += ((d == 0) & (sg != 0)).sum().item()
  partial += ((d > 0) & (d < 8)).sum().item()
  full += (d == 8).sum().item()
print('FGSM pixels: grad exactly 0', zero_grad, '| clamped to 0 change', sat, '| clamped partially (1-7)', partial, '| full 8', full)
# benign pixel values at the boundary
allpix = np.concatenate([np.asarray(__import__('PIL.Image', fromlist=['Image']).open(f)).ravel() for f in ds.images])
print('benign pixels == 0:', int((allpix == 0).sum()), '== 255:', int((allpix == 255).sum()), 'in [0,7]:', int((allpix <= 7).sum()), 'in [248,255]:', int((allpix >= 248).sum()), 'total', allpix.size)
# round trip benign -> transform -> back
back = []
for x, _ in loader:
  back.append(((x.to(device) * std + mean) * 255).cpu().numpy())
back = np.concatenate(back).transpose(0, 2, 3, 1).ravel()
print('round trip max |float - int| =', float(np.abs(back - allpix).max()), 'exact after round:', bool((np.round(back) == allpix).all()))
# eps without /std, in pixel units
print('epsilon if written 8/255 (normalized units) = pixel', [round(8 * s, 3) for s in std.flatten().tolist()])
# printed vs PNG accuracy over all grid runs
diffs = []
for l in open('../docs/tools/hw10_runs.jsonl'):
  r = json.loads(l)
  if len(r['surrogates']) == 1 and r['attack'] != 'none':
    s = r['surrogates'][0]
    diffs.append((round(r['acc'][s] - r['printed_acc'], 3), r['tag']))
nz = [d for d in diffs if d[0] != 0]
print('runs', len(diffs), 'printed != png', len(nz), 'max |diff|', max(abs(d[0]) for d in diffs), 'png higher', sum(d[0] > 0 for d in nz), 'png lower', sum(d[0] < 0 for d in nz))
print(sorted(nz, key=lambda d: -abs(d[0]))[:8])
