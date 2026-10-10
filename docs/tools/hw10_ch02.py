# ch02 facts: loss and accuracy along the FGSM direction vs along random sign directions (resnet110, white box);
# first-order (linear) prediction of the loss change vs the real change; gradient statistics.
# usage (from HW10/, deterministic): CUBLAS_WORKSPACE_CONFIG=:4096:8 python ../docs/tools/hw10_det.py ../docs/tools/hw10_ch02.py
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root, std
from dataset import AdvDataset, transform

m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device).eval()
ds = AdvDataset(root, transform=transform)
X = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
Y = torch.tensor(ds.labels, device=device)
ce = nn.CrossEntropyLoss(reduction='none')
# per-image gradient with batch 8 exactly like fgsm() (mean loss over the batch of 8 -> grad scaled by 1/8, sign unaffected)
G = []
for i in range(0, 200, batch_size):
  x = X[i:i + batch_size].clone().requires_grad_(True)
  nn.CrossEntropyLoss()(m(x), Y[i:i + batch_size]).backward()
  G.append(x.grad.detach())
G = torch.cat(G)
S = G.sign()
g = torch.Generator(device='cpu').manual_seed(0)
R = (torch.randint(0, 2, X.shape, generator=g) * 2 - 1).float().to(device)
print('grad: exactly zero', int((G == 0).sum()), 'of', G.numel())
with torch.no_grad():
  l0 = ce(m(X), Y)
  print('t(pixels)  fgsm_acc  fgsm_loss  rand_acc  rand_loss  linear_pred_loss')
  for t in (0, 1, 2, 4, 8, 16, 32):
    e = t / 255 / std
    lf = ce(m(X + e * S), Y); af = (m(X + e * S).argmax(1) == Y).float().mean().item()
    lr = ce(m(X + e * R), Y); ar = (m(X + e * R).argmax(1) == Y).float().mean().item()
    # first-order Taylor: L(x + d) ~ L(x) + grad . d ; G is the grad of the batch-mean loss, so multiply by 8
    pred = l0 + (G * batch_size * e * S).flatten(1).sum(1)
    print(f'{t:3d} {af:.3f} {lf.mean().item():.3f} {ar:.3f} {lr.mean().item():.3f} {pred.mean().item():.3f}')
  # how many of the 190 correctly classified flip with FGSM eps 8 (float, no clamp)
  e = 8 / 255 / std
  p0 = m(X).argmax(1); p1 = m(X + e * S).argmax(1)
  print('benign correct', int((p0 == Y).sum()), '-> still correct after FGSM8 (float)', int(((p0 == Y) & (p1 == Y)).sum()),
        '; wrong->correct', int(((p0 != Y) & (p1 == Y)).sum()))
  # predicted class of the fooled images
  fooled = (p0 == Y) & (p1 != Y)
  pairs = {}
  for a, b in zip(Y[fooled].tolist(), p1[fooled].tolist()):
    pairs[(a, b)] = pairs.get((a, b), 0) + 1
  print('top fooled (true->pred)', sorted(pairs.items(), key=lambda kv: -kv[1])[:8])
