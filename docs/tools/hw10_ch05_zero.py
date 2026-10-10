# ch05: why the 16-model logits-sum ensemble left most images untouched. For the first I-FGSM step of ens_k16_d0,
# count images whose input gradient is exactly zero, and show the logit margin / softmax probability of the true class,
# for the sum of logits vs the mean of logits.
# usage (from HW10/, deterministic): CUBLAS_WORKSPACE_CONFIG=:4096:8 python ../docs/tools/hw10_det.py ../docs/tools/hw10_ch05_zero.py
import json
import os
import sys

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root
from dataset import AdvDataset, transform

names = [json.loads(l) for l in open('../docs/tools/hw10_runs.jsonl') if '"tag": "ens_k16_d0"' in l][0]['surrogates']
ms = [ptcv_get_model(n, pretrained=True).to(device).eval() for n in names]
ds = AdvDataset(root, transform=transform)
X = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
Y = torch.tensor(ds.labels, device=device)
for mode in ('sum', 'mean'):
  zero, margins, pmax = 0, [], []
  for i in range(0, 200, batch_size):
    x = X[i:i + batch_size].clone().requires_grad_(True)
    out = sum(m(x) for m in ms)
    if mode == 'mean':
      out = out / len(ms)
    nn.CrossEntropyLoss()(out, Y[i:i + batch_size]).backward()
    g = x.grad.detach()
    zero += int((g.flatten(1).abs().sum(1) == 0).sum())
    o = out.detach()
    top2 = o.topk(2, 1).values
    margins += (top2[:, 0] - top2[:, 1]).tolist()
    pmax += o.softmax(1).max(1).values.tolist()
  margins.sort()
  print(mode, 'images with all-zero input gradient:', zero, '| median top1-top2 logit margin', round(margins[100], 1),
        '| images with max softmax prob == 1.0 in float32:', sum(p == 1.0 for p in pmax))
