# ch07: evaluate saved PNG folders (from the grid runs) on the 8 victims behind JPEG at several compression rates
# (imgaug convention; quality = round(1 + 99 * (1 - rate/101))). One JSON line per (run, rate).
# usage (from HW10/): python ../docs/tools/hw10_jpeg_sweep.py <runs_dir> <tag>[,<tag>...] <rates> <out.jsonl>
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device
from dataset import AdvDataset, transform

V = ['wrn28_10_cifar10', 'wrn40_8_cifar10', 'pyramidnet110_a48_cifar10', 'resnext29_32x4d_cifar10', 'ror3_110_cifar10',
     'rir_cifar10', 'shakeshakeresnet26_2x32d_cifar10', 'diaresnet56_cifar10']
runs_dir, tags, rates, out = sys.argv[1], sys.argv[2].split(','), [float(r) for r in sys.argv[3].split(',')], sys.argv[4]


def jpeg(arr, rate):
  q = int(np.clip(np.round(1 + 99 * (1.0 - rate / 101)), 1, 100))
  buf = io.BytesIO()
  Image.fromarray(arr).save(buf, format='JPEG', quality=q)
  buf.seek(0)
  return np.array(Image.open(buf).convert('RGB')), q


models = [ptcv_get_model(v, pretrained=True).to(device).eval() for v in V]
for tag in tags:
  d = './data' if tag == 'clean' else os.path.join(runs_dir, tag)
  ds = AdvDataset(d, transform=None)
  imgs = [np.array(Image.open(f).convert('RGB')) for f in ds.images]
  labels = torch.tensor(ds.labels, device=device)
  for r in [0.0] + rates:
    if r == 0:
      X, q = torch.stack([transform(a) for a in imgs]).to(device), None
    else:
      js = [jpeg(a, r) for a in imgs]
      X, q = torch.stack([transform(a) for a, _ in js]).to(device), js[0][1]
    with torch.no_grad():
      outs = [m(X) for m in models]
    accs = {v: (o.argmax(1) == labels).float().mean().item() for v, o in zip(V, outs)}
    rec = {'tag': tag, 'rate': r, 'quality': q, 'victims_mean': sum(accs.values()) / len(V),
           'victim_ens': (sum(outs).argmax(1) == labels).float().mean().item(), 'acc': accs}
    print(json.dumps({k: rec[k] for k in ('tag', 'rate', 'quality', 'victims_mean', 'victim_ens')}), flush=True)
    with open(out, 'a') as f:
      f.write(json.dumps(rec) + '\n')
