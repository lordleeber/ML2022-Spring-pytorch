# HW13 ch06: why masking whole channels collapses the teacher. ln_structured (L2, dim=0) on every conv, then
# (a) leave BatchNorm alone, (b) also zero gamma/beta of the BatchNorm that follows each pruned channel,
# (c) choose channels by BatchNorm |gamma| instead of the conv weight norm.
# usage (from HW13/): python ../docs/tools/hw13_prune_bn.py      Inference only.
import io, sys
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_teacher_model
with redirect_stdout(io.StringIO()):
  ds = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
vb = [(x, y) for x, y in DataLoader(ds, batch_size=128, num_workers=8)]

@torch.no_grad()
def acc(m):
  m.cuda().eval()
  return round(sum((m(x.cuda()).argmax(-1).cpu() == y).sum().item() for x, y in vb) / len(ds), 5)

def conv_bn_pairs(m):
  mods = list(m.named_modules())
  pairs = []
  for (n1, a), (n2, b) in zip(mods, mods[1:]):
    if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
      pairs.append((a, b))
  return pairs

for r in [0.1, 0.2, 0.3, 0.5]:
  out = {}
  for fix_bn in [False, True]:
    m = get_teacher_model(cfg['dataset_root']).eval()
    pairs = conv_bn_pairs(m)
    consts = []
    for conv, bn in pairs:
      prune.ln_structured(conv, 'weight', amount=r, n=2, dim=0)
      dead = conv.weight_mask.flatten(1).sum(1) == 0
      # what a dead channel outputs after BN (input is exactly 0): beta - gamma * mean / sqrt(var + eps)
      c = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
      consts.append(c[dead].abs())
      if fix_bn:
        with torch.no_grad():
          bn.weight[dead] = 0; bn.bias[dead] = 0
    out[fix_bn] = acc(m)
    if not fix_bn:
      cc = torch.cat(consts)
      stat = f'pairs {len(pairs)}  dead channels {len(cc)}  |BN output of a dead channel| mean {cc.mean().item():.3f} max {cc.max().item():.3f}'
  print(f'ratio {r}: BN untouched {out[False]}  BN zeroed {out[True]}  ({stat})', flush=True)

# (c) choose the channels by the BatchNorm scale |gamma| (network slimming) instead of the conv weight norm;
#     zero the conv channel and its gamma/beta, so the channel really outputs 0
for r in [0.1, 0.2, 0.3, 0.5]:
  m = get_teacher_model(cfg['dataset_root']).eval()
  with torch.no_grad():
    for conv, bn in conv_bn_pairs(m):
      k = int(round(r * bn.num_features))
      dead = bn.weight.abs().argsort()[:k]
      conv.weight[dead] = 0; bn.weight[dead] = 0; bn.bias[dead] = 0
  print(f'ratio {r}: channels chosen by |gamma|, BN zeroed {acc(m)}', flush=True)
import os; sys.stdout.flush(); os._exit(0)
