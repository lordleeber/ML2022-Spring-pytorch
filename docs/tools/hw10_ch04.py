# ch04 facts: the I-FGSM trajectory (resnet110, step 0.8, 100 steps): after selected steps, white-box and victim
# accuracy of the PNG-equivalent images (clamped to 0-255 and rounded), loss, how many pixel values sit on the eps boundary.
# Plus two variants at 20 steps: clamp to the valid pixel range every step; random start in the eps ball (PGD, 3 seeds).
# usage (from HW10/, deterministic): CUBLAS_WORKSPACE_CONFIG=:4096:8 python ../docs/tools/hw10_det.py ../docs/tools/hw10_ch04.py
import os
import sys

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root, mean, std, epsilon
from dataset import AdvDataset, transform
from attack import alpha

V = ['wrn28_10_cifar10', 'wrn40_8_cifar10', 'pyramidnet110_a48_cifar10', 'resnext29_32x4d_cifar10', 'ror3_110_cifar10',
     'rir_cifar10', 'shakeshakeresnet26_2x32d_cifar10', 'diaresnet56_cifar10']
m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device).eval()
vic = [ptcv_get_model(v, pretrained=True).to(device).eval() for v in V]
ds = AdvDataset(root, transform=transform)
X = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
Y = torch.tensor(ds.labels, device=device)
ce = nn.CrossEntropyLoss()
lo, hi = (0 - mean) / std, (1 - mean) / std   # valid pixel range in normalized units, per channel


def png(xa):
  # what create_dir would save, back in normalized units
  p = (((xa * std + mean).clamp(0, 1) * 255).round()) / 255
  return (p - mean) / std


@torch.no_grad()
def evaluate(xa):
  xp = png(xa)
  w = (m(xp).argmax(1) == Y).float().mean().item()
  outs = [v(xp) for v in vic]
  accs = [(o.argmax(1) == Y).float().mean().item() for o in outs]
  ens = (sum(outs).argmax(1) == Y).float().mean().item()
  vloss = sum(nn.functional.cross_entropy(o, Y).item() for o in outs) / len(outs)
  d = ((xp - X) * std * 255).abs().round()
  return w, sum(accs) / len(accs), ens, vloss, (d == 8).float().mean().item()


def run(steps, record, clamp_valid=False, rand_start=None):
  xa_all, rec = [], {}
  for i in range(0, 200, batch_size):
    x, y = X[i:i + batch_size], Y[i:i + batch_size]
    xa = x
    if rand_start is not None:
      g = torch.Generator(device='cpu').manual_seed(rand_start * 1000 + i)
      xa = x + (torch.rand(x.shape, generator=g).to(device) * 2 - 1) * epsilon
    traj = []
    for t in range(1, steps + 1):
      xa = xa.detach().clone(); xa.requires_grad = True
      loss = ce(m(xa), y); loss.backward()
      xa = xa + alpha * xa.grad.detach().sign()
      xa = torch.max(torch.min(xa, x + epsilon), x - epsilon)
      if clamp_valid:
        xa = torch.max(torch.min(xa, hi), lo)
      if t in record:
        traj.append(xa.detach())
    xa_all.append(traj)
  for k, t in enumerate(record):
    rec[t] = evaluate(torch.cat([tr[k] for tr in xa_all]))
  return rec


steps = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70, 100]
print('step white victims8 victim_ens victims_loss frac_at_eps')
with torch.no_grad():
  w0 = evaluate(X)
print(0, *[f'{v:.4f}' for v in w0])
for t, v in run(100, steps).items():
  print(t, *[f'{u:.4f}' for u in v])
print('clamp to valid range every step, 20 steps:', *[f'{u:.4f}' for u in run(20, [20], clamp_valid=True)[20]])
for s in range(3):
  print(f'random start seed {s}, 20 steps:', *[f'{u:.4f}' for u in run(20, [20], rand_start=s)[20]])
