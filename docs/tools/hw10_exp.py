# HW10 experiment tool: attack the 200 images with one surrogate or an ensemble, save the PNGs
# exactly like hw10.py, then evaluate the saved PNGs on a local "black-box" pool of models
# (plain, and behind a JPEG pre-processing defence). One JSON line per run.
# With --surrogates resnet110_cifar10 --attack fgsm|ifgsm it writes the same PNGs as hw10.py
# (checked bit for bit under docs/tools/hw10_det.py).
# usage (from HW10/): python ../docs/tools/hw10_exp.py --tag T --surrogates a,b --attack ifgsm --out_dir D --jsonl J
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root, mean, std
import attack as A
from dataset import AdvDataset, transform

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--surrogates', default='resnet110_cifar10')
p.add_argument('--ens', default='logits_sum', choices=['logits_sum', 'logits_mean', 'prob_mean', 'loss_sum'])
p.add_argument('--attack', default='fgsm', choices=['none', 'fgsm', 'ifgsm', 'mifgsm', 'dim_mifgsm', 'dim_ifgsm'])
p.add_argument('--eps', type=float, default=8)       # 0-255 units
p.add_argument('--alpha', type=float, default=0.8)   # 0-255 units
p.add_argument('--iters', type=int, default=20)
p.add_argument('--decay', type=float, default=1.0)
p.add_argument('--dim_p', type=float, default=0.5)
p.add_argument('--dim_max', type=int, default=36)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--victims', default='')
p.add_argument('--jpeg', default='')                 # imgaug compression rates, e.g. 70
p.add_argument('--out_dir', required=True)
p.add_argument('--jsonl', required=True)
opt = p.parse_args()

eps = opt.eps / 255 / std
alpha = opt.alpha / 255 / std
loss_fn = nn.CrossEntropyLoss()


def get_m(name):
  # 'arch@path.pth' = a pytorchcv architecture with our own (undertrained) weights from hw10_train_surrogate.py
  if '@' in name:
    arch, path = name.split('@')
    m = ptcv_get_model(arch, pretrained=False)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m
  return ptcv_get_model(name, pretrained=True)


class Ens(nn.Module):
  def __init__(self, names, mode):
    super().__init__()
    self.models = nn.ModuleList([get_m(n) for n in names])
    self.mode = mode
  def forward(self, x):
    outs = [m(x) for m in self.models]
    if self.mode == 'logits_sum':
      return sum(outs)
    if self.mode == 'logits_mean':
      return sum(outs) / len(outs)
    if self.mode == 'prob_mean':  # log of the averaged probabilities, so CrossEntropyLoss = NLL of the mean prob
      return torch.log(sum(o.softmax(1) for o in outs) / len(outs))
    return torch.stack(outs)      # loss_sum: handled by ens_loss


def ens_loss(out, y):
  if out.dim() == 3:
    return sum(loss_fn(o, y) for o in out)
  return loss_fn(out, y)


def diverse(x):
  # DIM (Xie et al. 2019): with probability p, resize to a random size in [32, dim_max) and zero-pad to dim_max,
  # then resize back to 32 so pytorchcv CIFAR models still see 32x32
  if torch.rand(1).item() >= opt.dim_p:
    return x
  rnd = int(torch.randint(32, opt.dim_max, (1,)).item())
  x2 = F.interpolate(x, size=(rnd, rnd), mode='nearest')
  rem = opt.dim_max - rnd
  top = int(torch.randint(0, rem + 1, (1,)).item())
  left = int(torch.randint(0, rem + 1, (1,)).item())
  x2 = F.pad(x2, (left, rem - left, top, rem - top), value=0)
  return F.interpolate(x2, size=(32, 32), mode='bilinear', align_corners=False)


def gen_iter(model, x, y, momentum_on, dim_on):
  x_adv = x
  momentum = torch.zeros_like(x).detach().to(device)
  for i in range(opt.iters):
    x_adv = x_adv.detach().clone()
    x_adv.requires_grad = True
    inp = diverse(x_adv) if dim_on else x_adv
    loss = ens_loss(model(inp), y)
    loss.backward()
    grad = x_adv.grad.detach()
    if momentum_on:
      grad = opt.decay * momentum + grad / grad.abs().sum(dim=(1, 2, 3), keepdim=True)
      momentum = grad
    x_adv = x_adv + alpha * grad.sign()
    x_adv = torch.max(torch.min(x_adv, x+eps), x-eps)
  return x_adv


def attack_fn(model, x, y, _loss_fn):
  if opt.attack == 'none':
    return x
  if opt.attack == 'fgsm' and opt.ens != 'loss_sum':
    return A.fgsm(model, x, y, loss_fn, eps)
  if opt.attack == 'ifgsm' and opt.ens != 'loss_sum':
    return A.ifgsm(model, x, y, loss_fn, eps, alpha, opt.iters)
  if opt.attack == 'fgsm':
    x_adv = x.detach().clone(); x_adv.requires_grad = True
    ens_loss(model(x_adv), y).backward()
    return x_adv + eps * x_adv.grad.detach().sign()
  return gen_iter(model, x, y, 'mifgsm' in opt.attack, opt.attack.startswith('dim'))


def jpeg(arr, rate):
  # same mapping as imgaug 0.4.0 JpegCompression(compression=rate): PIL quality = round(1 + 99 * (1 - rate/101))
  q = int(np.clip(np.round(1 + 99 * (1.0 - rate / 101)), 1, 100))
  buf = io.BytesIO()
  Image.fromarray(arr).save(buf, format='JPEG', quality=q)
  buf.seek(0)
  return np.array(Image.open(buf).convert('RGB'))


t0 = time.time()
torch.manual_seed(opt.seed)
names = opt.surrogates.split(',')
if len(names) == 1:
  model = get_m(names[0]).to(device)
else:
  model = Ens(names, opt.ens).to(device)
adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)
if opt.ens == 'loss_sum' and len(names) > 1:
  model.eval()
  adv_examples, acc, loss = None, float('nan'), float('nan')
  exs = []
  correct = 0
  for x, y in adv_loader:
    x, y = x.to(device), y.to(device)
    x_adv = attack_fn(model, x, y, loss_fn)
    correct += (model(x_adv).sum(0).argmax(1) == y).sum().item()
    ex = ((x_adv) * std + mean).clamp(0, 1)
    ex = (ex * 255).clamp(0, 255).detach().cpu().data.numpy().round().transpose((0, 2, 3, 1))
    exs.append(ex)
  adv_examples, acc = np.concatenate(exs), correct / len(adv_set)
else:
  adv_examples, acc, loss = A.gen_adv_examples(model, adv_loader, attack_fn, loss_fn)
attack_s = time.time() - t0
A.create_dir(root, opt.out_dir, adv_examples, adv_names)
del model
torch.cuda.empty_cache()

rec = {'tag': opt.tag, 'surrogates': names, 'ens': opt.ens, 'attack': opt.attack, 'eps': opt.eps,
       'alpha': opt.alpha, 'iters': opt.iters, 'decay': opt.decay, 'dim_p': opt.dim_p, 'dim_max': opt.dim_max,
       'seed': opt.seed, 'printed_acc': acc, 'attack_s': round(attack_s, 1)}

# evaluate the saved PNGs (what JudgeBoi receives)
ds = AdvDataset(opt.out_dir, transform=None)
benign = AdvDataset(root, transform=None)
imgs = [np.array(Image.open(f).convert('RGB')) for f in ds.images]
rec['linf'] = int(max(np.abs(a.astype(np.int16) - np.asarray(Image.open(b).convert('RGB'), dtype=np.int16)).max()
                      for a, b in zip(imgs, benign.images)))
labels = torch.tensor(ds.labels, device=device)
X = torch.stack([transform(a) for a in imgs]).to(device)
jx = {r: torch.stack([transform(jpeg(a, float(r))) for a in imgs]).to(device) for r in opt.jpeg.split(',') if r}
rec['acc'] = {}
victims = [n for n in opt.victims.split(',') if n]
ens_logits = {}  # the victims as one ensemble (sum of logits): a stand-in for the TA's "ensemble of vanilla models"
for v in [n for n in (names + victims) if n]:
  if v in rec['acc']:
    continue
  m = get_m(v).to(device).eval()
  with torch.no_grad():
    out = {'': m(X)}
    out.update({f'+jpeg{r}': m(xj) for r, xj in jx.items()})
    for k, o in out.items():
      rec['acc'][v + k] = (o.argmax(1) == labels).float().mean().item()
      if v in victims:
        ens_logits[k] = ens_logits.get(k, 0) + o
  del m
  torch.cuda.empty_cache()
for k, o in ens_logits.items():
  rec['acc']['victim_ens' + k] = (o.argmax(1) == labels).float().mean().item()
rec['total_s'] = round(time.time() - t0, 1)
print(json.dumps(rec), flush=True)
with open(opt.jsonl, 'a') as f:
  f.write(json.dumps(rec) + '\n')
