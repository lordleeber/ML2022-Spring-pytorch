# Facts for ch00/ch01: resnet110_cifar10 structure, shapes and parameter counts; dataset order;
# epsilon per channel; what rounding/clamping does to the FGSM and I-FGSM images.
# usage (from HW10/, deterministic): CUBLAS_WORKSPACE_CONFIG=:4096:8 python ../docs/tools/hw10_det.py ../docs/tools/hw10_facts.py
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root, mean, std, epsilon, cifar_10_std
from dataset import AdvDataset, transform
import attack as A

print('== model ==')
m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device).eval()
tot = sum(p.numel() for p in m.parameters())
buf = sum(b.numel() for b in m.buffers())
print('params', tot, 'buffers', buf)
for name, mod in m.features.named_children():
  print(name, type(mod).__name__, sum(p.numel() for p in mod.parameters()), len(list(mod.children())) if isinstance(mod, nn.Sequential) else '')
print('output', sum(p.numel() for p in m.output.parameters()))
n_conv = sum(1 for x in m.modules() if isinstance(x, nn.Conv2d))
n_conv3 = sum(1 for x in m.modules() if isinstance(x, nn.Conv2d) and x.kernel_size == (3, 3))
n_bn = sum(1 for x in m.modules() if isinstance(x, nn.BatchNorm2d))
print('conv', n_conv, 'conv3x3', n_conv3, 'bn', n_bn, 'linear', sum(1 for x in m.modules() if isinstance(x, nn.Linear)))
print(m.features.stage2.unit1)
shapes = []
hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: shapes.append((n, tuple(o.shape))))
         for n, mod in list(m.features.named_children())]
hooks.append(m.output.register_forward_hook(lambda mod, i, o: shapes.append(('output', tuple(o.shape)))))
with torch.no_grad():
  m(torch.zeros(8, 3, 32, 32, device=device))
for h in hooks:
  h.remove()
for s in shapes:
  print('shape', s)
# hand count: 3x3 convs without bias + BN (2 params/channel) + 1x1 shortcut convs + linear
def conv(ci, co, k): return ci * co * k * k
hand = conv(3, 16, 3) + 2 * 16
for ci, co, n in ((16, 16, 18), (16, 32, 18), (32, 64, 18)):
  for u in range(n):
    cin = ci if u == 0 else co
    hand += conv(cin, co, 3) + 2 * co + conv(co, co, 3) + 2 * co
    if u == 0 and cin != co:
      hand += conv(cin, co, 1) + 2 * co
hand += 64 * 10 + 10
print('hand count', hand)

print('== data ==')
ds = AdvDataset(root, transform=transform)
print('n', len(ds), 'classes', Counter(ds.labels))
print('first names', ds.names[:4], '... dog idx', ds.names.index('dog/dog2.png'))
print('class order', sorted(os.listdir(root)))
print('epsilon (normalized units)', [round(v, 6) for v in epsilon.flatten().tolist()], 'cifar std', cifar_10_std)
print('8/255 =', 8 / 255)
x0, _ = ds[0]
print('x range', [round(v, 4) for v in (x0.min().item(), x0.max().item())])
allx = torch.stack([ds[i][0] for i in range(len(ds))])
print('normalized range over 200 imgs', allx.amin(dim=(0, 2, 3)).tolist(), allx.amax(dim=(0, 2, 3)).tolist())

print('== rounding / clamping ==')
loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
loss_fn = nn.CrossEntropyLoss()
for nm, fn in (('fgsm', A.fgsm), ('ifgsm', A.ifgsm)):
  out_range, clamped, rounded_changed, total = 0, 0, 0, 0
  c_float, c_png = 0, 0
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    xa = fn(m, x, y, loss_fn)
    pix = (xa * std + mean) * 255           # float pixels, before clamp and rounding
    outside = (pix < 0) | (pix > 255)
    out_range += outside.sum().item()
    final = pix.clamp(0, 255).round()
    rounded_changed += (final != pix).sum().item()
    total += pix.numel()
    c_float += (m(xa).argmax(1) == y).sum().item()
    xr = (final / 255 - mean) / std
    c_png += (m(xr).argmax(1) == y).sum().item()
  print(nm, 'pixels', total, 'outside[0,255]', out_range, 'float acc', c_float / 200, 'rounded acc', c_png / 200)
  # distribution of |adv - benign| in the saved PNGs
  from PIL import Image
  d = Counter()
  for f, g in zip(AdvDataset(nm, None).images, AdvDataset(root, None).images):
    diff = np.abs(np.asarray(Image.open(f), dtype=np.int16) - np.asarray(Image.open(g), dtype=np.int16))
    d.update(diff.flatten().tolist())
  print(nm, '|diff| histogram', sorted(d.items()))
