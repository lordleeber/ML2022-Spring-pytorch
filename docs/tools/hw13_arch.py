# HW13 ch05: per-layer parameters, MACs and "MACs per number moved" for the students in hw13_students.py.
# usage (from HW13/): python ../docs/tools/hw13_arch.py > ../docs/tools/hw13_arch.txt
# For each Conv2d: params, MACs, and an arithmetic-intensity proxy = MACs / (input + output activation
# elements + weights), i.e. how much work is done per number that has to be read or written. CPU only.
import os, sys
import torch
import torch.nn as nn
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw13_students import STUDENTS

torch.manual_seed(0)
for name in ['sample', 'dw', 'plain', 'mbv2']:
  m = STUDENTS[name]().eval()
  rows = []
  def hook(mod, inp, out):
    x = inp[0]
    w = mod.weight
    macs = out.numel() * (w.shape[1] * w.shape[2] * w.shape[3])   # per output element: in/groups * k * k
    kind = 'depthwise' if mod.groups > 1 else ('pointwise' if w.shape[2] == 1 else f'{w.shape[2]}x{w.shape[3]}')
    moved = x.numel() + out.numel() + w.numel()
    rows.append((kind, tuple(x.shape[1:]), tuple(out.shape[1:]), sum(p.numel() for p in mod.parameters()), macs, macs / moved))
  hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
  with torch.no_grad():
    m(torch.zeros(1, 3, 224, 224))
  for h in hs:
    h.remove()
  tot_p = sum(p.numel() for p in m.parameters())
  tot_m = sum(r[4] for r in rows)
  print(f'== {name}: params {tot_p:,}  conv MACs {tot_m:,}  conv layers {len(rows)}')
  by = {}
  for kind, i, o, p, mac, ai in rows:
    k = 'depthwise' if kind == 'depthwise' else ('pointwise' if kind == 'pointwise' else 'kxk')
    b = by.setdefault(k, [0, 0, 0, 0])
    b[0] += 1; b[1] += p; b[2] += mac; b[3] += mac / ai
  for k, (n, p, mac, moved) in by.items():
    print(f'   {k:10s} layers {n:2d}  params {p:7,} ({100*p/tot_p:5.1f}%)  MACs {mac:12,} ({100*mac/tot_m:5.1f}%)  MACs per number moved {mac/moved:7.1f}')
  for kind, i, o, p, mac, ai in rows:
    print(f'     {kind:10s} {str(i):16s} -> {str(o):16s} params {p:7,}  MACs {mac:12,}  intensity {ai:7.1f}')
