# HW13 ch04: properties of loss_fn_kd on real logits (Simple student checkpoint vs teacher, training images,
# test_tfm, the whole training set). Inference only, no randomness.
# usage (from HW13/): python ../docs/tools/hw13_kdcheck.py > ../docs/tools/hw13_kdcheck.txt
import io, sys, warnings
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_student_model, get_teacher_model
from kd import loss_fn_kd

teacher = get_teacher_model(cfg['dataset_root']).cuda().eval()
student = get_student_model()
student.load_state_dict(torch.load('outputs/simple_baseline/student_best.ckpt'))
student.cuda().eval()
with redirect_stdout(io.StringIO()):
  ds = FoodDataset(f"{cfg['dataset_root']}/training", tfm=test_tfm)
zs, zt, ys = [], [], []
with torch.no_grad():
  for x, y in DataLoader(ds, batch_size=128, num_workers=8):
    zs.append(student(x.cuda()).cpu()); zt.append(teacher(x.cuda()).cpu()); ys.append(y)
s, t, y = torch.cat(zs), torch.cat(zt), torch.cat(ys)
print('images', len(y), 'student acc', (s.argmax(-1) == y).float().mean().item(), 'teacher acc', (t.argmax(-1) == y).float().mean().item())

print('== 1. direction: kl_div(input=log q_student, target=p_teacher) = KL(teacher || student)')
for T in [1, 4]:
  ls, pt = F.log_softmax(s / T, -1), F.softmax(t / T, -1)
  lt, ps = F.log_softmax(t / T, -1), F.softmax(s / T, -1)
  lib = F.kl_div(ls, pt, reduction='batchmean').item()
  kl_ts = (pt * (lt - ls)).sum(-1).mean().item()
  kl_st = (ps * (ls - lt)).sum(-1).mean().item()
  print(f'T={T}: F.kl_div {lib:.6f}  KL(teacher||student) {kl_ts:.6f}  KL(student||teacher) {kl_st:.6f}')

print('== 2. reduction')
ls, pt = F.log_softmax(s, -1), F.softmax(t, -1)
with warnings.catch_warnings(record=True) as w:
  warnings.simplefilter('always')
  mean = F.kl_div(ls, pt, reduction='mean').item()
  print('mean', round(mean, 6), 'warning:', str(w[0].message)[:160] if w else None)
print('batchmean', round(F.kl_div(ls, pt, reduction='batchmean').item(), 6), 'sum', round(F.kl_div(ls, pt, reduction='sum').item(), 4),
      'nn.KLDivLoss() default', round(nn.KLDivLoss()(ls, pt).item(), 6))

print('== 3. gradient size of the KL term w.r.t. student logits, with and without T^2 (mean L2 norm per image)')
for T in [1, 2, 4, 8, 20]:
  sl = s.clone().requires_grad_(True)
  kl = F.kl_div(F.log_softmax(sl / T, -1), F.softmax(t / T, -1), reduction='batchmean')
  g, = torch.autograd.grad(kl, sl)
  g = g * len(y)   # per-image gradient (batchmean divides by N)
  n = g.norm(dim=-1).mean().item()
  print(f'T={T:2d}: KL {kl.item():.5f}  grad norm {n:.5f}  x T^2 -> {n * T * T:.5f}')
sl = s.clone().requires_grad_(True)
ce = F.cross_entropy(sl, y)
g, = torch.autograd.grad(ce, sl)
print(f'CE (hard labels): {ce.item():.5f} grad norm {(g * len(y)).norm(dim=-1).mean().item():.5f}')

print('== 4. loss_fn_kd terms at the report setting and at T=4, alpha=0.5')
for T in [1, 4]:
  kd = loss_fn_kd(s, y, t, alpha=0.5, temperature=T).item()
  klv = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction='batchmean').item()
  print(f'T={T}: loss_fn_kd {kd:.5f} = 0.5*{T*T}*{klv:.5f} + 0.5*{F.cross_entropy(s, y).item():.5f}')
sys.stdout.flush(); import os; os._exit(0)
