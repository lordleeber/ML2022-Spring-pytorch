# HW13 ch02: how torchsummary counts, FLOPs/MACs per layer, activation memory, checkpoint contents.
# usage (from HW13/): python ../docs/tools/hw13_count.py > ../docs/tools/hw13_count.txt
import io, os, sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from torchsummary import summary

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from model import get_student_model, get_teacher_model
from hw13_students import STUDENTS


def ts_total(model):
  buf = io.StringIO()
  with redirect_stdout(buf):
    summary(model, (3, 224, 224), device='cpu')
  lines = buf.getvalue().splitlines()
  get = lambda key: [l for l in lines if l.startswith(key)][0].split(':')[1].strip()
  return dict(total=get('Total params'), trainable=get('Trainable params'), non_trainable=get('Non-trainable params'),
              fb_mb=get('Forward/backward pass size (MB)'))


def numel(model):
  return sum(p.numel() for p in model.parameters())


print('== 1. torchsummary uses the global RNG (torch.rand(2, ...) as input)')
torch.manual_seed(0)
before = torch.get_rng_state()
with redirect_stdout(io.StringIO()):
  summary(get_student_model(), (3, 224, 224), device='cpu')
after_model_and_summary = torch.get_rng_state()
torch.manual_seed(0)
get_student_model()
after_model_only = torch.get_rng_state()
print('state after model == after model+summary:', torch.equal(after_model_only, after_model_and_summary))


print('== 2. counting corner cases: numel vs torchsummary')
class Shared(nn.Module):          # one conv used twice
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 3, 3, padding=1)
    self.fc = nn.Linear(3, 11)
  def forward(self, x):
    x = self.conv(self.conv(x))
    return self.fc(x.mean((2, 3)))

class TopLevelParam(nn.Module):   # an nn.Parameter on the model itself
  def __init__(self):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(50_000))
    self.fc = nn.Linear(3, 11)
  def forward(self, x):
    return self.fc(x.mean((2, 3))) * self.scale[:11]

class Unused(nn.Module):          # a layer defined but never called
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(3, 11)
    self.spare = nn.Linear(300, 300)
  def forward(self, x):
    return self.fc(x.mean((2, 3)))

class Frozen(nn.Module):          # sample student with the first conv frozen
  def __init__(self):
    super().__init__()
    self.net = get_student_model()
    for p in self.net.cnn[0].parameters():
      p.requires_grad = False
  def forward(self, x):
    return self.net(x)

def no_affine_student():
  m = get_student_model()
  m.cnn[1] = nn.BatchNorm2d(32, affine=False)
  return m

for name, m in [('Shared', Shared()), ('TopLevelParam', TopLevelParam()), ('Unused', Unused()),
                ('Frozen', Frozen()), ('student BN affine=False on cnn[1]', no_affine_student())]:
  print(f'{name}: numel {numel(m):,}  torchsummary {ts_total(m)}')


print('== 3. FLOPs (torch.utils.flop_counter) per layer, one 224x224 image; MACs = FLOPs / 2')
def flops_by_module(model):
  model.eval()
  x = torch.zeros(1, 3, 224, 224)
  with FlopCounterMode(display=False, depth=None) as fc:
    model(x)
  return fc.get_total_flops(), fc.get_flop_counts()

total, by = flops_by_module(get_student_model())
print(f'student total FLOPs {total:,}  MACs {total // 2:,}')
for k, v in by.items():
  if k.count('.') == 2:   # e.g. StudentNet.cnn.0
    print('  ', k, f'{sum(v.values()) // 2:,} MACs')
for name in ['dw', 'plain', 'mbv2']:
  t, _ = flops_by_module(STUDENTS[name]())
  print(f'{name}: params {numel(STUDENTS[name]()):,} MACs {t // 2:,}')
t, by = flops_by_module(get_teacher_model('food11-hw13'))
print(f'teacher total MACs {t // 2:,}')
for k, v in by.items():
  if k.count('.') == 1:
    print('  ', k, f'{sum(v.values()) // 2:,} MACs')


print('== 4. activation sizes of the sample student (floats per image, forward only)')
m = get_student_model().eval()
x = torch.zeros(1, 3, 224, 224)
for i, layer in enumerate(m.cnn):
  x = layer(x)
  print(f'  cnn.{i} {layer.__class__.__name__:18s} {tuple(x.shape[1:])} {x.numel():,}')
for name in ['sample', 'dw', 'plain', 'mbv2']:
  print(name, 'torchsummary', ts_total(STUDENTS[name]()))
print('teacher torchsummary', ts_total(get_teacher_model('food11-hw13')))


print('== 5. what is inside student_best.ckpt')
sd = torch.load('outputs/simple_baseline/student_best.ckpt')
kinds = {}
for k, v in sd.items():
  kind = k.split('.')[-1]
  kinds.setdefault(kind, [0, 0, str(v.dtype)])
  kinds[kind][0] += v.numel(); kinds[kind][1] += v.numel() * v.element_size()
for k, (n, b, dt) in kinds.items():
  print(f'  {k:22s} {dt:12s} {n:7,} numbers {b:9,} bytes')
print('  file', os.path.getsize('outputs/simple_baseline/student_best.ckpt'), 'bytes')
half = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
torch.save(half, '/tmp/hw13_half.ckpt')
print('  fp16 copy file', os.path.getsize('/tmp/hw13_half.ckpt'), 'bytes')
os.remove('/tmp/hw13_half.ckpt')
