# HW13 ch02/ch06/ch07: inference time. Run it alone on the machine (no training, nothing else on the GPU).
# usage (from HW13/): python ../docs/tools/hw13_speed.py ../docs/tools/hw13_speed.json [--quick]
# Timing: warm-up, then N timed iterations with torch.cuda.synchronize(); report the median in ms.
# Inputs are fixed random tensors (timing does not depend on the values). Accuracy of the physically
# pruned student is measured on the validation set.
import copy, io, json, os, statistics, sys, time
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_student_model, get_teacher_model
from hw13_students import STUDENTS
from hw13_shrink import shrink_student

QUICK = '--quick' in sys.argv
N_GPU, N_CPU = (5, 2) if QUICK else (100, 20)
torch.backends.cudnn.benchmark = False
CPU_THREADS = 4


def time_ms(model, x, n, device):
  model.eval()
  with torch.no_grad():
    for _ in range(max(3, n // 10)):
      model(x)
    ts = []
    for _ in range(n):
      if device == 'cuda':
        torch.cuda.synchronize()
      t0 = time.perf_counter()
      model(x)
      if device == 'cuda':
        torch.cuda.synchronize()
      ts.append((time.perf_counter() - t0) * 1000)
  return round(statistics.median(ts), 3)


def macs(model):
  model.eval()
  with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
      model(torch.zeros(1, 3, 224, 224))
  return fc.get_total_flops() // 2


def load_student():
  m = get_student_model(); m.load_state_dict(torch.load('outputs/simple_baseline/student_best.ckpt')); return m.eval()


with redirect_stdout(io.StringIO()):
  valid = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
vb = [(x, y) for x, y in DataLoader(valid, batch_size=128, num_workers=8)]


@torch.no_grad()
def acc(model):
  model.cuda().eval()
  r = sum((model(x.cuda()).argmax(-1).cpu() == y).sum().item() for x, y in vb) / len(valid)
  model.cpu()
  return round(r, 5)


models = {}
models['teacher'] = get_teacher_model(cfg['dataset_root']).eval()
def pruned_teacher(remove):
  # a pruned module cannot be deep-copied (its .weight is computed, not a leaf tensor), so prune a fresh copy
  m = get_teacher_model(cfg['dataset_root']).eval()
  for mod in m.modules():
    if isinstance(mod, nn.Conv2d):
      prune.l1_unstructured(mod, 'weight', amount=0.5)
      if remove:
        prune.remove(mod, 'weight')
  return m
models['teacher_pruned50_mask'] = pruned_teacher(False)
models['teacher_pruned50_removed'] = pruned_teacher(True)
models['student_sample'] = load_student()
for name in ['dw', 'plain', 'mbv2']:
  models[f'student_{name}'] = STUDENTS[name]().eval()
shrink_info = {}
for keep in [0.75, 0.5]:
  s = shrink_student(load_student(), keep)
  models[f'student_sample_keep{int(keep * 100)}'] = s
  shrink_info[f'keep{int(keep * 100)}'] = dict(params=sum(p.numel() for p in s.parameters()), macs=macs(s), acc_no_finetune=acc(s),
                                               channels=[l.out_channels for l in s.cnn if isinstance(l, nn.Conv2d)])
print('shrink', shrink_info, flush=True)

res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cpu_threads=CPU_THREADS, quick=QUICK, shrink=shrink_info, rows={})
torch.set_num_threads(CPU_THREADS)
xg64, xg1, xc1 = torch.randn(64, 3, 224, 224, device='cuda'), torch.randn(1, 3, 224, 224, device='cuda'), torch.randn(1, 3, 224, 224)
for name, m in models.items():
  row = dict(params=sum(p.numel() for p in m.parameters()))
  try:
    row['macs'] = macs(m)
  except Exception as e:
    row['macs'] = None
  g = m.cuda()
  row['gpu_b64_fp32_ms'] = time_ms(g, xg64, N_GPU, 'cuda')
  row['gpu_b1_fp32_ms'] = time_ms(g, xg1, N_GPU, 'cuda')
  m.cpu()
  if 'mask' not in name:
    gh = copy.deepcopy(m).cuda().half()
    row['gpu_b64_fp16_ms'] = time_ms(gh, xg64.half(), N_GPU, 'cuda')
    del gh
  row['cpu_b1_fp32_ms'] = time_ms(m, xc1, N_CPU, 'cpu')
  res['rows'][name] = row
  print(name, row, flush=True)

# static int8 on CPU (same recipe as hw13_quant.py)
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import warnings
warnings.simplefilter('ignore')
with redirect_stdout(io.StringIO()):
  train = FoodDataset(f"{cfg['dataset_root']}/training", tfm=test_tfm)
calib = [x for x, _ in DataLoader(torch.utils.data.Subset(train, range(0, len(train), 10)), batch_size=128, num_workers=8)]
torch.backends.quantized.engine = 'x86'
for name in ['teacher', 'student_sample']:
  p = prepare_fx(copy.deepcopy(models[name]), get_default_qconfig_mapping('x86'), example_inputs=(xc1,))
  with torch.no_grad():
    for x in calib:
      p(x)
  q = convert_fx(p)
  res['rows'][name]['cpu_b1_int8_ms'] = time_ms(q, xc1, N_CPU, 'cpu')
  print(name, 'int8 cpu', res['rows'][name]['cpu_b1_int8_ms'], flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
sys.stdout.flush(); os._exit(0)
