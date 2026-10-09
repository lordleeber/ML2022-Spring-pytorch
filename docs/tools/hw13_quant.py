# HW13 ch07: quantization, accuracy and file size (timing lives in hw13_speed.py).
# usage (from HW13/): python ../docs/tools/hw13_quant.py ../docs/tools/hw13_quant.json
# Inference only; the calibration set is fixed (every 10th training image, test_tfm): no randomness.
import copy, io, json, os, sys, warnings
from contextlib import redirect_stdout
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_student_model, get_teacher_model

torch.set_num_threads(4)
with redirect_stdout(io.StringIO()):
  valid = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
  train = FoodDataset(f"{cfg['dataset_root']}/training", tfm=test_tfm)
vb = [(x, y) for x, y in DataLoader(valid, batch_size=128, num_workers=8)]
calib = [x for x, _ in DataLoader(Subset(train, range(0, len(train), 10)), batch_size=128, num_workers=8)]


@torch.no_grad()
def acc(model, device, dtype=torch.float32):
  model.eval()
  c = sum((model(x.to(device, dtype)).argmax(-1).cpu() == y).sum().item() for x, y in vb)
  return round(c / len(valid), 5)


def size(obj):
  torch.save(obj, '/tmp/hw13_q.pt'); n = os.path.getsize('/tmp/hw13_q.pt'); os.remove('/tmp/hw13_q.pt'); return n


def load_student():
  m = get_student_model(); m.load_state_dict(torch.load('outputs/simple_baseline/student_best.ckpt')); return m


def fake_quant_weights(model, bits):
  # weight-only, symmetric, per output channel: w -> round(w / s) * s with s = max|w| / (2^(bits-1) - 1)
  m = copy.deepcopy(model)
  qmax = 2 ** (bits - 1) - 1
  with torch.no_grad():
    for mod in m.modules():
      if isinstance(mod, (nn.Conv2d, nn.Linear)):
        w = mod.weight
        s = w.abs().flatten(1).max(1).values.clamp_min(1e-12) / qmax
        s = s.view(-1, *([1] * (w.dim() - 1)))
        w.copy_((w / s).round().clamp(-qmax, qmax) * s)
  return m


res = {}
warns = []
for who, load in [('teacher', lambda: get_teacher_model(cfg['dataset_root']).eval()), ('student', lambda: load_student().eval())]:
  r = {}
  base = load()
  r['fp32_gpu'] = acc(copy.deepcopy(base).cuda(), 'cuda')
  r['fp32_file'] = size(base.state_dict())
  r['fp16_gpu'] = acc(copy.deepcopy(base).cuda().half(), 'cuda', torch.float16)
  r['bf16_gpu'] = acc(copy.deepcopy(base).cuda().to(torch.bfloat16), 'cuda', torch.bfloat16)
  r['fp16_file'] = size({k: (v.half() if v.is_floating_point() else v) for k, v in base.state_dict().items()})
  r['fp32_cpu'] = acc(copy.deepcopy(base), 'cpu')
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    # dynamic int8: only nn.Linear is supported for this kind of model
    from torch.ao.quantization import quantize_dynamic
    dq = quantize_dynamic(copy.deepcopy(base), {nn.Linear}, dtype=torch.qint8)
    r['dynamic_int8_cpu'] = acc(dq, 'cpu')
    r['dynamic_int8_file'] = size(dq.state_dict())
    # static int8 (FX graph mode, x86 backend): calibrate activation ranges on 987 training images
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    torch.backends.quantized.engine = 'x86'
    qmap = get_default_qconfig_mapping('x86')
    prepared = prepare_fx(copy.deepcopy(base), qmap, example_inputs=(calib[0][:1],))
    with torch.no_grad():
      for x in calib:
        prepared(x)
    sq = convert_fx(prepared)
    r['static_int8_cpu'] = acc(sq, 'cpu')
    r['static_int8_file'] = size(sq.state_dict())
    warns += sorted({str(x.message)[:200] for x in w})
  for bits in [8, 6, 4, 3, 2]:
    r[f'weight_only_int{bits}_gpu'] = acc(fake_quant_weights(base, bits).cuda(), 'cuda')
  res[who] = r
  print(who, json.dumps(r), flush=True)
res['warnings'] = sorted(set(warns))
print('warnings:', *res['warnings'], sep='\n  ')
json.dump(res, open(sys.argv[1], 'w'), indent=1)
sys.stdout.flush(); os._exit(0)
