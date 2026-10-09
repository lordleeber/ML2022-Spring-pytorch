# HW13 ch06: pruning (report Q3). Accuracy part only (no timing; timing lives in hw13_speed.py).
# usage (from HW13/): python ../docs/tools/hw13_prune.py ../docs/tools/hw13_prune.json
# Inference only on the validation set (test_tfm, shuffle=False): no randomness.
import copy, io, json, os, sys
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchsummary import summary
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_student_model, get_teacher_model

with redirect_stdout(io.StringIO()):
  ds = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
batches = [(x, y) for x, y in DataLoader(ds, batch_size=128, num_workers=8)]


@torch.no_grad()
def acc(model):
  model.cuda().eval()
  c = sum((model(x.cuda()).argmax(-1).cpu() == y).sum().item() for x, y in batches)
  return round(c / len(ds), 5)


def convs(model):
  return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def zero_frac(model):
  ws = [m.weight for m in convs(model)]
  return round(sum((w == 0).sum().item() for w in ws) / sum(w.numel() for w in ws), 4)


def load_student():
  m = get_student_model()
  m.load_state_dict(torch.load('outputs/simple_baseline/student_best.ckpt'))
  return m


RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
res = {}
for who, load in [('teacher', lambda: get_teacher_model(cfg['dataset_root']).eval()), ('student', lambda: load_student().eval())]:
  base = load()
  res[who] = {'per_layer_l1': [], 'global_l1': [], 'per_layer_ln_structured': []}
  for r in RATIOS:
    # 1) the sample code: l1_unstructured on every Conv2d, same ratio per layer
    m = copy.deepcopy(base)
    for mod in convs(m):
      prune.l1_unstructured(mod, name='weight', amount=r)
    res[who]['per_layer_l1'].append(dict(ratio=r, acc=acc(m), zeros=zero_frac(m)))
    # 2) global_unstructured: one threshold over all conv weights together
    m = copy.deepcopy(base)
    prune.global_unstructured([(mod, 'weight') for mod in convs(m)], pruning_method=prune.L1Unstructured, amount=r)
    res[who]['global_l1'].append(dict(ratio=r, acc=acc(m), zeros=zero_frac(m)))
    # 3) ln_structured: drop whole output channels (L2 norm), same ratio per layer; still a mask
    m = copy.deepcopy(base)
    for mod in convs(m):
      prune.ln_structured(mod, name='weight', amount=r, n=2, dim=0)
    res[who]['per_layer_ln_structured'].append(dict(ratio=r, acc=acc(m), zeros=zero_frac(m)))
    print(who, r, res[who]['per_layer_l1'][-1]['acc'], res[who]['global_l1'][-1]['acc'], res[who]['per_layer_ln_structured'][-1]['acc'], flush=True)

# what pruning does to the module, the parameter count and the file (teacher, l1_unstructured 0.5)
m = get_teacher_model(cfg['dataset_root']).eval()
for mod in convs(m):
  prune.l1_unstructured(mod, name='weight', amount=0.5)
c1 = m.conv1
info = dict(
  params_names=[n for n, _ in c1.named_parameters()], buffer_names=[n for n, _ in c1.named_buffers()],
  weight_is_parameter=isinstance(c1.weight, nn.Parameter), hooks=len(c1._forward_pre_hooks),
  numel_params=sum(p.numel() for p in m.parameters()))
buf = io.StringIO()
with redirect_stdout(buf):
  summary(m, (3, 224, 224), device='cpu')    # eval mode, so BN statistics stay untouched
info['torchsummary_total'] = [l for l in buf.getvalue().splitlines() if l.startswith('Total params')][0]
torch.save(m.state_dict(), '/tmp/hw13_pruned_masked.ckpt')
info['file_masked'] = os.path.getsize('/tmp/hw13_pruned_masked.ckpt')
info['state_dict_keys_masked'] = len(m.state_dict())
for mod in convs(m):
  prune.remove(mod, 'weight')
torch.save(m.state_dict(), '/tmp/hw13_pruned_removed.ckpt')
info['file_after_remove'] = os.path.getsize('/tmp/hw13_pruned_removed.ckpt')
info['state_dict_keys_after_remove'] = len(m.state_dict())
info['acc_after_remove'] = acc(m)
info['zeros_after_remove'] = zero_frac(m)
sd = {k: (v.to_sparse() if v.dim() == 4 else v) for k, v in m.state_dict().items()}
torch.save(sd, '/tmp/hw13_pruned_sparse.ckpt')
info['file_sparse_coo'] = os.path.getsize('/tmp/hw13_pruned_sparse.ckpt')
for f in ['/tmp/hw13_pruned_masked.ckpt', '/tmp/hw13_pruned_removed.ckpt', '/tmp/hw13_pruned_sparse.ckpt']:
  os.remove(f)
res['mechanics_teacher_l1_0.5'] = info
print(json.dumps(info, indent=1))
json.dump(res, open(sys.argv[1], 'w'), indent=1)
sys.stdout.flush(); os._exit(0)
