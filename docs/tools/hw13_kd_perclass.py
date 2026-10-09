# HW13 ch04: per-class validation accuracy and agreement with the teacher, from the best-epoch validation logits
# that hw13_exp.py --save stored (best_valid_logits.pt, validation order = sorted file names).
# usage (from HW13/): python ../docs/tools/hw13_kd_perclass.py <ckpt_dir> run1 run2 ...
import io, json, sys
from contextlib import redirect_stdout
import torch
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
with redirect_stdout(io.StringIO()):
  ds = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
y = torch.tensor([int(f.split('/')[-1].split('_')[0]) for f in ds.files])
t = json.load(open('../docs/tools/hw13_teacher.json'))
N = ['麵包', '乳製品', '甜點', '蛋', '炸物', '肉類', '麵食', '米飯', '海鮮', '湯', '蔬果']
# teacher predictions on validation (recomputed from the teacher)
from model import get_teacher_model
from torch.utils.data import DataLoader
tm = get_teacher_model(cfg['dataset_root']).cuda().eval()
with torch.no_grad():
  tp = torch.cat([tm(x.cuda()).argmax(-1).cpu() for x, _ in DataLoader(ds, batch_size=128, num_workers=8)])
for run in sys.argv[2:]:
  z = torch.load(f'{sys.argv[1]}/{run}/best_valid_logits.pt')
  p = z.argmax(-1)
  pc = [round((p[y == c] == c).float().mean().item(), 3) for c in range(11)]
  pred_share = [round((p == c).float().mean().item() * 100, 1) for c in range(11)]
  print(json.dumps(dict(run=run, acc=round((p == y).float().mean().item(), 5), agree_teacher=round((p == tp).float().mean().item(), 4),
                        per_class=dict(zip(N, pc)), pred_share_pct=dict(zip(N, pred_share))), ensure_ascii=False), flush=True)
import os; sys.stdout.flush(); os._exit(0)
