# HW13 ch04: average soft target per class over the training set, by temperature (vs the hard-label shares).
# usage (from HW13/): python ../docs/tools/hw13_soft_mass.py
import io, sys, torch
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
sys.path.insert(0,'.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_teacher_model
t = get_teacher_model(cfg['dataset_root']).cuda().eval()
with redirect_stdout(io.StringIO()):
    ds = FoodDataset('food11-hw13/training', tfm=test_tfm)
with torch.no_grad():
    z = torch.cat([t(x.cuda()).cpu() for x, _ in DataLoader(ds, batch_size=128, num_workers=8)])
y = torch.tensor([int(f.split('/')[-1].split('_')[0]) for f in ds.files])
N=['麵包','乳製品','甜點','蛋','炸物','肉類','麵食','米飯','海鮮','湯','蔬果']
for T in [1, 2, 4, 8]:
    q = (z / T).softmax(-1)
    target = q.mean(0) * 100           # average soft target per class over the training set
    hard = torch.bincount(y, minlength=11).float() / len(y) * 100
    print(f'T={T}', ' '.join(f'{N[c]} {target[c]:.2f}' for c in range(11)))
print('hard', ' '.join(f'{N[c]} {hard[c]:.2f}' for c in range(11)))
sys.stdout.flush(); import os; os._exit(0)
