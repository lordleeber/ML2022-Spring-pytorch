# HW13 ch03: the teacher on the TRAINING set (what distillation actually sees): errors, two examples at T = 1/4/8,
# average soft label per class, probability mass on the other 10 classes. usage (from HW13/): python ../docs/tools/hw13_teacher_train.py
import io, sys, torch, json
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
sys.path.insert(0,'.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_teacher_model
t = get_teacher_model(cfg['dataset_root']).cuda().eval()
print('num_batches_tracked', t.bn1.num_batches_tracked.item())
with redirect_stdout(io.StringIO()):
    ds = FoodDataset('food11-hw13/training', tfm=test_tfm)
zs, ys = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=128, num_workers=8): zs.append(t(x.cuda()).cpu()); ys.append(y)
z = torch.cat(zs); y = torch.cat(ys)
N=['麵包','乳製品','甜點','蛋','炸物','肉類','麵食','米飯','海鮮','湯','蔬果']
wrong = (z.argmax(-1) != y).nonzero().flatten().tolist()
print('wrong on training:', [(ds.files[i].split('/')[-1], N[y[i]], N[z[i].argmax()]) for i in wrong])
for name in ['3_0.jpg', '7_0.jpg']:
    i = [k for k, f in enumerate(ds.files) if f.endswith('/' + name)][0]
    out = {}
    for T in [1, 4, 8]:
        out[T] = [round(100 * v, 2) for v in (z[i] / T).softmax(-1).tolist()]
    print(name, 'logits', [round(v, 2) for v in z[i].tolist()]); print(json.dumps(out))
for T in [1, 4]:
    print('training soft label T', T)
    for c in range(11): print(' ', N[c], [round(100 * v, 2) for v in (z[y == c] / T).softmax(-1).mean(0).tolist()])
# non-true mass: average probability on the wrong classes, training, by T
for T in [1, 2, 4, 8]:
    q = (z / T).softmax(-1); m = 1 - q[torch.arange(len(y)), y]
    print('T', T, 'mean mass on other 10 classes', round(m.mean().item(), 4))
sys.stdout.flush(); import os; os._exit(0)
