# Gradient norms of the Simple run: train.py's loop (CE, sample student, 10 epochs, seed 20220013) with 16
# DataLoader workers, recording the total norm that clip_grad_norm_ returns (the norm BEFORE clipping).
# usage (from HW13/): python ../docs/tools/hw13_gradnorm.py <out.json>
import json, random, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, train_tfm
from model import get_student_model

torch.backends.cudnn.deterministic = True
np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
loader = DataLoader(FoodDataset(f"{cfg['dataset_root']}/training", tfm=train_tfm), batch_size=cfg['batch_size'],
                    shuffle=True, num_workers=16, pin_memory=True)
model = get_student_model().cuda()
opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
loss_fn = nn.CrossEntropyLoss()
norms = []
for epoch in range(cfg['n_epochs']):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        loss = loss_fn(model(imgs), labels)
        opt.zero_grad()
        loss.backward()
        norms.append(nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_norm_max']).item())
        opt.step()
    print(epoch + 1, f'max {max(norms[-len(loader):]):.3f}', flush=True)
a = np.array(norms)
out = dict(steps=len(a), clipped=int((a > cfg['grad_norm_max']).sum()), max=float(a.max()), mean=float(a.mean()),
           median=float(np.median(a)), first10=[round(x, 3) for x in norms[:10]], per_epoch_max=[round(float(a[i*155:(i+1)*155].max()), 3) for i in range(cfg['n_epochs'])])
print(out)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
sys.stdout.flush(); import os; os._exit(0)
