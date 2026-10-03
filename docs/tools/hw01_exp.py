"""Replicates HW01 train.py's trainer with switchable improvements.

Run from HW01/ with PYTHONPATH=. so utils/model/data_loader import from the repo:
    cd HW01 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw01_exp.py --name orig
Prints one JSON line with the results. Nothing is written to disk, so models/model.ckpt,
runs/ and pred.csv are left alone. hw01_run_grid.sh runs many configurations in parallel.

Default arguments are the original train.py. Verified on 2026-10-03 to reproduce it bit for
bit: printed best 1.6611 at epoch 1483, stop at 1883, 49 saves, full-set valid MSE 2.0685.
With --fix 1 (valid shuffle=False + per-sample weighting, ch03) it gives 1.7174 / 2968 / 3000.
The results behind docs/HW01/FACTS.md「ch07 實測」 come from hw01_ch07_runs.txt.
"""
import argparse, json, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from utils import same_seed, train_valid_split
from data_loader import COVID19Dataset
from model import My_Model

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--feat', default='all')          # all | noid | tp4 | survey | corr
ap.add_argument('--std', type=int, default=0)       # standardize with train-split stats
ap.add_argument('--opt', default='sgd')             # sgd | adam | adamw
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--mom', type=float, default=0.9)
ap.add_argument('--wd', type=float, default=0.0)
ap.add_argument('--arch', default='16-8')           # hidden sizes, '-' separated; '' = linear
ap.add_argument('--act', default='relu')            # relu | leaky | gelu
ap.add_argument('--fix', type=int, default=0)       # valid shuffle=False + per-sample weighting
ap.add_argument('--split', default='random')        # random | time
ap.add_argument('--seed', type=int, default=5201314)
ap.add_argument('--epochs', type=int, default=3000)
ap.add_argument('--es', type=int, default=400)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--perm', type=int, default=0)       # 1: permutation importance on valid, np.random.default_rng(0)
a = ap.parse_args()
dev = 'cuda'
t0 = time.time()

same_seed(a.seed)
df = pd.read_csv('./covid.train.csv'); full = df.values
te = pd.read_csv('./covid.test.csv').values
if a.split == 'random':
    tr, va = train_valid_split(full, 0.2, 5201314)       # split seed fixed, like config
else:
    state = df.iloc[:, 1:38].idxmax(axis=1)
    rank = df.groupby(state)['id'].rank(pct=True)
    tr, va = full[(rank <= 0.8).values], full[(rank > 0.8).values]

ytr, yva = tr[:, -1], va[:, -1]
Xtr, Xva = tr[:, :-1], va[:, :-1]
if a.feat == 'all':    idx = list(range(117))
elif a.feat == 'noid': idx = list(range(1, 117))
elif a.feat == 'tp4':  idx = [53, 69, 85, 101]
elif a.feat == 'survey': idx = list(range(38, 117))
elif a.feat == 'corr':
    c = pd.DataFrame(Xtr[:, 1:]).corrwith(pd.Series(ytr)).abs()
    idx = [i + 1 for i in c.index if c[i] > 0.5]
Xtr, Xva, Xte = Xtr[:, idx], Xva[:, idx], te[:, idx]
if a.std:
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

tl = DataLoader(COVID19Dataset(Xtr, ytr), batch_size=a.bs, shuffle=True, pin_memory=True)
vl = DataLoader(COVID19Dataset(Xva, yva), batch_size=a.bs, shuffle=not a.fix, pin_memory=True)

if a.arch == '16-8' and a.act == 'relu':
    model = My_Model(input_dim=len(idx))
else:
    Act = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'gelu': nn.GELU}[a.act]
    hs = [int(h) for h in a.arch.split('-')] if a.arch not in ('', 'none') else []
    layers, d = [], len(idx)
    for h in hs:
        layers += [nn.Linear(d, h), Act()]; d = h
    layers += [nn.Linear(d, 1)]
    class M(nn.Module):
        def __init__(self):
            super().__init__(); self.layers = nn.Sequential(*layers)
        def forward(self, x): return self.layers(x).squeeze(1)
    model = M()
model = model.to(dev)
nparams = sum(p.numel() for p in model.parameters())

crit = nn.MSELoss(reduction='mean')
if a.opt == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.mom, weight_decay=a.wd)
elif a.opt == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

best, best_ep, cnt, saves, best_sd = math.inf, 0, 0, 0, None
first_ep = None
for ep in range(a.epochs):
    model.train(); rec = []
    for x, y in tl:
        opt.zero_grad(); x, y = x.to(dev), y.to(dev)
        loss = crit(model(x), y); loss.backward(); opt.step(); rec.append(loss.item())
    mtl = sum(rec) / len(rec)
    model.eval(); rec = []; n = 0
    for x, y in vl:
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            l = crit(model(x), y)
        if a.fix: rec.append(l.item() * len(y)); n += len(y)
        else: rec.append(l.item())
    mvl = sum(rec) / (n if a.fix else len(rec))
    if first_ep is None: first_ep = (round(mtl, 4), round(mvl, 4))
    if mvl < best:
        best, best_ep, cnt = mvl, ep + 1, 0; saves += 1
        best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        cnt += 1
    if cnt >= a.es: break
stop = ep + 1

model.load_state_dict(best_sd); model.eval()
f = lambda X: model(torch.tensor(X, dtype=torch.float32, device=dev)).detach().cpu().numpy()
with torch.no_grad():
    pv, pte, ptr = f(Xva), f(Xte), f(Xtr)
    h = torch.tensor(np.concatenate([Xtr, Xva]), dtype=torch.float32, device=dev)
    dead = None
    if len(model.layers) > 1:
        a1 = model.layers[1](model.layers[0](h)); dead = int((a1 == 0).all(0).sum())
d4 = te[:, 101]
out = dict(name=a.name, feat=a.feat, nfeat=len(idx), std=a.std, opt=a.opt, lr=a.lr, mom=a.mom, wd=a.wd,
           arch=a.arch, act=a.act, fix=a.fix, split=a.split, seed=a.seed, nparams=nparams,
           printed_best=round(best, 4), best_epoch=best_ep, stop_epoch=stop, saves=saves,
           true_valid_mse=round(float(((pv - yva) ** 2).mean()), 4),
           true_train_mse=round(float(((ptr - ytr) ** 2).mean()), 4),
           dead_l1=dead, first_epoch=first_ep,
           test_mse_vs_day4=round(float(((pte - d4) ** 2).mean()), 4),
           test_pred_mean=round(float(pte.mean()), 4), test_pred_max=round(float(pte.max()), 4),
           n_valid=len(yva), n_train=len(ytr), secs=round(time.time() - t0, 1))
if a.feat == 'corr': out['corr_idx'] = idx
if a.perm:
    names = list(df.columns)
    base = float(((pv - yva) ** 2).mean()); rng = np.random.default_rng(0); imp = []
    with torch.no_grad():
        for j in range(len(idx)):
            Xp = Xva.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            imp.append((round(float(((f(Xp) - yva) ** 2).mean()) - base, 4), names[idx[j]]))
    out['perm_top'] = sorted(imp, reverse=True)[:8]; out['perm_bottom'] = sorted(imp)[:3]
print(json.dumps(out))
