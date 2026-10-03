"""LightGBM reference for ch07 challenge 4: noid 116 columns, random vs time split.

LightGBM is not in requirements.txt. Install it somewhere outside the project venv, e.g.
    uv pip install --python .venv/bin/python --target /tmp/lgbm lightgbm
then run from HW01/:
    PYTHONPATH=/tmp/lgbm:. ../.venv/bin/python ../docs/tools/hw01_lgbm.py
Measured 2026-10-03 with lightgbm 4.7.0 (docs/HW01/FACTS.md「ch07 審稿補測」).
"""
import numpy as np, pandas as pd, lightgbm as lgb
from utils import train_valid_split

df = pd.read_csv('covid.train.csv'); full = df.values; te = pd.read_csv('covid.test.csv').values
idx = list(range(1, 117)); d4 = te[:, 101]
mse = lambda a, b: float(((a - b) ** 2).mean())

def splits():
    yield 'random', train_valid_split(full, 0.2, 5201314)
    state = df.iloc[:, 1:38].idxmax(axis=1); rank = df.groupby(state)['id'].rank(pct=True)
    yield 'time', (full[(rank <= 0.8).values], full[(rank > 0.8).values])

for name, (tr, va) in splits():
    Xt, yt, Xv, yv, Xte = tr[:, idx], tr[:, -1], va[:, idx], va[:, -1], te[:, idx]
    m = lgb.LGBMRegressor(verbose=-1).fit(Xt, yt)
    print(f"{name:6} default(100 trees): valid {mse(m.predict(Xv), yv):.4f} train {mse(m.predict(Xt), yt):.4f} "
          f"test_vs_d4 {mse(m.predict(Xte), d4):.4f} test_max {m.predict(Xte).max():.4f}")
    m = lgb.LGBMRegressor(n_estimators=10000, verbose=-1).fit(
        Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = m.best_iteration_
    print(f"{name:6} early-stop(100):     valid {mse(m.predict(Xv, num_iteration=it), yv):.4f} best_iter {it} "
          f"train {mse(m.predict(Xt, num_iteration=it), yt):.4f} test_vs_d4 {mse(m.predict(Xte, num_iteration=it), d4):.4f} "
          f"test_max {m.predict(Xte, num_iteration=it).max():.4f}")
