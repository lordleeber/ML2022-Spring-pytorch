# HW12 chapter 5: where are the truncated (1,000-step) episodes of the better policies -- in the air or on the ground?
# usage: python docs/tools/hw12_ch05_timeouts.py > docs/tools/hw12_ch05_timeouts.txt
import os
import sys

import gymnasium
import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw12_ch07 import load  # same MLP / weight loading

torch.set_num_threads(1)
import json
runs = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hw12_runs.jsonl'))]


def worst(tag):
    # the saved seed whose greedy policy was truncated most often at the 1M-step snapshot
    r = max((r for r in runs if r['tag'] == tag and r['seed'] < 10), key=lambda r: r['snapshots']['steps']['closed_greedy']['timeout'])
    return f'{tag}_s{r["seed"]}', r['snapshots']['steps']['closed_greedy']['timeout']


picks = [('B_rtg0.99_s1', 'ep')] + [(worst(t)[0], 'steps') for t in ('C_rtg_adam', 'C_mc', 'C_gae0.99')]
print('policies:', picks)
for name, snap in picks:
    net, dqn = load(name, snap)
    for greedy in (False, True):
        env = gymnasium.make('LunarLander-v3')
        rows = []
        with torch.no_grad():
            for i in range(100):
                sd = 100000 + i
                torch.manual_seed(sd)
                s, _ = env.reset(seed=sd)
                tot, term, trunc, ground = 0.0, False, False, 0
                while not (term or trunc):
                    p = net(torch.FloatTensor(s))
                    a = int(p.argmax()) if greedy else Categorical(p).sample().item()
                    s, r, term, trunc, _ = env.step(a)
                    tot += r
                    ground += int(s[6] == 1 and s[7] == 1)
                if trunc and not term:
                    rows.append((s[1], s[6] + s[7], ground, tot, abs(s[0])))
        if not rows:
            print(f'{name:15s} {"greedy" if greedy else "stoch."}: no truncated episodes'); continue
        y, legs, g, tot, x = map(np.array, zip(*rows))
        on = legs == 2
        print(f'{name:15s} {"greedy" if greedy else "stoch."}: {len(rows):3d} truncated; ended with both legs down {on.sum():3d} '
              f'(mean total {tot[on].mean() if on.any() else float("nan"):6.1f}, |x| {x[on].mean() if on.any() else float("nan"):.2f}, steps with both legs down {g[on].mean() if on.any() else float("nan"):5.0f}); '
              f'in the air {(~on).sum():3d} (mean height {y[~on].mean() if (~on).any() else float("nan"):.2f}, mean total {tot[~on].mean() if (~on).any() else float("nan"):6.1f})')
