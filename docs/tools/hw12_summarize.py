# Summarise docs/tools/hw12_runs.jsonl by tag: mean ± std (min..max) over training seeds, for both snapshots.
# usage: python docs/tools/hw12_summarize.py [tag prefix ...]
import json
import os
import sys
from collections import defaultdict

import numpy as np

runs = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hw12_runs.jsonl'))]
prefixes = sys.argv[1:]
groups = defaultdict(list)
for r in runs:
    if not prefixes or any(r['tag'].startswith(p) for p in prefixes):
        groups[r['tag']].append(r)


def stat(v):
    v = np.array(v, dtype=float)
    return f'{v.mean():7.1f} ±{v.std():5.1f} ({v.min():6.1f}..{v.max():6.1f})'


print(f'{"tag":24s} {"snap":5s} {"n":>2s}  {"judge":28s} {"closed stochastic":28s} {"closed greedy":28s} '
      f'{"solved_g":>8s} {"land/crash/tmo g":>16s} {"episodes":>8s} {"steps":>8s} {"train_s":>7s}')
for tag, rs in groups.items():
    for snap in ('ep', 'steps'):
        ss = [r['snapshots'][snap] for r in rs if snap in r['snapshots']]
        if not ss:
            continue
        g = [s['closed_greedy'] for s in ss]
        print(f'{tag:24s} {snap:5s} {len(ss):2d}  {stat([s["judge_mean"] for s in ss])} '
              f'{stat([s["closed_stochastic"]["mean"] for s in ss])} {stat([x["mean"] for x in g])} '
              f'{np.mean([x["solved_rate"] for x in g]):8.2f} '
              f'{np.mean([x["landed"] for x in g]):5.1f}/{np.mean([x["crashed"] for x in g]):4.1f}/{np.mean([x["timeout"] for x in g]):4.1f} '
              f'{np.mean([s["episodes"] for s in ss]):8.0f} {np.mean([s["env_steps"] for s in ss]):8.0f} '
              f'{np.mean([s["train_s"] for s in ss]):7.0f}')
