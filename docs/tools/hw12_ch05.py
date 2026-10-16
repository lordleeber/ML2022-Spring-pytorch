# HW12 chapter 5 facts: group C (baseline and critic), both snapshots, outcomes and how well the critic fits.
# usage: python docs/tools/hw12_ch05.py > docs/tools/hw12_ch05.txt
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(l) for l in open(os.path.join(here, 'hw12_runs.jsonl'))]
TAGS = ['B_rtg0.99', 'C_rtg_adam', 'C_mc_sgd', 'C_mc_ci5', 'C_mc', 'C_td0', 'C_n5', 'C_n20', 'C_gae0.95_ci5', 'C_gae0.95', 'C_gae0.99']


def f(xs):
    xs = np.array(xs, dtype=float)
    return f'{xs.mean():7.1f} ±{xs.std():5.1f} [{xs.min():7.1f},{xs.max():7.1f}]'


for tag in TAGS:
    rs = [r for r in runs if r['tag'] == tag and r['seed'] < 10]
    for snap in ('ep', 'steps'):
        ss = [r['snapshots'][snap] for r in rs]
        if not ss:
            continue
        sto, gr = [s['closed_stochastic'] for s in ss], [s['closed_greedy'] for s in ss]
        ev = [s['critic_ev'] for s in ss if s.get('critic_ev') is not None]
        print(f'{tag:14s} {snap:5s} n={len(ss)} judge {f([s["judge_mean"] for s in ss])} sto {f([x["mean"] for x in sto])} '
              f'greedy {f([x["mean"] for x in gr])}')
        print(f'{"":20s} sto L/C/T {np.mean([x["landed"] for x in sto]):4.1f}/{np.mean([x["crashed"] for x in sto]):4.1f}/{np.mean([x["timeout"] for x in sto]):4.1f}'
              f'  greedy L/C/T {np.mean([x["landed"] for x in gr]):4.1f}/{np.mean([x["crashed"] for x in gr]):4.1f}/{np.mean([x["timeout"] for x in gr]):4.1f}'
              f'  sto>=200 {np.mean([x["solved_rate"] for x in sto]):.2f}  episodes {np.mean([s["episodes"] for s in ss]):5.0f}'
              f'  steps {np.mean([s["env_steps"] for s in ss]):7.0f}  train_s {np.mean([s["train_s"] for s in ss]):5.0f}'
              + (f'  critic EV {np.mean(ev):.3f} [{np.min(ev):.3f},{np.max(ev):.3f}]' if ev else ''))
