# HW12 chapter 6 facts: groups D (tricks stacked on GAE 0.99) and E (PPO, DQN), both snapshots, outcomes and how well the critic fits.
# usage: python docs/tools/hw12_ch06.py > docs/tools/hw12_ch06.txt
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(l) for l in open(os.path.join(here, 'hw12_runs.jsonl'))]
TAGS = ['C_gae0.99', 'D_h64', 'D_h64_ent', 'D_h64_ent_clip', 'D_h64_ent_clip_b10', 'C_mc', 'E_ppo_h16', 'E_ppo', 'E_ppo_ent', 'E_dqn']


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
