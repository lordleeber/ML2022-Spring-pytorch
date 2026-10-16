# HW12 chapter 4 facts: group B summary by gamma, and what a "hovering" greedy policy actually does.
# usage: python docs/tools/hw12_ch04.py > docs/tools/hw12_ch04.txt   (needs hw12_runs.jsonl group B and hw12_models/B_rtg*_s*.pt)
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import gymnasium
import numpy as np
import torch
from torch.distributions import Categorical

from model import PolicyGradientNetwork

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(l) for l in open(os.path.join(here, 'hw12_runs.jsonl'))]

print('== group B and the sample (seeds 0-9), both snapshots')
tags = ['A_sample', 'B_total', 'B_rtg0.9', 'B_rtg0.95', 'B_rtg0.99', 'B_rtg0.999', 'B_rtg1.0', 'B_rtg0.99_nonorm']
for tag in tags:
    rs = [r for r in runs if r['tag'] == tag and r['seed'] < 10]
    for snap in ('ep', 'steps'):
        ss = [r['snapshots'][snap] for r in rs]
        f = lambda xs: '%7.1f ±%5.1f' % (np.mean(xs), np.std(xs))
        g, s = [x['closed_greedy'] for x in ss], [x['closed_stochastic'] for x in ss]
        print(f'{tag:17s} {snap:5s} n={len(ss)} judge {f([x["judge_mean"] for x in ss])} sto {f([x["mean"] for x in s])} '
              f'greedy {f([x["mean"] for x in g])} | sto L/C/T {np.mean([x["landed"] for x in s]):4.1f}/{np.mean([x["crashed"] for x in s]):4.1f}/{np.mean([x["timeout"] for x in s]):4.1f}'
              f' greedy L/C/T {np.mean([x["landed"] for x in g]):4.1f}/{np.mean([x["crashed"] for x in g]):4.1f}/{np.mean([x["timeout"] for x in g]):4.1f}'
              f' | eps {np.mean([x["episodes"] for x in ss]):5.0f} steps {np.mean([x["env_steps"] for x in ss]):7.0f}'
              f' sto>=200 {np.mean([x["solved_rate"] for x in s]):.2f}')
    if tag.startswith('B_rtg') and not tag.endswith('nonorm'):
        hov = [r['seed'] for r in rs if r['snapshots']['ep']['closed_greedy']['timeout'] >= 50]
        print(f'   seeds whose greedy policy times out on >= 50 of 100 seeds (2,000 episodes): {hov}')

print('\n== one hovering greedy policy, replayed: B_rtg0.99, the seed with the most greedy timeouts')
rs = [r for r in runs if r['tag'] == 'B_rtg0.99']
r = max(rs, key=lambda r: r['snapshots']['ep']['closed_greedy']['timeout'])
print('seed', r['seed'], 'greedy timeouts', r['snapshots']['ep']['closed_greedy']['timeout'], 'stochastic timeouts',
      r['snapshots']['ep']['closed_stochastic']['timeout'])
w = torch.load(os.path.join(here, f'hw12_models/B_rtg0.99_s{r["seed"]}.pt'), weights_only=False)['weights']['ep']
pol = PolicyGradientNetwork()
pol.load_state_dict(w)
env = gymnasium.make('LunarLander-v3')
for greedy in (True, False):
    torch.manual_seed(100000)
    s, _ = env.reset(seed=100000)
    acts, ys, xs, speeds, total, t = [], [], [], [], 0.0, 0
    term = trunc = False
    with torch.no_grad():
        while not (term or trunc):
            p = pol(torch.FloatTensor(s))
            a = int(p.argmax()) if greedy else Categorical(p).sample().item()
            s, rew, term, trunc, _ = env.step(a)
            acts.append(a); xs.append(s[0]); ys.append(s[1]); speeds.append(float(np.hypot(s[2], s[3])))
            total += rew
            t += 1
    acts = np.array(acts)
    print(f'{"greedy" if greedy else "stochastic"}: env seed 100000, {t} steps, total {total:.1f}, '
          f'{"truncated" if trunc and not term else ("landed" if rew > 0 else "crashed")}, action shares {np.round(np.bincount(acts, minlength=4) / t, 3)}')
    for k in (0, 100, 200, 400, 600, 800, t - 1):
        if k < t:
            print(f'   t={k:4d} x {xs[k]:+.3f} y {ys[k]:.3f} speed {speeds[k]:.3f} last-20 actions {acts[max(0, k - 19):k + 1].tolist()}')
    if greedy:
        tail = acts[200:]
        print(f'   steps 200..end: main engine share {np.mean(tail == 2):.3f}, mean height {np.mean(ys[200:]):.3f}, mean speed {np.mean(speeds[200:]):.3f}')
        # potential-based shaping: a hovering policy pays fuel every step it fires the main engine
        print(f'   fuel paid for the main engine over the episode: {0.3 * np.sum(acts == 2):.1f}')
