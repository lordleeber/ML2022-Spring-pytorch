# HW12 chapter 1 facts: what each action does, constant-action policies on the 100 evaluation seeds,
# the raw Gymnasium API next to the shim, and what the sample's "Final" number can be.
# usage: python docs/tools/hw12_ch01.py > docs/tools/hw12_ch01.txt
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import gymnasium
import numpy as np

import gym_compat as gym

np.set_printoptions(precision=3, suppress=True, linewidth=120)
NAMES = ['0 不動', '1 左側引擎', '2 主引擎', '3 右側引擎']

print('== raw Gymnasium API vs the shim')
raw = gymnasium.make('LunarLander-v3')
out = raw.reset(seed=7)
print('gymnasium reset ->', type(out).__name__, len(out), 'items; info =', out[1])
out = raw.step(0)
print('gymnasium step  ->', len(out), 'items: obs, reward, terminated, truncated, info =', out[1:])
print('wrappers:', raw)
sh = gym.make('LunarLander-v2')
sh.seed(7)
o = sh.reset()
print('shim reset      ->', type(o).__name__, o.shape)
print('shim step       ->', len(sh.step(0)), 'items: obs, reward, done, info')

print('\n== one action held for 20 steps from the same start (seed 7)')
for a in range(4):
    env = gymnasium.make('LunarLander-v3')
    s0, _ = env.reset(seed=7)
    s = s0
    rs = 0.0
    for t in range(20):
        s, r, term, trunc, _ = env.step(a)
        rs += r
    print(f'{NAMES[a]:8s} dvx {s[2] - s0[2]:+.3f} dvy {s[3] - s0[3]:+.3f} dangle {s[4] - s0[4]:+.3f} '
          f'dangvel {s[5] - s0[5]:+.3f} reward sum {rs:+.2f}')

print('\n== constant-action policies on the 100 evaluation seeds (100000-100099)')
for a in range(4):
    env = gymnasium.make('LunarLander-v3')
    tot, steps, outc = [], [], Counter()
    for i in range(100):
        s, _ = env.reset(seed=100000 + i)
        R, n, term, trunc = 0.0, 0, False, False
        while not (term or trunc):
            s, r, term, trunc, _ = env.step(a)
            R += r
            n += 1
        tot.append(R); steps.append(n)
        # outcome as in hw12_exp.py: the last reward is +-100 when the episode terminates
        outc['timeout' if trunc and not term else ('landed' if r >= 50 else ('crashed' if r <= -50 else 'other'))] += 1
    print(f'always {NAMES[a]:8s} mean {np.mean(tot):8.1f} min {np.min(tot):8.1f} max {np.max(tot):8.1f} '
          f'steps {np.mean(steps):6.1f} {dict(outc)}')

print('\n== the sample\'s "Final" (mean last reward of 5 episodes) over its 400 batches, seed 543')
log = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hw12_logs/ch00/train_log.json')))
fin = log['avg_final_rewards']
c = Counter(round(f, 6) for f in fin)
on_grid = sum(v for k, v in c.items() if abs(k / 40 - round(k / 40)) < 1e-9 or abs((k - 20) / 40 - round((k - 20) / 40)) < 1e-9)
print('distinct values:', len(c), 'batches whose Final is a multiple of 20 (every episode ended at +-100):', on_grid, 'of', len(fin))
print('most common:', sorted(c.items(), key=lambda kv: -kv[1])[:8])
print('values off the +-100 grid:', sorted(k for k in c if abs(k / 20 - round(k / 20)) > 1e-9)[:10])
