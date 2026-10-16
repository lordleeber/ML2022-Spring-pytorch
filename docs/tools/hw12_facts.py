# HW12 facts for chapters 0-2: model shapes and parameters, what the environment returns, one test episode
# step by step with its reward split into the terms of LunarLander.step, the random policy, and
# Gymnasium's hand-written heuristic controller on the same 100 evaluation seeds as hw12_exp.py.
# usage: python docs/tools/hw12_facts.py docs/tools/hw12_models/A_sample_s543.pt > docs/tools/hw12_facts.txt
#        (or HW12/outputs/network.pt from train.py, or a hw12_exp.py --dump file: the same seed-543 sample network)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch
from torch.distributions import Categorical
from gymnasium.envs.box2d.lunar_lander import heuristic

import gym_compat as gym
from env_setup import fix, seed
from model import PolicyGradientNetwork

torch.set_num_threads(1)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print('== model')
net = PolicyGradientNetwork()
for name, prm in net.named_parameters():
    print(f'{name:12s} {str(tuple(prm.shape)):10s} {prm.numel()}')
print('total', sum(prm.numel() for prm in net.parameters()))
x = torch.zeros(3, 8)
print('shapes: in', tuple(x.shape), 'fc1', tuple(net.fc1(x).shape), 'out', tuple(net(x).shape))
print(net)

print('\n== environment (gym_compat over Gymnasium)')
env = gym.make('LunarLander-v2')
fix(env, seed)
print('observation_space', env.observation_space)
print('action_space', env.action_space)
s0 = env.reset()
print('first reset after fix(env, 543):', s0)
a = env.action_space.sample()
print('action_space.sample():', a)
s1, r1, d1, _ = env.step(a)
print('step -> reward', r1, 'done', d1)
env.reset()
n, done = 0, False
while not done:
    _, _, done, _ = env.step(env.action_space.sample())
    n += 1
print('random-agent episode of cell [24]:', n, 'steps')
print('max_episode_steps', env.env.spec.max_episode_steps, 'reward_threshold', env.env.spec.reward_threshold)

ckpt = torch.load(sys.argv[1], weights_only=False)
# a train.py network.pt, a hw12_exp.py --dump file, or a hw12_exp.py --save file (its 2,000-episode snapshot)
state_dict = ckpt['weights']['ep'] if 'weights' in ckpt else (ckpt['network'] if 'network' in ckpt else ckpt)
policy = PolicyGradientNetwork()
policy.load_state_dict(state_dict)
policy.eval()


def shaping(s):
    return (-100 * np.sqrt(s[0] ** 2 + s[1] ** 2) - 100 * np.sqrt(s[2] ** 2 + s[3] ** 2)
            - 100 * abs(s[4]) + 10 * s[6] + 10 * s[7])


print('\n== test episodes of the sample policy (fix(env, 543), sampled actions, as test.py)')
env = gym.make('LunarLander-v2')
fix(env, seed)
with torch.no_grad():
    for ep in range(5):
        state, done, total, t, rows = env.reset(), False, 0.0, 0, []
        counts = [0, 0, 0, 0]
        parts = {'shaping': 0.0, 'main': 0.0, 'side': 0.0, 'end': 0.0}
        prev = shaping(state)
        while not done:
            probs = policy(torch.FloatTensor(state))
            action = Categorical(probs).sample().item()
            counts[action] += 1
            nxt, r, done, _ = env.step(action)
            fuel = -0.3 if action == 2 else (-0.03 if action in (1, 3) else 0.0)
            sh = shaping(nxt)
            if done and abs(r) == 100:
                parts['end'] += r
            else:
                parts['shaping'] += sh - prev
                parts['main' if action == 2 else 'side'] += fuel
                # the env computes the shaping difference against its own previous shaping; check it
                assert abs((sh - prev + fuel) - r) < 1e-3, (t, r, sh - prev + fuel)
            if ep == 3 and (t < 3 or t % 40 == 0 or done):
                rows.append((t, action, np.round(probs.numpy(), 3), np.round(nxt, 3), round(r, 3), round(total + r, 2)))
            prev, state = sh, nxt
            total += r
            t += 1
        print(f'episode {ep}: steps {t} total {total:.2f} last reward {r:.2f} actions {counts} parts',
              {k: round(float(v), 2) for k, v in parts.items()})
        for row in rows:
            print('   t=%d a=%d probs=%s state=%s r=%s cum=%s' % row)

print('\n== uniform random policy and the Gymnasium heuristic on the 100 closed-loop seeds (100000-100099)')
genv = gym.make('LunarLander-v2').env
for name in ('random', 'heuristic'):
    totals, lengths, outcomes = [], [], []
    for i in range(100):
        sd = 100000 + i
        torch.manual_seed(sd)
        st, _ = genv.reset(seed=sd)
        tot, n, term, trunc = 0.0, 0, False, False
        while not (term or trunc):
            act = heuristic(genv.unwrapped, st) if name == 'heuristic' else Categorical(probs=torch.full((4,), 0.25)).sample().item()
            st, r, term, trunc, _ = genv.step(act)
            tot += r
            n += 1
        totals.append(tot); lengths.append(n)
        # outcome as in hw12_exp.py: the last reward is +-100 when the episode terminates
        outcomes.append('timeout' if trunc and not term else ('landed' if r >= 50 else ('crashed' if r <= -50 else 'other')))
    t = np.array(totals)
    print(f'{name:9s} mean {t.mean():.1f} std {t.std():.1f} min {t.min():.1f} max {t.max():.1f} '
          f'>=200 {(t >= 200).mean():.2f} mean steps {np.mean(lengths):.0f} '
          f'landed {outcomes.count("landed")} crashed {outcomes.count("crashed")} timeout {outcomes.count("timeout")} other {outcomes.count("other")}')
