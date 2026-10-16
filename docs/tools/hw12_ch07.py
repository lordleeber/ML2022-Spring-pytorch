# HW12 chapter 7 facts (inference only, on saved weights): how much of the judge score is luck.
#  1. rerun test.py's 5 episodes with K different torch seeds (environment seed stays 543): spread and best-of-K
#  2. open loop: replay the seed-543 action list on other environment seeds
#  3. what judge.py does with a too-short / too-long action list
#  4. judge vs closed-loop over every run in hw12_runs.jsonl
# usage: python docs/tools/hw12_ch07.py > docs/tools/hw12_ch07.txt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym_compat as gym
from env_setup import fix

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(l) for l in open(os.path.join(here, 'hw12_runs.jsonl'))]


class MLP(nn.Module):
    # same layers as hw12_exp.py's MLP (and HW12/model.py when hidden=16)
    def __init__(self, hidden, out, act=torch.tanh, softmax=False):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(8, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, out)
        self.act, self.softmax = act, softmax

    def forward(self, s):
        h = self.act(self.fc2(self.act(self.fc1(s))))
        o = self.fc3(h)
        return F.softmax(o, -1) if self.softmax else o


def load(name, snap):
    d = torch.load(os.path.join(here, 'hw12_models', name + '.pt'), weights_only=False)
    cfg = d['config']
    dqn = cfg['algo'] == 'dqn'
    net = MLP(cfg['hidden'], 4, act=F.relu if dqn else torch.tanh, softmax=not dqn)
    net.load_state_dict(d['weights'][snap])
    return net, dqn


def play(net, dqn, torch_seed, greedy=False, env_seed=543):
    # test.py: fix(env, 543) then 5 episodes in a row on one environment; torch reseeded to torch_seed
    env = gym.make('LunarLander-v2')
    fix(env, env_seed)
    torch.manual_seed(torch_seed)
    rewards, actions = [], []
    with torch.no_grad():
        for _ in range(5):
            s, done, tot, acts = env.reset(), False, 0.0, []
            while not done:
                out = net(torch.FloatTensor(s))
                a = int(out.argmax()) if (dqn or greedy) else Categorical(out).sample().item()
                s, r, done, _ = env.step(a)
                tot += r
                acts.append(a)
            rewards.append(tot)
            actions.append(acts)
    return rewards, actions


def replay(actions, env_seed=543, mode='judge'):
    # judge.py: fix, then for each list reset and step until done (extra actions ignored, missing ones just stop)
    env = gym.make('LunarLander-v2')
    fix(env, env_seed)
    out, ended = [], []
    for acts in actions:
        env.reset()
        tot, done = 0.0, False
        for a in acts:
            _, r, done, _ = env.step(a)
            tot += r
            if done:
                break
        out.append(tot)
        ended.append(done)
    return out, ended


def main():
    POLICIES = [('A_sample_s543', 'ep'), ('A_sample_s0', 'ep'), ('B_rtg0.99_s0', 'steps'), ('C_gae0.99_s0', 'steps'),
                ('E_ppo_s0', 'steps'), ('E_dqn_s0', 'steps')]
    K = 100
    print(f'== 1. the same policy, test.py rerun with {K} torch seeds (environment seed 543 every time)')
    for name, snap in POLICIES:
        if not os.path.exists(os.path.join(here, 'hw12_models', name + '.pt')):
            print(name, 'missing'); continue
        net, dqn = load(name, snap)
        means = np.array([np.mean(play(net, dqn, ts)[0]) for ts in range(K)])
        g = np.mean(play(net, dqn, 0, greedy=True)[0])
        best = [np.mean([means[np.random.default_rng(j).choice(K, k, replace=False)].max() for j in range(200)]) for k in (1, 5, 20, 100)]
        print(f'{name:16s} {snap:5s} judge over {K} reruns: mean {means.mean():7.1f} std {means.std():5.1f} min {means.min():7.1f} '
              f'max {means.max():7.1f} | greedy judge {g:7.1f} | expected best of 1/5/20/100: {np.round(best, 1).tolist()} '
              f'| share of reruns >= 200: {(means >= 200).mean():.2f}')

    print('\n== 2. open loop: the seed-543 action list replayed on other environment seeds (A_sample_s543 and C_gae0.99_s0)')
    for name, snap in [('A_sample_s543', 'ep'), ('C_gae0.99_s0', 'steps')]:
        if not os.path.exists(os.path.join(here, 'hw12_models', name + '.pt')):
            continue
        net, dqn = load(name, snap)
        rew, acts = play(net, dqn, 543)
        same, _ = replay(acts, 543)
        others = [np.mean(replay(acts, es)[0]) for es in range(1, 101)]
        closed = [np.mean(play(net, dqn, 543, env_seed=es)[0]) for es in range(1, 101)]
        print(f'{name}: judge (seed 543) {np.mean(same):.1f} == test {np.mean(rew):.1f}; same actions replayed on env seeds 1-100: '
              f'mean {np.mean(others):.1f} (min {np.min(others):.1f}, max {np.max(others):.1f}); the policy itself playing those seeds: '
              f'mean {np.mean(closed):.1f}')

    print('\n== 3. judge.py and the action-list length (A_sample_s543)')
    net, dqn = load('A_sample_s543', 'ep')
    rew, acts = play(net, dqn, 543)
    print('lengths', [len(a) for a in acts], 'judge', [round(float(x), 2) for x in replay(acts)[0]])
    half = [a[:len(a) // 2] for a in acts]
    r, e = replay(half)
    print('first half of every list: judge', [round(float(x), 2) for x in r], 'mean', round(float(np.mean(r)), 2), 'episodes that reached done:', e)
    longer = [a + [0] * 50 for a in acts]
    r, e = replay(longer)
    print('50 extra actions per list: judge', [round(float(x), 2) for x in r], 'mean', round(float(np.mean(r)), 2))
    shifted = acts[1:] + acts[:1]
    r, e = replay(shifted)
    print('lists rotated by one episode: judge', [round(float(x), 2) for x in r], 'mean', round(float(np.mean(r)), 2), 'ended:', e)

    print('\n== 4. judge vs closed-loop stochastic over every training run and snapshot in hw12_runs.jsonl')
    J, Cs = [], []
    for r in runs:
        for s in r['snapshots'].values():
            J.append(s['judge_mean']); Cs.append(s['closed_stochastic']['mean'])
    J, Cs = np.array(J), np.array(Cs)
    print(f'{len(J)} snapshots, corr {np.corrcoef(J, Cs)[0, 1]:.3f}; |judge - closed| mean {np.mean(np.abs(J - Cs)):.1f}, '
          f'share > 50: {(np.abs(J - Cs) > 50).mean():.2f}')
    ok = Cs > -150
    print(f'excluding collapsed runs (closed > -150): {ok.sum()} snapshots, corr {np.corrcoef(J[ok], Cs[ok])[0, 1]:.3f}')


if __name__ == '__main__':
    main()
