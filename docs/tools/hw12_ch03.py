# HW12 chapter 3 facts: replay the sample's training (seed 543, same RNG calls as train.py) and record, per batch,
# what the weights that multiply the log-probs look like before and after the sample's normalisation.
# usage: python docs/tools/hw12_ch03.py > docs/tools/hw12_ch03.txt   (checks the result against hw12_models/A_sample_s543.pt)
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch

from env_setup import make_env
from model import PolicyGradientNetwork, PolicyGradientAgent

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))

env = make_env()
env.reset(); env.step(env.action_space.sample()); env.reset()          # train.py's exploration cells
done = False
while not done:
    _, _, done, _ = env.step(env.action_space.sample())

agent = PolicyGradientAgent(PolicyGradientNetwork())
stats = []
for batch in range(400):
    log_probs, rewards, actions, terminal = [], [], [], []
    for episode in range(5):
        state = env.reset()
        while True:
            action, log_prob = agent.sample(state)
            state, reward, done, _ = env.step(action)
            log_probs.append(log_prob); rewards.append(reward); actions.append(action)
            terminal.append(done and abs(reward) == 100)
            if done:
                break
    r, a, term = np.array(rewards), np.array(actions), np.array(terminal)
    w = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)              # train.py line 98
    stats.append({'n': len(r), 'mean': r.mean(), 'std': r.std(), 'std_no_end': r[~term].std(),
                  'n_end': int(term.sum()), 'w_end_crash': w[term & (r < 0)].tolist(),
                  'w_mean_by_action': [float(w[a == k].mean()) if (a == k).any() else None for k in range(4)],
                  'share_by_action': [float((a == k).mean()) for k in range(4)],
                  'w_pos': float((w > 0).mean())})
    agent.learn(torch.stack(log_probs), torch.from_numpy(w))

ref = torch.load(os.path.join(here, 'hw12_models/A_sample_s543.pt'), weights_only=False)['weights']['ep']
print('same network as hw12_exp.py / train.py (seed 543):', all(torch.equal(ref[k], agent.network.state_dict()[k]) for k in ref))


def show(name, rows):
    m = lambda k: np.mean([s[k] for s in rows])
    crash_w = [x for s in rows for x in s['w_end_crash']]
    wa = [np.mean([s['w_mean_by_action'][k] for s in rows if s['w_mean_by_action'][k] is not None]) for k in range(4)]
    sh = [np.mean([s['share_by_action'][k] for s in rows]) for k in range(4)]
    print(f'{name}: steps/batch {m("n"):.0f}, reward mean {m("mean"):.3f}, std {m("std"):.2f}, std without the +-100 steps '
          f'{m("std_no_end"):.2f}, +-100 steps/batch {m("n_end"):.2f}, weight of a crash step {np.mean(crash_w):.2f} '
          f'(min {np.min(crash_w):.2f}), share of positive weights {m("w_pos"):.3f}')
    print('   mean normalised weight by action [0,1,2,3]:', np.round(wa, 3), ' share of steps:', np.round(sh, 3))


show('batches 1-50', stats[:50])
show('batches 351-400', stats[350:])

print('\n== learning rate (group A, 10 seeds each, closed-loop stochastic at the 2,000-episode snapshot)')
runs = [json.loads(l) for l in open(os.path.join(here, 'hw12_runs.jsonl'))]
for tag in ('A_lr1e-4', 'A_sample', 'A_lr1e-2'):
    rs = sorted((r for r in runs if r['tag'] == tag and r['seed'] < 10), key=lambda r: r['seed'])
    print(tag, [(r['seed'], round(r['snapshots']['ep']['closed_stochastic']['mean'], 1),
                 round(r['snapshots']['ep']['closed_stochastic']['mean_steps'])) for r in rs])
    print('   episodes per 1M steps:', [r['snapshots']['steps']['episodes'] for r in rs])
