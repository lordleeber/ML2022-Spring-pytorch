# HW12 chapter 6: how large is the policy-gradient norm of the summed loss, compared with --clip_grad 1?
# Uses the saved D_full (hidden 64) policies; weights = batch-normalised reward-to-go (gamma 0.99), 5 episodes per batch.
# usage: python docs/tools/hw12_ch06_gradnorm.py > docs/tools/hw12_ch06_gradnorm.txt
import os
import sys

import gymnasium
import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw12_ch07 import load

torch.set_num_threads(1)
for seed in range(3):
    for snap in ('ep', 'steps'):
        net, _ = load(f'D_full_s{seed}', snap)
        env = gymnasium.make('LunarLander-v3')
        torch.manual_seed(seed)
        norms, steps = [], []
        for b in range(5):
            lps, rews = [], []
            for e in range(5):
                s, _ = env.reset(seed=1000 * seed + 10 * b + e)
                rs, term, trunc = [], False, False
                while not (term or trunc):
                    d = Categorical(net(torch.FloatTensor(s)))
                    a = d.sample()
                    lps.append(d.log_prob(a))
                    s, r, term, trunc, _ = env.step(a.item())
                    rs.append(r)
                g, out = 0.0, []
                for r in reversed(rs):
                    g = r + 0.99 * g
                    out.append(g)
                rews += out[::-1]
            w = (np.array(rews) - np.mean(rews)) / (np.std(rews) + 1e-9)
            net.zero_grad()
            (-torch.stack(lps) * torch.from_numpy(w)).sum().backward()
            norms.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters()))))
            steps.append(len(rews))
        print(f'D_full_s{seed} {snap:5s}: steps per batch {np.mean(steps):6.0f}, grad norm of the summed loss mean {np.mean(norms):7.1f} '
              f'(min {np.min(norms):6.1f}, max {np.max(norms):7.1f}) -> clip 1 scales the step by about {1 / np.mean(norms):.3f}')
