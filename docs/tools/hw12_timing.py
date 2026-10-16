# HW12 clean timing (run with nothing else on the machine): one policy step on CPU vs GPU, and one environment step.
# usage: OMP_NUM_THREADS=1 python docs/tools/hw12_timing.py > docs/tools/hw12_timing.txt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch
from torch.distributions import Categorical

import gym_compat as gym
from model import PolicyGradientNetwork

torch.set_num_threads(1)
N = 5000
state = np.zeros(8, dtype=np.float32)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
for dev in ['cpu', 'cuda']:
    if dev == 'cuda' and not torch.cuda.is_available():
        continue
    net = PolicyGradientNetwork().to(dev)
    for rep in range(3):  # the first repetition warms up (CUDA context, kernels)
        t = time.perf_counter()
        for _ in range(N):
            probs = net(torch.FloatTensor(state).to(dev))
            dist = Categorical(probs)
            a = dist.sample()
            lp = dist.log_prob(a)
            act = a.item()
        if dev == 'cuda':
            torch.cuda.synchronize()
        us = (time.perf_counter() - t) / N * 1e6
    print(f'policy forward + sample + log_prob + item on {dev}: {us:.1f} us/step (last of 3 repetitions)')

env = gym.make('LunarLander-v2')
env.seed(0)
env.reset()
t, n = time.perf_counter(), 0
rng = np.random.default_rng(0)
while n < N:
    _, _, done, _ = env.step(int(rng.integers(4)))
    n += 1
    if done:
        env.reset()
print(f'LunarLander step (random actions, with resets): {(time.perf_counter() - t) / N * 1e6:.1f} us/step')
