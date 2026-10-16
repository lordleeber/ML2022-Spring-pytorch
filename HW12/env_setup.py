# HW12 Reinforcement Learning: random seed and the Lunar Lander environment (HW12_EN.ipynb cells [6], [8]).
# The sample is the 2021 one (ML2021-Spring); the 2022 sample code was never published.
import random

import numpy as np
import torch

import gym_compat as gym  # notebook: import gym (gym 0.18.3); see gym_compat.py

seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms(True)  # notebook: torch.set_deterministic(True), removed from torch
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def make_env():
  env = gym.make('LunarLander-v2')
  fix(env, seed) # fix the environment Do not revise this !!!
  return env
