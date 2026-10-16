# HW12 Reinforcement Learning: plays 5 test episodes and saves the action list for JudgeBoi
# (HW12_EN.ipynb cells [38], [39], [41], [43], [45]). Loads the network saved by train.py.
# fix() reseeds the environment and torch here, so the result depends only on the trained weights.
import os

import numpy as np
import torch

from env_setup import make_env, fix, seed
from model import PolicyGradientNetwork, PolicyGradientAgent

OUT_DIR = 'outputs'

env = make_env()
network = PolicyGradientNetwork()
network.load_state_dict(torch.load(os.path.join(OUT_DIR, 'network.pt')))
agent = PolicyGradientAgent(network)

fix(env, seed)
agent.network.eval()  # set the network into evaluation mode
NUM_OF_TEST = 5 # Do not revise this !!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
  actions = []
  state = env.reset()

  total_reward = 0

  done = False
  while not done:
      action, _ = agent.sample(state)
      actions.append(action)
      state, reward, done, _ = env.step(action)

      total_reward += reward

  print(total_reward)
  test_total_reward.append(total_reward)

  action_list.append(actions) # save the result of testing

print(np.mean(test_total_reward))

print("Action list looks like ", action_list)
print("Action list's shape looks like ", np.shape(np.array(action_list, dtype=object)))  # notebook: np.shape(action_list)

distribution = {}
for actions in action_list:
  for action in actions:
    if action not in distribution.keys():
      distribution[action] = 1
    else:
      distribution[action] += 1
print(distribution)

PATH = "Action_List.npy" # Can be modified into the name or path you want
np.save(PATH ,np.array(action_list, dtype=object))  # notebook: np.array(action_list); NumPy 2 needs dtype=object for ragged lists
