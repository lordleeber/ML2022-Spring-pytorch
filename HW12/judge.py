# HW12 Reinforcement Learning: the notebook's simulation of the judge server (HW12_EN.ipynb cells [49], [51]).
# Replays an action list in the seeded environment and prints the rewards.
# usage: python judge.py [Action_List.npy]
import sys

import numpy as np

from env_setup import make_env, fix

PATH = sys.argv[1] if len(sys.argv) > 1 else "Action_List.npy"
env = make_env()

action_list = np.load(PATH,allow_pickle=True) # The action list you upload
seed = 543 # Do not revise this
fix(env, seed)
# notebook: agent.network.eval() here; replaying actions never calls the network, so it is dropped

test_total_reward = []
if len(action_list) != 5:
  print("Wrong format of file !!!")
  exit(0)
for actions in action_list:
  state = env.reset()

  total_reward = 0

  done = False

  for action in actions:

      state, reward, done, _ = env.step(action)
      total_reward += reward
      if done:
        break

  print(f"Your reward is : %.2f"%total_reward)
  test_total_reward.append(total_reward)

print(f"Your final reward is : %.2f"%np.mean(test_total_reward))
