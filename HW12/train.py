# HW12 Reinforcement Learning: trains the policy gradient agent (HW12_EN.ipynb cells [11]-[24], [30], [32], [34], [36]).
# Prints the same output as the notebook; saves the network and the training curves to outputs/.
# The exploration cells [15]-[24] stay in front of training: their reset() and step() calls advance
# the environment's RNG, so removing them changes every training episode.
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # notebook: tqdm.notebook

from env_setup import make_env
from model import PolicyGradientNetwork, PolicyGradientAgent

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

env = make_env()

# ---- What Lunar Lander? (cells [11]-[24]) ----
print(env.observation_space)
print(env.action_space)
initial_state = env.reset()
print(initial_state)
random_action = env.action_space.sample()
print(random_action)
observation, reward, done, info = env.step(random_action)
print(done)
print(reward)

# Random agent (cell [24]); the frames it drew are dropped, see gym_compat.py
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

# ---- Build and train (cells [30], [32]) ----
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

agent.network.train()  # Switch network into training mode
EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
NUM_BATCH = 400        # totally update the agent for 400 time

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):

        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) # change here
            # ! IMPORTANT !
            # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
            #                                                         rewards :     r1, r2 ,r3 ......
            # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
            #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement DQN
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)

                break

    print(f"rewards looks like ", np.shape(rewards))
    print(f"log_probs looks like ", (len(log_probs),))  # notebook: np.shape(log_probs), which NumPy 2 rejects
    # record training process
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # update agent
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    print("logs prob looks like ", torch.stack(log_probs).size())
    print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

# ---- Training result (cells [34], [36]); saved instead of shown ----
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.savefig(os.path.join(OUT_DIR, 'total_rewards.png'))
plt.close()

plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.savefig(os.path.join(OUT_DIR, 'final_rewards.png'))
plt.close()

# the notebook keeps the agent in memory for testing; the script hands it to test.py
torch.save(agent.network.state_dict(), os.path.join(OUT_DIR, 'network.pt'))
json.dump({'avg_total_rewards': avg_total_rewards, 'avg_final_rewards': avg_final_rewards},
          open(os.path.join(OUT_DIR, 'train_log.json'), 'w'))
