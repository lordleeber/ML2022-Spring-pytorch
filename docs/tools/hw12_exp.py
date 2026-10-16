# HW12 experiment tool. With the defaults it trains exactly like HW12/train.py (same RNG calls in the
# same order); `--step_budget 0 --dump d.pt` stops at the notebook's 2,000 episodes and saves what
# docs/tools/hw12_check_split.py-style checks compare against the reference script.
#
# One training run, two snapshots: after the update that completes --ep_budget episodes (the sample's
# 400 x 5), and after the update that reaches --step_budget environment steps. Each snapshot is scored
#   judge  : HW12/test.py + judge.py -- fix(env, 543), 5 episodes, sampled actions, mean reward
#   closed : 100 unseen environment seeds, the policy acting live, sampled and greedy
# Evaluation uses its own environments and restores every RNG afterwards, so it never changes training.
# usage: python docs/tools/hw12_exp.py [--algo pg|ppo|dqn|random] [options] --seed 0 --out runs.jsonl
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym_compat as gym
from env_setup import fix

p = argparse.ArgumentParser()
p.add_argument('--algo', default='pg', choices=['pg', 'ppo', 'dqn', 'random'])
p.add_argument('--seed', type=int, default=543, help='training seed (the notebook fixes 543)')
p.add_argument('--no_explore', action='store_true', help='skip the notebook exploration cells [15]-[24]')
p.add_argument('--ep_budget', type=int, default=2000, help='first snapshot: episodes (sample: 400 batches x 5)')
p.add_argument('--step_budget', type=int, default=1_000_000, help='second snapshot: env steps (0 = none)')
# policy gradient family (the sample is --returns immediate --adv none --norm batch --optim sgd --lr 0.001)
p.add_argument('--episode_per_batch', type=int, default=5)
p.add_argument('--returns', default='immediate', choices=['immediate', 'rtg', 'total'],
               help='weight of each log-prob: immediate r_t (sample), discounted reward-to-go (slide p.4), episode total')
p.add_argument('--gamma', type=float, default=0.99)
p.add_argument('--adv', default='none', choices=['none', 'mc', 'gae', 'nstep'],
               help='mc: R_t - V(s_t) with V fit to R_t (slide p.5); gae / nstep: TD critic (A2C)')
p.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
p.add_argument('--nstep', type=int, default=5)
p.add_argument('--norm', default='batch', choices=['batch', 'none'], help='standardise the weights over the batch')
p.add_argument('--optim', default='sgd', choices=['sgd', 'adam'])
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--critic_lr', type=float, default=0.001)
p.add_argument('--critic_iters', type=int, default=5)
p.add_argument('--hidden', type=int, default=16)
p.add_argument('--entropy', type=float, default=0.0, help='entropy bonus coefficient')
p.add_argument('--clip_grad', type=float, default=0.0, help='max grad norm (0 = off)')
# PPO
p.add_argument('--rollout', type=int, default=2048)
p.add_argument('--ppo_epochs', type=int, default=10)
p.add_argument('--minibatch', type=int, default=64)
p.add_argument('--clip', type=float, default=0.2)
p.add_argument('--vf_coef', type=float, default=0.5)
# DQN
p.add_argument('--buffer', type=int, default=100_000)
p.add_argument('--dqn_batch', type=int, default=64)
p.add_argument('--learn_start', type=int, default=10_000)
p.add_argument('--train_freq', type=int, default=4)
p.add_argument('--target_update', type=int, default=1000)
p.add_argument('--eps_end', type=float, default=0.01)
p.add_argument('--eps_decay', type=int, default=100_000)
# evaluation and output
p.add_argument('--eval_episodes', type=int, default=100)
p.add_argument('--eval_base', type=int, default=100000, help='closed-loop eval uses env seeds base..base+N-1')
p.add_argument('--tag', default='')
p.add_argument('--out')
p.add_argument('--dump', help='save curves, weights and judge actions at the episode snapshot')
p.add_argument('--save', help='save the network weights of both snapshots to this .pt file')
opt = p.parse_args()
torch.set_num_threads(1)

JUDGE_SEED = 543


# ---------------------------------------------------------------- networks and agents
class MLP(nn.Module):
    # hidden=16, act=tanh, softmax=True is HW12/model.py's PolicyGradientNetwork: same layers, same init order
    def __init__(self, hidden, out, act=torch.tanh, softmax=False):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out)
        self.act, self.softmax = act, softmax

    def forward(self, state):
        hid = self.act(self.fc1(state))
        hid = self.act(self.fc2(hid))
        out = self.fc3(hid)
        return F.softmax(out, dim=-1) if self.softmax else out


class PolicyAgent:
    def __init__(self, policy):
        self.policy = self.net = policy

    def act(self, state, greedy):
        probs = self.policy(torch.FloatTensor(state))
        return int(probs.argmax()) if greedy else Categorical(probs).sample().item()


class QAgent:
    # DQN acts greedily on Q in both evaluation modes
    def __init__(self, q):
        self.q = self.net = q

    def act(self, state, greedy):
        return int(self.q(torch.FloatTensor(state)).argmax())


class RandomAgent:
    def act(self, state, greedy):
        return Categorical(probs=torch.full((4,), 0.25)).sample().item()


# ---------------------------------------------------------------- evaluation
class PreservedRNG:
    # evaluation reseeds torch (fix, closed-loop episodes); put every global RNG back afterwards
    def __enter__(self):
        self.state = (torch.get_rng_state(), np.random.get_state(), random.getstate())

    def __exit__(self, *exc):
        torch.set_rng_state(self.state[0]); np.random.set_state(self.state[1]); random.setstate(self.state[2])


def judge_test(agent):
    # HW12/test.py on a fresh environment, then the replay of HW12/judge.py on the saved actions
    env = gym.make('LunarLander-v2')
    fix(env, JUDGE_SEED)
    rewards, action_list = [], []
    with torch.no_grad():
        for i in range(5):
            actions, state, total_reward, done = [], env.reset(), 0, False
            while not done:
                action = agent.act(state, greedy=False)
                actions.append(action)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
            action_list.append(actions)
    fix(env, JUDGE_SEED)
    replay = []
    for actions in action_list:
        env.reset()
        total_reward = 0
        for action in actions:
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        replay.append(total_reward)
    assert replay == rewards, 'judge replay differs from the test episodes'
    return rewards, action_list


def closed_loop(agent, greedy):
    # one environment seed per episode; torch reseeded per episode so episodes are independent
    env = gym.make('LunarLander-v2').env
    out = []
    with torch.no_grad():
        for i in range(opt.eval_episodes):
            s = opt.eval_base + i
            torch.manual_seed(s)
            state, _ = env.reset(seed=s)
            total, n, term, trunc = 0.0, 0, False, False
            while not (term or trunc):
                state, r, term, trunc, _ = env.step(agent.act(state, greedy))
                total += r
                n += 1
            # the last reward carries the +100 (came to rest) or -100 (crashed / flew off) bonus
            outcome = 'timeout' if trunc and not term else ('landed' if r >= 50 else ('crashed' if r <= -50 else 'other'))
            out.append({'reward': total, 'steps': n, 'outcome': outcome})
    return out


def summary(eps):
    r = np.array([e['reward'] for e in eps])
    oc = [e['outcome'] for e in eps]
    return {'mean': float(r.mean()), 'std': float(r.std()), 'median': float(np.median(r)),
            'solved_rate': float((r >= 200).mean()),
            'landed': oc.count('landed'), 'crashed': oc.count('crashed'),
            'timeout': oc.count('timeout'), 'other': oc.count('other'),
            'mean_steps': float(np.mean([e['steps'] for e in eps])),
            'rewards': [round(float(x), 2) for x in r]}


# ---------------------------------------------------------------- bookkeeping shared by all algorithms
class Run:
    def __init__(self):
        self.episodes, self.steps, self.updates = 0, 0, 0
        self.ep_returns, self.ep_lengths = [], []
        self.snapshots, self.dump, self.weights = {}, None, {}
        self.train_s, self.t0 = 0.0, time.time()

    def end_episode(self, ret, length):
        self.episodes += 1
        self.ep_returns.append(float(ret))
        self.ep_lengths.append(length)

    def check(self, agent, extra=None):
        # called after every update; returns True when training should stop
        due = []
        if 'ep' not in self.snapshots and self.episodes >= opt.ep_budget:
            due.append('ep')
        if opt.step_budget and 'steps' not in self.snapshots and self.steps >= opt.step_budget:
            due.append('steps')
        if due:
            self.train_s += time.time() - self.t0
            with PreservedRNG():
                judge_rewards, action_list = judge_test(agent)
                snap = {'episodes': self.episodes, 'env_steps': self.steps, 'updates': self.updates,
                        'critic_ev': getattr(self, 'critic_ev', None),
                        'train_s': round(self.train_s, 1),
                        'last250_train_avg': float(np.mean(self.ep_returns[-250:])) if self.ep_returns else None,
                        'judge_rewards': judge_rewards, 'judge_mean': float(np.mean(judge_rewards)),
                        'judge_lengths': [len(a) for a in action_list],
                        'closed_stochastic': summary(closed_loop(agent, greedy=False)),
                        'closed_greedy': summary(closed_loop(agent, greedy=True))}
            for k in due:
                self.snapshots[k] = snap
                if opt.save and hasattr(agent, 'net'):
                    self.weights[k] = copy.deepcopy(agent.net.state_dict())
            if 'ep' in due and opt.dump:
                self.dump = dict(extra or {}, action_list=action_list, judge_rewards=judge_rewards)
            self.t0 = time.time()
        return 'ep' in self.snapshots and ('steps' in self.snapshots or not opt.step_budget)


def make_optimizer(params, lr):
    return optim_cls(params, lr=lr)


optim_cls = torch.optim.SGD if opt.optim == 'sgd' else torch.optim.Adam


def clip(params):
    if opt.clip_grad > 0:
        nn.utils.clip_grad_norm_(params, opt.clip_grad)


# ---------------------------------------------------------------- policy gradient family (sample, p.4, p.5, A2C)
def discounted(rs, gamma):
    out, g = [], 0.0
    for r in reversed(rs):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def episode_weights(ep, values):
    # ep: one episode's rewards / last_value (bootstrap if truncated); values: V(s_t) or None
    rs = ep['rewards']
    if opt.adv == 'none':
        if opt.returns == 'immediate':
            return rs, None
        if opt.returns == 'total':
            return [sum(rs)] * len(rs), None
        return discounted(rs, opt.gamma), None
    v = values.tolist() + [ep['last_value']]
    T = len(rs)
    if opt.adv == 'mc':
        g = discounted(rs[:-1] + [rs[-1] + opt.gamma * ep['last_value']], opt.gamma)
        return [g[t] - v[t] for t in range(T)], g
    if opt.adv == 'gae':
        adv, a = [0.0] * T, 0.0
        for t in reversed(range(T)):
            delta = rs[t] + opt.gamma * v[t + 1] - v[t]
            a = delta + opt.gamma * opt.lam * a
            adv[t] = a
        return adv, [adv[t] + v[t] for t in range(T)]
    # n-step: sum of n rewards, then bootstrap from V n steps later (or the episode end)
    adv, target = [], []
    for t in range(T):
        k = min(opt.nstep, T - t)
        g = sum(opt.gamma ** i * rs[t + i] for i in range(k)) + opt.gamma ** k * v[t + k]
        target.append(g)
        adv.append(g - v[t])
    return adv, target


def train_pg(env, run):
    policy = MLP(opt.hidden, 4, softmax=True)
    optimizer = make_optimizer(policy.parameters(), opt.lr)
    critic = critic_opt = None
    if opt.adv != 'none':
        critic = MLP(opt.hidden, 1)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=opt.critic_lr)
    agent = PolicyAgent(policy)
    curve_total, curve_final = [], []
    while True:
        log_probs, entropies, states, episodes = [], [], [], []
        total_rewards, final_rewards = [], []
        for episode in range(opt.episode_per_batch):
            state = env.reset()
            ep = {'rewards': [], 'last_value': 0.0}
            while True:
                probs = policy(torch.FloatTensor(state))
                dist = Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                states.append(state)
                state, reward, term, trunc, _ = env.env.step(action.item())
                ep['rewards'].append(reward)
                run.steps += 1
                if term or trunc:
                    if trunc and not term and critic is not None:
                        with torch.no_grad():
                            ep['last_value'] = float(critic(torch.FloatTensor(state)))
                    final_rewards.append(reward)
                    total_rewards.append(sum(ep['rewards']))
                    run.end_episode(sum(ep['rewards']), len(ep['rewards']))
                    episodes.append(ep)
                    break
        curve_total.append(sum(total_rewards) / len(total_rewards))
        curve_final.append(sum(final_rewards) / len(final_rewards))

        values = None
        if critic is not None:
            with torch.no_grad():
                values = critic(torch.FloatTensor(np.array(states))).squeeze(-1).double()
        weights, targets, i = [], [], 0
        for ep in episodes:
            T = len(ep['rewards'])
            w, tg = episode_weights(ep, values[i:i + T] if values is not None else None)
            weights += w
            targets += tg or []
            i += T
        if opt.norm == 'batch':
            weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-9)  # the sample's normalisation
        else:
            weights = np.array(weights)

        loss = (-torch.stack(log_probs) * torch.from_numpy(weights)).sum()
        if opt.entropy > 0:
            loss = loss - opt.entropy * torch.stack(entropies).sum()
        optimizer.zero_grad()
        loss.backward()
        clip(policy.parameters())
        optimizer.step()

        if critic is not None:
            s = torch.FloatTensor(np.array(states))
            y = torch.tensor(targets, dtype=torch.float32)
            # how much of the targets' variance the critic explained before this update (diagnostic only)
            run.critic_ev = float(1 - (y - values.float()).var() / (y.var() + 1e-8))
            for _ in range(opt.critic_iters):
                critic_loss = F.mse_loss(critic(s).squeeze(-1), y)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
        run.updates += 1
        extra = {'avg_total_rewards': list(curve_total), 'avg_final_rewards': list(curve_final),
                 'network': copy.deepcopy(policy.state_dict())} if opt.dump and 'ep' not in run.snapshots else None
        if run.check(agent, extra):
            return


# ---------------------------------------------------------------- PPO
def train_ppo(env, run):
    actor = MLP(opt.hidden, 4, softmax=True)
    critic = MLP(opt.hidden, 1)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=opt.lr)
    agent = PolicyAgent(actor)
    state, ep_ret, ep_len = env.reset(), 0.0, 0
    while True:
        S, A, LP, R, V, D, NV = [], [], [], [], [], [], []   # NV: bootstrap value after a truncation
        with torch.no_grad():
            for t in range(opt.rollout):
                st = torch.FloatTensor(state)
                dist = Categorical(actor(st))
                a = dist.sample()
                S.append(state); A.append(a.item()); LP.append(float(dist.log_prob(a))); V.append(float(critic(st)))
                state, r, term, trunc, _ = env.env.step(a.item())
                R.append(r); D.append(term or trunc); run.steps += 1
                NV.append(float(critic(torch.FloatTensor(state))) if trunc and not term else 0.0)
                ep_ret += r; ep_len += 1
                if term or trunc:
                    run.end_episode(ep_ret, ep_len)
                    state, ep_ret, ep_len = env.reset(), 0.0, 0
            last_v = float(critic(torch.FloatTensor(state)))
        adv, a_ = [0.0] * opt.rollout, 0.0
        for t in reversed(range(opt.rollout)):
            next_v = (NV[t] if D[t] else (last_v if t == opt.rollout - 1 else V[t + 1]))
            delta = R[t] + opt.gamma * next_v - V[t]
            a_ = delta + opt.gamma * opt.lam * (0.0 if D[t] else a_)
            adv[t] = a_
        S_t = torch.FloatTensor(np.array(S)); A_t = torch.tensor(A); LP_t = torch.tensor(LP)
        ADV = torch.tensor(adv, dtype=torch.float32); RET = ADV + torch.tensor(V, dtype=torch.float32)
        for epoch in range(opt.ppo_epochs):
            perm = torch.randperm(opt.rollout)
            for k in range(0, opt.rollout, opt.minibatch):
                idx = perm[k:k + opt.minibatch]
                dist = Categorical(actor(S_t[idx]))
                ratio = torch.exp(dist.log_prob(A_t[idx]) - LP_t[idx])
                adv_mb = (ADV[idx] - ADV[idx].mean()) / (ADV[idx].std() + 1e-8)
                pg_loss = -torch.min(ratio * adv_mb, torch.clamp(ratio, 1 - opt.clip, 1 + opt.clip) * adv_mb).mean()
                v_loss = F.mse_loss(critic(S_t[idx]).squeeze(-1), RET[idx])
                loss = pg_loss + opt.vf_coef * v_loss - opt.entropy * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                clip(params)
                optimizer.step()
        run.updates += 1
        if run.check(agent):
            return


# ---------------------------------------------------------------- DQN (the sample's "boss" hint)
def train_dqn(env, run):
    q = MLP(opt.hidden, 4, act=F.relu)
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=opt.lr)
    agent = QAgent(q)
    N = opt.buffer
    bs, ba = np.zeros((N, 8), np.float32), np.zeros(N, np.int64)
    br, bs2, bd = np.zeros(N, np.float32), np.zeros((N, 8), np.float32), np.zeros(N, np.float32)
    n_stored, ptr = 0, 0
    state, ep_ret, ep_len = env.reset(), 0.0, 0
    while True:
        eps = max(opt.eps_end, 1.0 - (1.0 - opt.eps_end) * run.steps / opt.eps_decay)
        if torch.rand(1).item() < eps:
            action = torch.randint(4, (1,)).item()
        else:
            with torch.no_grad():
                action = int(q(torch.FloatTensor(state)).argmax())
        next_state, r, term, trunc, _ = env.env.step(action)
        bs[ptr], ba[ptr], br[ptr], bs2[ptr], bd[ptr] = state, action, r, next_state, float(term)
        ptr, n_stored = (ptr + 1) % N, min(n_stored + 1, N)
        run.steps += 1
        ep_ret += r; ep_len += 1
        state = next_state
        if term or trunc:
            run.end_episode(ep_ret, ep_len)
            state, ep_ret, ep_len = env.reset(), 0.0, 0
        if run.steps >= opt.learn_start and run.steps % opt.train_freq == 0:
            idx = torch.randint(n_stored, (opt.dqn_batch,)).numpy()
            s, a = torch.from_numpy(bs[idx]), torch.from_numpy(ba[idx])
            with torch.no_grad():
                y = torch.from_numpy(br[idx]) + opt.gamma * (1 - torch.from_numpy(bd[idx])) * target(torch.from_numpy(bs2[idx])).max(1).values
            loss = F.smooth_l1_loss(q(s).gather(1, a[:, None]).squeeze(1), y)
            optimizer.zero_grad()
            loss.backward()
            clip(q.parameters())
            optimizer.step()
            run.updates += 1
        if run.steps % opt.target_update == 0:
            target.load_state_dict(q.state_dict())
        if (term or trunc or opt.step_budget and run.steps >= opt.step_budget) and run.check(agent):
            return


# ---------------------------------------------------------------- main
env = gym.make('LunarLander-v2')
fix(env, opt.seed)
if not opt.no_explore:
    # notebook cells [15]-[24]: they advance the environment RNG before training
    env.reset()
    random_action = env.action_space.sample()
    env.step(random_action)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())

run = Run()
if opt.algo == 'random':
    opt.step_budget = 0
    run.episodes = opt.ep_budget  # nothing to train: score the uniform policy once
    run.check(RandomAgent())
else:
    {'pg': train_pg, 'ppo': train_ppo, 'dqn': train_dqn}[opt.algo](env, run)

bins = 20  # training curve: mean return of every 20 episodes and the env steps at the end of each bin
cum = np.cumsum(run.ep_lengths) if run.ep_lengths else np.array([])
rec = {'algo': opt.algo, 'seed': opt.seed, 'tag': opt.tag, 'config': vars(opt),
       'snapshots': run.snapshots, 'episodes': run.episodes, 'env_steps': run.steps,
       'curve_return': [round(float(np.mean(run.ep_returns[i:i + bins])), 2) for i in range(0, len(run.ep_returns), bins)],
       'curve_steps': [int(cum[min(i + bins, len(cum)) - 1]) for i in range(0, len(cum), bins)]}
if opt.out:
    with open(opt.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
if opt.dump:
    torch.save(run.dump, opt.dump)
if opt.save:
    torch.save({'config': vars(opt), 'weights': run.weights}, opt.save)
print(json.dumps({'algo': opt.algo, 'seed': opt.seed, 'tag': opt.tag}),
      ' | '.join('%s: ep %d steps %d judge %.1f sto %.1f greedy %.1f train %.0fs' % (
          k, s['episodes'], s['env_steps'], s['judge_mean'], s['closed_stochastic']['mean'],
          s['closed_greedy']['mean'], s['train_s']) for k, s in run.snapshots.items()))
