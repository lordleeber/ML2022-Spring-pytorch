# HW12 chapter 2 facts: the untrained policy (seed 543), what Categorical does, train()/eval(), and how peaked
# the trained sample policy is on the states of its own 5 test episodes.
# usage: python docs/tools/hw12_ch02.py > docs/tools/hw12_ch02.txt   (needs docs/tools/hw12_models/A_sample_s543.pt)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'HW12'))

import numpy as np
import torch
from torch.distributions import Categorical

from env_setup import make_env, fix, seed
from model import PolicyGradientNetwork, PolicyGradientAgent

torch.set_num_threads(1)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print('== the untrained network, built as in train.py (torch seeded by fix(env, 543), exploration cells use no torch RNG)')
env = make_env()
s0 = env.reset()
net = PolicyGradientNetwork()
with torch.no_grad():
    p = net(torch.FloatTensor(s0))
print('first state', s0)
print('probs', p.numpy(), 'sum', float(p.sum()), 'entropy', float(Categorical(p).entropy()), 'max entropy ln4', float(np.log(4)))
d = Categorical(p)
a = d.sample()
print('one sample:', a.item(), 'log_prob', float(d.log_prob(a)), '= log(p[a])', float(torch.log(p[a])))
torch.manual_seed(0)
draws = Categorical(p).sample((100000,))
print('100000 draws, frequency', (torch.bincount(draws, minlength=4) / 100000).numpy())
print('log_prob requires_grad when the net is not under no_grad:', Categorical(net(torch.FloatTensor(s0))).log_prob(torch.tensor(0)).requires_grad)

print('\n== train() vs eval(): the network has no dropout and no batch norm')
x = torch.randn(5, 8)
net.train(); y1 = net(x)
net.eval(); y2 = net(x)
print('identical outputs:', torch.equal(y1, y2), '| modules:', [type(m).__name__ for m in net.modules()])

print('\n== the trained sample policy (seed 543, 2,000 episodes) on its own 5 judge episodes')
w = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hw12_models/A_sample_s543.pt'), weights_only=False)
pol = PolicyGradientNetwork()
pol.load_state_dict(w['weights']['ep'])
agent = PolicyGradientAgent(pol)
fix(env, seed)
maxp, ent, agree, steps, rewards = [], [], [], 0, []
with torch.no_grad():
    for ep in range(5):
        st, done, tot = env.reset(), False, 0.0
        while not done:
            probs = pol(torch.FloatTensor(st))
            act, _ = agent.sample(st)
            maxp.append(float(probs.max()))
            ent.append(float(Categorical(probs).entropy()))
            agree.append(act == int(probs.argmax()))
            st, r, done, _ = env.step(act)
            tot += r
            steps += 1
        rewards.append(round(tot, 2))
maxp, ent = np.array(maxp), np.array(ent)
print('judge rewards (same as test.py):', rewards, 'steps', steps)
print('max prob: mean %.3f median %.3f; share of steps with max prob > 0.9: %.3f, > 0.99: %.3f'
      % (maxp.mean(), np.median(maxp), (maxp > 0.9).mean(), (maxp > 0.99).mean()))
print('entropy: mean %.3f (ln4 = %.3f); sampled action == argmax on %.3f of the steps (%d of %d)'
      % (ent.mean(), np.log(4), np.mean(agree), sum(not a for a in agree), len(agree)))

print('\n== gradient of log pi(a|s) with respect to the 4 logits (fc3 output), untrained network, first state, a = 3')
torch.manual_seed(543)
net2 = PolicyGradientNetwork()
h = torch.tanh(net2.fc2(torch.tanh(net2.fc1(torch.FloatTensor(s0)))))
z = net2.fc3(h).detach().requires_grad_(True)
pz = torch.softmax(z, -1)
lp3 = Categorical(pz).log_prob(torch.tensor(3))
lp3.backward()
print('probs', pz.detach().numpy(), 'grad wrt logits', z.grad.numpy(), 'onehot(3) - probs', (torch.eye(4)[3] - pz).detach().numpy())
for w in (+1.0, -1.0):
    # loss = -w * log_prob, so one gradient-descent step moves the logits along -d(loss)/dz = w * grad
    print('weight %+.1f: descent direction of the logits' % w, (w * z.grad).numpy())
