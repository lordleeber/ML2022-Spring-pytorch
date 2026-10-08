# HW14 chapter 3: is a forgotten task really gone, or only the last layer misaligned?
#
# 1. Reruns one method exactly like hw14_exp.py (same random stream) and keeps a copy of the
#    weights at the end of every task (copying weights draws no random numbers, so the printed
#    accuracies must equal the hw14_exp.py run with the same options).
# 2. Parameter drift: per layer, ||theta_k - theta_1|| / ||theta_1|| after task k.
# 3. Linear probe: freeze fc1-fc3 of the final model, train a fresh fc4 on one task's training
#    data, and test on that task. Control: the same probe on a randomly initialised network.
#
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_probe.py --method baseline --tag P_baseline_s0
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw14_exp  # noqa: E402
from config import args, angle_list  # noqa: E402
from dataset import Data  # noqa: E402
from model import Model  # noqa: E402
from utils import same_seeds  # noqa: E402

LAYERS = ['fc1', 'fc2', 'fc3', 'fc4']


def features(model, x):
  x = x.view(-1, 784)
  x = model.relu(model.fc1(x))
  x = model.relu(model.fc2(x))
  return model.relu(model.fc3(x))


def probe(model, train_ds, test_xy, device, epochs, seed):
  """Train a fresh Linear(256, 10) on frozen features; return test accuracy (%)."""
  g = torch.Generator().manual_seed(seed)
  torch.manual_seed(seed)
  head = nn.Linear(256, 10).to(device)
  opt = torch.optim.Adam(head.parameters(), lr=1e-3)
  loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
  model.eval()
  for _ in range(epochs):
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
        f = features(model, x)
      loss = nn.functional.cross_entropy(head(f), y)
      opt.zero_grad()
      loss.backward()
      opt.step()
  with torch.no_grad():
    x, y = test_xy
    return (head(features(model, x)).argmax(1) == y).float().mean().item() * 100


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--method', default='baseline', choices=hw14_exp.ORDER)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--probe_epochs', type=int, default=3)
  p.add_argument('--tag', required=True)
  p.add_argument('--out', default='../docs/tools/hw14_probe.jsonl')
  opt = p.parse_args()

  # same setup as hw14_exp.main(), so the random stream is identical
  exp_opt = argparse.Namespace(scp_per_slice=False, lam=None, guards='accum')
  same_seeds(opt.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
  train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]
  test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
  test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]
  test_tensors = []
  for d in test_datasets:
    xs, ys = zip(*[d.dataset[i] for i in range(len(d.dataset))])
    test_tensors.append((torch.stack(xs).to(device), torch.tensor(ys).to(device)))
  example = Model()
  print(example)

  snapshots = []
  orig_make = hw14_exp.make_lll

  def make_and_snapshot(name, model, dataloader, device, prev_guards, o):
    if dataloader is not None:  # called right after a task finished
      snapshots.append(copy.deepcopy(model.state_dict()))
    return orig_make(name, model, dataloader, device, prev_guards, o)

  hw14_exp.make_lll = make_and_snapshot
  r = hw14_exp.run_method(opt.method, train_dataloaders, test_dataloaders, test_tensors, device, exp_opt)
  hw14_exp.make_lll = orig_make
  print(r['acc'], flush=True)

  # parameter drift relative to the end of task 1
  drift = []
  for k, sd in enumerate(snapshots):
    row = {}
    for L in LAYERS:
      w1, wk = snapshots[0][L + '.weight'], sd[L + '.weight']
      row[L] = ((wk - w1).norm() / w1.norm()).item()
    drift.append(row)

  # linear probes on the final model and on a random network
  final = Model().to(device)
  final.load_state_dict(snapshots[-1])
  same_seeds(1234)
  rand = Model().to(device)
  probe_final, probe_rand = [], []
  for j in range(args.task_number):
    probe_final.append(probe(final, train_datasets[j].dataset, test_tensors[j], device, opt.probe_epochs, 100 + j))
    probe_rand.append(probe(rand, train_datasets[j].dataset, test_tensors[j], device, opt.probe_epochs, 100 + j))
    print('task', j + 1, 'probe final %.2f random %.2f' % (probe_final[-1], probe_rand[-1]), flush=True)

  rec = {'tag': opt.tag, 'method': opt.method, 'seed': opt.seed, 'acc': r['acc'], 'matrix': r['matrix'],
         'drift': drift, 'probe_epochs': opt.probe_epochs, 'probe_final': probe_final, 'probe_random': probe_rand}
  with open(opt.out, 'a') as f:
    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
  main()
