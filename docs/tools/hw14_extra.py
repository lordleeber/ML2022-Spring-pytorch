# HW14 experiment E (added by this book, not part of the homework):
#   joint  - E1: train on all five rotated tasks mixed together (not continual; an upper bound)
#   replay - E2: sequential training like the baseline (lambda 0), but keep `--mem` random
#            training images of every finished task and mix a batch of them into every step
# Same model, optimizer (Adam lr 1e-4, rebuilt per task), batch 128 and number of samples
# seen (10 epochs x 5 tasks x 60,000) as the homework's methods. Results go to the same
# jsonl as hw14_exp.py, with the same per-epoch accuracy matrix.
#
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_extra.py --mode replay --mem 200 --tag E2_m200
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import args, angle_list  # noqa: E402
from dataset import Data  # noqa: E402
from model import Model  # noqa: E402
from utils import same_seeds  # noqa: E402
from hw14_exp import acc_all  # noqa: E402


def avg_seen(matrix, epochs_per_task):
  # the notebook's metric: mean accuracy over the tasks learned so far, in %
  return [float(np.mean(row[:e // epochs_per_task + 1]) * 100.0) for e, row in enumerate(matrix)]


def run_joint(train_datasets, test_tensors, device, epochs):
  model = Model().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  objective = nn.CrossEntropyLoss()
  loader = DataLoader(ConcatDataset([d.dataset for d in train_datasets]), batch_size=args.batch_size, shuffle=True)
  matrix = []
  for epoch in range(epochs):
    model.train()
    for imgs, labels in loader:
      imgs, labels = imgs.to(device), labels.to(device)
      loss = objective(model(imgs), labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    matrix.append(acc_all(model, test_tensors))
  return matrix


def run_replay(train_datasets, test_tensors, device, mem, seed):
  model = Model().to(device)
  objective = nn.CrossEntropyLoss()
  g = torch.Generator().manual_seed(seed)  # memory selection and replay batches
  mem_x, mem_y = [], []
  matrix = []
  for t, d in enumerate(train_datasets):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(d.dataset, batch_size=args.batch_size, shuffle=True)
    bank_x = torch.cat(mem_x) if mem_x else None
    bank_y = torch.cat(mem_y) if mem_y else None
    for epoch in range(args.epochs_per_task):
      model.train()
      for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if bank_x is not None:
          idx = torch.randint(len(bank_x), (min(args.batch_size, len(bank_x)),), generator=g)
          imgs = torch.cat([imgs, bank_x[idx]])
          labels = torch.cat([labels, bank_y[idx]])
        loss = objective(model(imgs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      matrix.append(acc_all(model, test_tensors))
    # keep `mem` random training images of the task just finished
    pick = torch.randperm(len(d.dataset), generator=g)[:mem].tolist()
    xs, ys = zip(*[d.dataset[i] for i in pick])
    mem_x.append(torch.stack(xs).to(device))
    mem_y.append(torch.tensor(ys).to(device))
  return matrix


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--mode', choices=['joint', 'replay'], required=True)
  p.add_argument('--mem', type=int, default=200, help='replay: images kept per finished task')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--epochs', type=int, default=args.epochs_per_task, help='joint: epochs over the mixed data')
  p.add_argument('--tag', required=True)
  p.add_argument('--out', default='../docs/tools/hw14_runs.jsonl')
  opt = p.parse_args()

  same_seeds(opt.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
  test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
  test_tensors = []
  for d in test_datasets:
    xs, ys = zip(*[d.dataset[i] for i in range(len(d.dataset))])
    test_tensors.append((torch.stack(xs).to(device), torch.tensor(ys).to(device)))

  t0 = time.time()
  if opt.mode == 'joint':
    matrix = run_joint(train_datasets, test_tensors, device, opt.epochs)
    acc = [float(np.mean(row) * 100.0) for row in matrix]  # all five tasks are "seen" from the start
    method = 'joint'
  else:
    matrix = run_replay(train_datasets, test_tensors, device, opt.mem, opt.seed)
    acc = avg_seen(matrix, args.epochs_per_task)
    method = 'replay'
  print(acc, flush=True)
  rec = {'tag': opt.tag, 'method': method, 'seed': opt.seed, 'mem': opt.mem if method == 'replay' else None,
         'epochs': opt.epochs if method == 'joint' else args.epochs_per_task,
         'acc': acc, 'matrix': matrix, 'seconds': round(time.time() - t0, 1)}
  with open(opt.out, 'a') as f:
    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
  main()
