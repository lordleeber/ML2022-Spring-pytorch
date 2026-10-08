# HW14 experiment tool: reruns HW14/train.py's method loops with extra measurements.
#
# With no options it reproduces `python train.py` bit for bit (same random stream,
# same printed accuracy lists). On top of that it records, after every epoch, the
# accuracy on ALL five tasks (seen and unseen) from precomputed test tensors, so the
# extra evaluation never touches the global random stream or any .grad.
#
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_exp.py --tag ref
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from config import args, angle_list  # noqa: E402
from dataset import Data  # noqa: E402
from model import Model  # noqa: E402
from trainer import train, evaluate  # noqa: E402
from utils import same_seeds  # noqa: E402
from methods.baseline import baseline  # noqa: E402
from methods.ewc import ewc  # noqa: E402
from methods.mas import mas  # noqa: E402
from methods.si import si  # noqa: E402
from methods.rwalk import rwalk  # noqa: E402
from methods.scp import scp, sample_spherical  # noqa: E402

LAMBDA = {'baseline': 0.0, 'EWC': 100, 'MAS': 0.1, 'SI': 1, 'RWALK': 100, 'SCP': 100}
ORDER = list(LAMBDA)


class scp_per_slice(scp):
  """SCP variant: square the gradient of every slice, then average (instead of
  averaging the L projections first and squaring once)."""
  def calculate_importance(self):
    precision_matrices = {}
    for n, p in self.params.items():
      precision_matrices[n] = p.clone().detach().fill_(0)
      for i in range(len(self.previous_guards_list)):
        if self.previous_guards_list[i]:
          precision_matrices[n] += self.previous_guards_list[i][n]
    self.model.eval()
    if self.dataloader is not None:
      num_data = len(self.dataloader)
      for data in self.dataloader:
        output = self.model(data[0].to(self.device))
        mean_vec = output.mean(dim=0)
        L_vectors = sample_spherical(self.L, output.shape[-1])
        L_vectors = L_vectors.transpose(1, 0).to(self.device).float()
        for vec in L_vectors:
          self.model.zero_grad()
          torch.matmul(vec, mean_vec).backward(retain_graph=True)
          for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad ** 2 / num_data / L_vectors.shape[0]
    return precision_matrices


def make_lll(name, model, dataloader, device, prev_guards, opt):
  kw = {} if prev_guards is None else {'prev_guards': prev_guards}
  if name == 'baseline':
    return baseline(model=model, dataloader=dataloader, device=device)
  if name == 'EWC':
    return ewc(model=model, dataloader=dataloader, device=device, **kw)
  if name == 'MAS':
    return mas(model=model, dataloader=dataloader, device=device, **kw)
  if name == 'SI':
    return si(model=model, dataloader=dataloader, epsilon=0.1, device=device)
  if name == 'RWALK':
    return rwalk(model=model, dataloader=dataloader, epsilon=0.1, device=device, **kw)
  if name == 'SCP':
    cls = scp_per_slice if opt.scp_per_slice else scp
    return cls(model=model, dataloader=dataloader, L=100, device=device, **kw)
  raise ValueError(name)


@torch.no_grad()
def acc_all(model, test_tensors):
  was_training = model.training
  model.eval()
  accs = []
  for x, y in test_tensors:
    accs.append((model(x).argmax(1) == y).float().mean().item())
  model.train(was_training)
  return accs


def importance_stats(lll, name):
  """Size of the importance matrices (all parameters together)."""
  if name in ('SI',):
    mats = lll._n_omega
  elif name == 'RWALK':
    mats = {n: lll._n_omega[n] + lll._precision_matrices[n] for n in lll._n_omega}
  else:
    mats = lll._precision_matrices
  v = torch.cat([m.flatten() for m in mats.values()])
  return {'sum': v.sum().item(), 'mean': v.mean().item(), 'max': v.max().item(),
          'min': v.min().item(), 'nonzero': int((v != 0).sum().item()), 'numel': v.numel()}


def run_method(name, train_dataloaders, test_dataloaders, test_tensors, device, opt):
  lam = LAMBDA[name] if opt.lam is None else opt.lam
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  lll_object = make_lll(name, model, None, device, None, opt)
  uses_guards = name not in ('baseline', 'SI')
  prev_guards = []
  acc = []          # what the notebook prints (average over seen tasks)
  matrix = []       # per epoch: accuracy on every task
  stats = []
  t0 = time.time()

  def evaluate_hook(m, loader, dev):
    # train() evaluates the seen tasks in order after each epoch; record the full row once
    if loader is test_dataloaders[0]:
      matrix.append(acc_all(m, test_tensors))
    return evaluate(m, loader, dev)

  for train_indexes in range(len(train_dataloaders)):
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task,
                               lll_object, lam, evaluate=evaluate_hook, device=device,
                               test_dataloaders=test_dataloaders[:train_indexes+1])
    if uses_guards:
      if opt.guards == 'accum':
        prev_guards.append(lll_object._precision_matrices)
        guards = prev_guards
      else:  # 'latest': each task's importance counted once
        guards = [lll_object._precision_matrices]
    else:
      guards = None
    lll_object = make_lll(name, model, train_dataloaders[train_indexes], device, guards, opt)
    stats.append(importance_stats(lll_object, name))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    acc.extend(acc_list)
  return {'acc': [float(a) for a in acc], 'matrix': matrix, 'importance': stats,
          'lambda': lam, 'seconds': round(time.time() - t0, 1)}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--methods', nargs='+', default=ORDER, choices=ORDER)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--lam', type=float, default=None, help='override lambda of every method run')
  p.add_argument('--guards', choices=['accum', 'latest'], default='accum')
  p.add_argument('--scp_per_slice', action='store_true')
  p.add_argument('--test_noshuffle', action='store_true', help='test DataLoaders with shuffle=False (chapter 2)')
  p.add_argument('--tag', required=True)
  p.add_argument('--out', default='../docs/tools/hw14_runs.jsonl')
  opt = p.parse_args()

  same_seeds(opt.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
  train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]
  test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
  test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=not opt.test_noshuffle) for data in test_datasets]

  # precomputed test tensors: transforms are deterministic, so no random numbers are drawn here
  test_tensors = []
  for d in test_datasets:
    xs, ys = zip(*[d.dataset[i] for i in range(len(d.dataset))])
    test_tensors.append((torch.stack(xs).to(device), torch.tensor(ys).to(device)))

  example = Model()
  print(example)

  for name in opt.methods:
    print("RUN", name, flush=True)
    r = run_method(name, train_dataloaders, test_dataloaders, test_tensors, device, opt)
    print(r['acc'], flush=True)
    rec = {'tag': opt.tag, 'method': name, 'seed': opt.seed, 'guards': opt.guards,
           'scp_per_slice': opt.scp_per_slice, 'test_noshuffle': opt.test_noshuffle, 'methods_run': opt.methods, **r}
    with open(opt.out, 'a') as f:
      f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
  main()
