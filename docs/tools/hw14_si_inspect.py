# HW14 chapter 6: what SI accumulates. Reruns SI exactly like hw14_exp.py (seed 0) and, right
# before each new si object is built, reads the model buffers <name>_W and <name>_SI_prev_task:
# sign of W, size of the parameter change over the task, and how often epsilon dominates W/(d^2+eps).
# Reading buffers draws no random numbers, so the accuracies must equal A_SI_s0.
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_si_inspect.py
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw14_exp  # noqa: E402
from config import args, angle_list  # noqa: E402
from dataset import Data  # noqa: E402
from model import Model  # noqa: E402
from utils import same_seeds  # noqa: E402

EPS = 0.1


def main():
  exp_opt = argparse.Namespace(scp_per_slice=False, lam=None, guards='accum')
  same_seeds(0)
  device = torch.device("cuda")
  train_datasets = [Data('data', angle=angle_list[i]) for i in range(args.task_number)]
  train_dataloaders = [DataLoader(d.dataset, batch_size=args.batch_size, shuffle=True) for d in train_datasets]
  test_datasets = [Data('data', train=False, angle=angle_list[i]) for i in range(args.task_number)]
  test_dataloaders = [DataLoader(d.dataset, batch_size=args.test_size, shuffle=True) for d in test_datasets]
  test_tensors = []
  for d in test_datasets:
    xs, ys = zip(*[d.dataset[i] for i in range(len(d.dataset))])
    test_tensors.append((torch.stack(xs).to(device), torch.tensor(ys).to(device)))
  print(Model())

  stats = []
  orig = hw14_exp.make_lll

  def hook(name, model, dataloader, dev, prev_guards, o):
    if dataloader is not None:
      Ws, ds, oa, names = [], [], [], []
      for n, p in model.named_parameters():
        k = n.replace('.', '__')
        W = getattr(model, k + '_W').flatten()
        d = (p.detach() - getattr(model, k + '_SI_prev_task')).flatten()
        Ws.append(W); ds.append(d); oa.append(W / (d ** 2 + EPS))
      W, d, om = torch.cat(Ws), torch.cat(ds), torch.cat(oa)
      d2 = d ** 2
      stats.append({'W_sum': W.sum().item(), 'W_neg_frac': (W < 0).float().mean().item(),
                    'W_zero_frac': (W == 0).float().mean().item(), 'W_neg_sum': W[W < 0].sum().item(),
                    'W_min': W.min().item(), 'W_max': W.max().item(),
                    'abs_change_median': d.abs().median().item(), 'abs_change_p99': d.abs().quantile(0.99).item() if d.numel() < 16_000_000 else None,
                    'abs_change_max': d.abs().max().item(), 'd2_lt_eps_frac': (d2 < EPS).float().mean().item(),
                    'd2_lt_eps_over_100_frac': (d2 < EPS / 100).float().mean().item(),
                    'omega_add_sum': om.sum().item(), 'omega_add_neg_frac': (om < 0).float().mean().item(),
                    'omega_add_if_no_eps_sum': (W / d2.clamp_min(1e-30)).sum().item()})
      print('task', len(stats), json.dumps(stats[-1]), flush=True)
    return orig(name, model, dataloader, dev, prev_guards, o)

  hw14_exp.make_lll = hook
  r = hw14_exp.run_method('SI', train_dataloaders, test_dataloaders, test_tensors, device, exp_opt)
  hw14_exp.make_lll = orig
  print(r['acc'], flush=True)
  with open('../docs/tools/hw14_si_inspect.json', 'w') as f:
    json.dump({'acc': r['acc'], 'importance': r['importance'], 'per_task': stats}, f, indent=1)


if __name__ == '__main__':
  main()
