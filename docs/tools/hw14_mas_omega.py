# HW14 chapter 5: MAS importance as the TODO computes it (abs of the gradient of the
# batch-mean squared L2 output norm) vs the paper's per-sample version (mean over samples of
# the abs gradient of each sample's squared L2 output norm), on the same task-1 model as
# hw14_fisher.py; also how MAS and EWC (as methods/ewc.py computes it) rank the parameters.
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_mas_omega.py
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from config import args  # noqa: E402
from dataset import Data  # noqa: E402
from model import Model  # noqa: E402
from utils import same_seeds  # noqa: E402


def main():
  same_seeds(0)
  device = torch.device("cuda")
  train_ds = Data('data', angle=0).dataset
  model = Model().to(device)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)
  loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
  for _ in range(args.epochs_per_task):
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      loss = F.cross_entropy(model(x), y)
      opt.zero_grad()
      loss.backward()
      opt.step()
  model.eval()

  names = [n for n, _ in model.named_parameters()]
  mas_b = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
  mas_s = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
  ewc_b = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
  eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
  nb = len(eval_loader)
  out_sq = []
  for x, y in eval_loader:
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    out = model(x)
    out_sq.append(out.detach().pow(2).sum(1))
    out.pow(2).sum(dim=1).mean().backward()          # the TODO in methods/mas.py
    for n, p in model.named_parameters():
      mas_b[n] += p.grad.abs() / nb
    model.zero_grad()
    F.nll_loss(F.log_softmax(model(x), dim=1), y).backward()   # methods/ewc.py
    for n, p in model.named_parameters():
      ewc_b[n] += p.grad ** 2 / nb

  params = {n: p.detach() for n, p in model.named_parameters()}

  def sqnorm(p, x):
    return functional_call(model, p, (x.unsqueeze(0),)).pow(2).sum()

  per_sample = vmap(grad(sqnorm), in_dims=(None, 0))
  ns = 0
  for x, _ in eval_loader:
    g = per_sample(params, x.to(device))
    for n in names:
      mas_s[n] += g[n].abs().sum(0)
    ns += x.size(0)
  for n in names:
    mas_s[n] /= ns

  def flat(d):
    return torch.cat([d[n].flatten() for n in names])

  def corr(a, c):
    a, c = a - a.mean(), c - c.mean()
    return (a @ c / (a.norm() * c.norm())).item()

  def rank(a):
    r = torch.empty_like(a); r[a.argsort()] = torch.arange(len(a), device=a.device, dtype=a.dtype)
    return r

  b, s, e = flat(mas_b), flat(mas_s), flat(ewc_b)
  sq = torch.cat(out_sq)
  res = {'sum': {'mas_batch': b.sum().item(), 'mas_per_sample': s.sum().item(), 'ewc_batch': e.sum().item()},
         'max': {'mas_batch': b.max().item(), 'mas_per_sample': s.max().item(), 'ewc_batch': e.max().item()},
         'ratio_per_sample_over_batch': (s.sum() / b.sum()).item(),
         'pearson_mas_batch_vs_per_sample': corr(b, s), 'spearman_mas_batch_vs_per_sample': corr(rank(b), rank(s)),
         'pearson_mas_vs_ewc': corr(b, e), 'spearman_mas_vs_ewc': corr(rank(b), rank(e)),
         'output_sqnorm_mean': sq.mean().item(), 'output_sqnorm_median': sq.median().item(),
         'per_layer_sum': {L: {'mas_batch': (mas_b[L + '.weight'].sum() + mas_b[L + '.bias'].sum()).item(),
                               'mas_per_sample': (mas_s[L + '.weight'].sum() + mas_s[L + '.bias'].sum()).item()}
                           for L in ['fc1', 'fc2', 'fc3', 'fc4']}}
  print(json.dumps(res, indent=1))
  with open('../docs/tools/hw14_mas_omega.json', 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
