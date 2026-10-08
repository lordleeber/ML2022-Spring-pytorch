# HW14 chapter 4: how big is the Fisher that methods/ewc.py computes, compared with the
# per-sample definitions?
#   batch     : what ewc.py does - square the gradient of the batch-mean NLL, average over batches
#   empirical : square every sample's own NLL gradient (true label), average over samples
#   true      : like empirical, but the label is sampled from the model's own softmax
# A model is trained on task 1 only (10 epochs, Adam lr 1e-4, batch 128, seed 0) first.
#
# Run from HW14/:  ../.venv/bin/python ../docs/tools/hw14_fisher.py
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
  zeros = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
  batch_f = {n: z.clone() for n, z in zeros.items()}
  emp_f = {n: z.clone() for n, z in zeros.items()}
  true_f = {n: z.clone() for n, z in zeros.items()}

  # batch version, exactly as methods/ewc.py (unshuffled loader; the order does not matter for a sum)
  eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
  nb = len(eval_loader)
  for x, y in eval_loader:
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    F.nll_loss(F.log_softmax(model(x), dim=1), y).backward()
    for n, p in model.named_parameters():
      batch_f[n] += p.grad ** 2 / nb

  # per-sample versions with torch.func
  params = {n: p.detach() for n, p in model.named_parameters()}

  def nll(p, x, y):
    out = functional_call(model, p, (x.unsqueeze(0),))
    return F.nll_loss(F.log_softmax(out, dim=1), y.unsqueeze(0))

  per_sample = vmap(grad(nll), in_dims=(None, 0, 0))
  g = torch.Generator(device=device).manual_seed(0)
  ns = 0
  for x, y in eval_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
      probs = F.softmax(model(x), dim=1)
    y_true = torch.multinomial(probs, 1, generator=g).squeeze(1)
    ge = per_sample(params, x, y)
    gt = per_sample(params, x, y_true)
    for n in names:
      emp_f[n] += (ge[n] ** 2).sum(0)
      true_f[n] += (gt[n] ** 2).sum(0)
    ns += x.size(0)
  for n in names:
    emp_f[n] /= ns
    true_f[n] /= ns

  def flat(d):
    return torch.cat([d[n].flatten() for n in names])

  b, e, t = flat(batch_f), flat(emp_f), flat(true_f)

  def corr(a, c):
    a, c = a - a.mean(), c - c.mean()
    return (a @ c / (a.norm() * c.norm())).item()

  def rank_corr(a, c):
    ra = torch.empty_like(a); ra[a.argsort()] = torch.arange(len(a), device=a.device, dtype=a.dtype)
    rc = torch.empty_like(c); rc[c.argsort()] = torch.arange(len(c), device=c.device, dtype=c.dtype)
    return corr(ra, rc)

  per_layer = {L: {'batch': (batch_f[L + '.weight'].sum() + batch_f[L + '.bias'].sum()).item(),
                   'empirical': (emp_f[L + '.weight'].sum() + emp_f[L + '.bias'].sum()).item(),
                   'true': (true_f[L + '.weight'].sum() + true_f[L + '.bias'].sum()).item()}
               for L in ['fc1', 'fc2', 'fc3', 'fc4']}
  res = {'train_acc_note': 'model trained on task 1 only, 10 epochs, seed 0',
         'n_batches': nb, 'n_samples': ns,
         'sum': {'batch': b.sum().item(), 'empirical': e.sum().item(), 'true': t.sum().item()},
         'max': {'batch': b.max().item(), 'empirical': e.max().item(), 'true': t.max().item()},
         'ratio_empirical_over_batch': (e.sum() / b.sum()).item(),
         'pearson_batch_empirical': corr(b, e), 'spearman_batch_empirical': rank_corr(b, e),
         'pearson_empirical_true': corr(e, t), 'spearman_empirical_true': rank_corr(e, t),
         'per_layer': per_layer}
  print(json.dumps(res, indent=1))
  with open('../docs/tools/hw14_fisher.json', 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
