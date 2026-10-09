# HW13 ch03: the teacher's accuracy, confusions and output distribution (soft labels, temperature).
# usage (from HW13/): python ../docs/tools/hw13_teacher.py ../docs/tools/hw13_teacher.json
# Inference only (no_grad, fixed transforms, shuffle=False): no randomness involved.
import json, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_teacher_model

NAMES = ['麵包', '乳製品', '甜點', '蛋', '炸物', '肉類', '麵食', '米飯', '海鮮', '湯', '蔬果']
teacher = get_teacher_model(cfg['dataset_root']).cuda().eval()


@torch.no_grad()
def logits_of(split):
  ds = FoodDataset(f"{cfg['dataset_root']}/{split}", tfm=test_tfm)
  out, ys = [], []
  for x, y in DataLoader(ds, batch_size=128, shuffle=False, num_workers=8):
    out.append(teacher(x.cuda()).float().cpu()); ys.append(y)
  return torch.cat(out), torch.cat(ys)


res = {}
for split in ['validation', 'training']:
  z, y = logits_of(split)
  p = z.softmax(-1)
  pred = p.argmax(-1)
  r = dict(n=len(y), acc=(pred == y).float().mean().item(), correct=int((pred == y).sum()))
  r['per_class'] = {NAMES[c]: dict(n=int((y == c).sum()), acc=round((pred[y == c] == c).float().mean().item(), 4)) for c in range(11)}
  conf = torch.zeros(11, 11, dtype=torch.long)
  for a, b in zip(y.tolist(), pred.tolist()):
    conf[a, b] += 1
  pairs = sorted(((int(conf[a, b]), NAMES[a], NAMES[b]) for a in range(11) for b in range(11) if a != b), reverse=True)[:8]
  r['top_confusions'] = [dict(count=c, true=t, pred=q) for c, t, q in pairs]
  r['mean_max_prob'] = p.max(-1).values.mean().item()
  r['mean_true_prob'] = p[torch.arange(len(y)), y].mean().item()
  r['frac_max_prob_over_0.99'] = (p.max(-1).values > 0.99).float().mean().item()
  # temperature: entropy (nats) and max prob of softmax(z / T); uniform over 11 classes = ln 11
  r['temperature'] = {}
  for T in [1, 2, 4, 8, 20]:
    q = (z / T).softmax(-1)
    ent = -(q * q.clamp_min(1e-12).log()).sum(-1).mean().item()
    r['temperature'][str(T)] = dict(entropy=round(ent, 4), mean_max_prob=round(q.max(-1).values.mean().item(), 4),
                                    mean_true_prob=round(q[torch.arange(len(y)), y].mean().item(), 4))
  r['ln11'] = round(torch.log(torch.tensor(11.)).item(), 4)
  # average soft label per true class (T = 1 and T = 4), as percentages
  r['soft_label_T1'] = [[round(100 * v, 2) for v in p[y == c].mean(0).tolist()] for c in range(11)]
  r['soft_label_T4'] = [[round(100 * v, 2) for v in (z[y == c] / 4).softmax(-1).mean(0).tolist()] for c in range(11)]
  r['logit_stats'] = dict(mean_max=z.max(-1).values.mean().item(), mean_std=z.std(-1).mean().item())
  if split == 'validation':
    # one example image per class: index, T=1 and T=4 distributions
    ex = {}
    for c in [3, 7]:
      i = int((y == c).nonzero()[0])
      ex[NAMES[c]] = dict(index=i, file=f'{c}_?', T1=[round(100 * v, 2) for v in p[i].tolist()],
                          T4=[round(100 * v, 2) for v in (z[i] / 4).softmax(-1).tolist()], logits=[round(v, 3) for v in z[i].tolist()])
    r['examples'] = ex
  res[split] = r
  print(split, 'acc', round(r['acc'], 5), r['correct'], '/', r['n'], 'mean max prob', round(r['mean_max_prob'], 4),
        '>0.99:', round(r['frac_max_prob_over_0.99'], 4))
  print('  per class', {k: v['acc'] for k, v in r['per_class'].items()})
  print('  top confusions', r['top_confusions'][:5])
  print('  temperature', r['temperature'])
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
sys.stdout.flush(); import os; os._exit(0)
