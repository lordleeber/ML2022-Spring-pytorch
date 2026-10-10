# Are the 200 HW10 images taken from CIFAR-10? Look for pixel-identical images in the train and test sets.
# Writes the matching training-set indices to docs/tools/hw10_overlap.json so hw10_train_surrogate.py can drop them.
# usage (from HW10/): python ../docs/tools/hw10_overlap.py
import json
import os
import sys

import numpy as np
import torchvision
from PIL import Image

sys.path.insert(0, os.getcwd())
from dataset import AdvDataset

ds = AdvDataset('./data', transform=None)
out = {'train': [], 'test': [], 'label_mismatch': 0, 'unmatched': []}
sets = {k: torchvision.datasets.CIFAR10('cifar10', train=(k == 'train')) for k in ('train', 'test')}
index = {k: {v.data[i].tobytes(): i for i in range(len(v.data))} for k, v in sets.items()}
for f, lab, name in zip(ds.images, ds.labels, ds.names):
  b = np.asarray(Image.open(f).convert('RGB')).tobytes()
  hit = False
  for k in ('train', 'test'):
    if b in index[k]:
      i = index[k][b]
      out[k].append(i)
      out['label_mismatch'] += int(sets[k].targets[i] != lab)
      hit = True
  if not hit:
    out['unmatched'].append(name)
print('train matches', len(out['train']), 'test matches', len(out['test']), 'label mismatches', out['label_mismatch'], 'unmatched', len(out['unmatched']))
print('first test idx', sorted(out['test'])[:10])
json.dump(out, open('../docs/tools/hw10_overlap.json', 'w'))
