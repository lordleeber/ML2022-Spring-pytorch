# Download every pytorchcv *_cifar10 model, record whether it loads, its parameter count
# and its benign accuracy/loss on the 200 HW10 images (same loader and loop as hw10.py).
# usage (from HW10/): python ../docs/tools/hw10_zoo.py <out.jsonl>
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model, _models

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root
from dataset import AdvDataset, transform
from attack import epoch_benign

adv_set = AdvDataset(root, transform=transform)
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)
loss_fn = nn.CrossEntropyLoss()
done = set()
if os.path.exists(sys.argv[1]):
    done = {json.loads(l)['name'] for l in open(sys.argv[1])}
names = [k for k in _models if k.endswith('_cifar10')]
for name in names:
    if name in done:
        continue
    rec = {'name': name}
    t0 = time.time()
    try:
        model = ptcv_get_model(name, pretrained=True).to(device)
        rec['params'] = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            rec['benign_acc'], rec['benign_loss'] = epoch_benign(model, adv_loader, loss_fn)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'[:300]
    rec['sec'] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
    with open(sys.argv[1], 'a') as f:
        f.write(json.dumps(rec) + '\n')
