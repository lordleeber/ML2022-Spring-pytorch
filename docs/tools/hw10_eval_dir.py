# Evaluate saved adversarial PNG folders (what JudgeBoi actually receives) with one or more models,
# and check the L-infinity distance to the benign images in 0-255 pixel units.
# usage (from HW10/): python ../docs/tools/hw10_eval_dir.py <dir>[,<dir>...] <model>[,<model>...]
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device, batch_size, root
from dataset import AdvDataset, transform
from attack import epoch_benign

loss_fn = nn.CrossEntropyLoss()
dirs = sys.argv[1].split(',')
for d in dirs:
    ds = AdvDataset(d, transform=transform)
    linf = max(np.abs(np.asarray(Image.open(a), dtype=np.int16) - np.asarray(Image.open(b), dtype=np.int16)).max()
               for a, b in zip(ds.images, AdvDataset(root, transform=None).images))
    print(f'{d}: {len(ds)} images, max |adv - benign| = {linf}')
for name in sys.argv[2].split(','):
    model = ptcv_get_model(name, pretrained=True).to(device)
    for d in dirs:
        loader = DataLoader(AdvDataset(d, transform=transform), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            acc, loss = epoch_benign(model, loader, loss_fn)
        print(f'{name} on {d}: acc = {acc:.5f}, loss = {loss:.5f}')
