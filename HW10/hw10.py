# HW10 Adversarial Attack: attacks the 200 benign images with a proxy model and saves them
# (HW10.ipynb cells [8], [16], [18], [20], [22]).
# With no arguments this prints the same output as the notebook: benign accuracy, FGSM and I-FGSM
# on resnet110_cifar10, and writes fgsm/, ifgsm/, fgsm.tgz, ifgsm.tgz.
# --models a,b,c attacks an ensembleNet (cell [24]) instead; --attacks picks the attacks to run.
import argparse
import os
import subprocess

import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model

from config import device, batch_size, root
from dataset import AdvDataset, transform
from attack import epoch_benign, fgsm, ifgsm, mifgsm, gen_adv_examples, create_dir
from ensemble import ensembleNet

parser = argparse.ArgumentParser()
parser.add_argument('--models', default='resnet110_cifar10')
parser.add_argument('--attacks', default='fgsm,ifgsm')
opt = parser.parse_args()

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')

model_names = opt.models.split(',')
if len(model_names) == 1:
    model = ptcv_get_model(model_names[0], pretrained=True).to(device)
else:
    model = ensembleNet(model_names).to(device)
loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

attacks = {'fgsm': fgsm, 'ifgsm': ifgsm, 'mifgsm': mifgsm}
for name in opt.attacks.split(','):
    adv_examples, acc, loss = gen_adv_examples(model, adv_loader, attacks[name], loss_fn)
    print(f'{name}_acc = {acc:.5f}, {name}_loss = {loss:.5f}')

    create_dir(root, name, adv_examples, adv_names)

# Compress the images: submit the .tgz file to JudgeBoi
for name in opt.attacks.split(','):
    subprocess.run(f'tar zcvf ../{name}.tgz *', shell=True, cwd=name)
