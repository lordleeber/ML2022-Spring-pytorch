# ch07: report question with and without model.eval(). pytorchcv returns models in training mode; report.py's
# notebook cells relied on gen_adv_examples() having called model.eval(). A fresh model per measurement, because a
# forward pass in training mode also updates the BatchNorm running statistics (even under no_grad).
# usage (from HW10/, after hw10.py): python ../docs/tools/hw10_ch07_eval.py
import os
import sys

import torch
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model

sys.path.insert(0, os.getcwd())
from config import device
from dataset import transform

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('training flag right after get_model:', ptcv_get_model('resnet110_cifar10', pretrained=True).training)
for f in ('./data/dog/dog2.png', './fgsm/dog/dog2.png'):
  x = transform(Image.open(f)).unsqueeze(0).to(device)
  for mode in ('train', 'eval'):
    m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
    getattr(m, mode)()
    with torch.no_grad():
      logit = m(x)[0]
    p = logit.argmax().item()
    print(f'{f} {mode}: {classes[p]} {logit.softmax(-1)[p].item():.2%}')
# a single training-mode forward changes the running statistics
m = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
before = m.features.init_block.bn.running_mean.clone()
with torch.no_grad():
  m.train()(transform(Image.open('./data/dog/dog2.png')).unsqueeze(0).to(device))
print('init_block BN running_mean max change after one train-mode forward:', (m.features.init_block.bn.running_mean - before).abs().max().item())
m.eval()
with torch.no_grad():
  logit = m(transform(Image.open('./data/dog/dog2.png')).unsqueeze(0).to(device))[0]
p = logit.argmax().item()
print(f'then eval on benign dog2: {classes[p]} {logit.softmax(-1)[p].item():.2%}')
