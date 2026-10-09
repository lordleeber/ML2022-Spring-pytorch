# HW13 ch06: structured pruning that really removes channels from the sample student (model.py StudentNet).
import copy
import torch
import torch.nn as nn


def shrink_student(m, keep):
  """Physically remove output channels of every conv in the sample student (L1 norm), fix BN and the next layer."""
  layers = list(m.cnn)
  conv_idx = [i for i, l in enumerate(layers) if isinstance(l, nn.Conv2d)]
  new, prev_keep = copy.deepcopy(layers), torch.arange(3)
  for ci in conv_idx:
    conv, bn = layers[ci], layers[ci + 1]
    n_out = conv.out_channels
    k = max(1, round(n_out * keep))
    idx = conv.weight.detach().abs().sum((1, 2, 3)).argsort(descending=True)[:k].sort().values
    c = nn.Conv2d(len(prev_keep), k, conv.kernel_size, conv.stride, conv.padding)
    c.weight.data = conv.weight.data[idx][:, prev_keep].clone(); c.bias.data = conv.bias.data[idx].clone()
    b = nn.BatchNorm2d(k)
    for name in ['weight', 'bias', 'running_mean', 'running_var']:
      getattr(b, name).data = getattr(bn, name).data[idx].clone()
    new[ci], new[ci + 1] = c, b
    prev_keep = idx
  out = copy.deepcopy(m)
  out.cnn = nn.Sequential(*new)
  fc = m.fc[0]
  out.fc = nn.Sequential(nn.Linear(len(prev_keep), 11))
  out.fc[0].weight.data = fc.weight.data[:, prev_keep].clone(); out.fc[0].bias.data = fc.bias.data.clone()
  return out.eval()


def narrow_student(channels):
  """A fresh, randomly initialised StudentNet with the given conv widths (same layout as model.py)."""
  from model import get_student_model
  m = get_student_model()
  layers, c_in = list(m.cnn), 3
  conv_idx = [i for i, l in enumerate(layers) if isinstance(l, nn.Conv2d)]
  for ci, c_out in zip(conv_idx, channels):
    layers[ci] = nn.Conv2d(c_in, c_out, 3)
    layers[ci + 1] = nn.BatchNorm2d(c_out)
    c_in = c_out
  m.cnn = nn.Sequential(*layers)
  m.fc = nn.Sequential(nn.Linear(c_in, 11))
  return m
