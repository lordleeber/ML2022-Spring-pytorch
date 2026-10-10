# Run hw10_exp.py with JPEG-aware surrogates, without editing it: a surrogate named "jpegR+<model>" is <model> behind
# a JPEG layer (imgaug compression rate R). Forward pass uses the real JPEG; backward pass treats JPEG as the
# identity (BPDA, Athalye et al. 2018: x + (jpeg(x) - x).detach()).
# usage (from HW10/): python ../docs/tools/hw10_det.py ../docs/tools/hw10_bpda.py ../docs/tools/hw10_exp.py <exp args>
import io
import os
import runpy
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pytorchcv.model_provider as mp

sys.path.insert(0, os.getcwd())
from config import mean, std

_orig = mp.get_model


def jpeg_np(arr, rate):
  # same mapping as imgaug 0.4.0 JpegCompression(compression=rate) (and hw10_exp.py's jpeg())
  q = int(np.clip(np.round(1 + 99 * (1.0 - rate / 101)), 1, 100))
  buf = io.BytesIO()
  Image.fromarray(arr).save(buf, format='JPEG', quality=q)
  buf.seek(0)
  return np.array(Image.open(buf).convert('RGB'))


class JpegFront(nn.Module):
  def __init__(self, model, rate):
    super().__init__()
    self.model, self.rate = model, rate

  def load_state_dict(self, state_dict, *args, **kw):
    # hw10_exp.py loads our own checkpoints ("arch@path") after building the model: they belong to the inner model
    return self.model.load_state_dict(state_dict, *args, **kw)

  def forward(self, x):
    with torch.no_grad():
      pix = ((x * std + mean).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
      xj = torch.stack([torch.from_numpy(jpeg_np(a, self.rate)) for a in pix]).to(x.device)
      xj = (xj.permute(0, 3, 1, 2).float() / 255 - mean) / std
    return self.model(x + (xj - x).detach())


def get_model(name, **kw):
  if name.startswith('jpeg') and '+' in name:
    rate, base = name.split('+', 1)
    return JpegFront(_orig(base, **kw), float(rate[4:]))
  return _orig(name, **kw)


mp.get_model = get_model
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
