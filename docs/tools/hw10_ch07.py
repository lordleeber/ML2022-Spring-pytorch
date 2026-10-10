# ch07 facts: how much of the adversarial perturbation survives JPEG. For a saved attack folder, compare
# delta = adv - benign with delta_j = jpeg(adv) - jpeg(benign) (same quality), in 0-255 units:
# RMS size, cosine similarity between delta and delta_j, and the share of energy in the "high" spatial frequencies (2D FFT outside the central 8x8 of 32x32).
# usage (from HW10/): python ../docs/tools/hw10_ch07.py <runs_dir> <tag>[,<tag>...] <rate>[,<rate>...]
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
from dataset import AdvDataset


def jpeg(arr, rate):
  q = int(np.clip(np.round(1 + 99 * (1.0 - rate / 101)), 1, 100))
  buf = io.BytesIO()
  Image.fromarray(arr).save(buf, format='JPEG', quality=q)
  buf.seek(0)
  return np.array(Image.open(buf).convert('RGB')).astype(np.float64), q


def high_share(d):
  # d: (32, 32, 3); energy outside the lowest 8x8 frequencies (|fy|, |fx| < 4 cycles per image)
  F = np.abs(np.fft.fftshift(np.fft.fft2(d, axes=(0, 1)), axes=(0, 1))) ** 2
  low = F[12:20, 12:20].sum()
  return 1 - low / F.sum()


runs_dir, tags, rates = sys.argv[1], sys.argv[2].split(','), [float(r) for r in sys.argv[3].split(',')]
benign = AdvDataset('./data', transform=None)
B = [np.array(Image.open(f).convert('RGB')) for f in benign.images]
for tag in tags:
  A = [np.array(Image.open(f).convert('RGB')) for f in AdvDataset(os.path.join(runs_dir, tag), transform=None).images]
  D = [a.astype(np.float64) - b for a, b in zip(A, B)]
  print(f'{tag}: delta RMS {np.sqrt(np.mean([np.mean(d ** 2) for d in D])):.3f}  high-frequency share {np.mean([high_share(d) for d in D]):.3f}')
  for r in rates:
    DJ, q, cos = [], None, []
    for a, b in zip(A, B):
      ja, q = jpeg(a, r)
      jb, _ = jpeg(b, r)
      DJ.append(ja - jb)
      d0, d1 = (a.astype(np.float64) - b).ravel(), (ja - jb).ravel()
      cos.append(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-12))
    print(f'  rate {r:.0f} (quality {q}): delta after JPEG RMS {np.sqrt(np.mean([np.mean(d ** 2) for d in DJ])):.3f}'
          f'  high-frequency share {np.mean([high_share(d) for d in DJ]):.3f}  cosine(delta, delta_j) {np.mean(cos):.3f}'
          f'  | JPEG error on the benign image RMS {np.sqrt(np.mean([np.mean((jpeg(b, r)[0] - b) ** 2) for b in B])):.3f}')
