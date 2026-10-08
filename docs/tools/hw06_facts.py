"""HW06 data / loader facts for the textbook (ch01). Run from HW06/ with PYTHONPATH=.

  data     every file in faces/: size, mode, format, bytes; glob order vs sorted order;
           one image through get_dataset's transform step by step (dtype, shape, range)
  stats    per-channel mean / std of all 71,314 images after the transform (8 workers, CPU)
  loader   len(dataloader), last batch size, and the wall time of one epoch of the DataLoader
           alone (no model) for num_workers 0 / 2 / 4 / 8
  model    ch02: transposed-conv shape formula, conv_transpose2d == gradient of conv2d, which modules
           weights_init touches and the resulting weight statistics, outputs of the freshly
           initialized G and D (train.py's seed order: same_seeds(2022) -> Generator -> Discriminator)
  figs     PNG grids for the book: the first 24 files by number at 96x96 and after the transform
           (64x64), written to the directory given as the second argument
"""
import glob, os, sys, time, collections
import numpy as np
import torch, torchvision
from PIL import Image
from torch.utils.data import DataLoader

import utils

cmd = sys.argv[1]

if cmd == 'data':
    files = glob.glob(os.path.join('faces', '*'))
    print('glob count', len(files), 'first 5 in glob order', files[:5])
    byname = sorted(files, key=lambda p: int(os.path.basename(p)[:-4]))
    print('first 5 by number', byname[:5], 'last', byname[-1])
    print('glob order == numeric order:', files == byname, '; == string-sorted:', files == sorted(files))
    c, sizes = collections.Counter(), []
    for p in files:
        im = Image.open(p)
        c[(im.size, im.mode, im.format)] += 1
        sizes.append(os.path.getsize(p))
    print('size/mode/format', dict(c))
    sizes = np.array(sizes)
    print('bytes total', sizes.sum(), 'mean %.1f min %d max %d' % (sizes.mean(), sizes.min(), sizes.max()))
    ds = utils.get_dataset('faces')
    p = 'faces/0.jpg'
    x = torchvision.io.read_image(p)
    print('read_image', x.dtype, tuple(x.shape), int(x.min()), int(x.max()))
    t = ds.transform.transforms
    y = t[0](x); print('ToPILImage', type(y).__name__, y.size, y.mode)
    y = t[1](y); print('Resize', type(y).__name__, y.size, 'interpolation', t[1].interpolation, 'antialias', t[1].antialias)
    y = t[2](y); print('ToTensor', y.dtype, tuple(y.shape), '%.4f %.4f' % (y.min(), y.max()))
    y = t[3](y); print('Normalize', y.dtype, tuple(y.shape), '%.4f %.4f' % (y.min(), y.max()))
    print('values per channel after Normalize are (k/255 - 0.5)/0.5: first pixel', y[:, 0, 0].tolist())

elif cmd == 'stats':
    ds = utils.get_dataset('faces')
    dl = DataLoader(ds, batch_size=500, num_workers=8)
    s = torch.zeros(3, dtype=torch.float64); s2 = torch.zeros(3, dtype=torch.float64); n = 0
    lo, hi = 1e9, -1e9
    for x in dl:
        x = x.double()
        s += x.sum(dim=(0, 2, 3)); s2 += (x ** 2).sum(dim=(0, 2, 3)); n += x.numel() // 3
        lo, hi = min(lo, x.min().item()), max(hi, x.max().item())
    m = s / n; sd = (s2 / n - m ** 2).sqrt()
    print('images', len(ds), 'mean RGB (after Normalize)', [round(v, 4) for v in m.tolist()],
          'std', [round(v, 4) for v in sd.tolist()], 'min %.4f max %.4f' % (lo, hi))
    print('mean RGB in [0,1]', [round(v * 0.5 + 0.5, 4) for v in m.tolist()])

elif cmd == 'loader':
    ds = utils.get_dataset('faces')
    for w in (0, 2, 4, 8):
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=w)
        t0 = time.time(); nb = 0; last = None
        for x in dl:
            nb += 1; last = x.shape
        print(f'workers {w}: len {len(dl)} batches {nb} last batch {tuple(last)} one epoch {time.time() - t0:.1f}s', flush=True)

elif cmd == 'model':
    import torch.nn as nn, torch.nn.functional as F
    from generator import Generator
    from discriminator import Discriminator
    x = torch.randn(1, 512, 4, 4)
    for op in (1, 0):
        ct = nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=op, bias=False)
        print(f'ConvTranspose2d k5 s2 p2 output_padding={op}: 4x4 ->', tuple(ct(x).shape[2:]))
    for h in (4, 8, 16, 32):
        print(f'  in {h}: (h-1)*2 - 2*2 + 5 + 1 =', (h - 1) * 2 - 4 + 5 + 1)
    c = nn.Conv2d(3, 64, 4, 2, 1)
    print('Conv2d k4 s2 p1: 64x64 ->', tuple(c(torch.randn(1, 3, 64, 64)).shape[2:]), '; floor((64+2-4)/2)+1 =', (64 + 2 - 4) // 2 + 1)
    # conv_transpose2d is the gradient of conv2d w.r.t. its input
    torch.manual_seed(0)
    W = torch.randn(8, 4, 5, 5)           # conv2d weight (out=8, in=4); the same tensor is a conv_transpose weight (in=8, out=4)
    big = torch.randn(1, 4, 16, 16, requires_grad=True)
    small = F.conv2d(big, W, stride=2, padding=2)          # 16 -> 8
    g = torch.randn_like(small)
    grad, = torch.autograd.grad(small, big, g)
    ct = F.conv_transpose2d(g, W, stride=2, padding=2, output_padding=1)
    print('conv2d 16->8 shape', tuple(small.shape), '; conv_transpose2d 8->16 shape', tuple(ct.shape),
          '; max |grad - conv_transpose| = %.2e' % (grad - ct).abs().max().item())
    utils.same_seeds(2022)
    G = Generator(100); D = Discriminator(3)
    for name, m in [('G', G), ('D', D)]:
        for n, mod in m.named_modules():
            if len(list(mod.children())) or not any(True for _ in mod.parameters(recurse=False)):
                continue
            cls = mod.__class__.__name__
            hit = 'Conv' if cls.find('Conv') != -1 else ('BatchNorm' if cls.find('BatchNorm') != -1 else 'not touched')
            w = mod.weight.data
            print(f'{name} {n:8s} {cls:16s} weights_init:{hit:12s} weight mean {w.mean():+.4f} std {w.std():.4f}'
                  + (f' bias all zero {bool((mod.bias == 0).all())}' if mod.bias is not None else ''))
    lin = G.l1[0]
    print('default Linear init bound 1/sqrt(100) = %.4f; uniform std = bound/sqrt(3) = %.4f; measured |w| max %.4f'
          % (0.1, 0.1 / 3 ** 0.5, lin.weight.detach().abs().max()))
    G.train(); D.train()
    with torch.no_grad():
        z = torch.randn(64, 100)
        f = G(z)
        print('fresh G(z) train mode: shape', tuple(f.shape), 'mean %.4f std %.4f min %.4f max %.4f |x|>0.99: %.4f'
              % (f.mean(), f.std(), f.min(), f.max(), (f.abs() > 0.99).float().mean()))
        ds = utils.get_dataset('faces')
        r = torch.stack([ds[i] for i in range(64)])
        dr, df = D(r), D(f)
        print('fresh D(real) mean %.4f min %.4f max %.4f ; D(fake) mean %.4f min %.4f max %.4f'
              % (dr.mean(), dr.min(), dr.max(), df.mean(), df.min(), df.max()))
        bce = nn.BCELoss()
        print('BCE at init: D loss %.4f (ln2 = %.4f), G loss %.4f' % (
            ((bce(dr, torch.ones(64)) + bce(df, torch.zeros(64))) / 2).item(), torch.log(torch.tensor(2.)).item(),
            bce(df, torch.ones(64)).item()))

elif cmd == 'figs':
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ds = utils.get_dataset('faces')
    names = [f'faces/{i}.jpg' for i in range(24)]
    raw = torch.stack([torchvision.io.read_image(p) for p in names]).float() / 255
    torchvision.utils.save_image(raw, os.path.join(out, 'ch01_crypko96.png'), nrow=12, padding=2)
    t = torch.stack([ds.transform(torchvision.io.read_image(p)) for p in names]) * 0.5 + 0.5
    torchvision.utils.save_image(t, os.path.join(out, 'ch01_crypko64.png'), nrow=12, padding=2)
    print('wrote', sorted(os.listdir(out)))
