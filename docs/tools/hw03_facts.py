"""Measures the facts docs/HW03 cites (read-only on the repo; images go to docs/HW03/img/).

Run from HW03/:  PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw03_facts.py [part ...]
Parts: data thumbs aug shapes loader  (CPU only)
       timing                          (GPU; run when nothing else is training)
       tta CKDIR                       (GPU; needs the .ckpt/.npz files from hw03_run_grid.sh)
"""
import collections, os, sys, time
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

DS = './food11'
IMG = '../docs/HW03/img'
parts = sys.argv[1:] or ['data', 'shapes', 'loader']
AUG_A = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def files(split):
    d = os.path.join(DS, split)
    return sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith('.jpg'))


if 'data' in parts:
    for split in ['training', 'validation', 'test']:
        fs = files(split)
        sizes, modes, nbytes = collections.Counter(), collections.Counter(), 0
        ws, hs = [], []
        for f in fs:
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
                ws.append(im.size[0]); hs.append(im.size[1])
            nbytes += os.path.getsize(f)
        sq = sum(v for (w, h), v in sizes.items() if w == h)
        print(f'{split}: n={len(fs)} bytes={nbytes} modes={dict(modes)} distinct_sizes={len(sizes)} square={sq}')
        print('  top sizes (w,h):', sizes.most_common(6))
        print(f'  width min/median/max {min(ws)}/{int(np.median(ws))}/{max(ws)}  height {min(hs)}/{int(np.median(hs))}/{max(hs)}')
        if split != 'test':
            c = collections.Counter(int(os.path.basename(f).split('_')[0]) for f in fs)
            print('  per class:', [c[k] for k in range(11)])
    # labels: what dataset.py's try/except gives for each kind of name
    for name in ['./food11/training/3_120.jpg', './food11/test/0001.jpg']:
        try:
            print(name, '->', int(os.path.basename(name).split('_')[0]))
        except Exception as e:
            print(name, '->', type(e).__name__, e, '-> label -1')
    print('windows path with split("/"):', repr('food11\\training\\3_120.jpg'.split('/')[-1].split('_')[0]))

if 'thumbs' in parts:
    os.makedirs(IMG, exist_ok=True)
    tot = 0
    for k in range(11):
        f = os.path.join(DS, 'training', f'{k}_0.jpg')
        with Image.open(f) as im:
            print(k, os.path.basename(f), im.size)
            t = im.convert('RGB').copy()
        t.thumbnail((96, 96))
        out = os.path.join(IMG, f'class{k:02d}.jpg')
        t.save(out, quality=80)
        tot += os.path.getsize(out)
    print('thumbs bytes', tot)

if 'aug' in parts:
    os.makedirs(IMG, exist_ok=True)
    f = os.path.join(DS, 'training', '0_0.jpg')
    im = Image.open(f).convert('RGB')
    to_pil = transforms.ToPILImage()
    to_pil(test_tfm(im)).save(os.path.join(IMG, 'aug_resize.jpg'), quality=85)
    torch.manual_seed(0)
    outs = [AUG_A(im) for _ in range(5)]
    for i, t in enumerate(outs):
        to_pil(t).save(os.path.join(IMG, f'aug_{i + 1}.jpg'), quality=85)
    flat = torch.stack(outs).flatten(1)
    print('aug 5 outputs pairwise identical?', [[bool(torch.equal(flat[i], flat[j])) for j in range(5)] for i in range(5)])
    print('distinct outputs:', len({t.numpy().tobytes() for t in outs}))
    t = test_tfm(im)
    print('original size', im.size, 'tensor', tuple(t.shape), t.dtype, 'min/max', t.min().item(), t.max().item())
    print('pixel [0,0] raw RGB', im.resize((128, 128)).getpixel((0, 0)), '-> tensor', t[:, 0, 0].tolist())
    sizes = sum(os.path.getsize(os.path.join(IMG, x)) for x in os.listdir(IMG) if x.startswith('aug'))
    print('aug images bytes', sizes)

if 'shapes' in parts:
    from classifier import Classifier
    m = Classifier()
    x = torch.zeros(1, 3, 128, 128)
    print('input', tuple(x.shape))
    for i, layer in enumerate(m.cnn):
        x = layer(x)
        n = sum(p.numel() for p in layer.parameters())
        print(f'cnn[{i:2d}] {layer.__class__.__name__:12s} -> {tuple(x.shape)}  params={n}')
    x = x.view(1, -1)
    print('view ->', tuple(x.shape))
    for i, layer in enumerate(m.fc):
        x = layer(x)
        n = sum(p.numel() for p in layer.parameters())
        print(f'fc[{i}] {layer.__class__.__name__:8s} -> {tuple(x.shape)}  params={n}')
    print('total', sum(p.numel() for p in m.parameters()),
          'trainable', sum(p.numel() for p in m.parameters() if p.requires_grad),
          'buffers', sum(b.numel() for b in m.buffers()))
    print(m)
    # receptive field of the last conv output
    rf, jump = 1, 1
    for layer in m.cnn:
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            rf += (k - 1) * jump
            jump *= s
            print(f'  after {layer.__class__.__name__}: receptive field {rf}, jump {jump}')
    src = open('others.py').read()
    ns = {}
    exec(src[src.index('from torch import nn'):], ns)
    r = ns['Residual_Network']()
    x = torch.zeros(1, 3, 128, 128)
    for name in ['cnn_layer1', 'cnn_layer2', 'cnn_layer3', 'cnn_layer4', 'cnn_layer5', 'cnn_layer6']:
        x = getattr(r, name)(x)
        print(name, tuple(x.shape), sum(p.numel() for p in getattr(r, name).parameters()))
    print('fc_layer params', sum(p.numel() for p in r.fc_layer.parameters()), 'total', sum(p.numel() for p in r.parameters()))
    try:
        exec(src, {})
    except Exception as e:
        print('exec(others.py):', type(e).__name__, e)

if 'loader' in parts:
    for n, bs in [(9866, 256), (3430, 256), (3347, 256), (9866, 64)]:
        print(n, bs, 'batches', -(-n // bs), 'last', n - (n - 1) // bs * bs)

if 'timing' in parts:
    # how much of an epoch is JPEG decode + resize (CPU) vs the model step (GPU)
    from classifier import Classifier
    from dataset import FoodDataset
    from torch.utils.data import DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ds = FoodDataset(os.path.join(DS, 'training'), tfm=test_tfm)
    for nw in [0, 4, 8]:
        dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=nw, pin_memory=True)
        t = time.time()
        for x, y in dl:
            pass
        print(f'num_workers={nw}: one pass of training images (load only) {time.time() - t:.1f}s')
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    batches = [b for b in dl]
    m = Classifier().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    for rep in range(2):
        torch.cuda.synchronize(); t = time.time()
        for x, y in batches:
            loss = crit(m(x.cuda()), y.cuda())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 10); opt.step()
        torch.cuda.synchronize()
        print(f'model steps only (data already in RAM), pass {rep + 1}: {time.time() - t:.1f}s')
    print('peak GPU memory MiB', torch.cuda.max_memory_allocated() // 2**20)

if 'tta' in parts:
    ck = sys.argv[sys.argv.index('tta') + 1]
    print('TODO: tta/ensemble from', ck)
