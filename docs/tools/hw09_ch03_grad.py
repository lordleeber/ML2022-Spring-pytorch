"""Numbers and extra figures for docs/HW09 ch03 (Saliency map, SmoothGrad).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch03_grad.py
Figures are written to ../docs/HW09/img/ch03_*.png.
"""
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels
import explain_cnn as E

IMG = '../docs/HW09/img/'
model = Classifier().cuda()
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))


def row_figure(rows, name, cmaps):
    fig, axs = plt.subplots(len(rows), 10, figsize=(15, 3.2 * len(rows)))
    for r, (imgs, title) in enumerate(rows):
        for c in range(10):
            axs[r][c].imshow(imgs[c], cmap=cmaps[r], vmin=0, vmax=1) if cmaps[r] else axs[r][c].imshow(np.clip(imgs[c], 0, 1))
        axs[r][0].set_ylabel(title, fontsize=9)
    fig.savefig(IMG + name, bbox_inches='tight')
    plt.close(fig)


print('===== saliency: loss vs per-image quantities')
x = images.cuda().requires_grad_()
logits = model(x)
loss = torch.nn.CrossEntropyLoss()(logits, labels.cuda())
loss.backward()
p = logits.softmax(1).detach().cpu()
sal = x.grad.abs().max(dim=1).values.cpu()
for i in range(10):
    y = labels[i].item()
    per_loss = -torch.log_softmax(logits[i].detach(), 0)[y].item()
    print(f'img {i}: 1-p {1 - p[i, y].item():.2e} per-image CE {per_loss:.2e} saliency max {sal[i].max():.3e}')
print('batch mean CE %.6f' % loss.item())

print('===== normalization: per image (repo) vs one global min-max')
per_image = torch.stack([E.normalize(s) for s in sal])
glob = E.normalize(sal)
for i in range(10):
    print(f'img {i}: global-normalized max {glob[i].max():.2e} mean {glob[i].mean():.2e}')
row_figure([(images.permute(0, 2, 3, 1).numpy(), 'image'), (per_image.numpy(), 'per-image (repo)'), (glob.numpy(), 'one global min-max')], 'ch03_saliency_global.png', [None, plt.cm.hot, plt.cm.hot])

print('===== gradient of the label logit instead of the loss (slide p.8 wording)')
x2 = images.cuda().requires_grad_()
out = model(x2)
out.gather(1, labels.cuda().view(-1, 1)).sum().backward()
sal_logit = x2.grad.abs().max(dim=1).values.cpu()
for i in range(10):
    a = per_image[i].flatten()
    b = E.normalize(sal_logit[i]).flatten()
    corr = np.corrcoef(a.numpy(), b.numpy())[0, 1]
    print(f'img {i}: logit-grad max {sal_logit[i].max():.3e} | corr with repo saliency (both normalized) {corr:.3f}')
row_figure([(images.permute(0, 2, 3, 1).numpy(), 'image'), (per_image.numpy(), 'grad of CE loss (repo)'), (torch.stack([E.normalize(s) for s in sal_logit]).numpy(), 'grad of label logit')], 'ch03_saliency_logit.png', [None, plt.cm.hot, plt.cm.hot])

print('===== SmoothGrad variants (torch.manual_seed(0) before each variant)')


def smooth_variant(x, y, n, std_fn, do_norm=True):
    acc = np.zeros((1, 3, 128, 128))
    std = std_fn(x)
    for _ in range(n):
        noise = x.new_empty(x.size()).normal_(0, std)
        x_mod = (x + noise).unsqueeze(0).cuda().requires_grad_()
        torch.nn.CrossEntropyLoss()(model(x_mod), y.cuda().unsqueeze(0)).backward()
        acc += x_mod.grad.abs().detach().cpu().numpy()
    acc = acc / n
    return E.normalize(acc) if do_norm else acc


repo_std = lambda x: (0.4 / (x.max() - x.min()).item()) ** 2
paper_std = lambda x: 0.4 * (x.max() - x.min()).item()
variants = {}
for name, kw in [('repo', dict(n=500, std_fn=repo_std)), ('nonorm', dict(n=500, std_fn=repo_std, do_norm=False)), ('paper', dict(n=500, std_fn=paper_std))]:
    torch.manual_seed(0)
    t = time.time()
    variants[name] = [smooth_variant(images[i], labels[i], **kw) for i in range(10)]
    print(f'{name}: {time.time() - t:.1f} s')
for i in range(10):
    nn_ = variants['nonorm'][i]
    print(f'img {i}: no-normalize value range {nn_.min():.3e} .. {nn_.max():.3e} | pixels > 1 after clip: {(nn_ > 1).mean():.3f}')
to_hwc = lambda arr: np.transpose(arr.reshape(3, 128, 128), (1, 2, 0))
row_figure([(images.permute(0, 2, 3, 1).numpy(), 'image'), ([to_hwc(a) for a in variants['repo']], 'repo: std (0.4/range)^2'), ([to_hwc(a) for a in variants['nonorm']], 'no normalize'), ([to_hwc(a) for a in variants['paper']], 'paper: std 0.4*range')], 'ch03_smoothgrad_variants.png', [None, None, None, None])

print('===== SmoothGrad vs number of samples (img 0, repo std)')
torch.manual_seed(0)
ns = [1, 10, 50, 500]
maps = []
for n in ns:
    maps.append(to_hwc(smooth_variant(images[0], labels[0], n, repo_std)))
ref = maps[-1]
for n, m in zip(ns, maps):
    print(f'n={n}: mean |map - map(500)| {np.abs(m - ref).mean():.4f}')
fig, axs = plt.subplots(1, 5, figsize=(15, 3.5))
axs[0].imshow(images[0].permute(1, 2, 0).numpy())
axs[0].set_title('image 0', fontsize=9)
for ax, n, m in zip(axs[1:], ns, maps):
    ax.imshow(m)
    ax.set_title(f'{n} sample(s)', fontsize=9)
fig.savefig(IMG + 'ch03_smoothgrad_n.png', bbox_inches='tight')
plt.close(fig)

print('===== single noisy sample: how the gradient grows')
torch.manual_seed(0)
x = images[0]
for std in [0.0, 0.01, 0.05, 0.1613, 0.3984]:
    noise = x.new_empty(x.size()).normal_(0, std) if std else torch.zeros_like(x)
    x_mod = (x + noise).unsqueeze(0).cuda().requires_grad_()
    o = model(x_mod)
    l = torch.nn.CrossEntropyLoss()(o, labels[0].cuda().unsqueeze(0))
    l.backward()
    print(f'std {std}: p(label) {o.softmax(1)[0, 0].item():.4f} loss {l.item():.3e} grad max {x_mod.grad.abs().max().item():.3e}')

print('===== parameter grads accumulate (model.zero_grad is never called in saliency/smooth_grad)')
model.zero_grad()
E.compute_saliency_maps(images, labels, model)
g1 = model.fc[3].weight.grad.abs().sum().item()
E.compute_saliency_maps(images, labels, model)
g2 = model.fc[3].weight.grad.abs().sum().item()
print('fc[3].weight.grad |sum| after 1 call %.3e, after 2 calls %.3e (ratio %.2f)' % (g1, g2, g2 / g1))
