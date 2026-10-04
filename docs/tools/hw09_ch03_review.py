"""Fill the TODO(本機實測) markers of docs/HW09/ch03.html (review of PR #13).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch03_review.py
Writes ../docs/HW09/img/ch03_smoothgrad_small.png.
"""
import sys
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

print('===== saliency: which RGB channel holds the max (batch of 10, as in compute_saliency_maps)')
x = images.cuda().requires_grad_()
logits = model(x)
torch.nn.CrossEntropyLoss()(logits, labels.cuda()).backward()
g = x.grad.abs().detach().cpu()
mx = g.max(dim=1, keepdim=True).values
ties = ((g == mx).sum(1) > 1)
_, arg = torch.max(g, dim=1)
for i in range(10):
    c = torch.bincount(arg[i].flatten(), minlength=3).float() / arg[i].numel()
    print(f'img {i}: R {c[0]:.3f} G {c[1]:.3f} B {c[2]:.3f} ties {ties[i].sum().item()} zero-grad pixels {(mx[i] == 0).sum().item()}')
c = torch.bincount(arg.flatten(), minlength=3).float() / arg.numel()
print(f'all 10: R {c[0]:.4f} G {c[1]:.4f} B {c[2]:.4f}; tied pixels {ties.sum().item()} of {ties.numel()}')

print('===== softmax of the batch logits for images 1, 6, 8 (float32)')
p = logits.detach().softmax(1).cpu()
torch.set_printoptions(precision=3, sci_mode=True, linewidth=200)
for i in (1, 6, 8):
    y = labels[i].item()
    others = torch.cat([p[i, :y], p[i, y + 1:]])
    print(f'img {i}: p(label) == 1.0 exactly: {p[i, y].item() == 1.0}; sum of other p_j {others.sum().item():.3e}; largest other p_j {others.max().item():.3e}')
    print('   p =', p[i])

print('===== repo noise: mean softmax over 500 noisy samples (torch.manual_seed(0) per image)')
for i in range(10):
    torch.manual_seed(0)
    xi = images[i]
    std = (0.4 / (xi.max() - xi.min()).item()) ** 2
    noisy = xi + torch.randn(500, 3, 128, 128) * std
    with torch.no_grad():
        pi = model(noisy.cuda()).softmax(1).cpu()
    print(f'img {i}: mean p(Vegetable/Fruit) {pi[:, 10].mean():.4f} min {pi[:, 10].min():.4f} | mean p(label) {pi[:, labels[i]].mean():.2e}')


def smooth_variant(x, y, n, std):
    acc = np.zeros((1, 3, 128, 128))
    for _ in range(n):
        noise = x.new_empty(x.size()).normal_(0, std)
        x_mod = (x + noise).unsqueeze(0).cuda().requires_grad_()
        torch.nn.CrossEntropyLoss()(model(x_mod), y.cuda().unsqueeze(0)).backward()
        acc += x_mod.grad.abs().detach().cpu().numpy()
    return acc / n


print('===== SmoothGrad channel means: repo std (seed 0) vs std 0.01 (seed 0)')
to_hwc = lambda a: np.transpose(a.reshape(3, 128, 128), (1, 2, 0))
repo, small = [], []
torch.manual_seed(0)
for i in range(10):
    repo.append(E.normalize(smooth_variant(images[i], labels[i], 500, (0.4 / (images[i].max() - images[i].min()).item()) ** 2)))
torch.manual_seed(0)
for i in range(10):
    small.append(E.normalize(smooth_variant(images[i], labels[i], 500, 0.01)))
for name, maps in (('repo', repo), ('std 0.01', small)):
    for i, m in enumerate(maps):
        r, gg, b = m.reshape(3, -1).mean(1)
        where = np.unravel_index(m.argmax(), m.shape)[1]
        print(f'{name} img {i}: channel means R {r:.3f} G {gg:.3f} B {b:.3f}; overall max in channel {"RGB"[where]}')
    allm = np.stack(maps).reshape(10, 3, -1)
    print(f'{name} all 10: R {allm[:, 0].mean():.3f} G {allm[:, 1].mean():.3f} B {allm[:, 2].mean():.3f}')

print('===== std 0.01 vs saliency (both reduced to one channel by max over RGB, per-image normalized)')
sal = E.compute_saliency_maps(images, labels, model)
for i in range(10):
    s1 = E.normalize(small[i].reshape(3, 128, 128).max(0))
    s2 = E.normalize(repo[i].reshape(3, 128, 128).max(0))
    print(f'img {i}: corr(std0.01, saliency) {np.corrcoef(s1.ravel(), sal[i].numpy().ravel())[0, 1]:.3f} corr(repo smoothgrad, saliency) {np.corrcoef(s2.ravel(), sal[i].numpy().ravel())[0, 1]:.3f}')

fig, axs = plt.subplots(4, 10, figsize=(15, 6.4 * 2))
rows = [(images.permute(0, 2, 3, 1).numpy(), 'image', None), (sal.numpy(), 'saliency (repo)', plt.cm.hot),
        ([to_hwc(m) for m in repo], 'repo: std (0.4/range)^2', None), ([to_hwc(m) for m in small], 'std 0.01', None)]
for r, (imgs, title, cmap) in enumerate(rows):
    for c in range(10):
        axs[r][c].imshow(imgs[c], cmap=cmap) if cmap else axs[r][c].imshow(np.clip(imgs[c], 0, 1))
    axs[r][0].set_ylabel(title, fontsize=9)
fig.savefig(IMG + 'ch03_smoothgrad_small.png', bbox_inches='tight')
plt.close(fig)

print('===== single noisy sample, img 0, std 0.01 (same draw order as hw09_ch03_grad.py)')
torch.manual_seed(0)
x0 = images[0]
for std in [0.0, 0.01]:
    noise = x0.new_empty(x0.size()).normal_(0, std) if std else torch.zeros_like(x0)
    x_mod = (x0 + noise).unsqueeze(0).cuda().requires_grad_()
    o = model(x_mod)
    l = torch.nn.CrossEntropyLoss()(o, labels[0].cuda().unsqueeze(0))
    l.backward()
    print(f'std {std}: p(label) {o.softmax(1)[0, 0].item():.6f} loss {l.item():.3e} grad max {x_mod.grad.abs().max().item():.3e}')
