"""Numbers and extra figures for docs/HW09 ch05 (Integrated Gradients).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch05_ig.py
Figures are written to ../docs/HW09/img/ch05_*.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
X = images.cuda()


def f(x, c):
    with torch.no_grad():
        return model(x)[:, c]


def grad_at(x, c):
    x = x.clone().requires_grad_()
    model(x)[:, c].sum().backward()
    return x.grad.detach()


def ig(x, c, steps, rule='left', baseline=None):
    """Average gradient on the straight path from baseline to x (no multiplication)."""
    b = torch.zeros_like(x) if baseline is None else baseline
    if rule == 'left':
        alphas = [k / steps for k in range(steps)]
    elif rule == 'mid':
        alphas = [(k + 0.5) / steps for k in range(steps)]
    acc = torch.zeros_like(x)
    for a in alphas:
        acc += grad_at(b + a * (x - b), c) / steps
    return acc


print('===== repo IG (IntegratedGradients class, 10 steps) and completeness')
IG = E.IntegratedGradients(model)
repo = []
rows = []
for i in range(10):
    x = X[i:i + 1]
    c = labels[i].item()
    g = IG.generate_integrated_gradients(x.clone(), labels[i], 10)   # numpy (3,128,128), float64
    repo.append(g)
    fx, f0 = f(x, c).item(), f(torch.zeros_like(x), c).item()
    s_prog = g.sum()
    s_paper = (g * x[0].cpu().numpy()).sum()
    rows.append((i, c, fx, f0, fx - f0, s_prog, s_paper))
    print(f'img {i} class {c}: f(x) {fx:.3f} f(0) {f0:.3f} diff {fx - f0:.3f} | sum(repo) {s_prog:.3f} | sum(repo * x) {s_paper:.3f} | max|repo| {np.abs(g).max():.3e} frac<0 {(g < 0).mean():.3f}')

print('===== f(0) per class: the all-black image')
with torch.no_grad():
    z = model(torch.zeros(1, 3, 128, 128).cuda())[0]
print('logits of the black image:', [round(v, 3) for v in z.tolist()], '| argmax', z.argmax().item(), '| p max %.4f' % z.softmax(0).max().item())

print('===== check: IG class == my left-Riemann ig()')
g_mine = ig(X[0:1], labels[0].item(), 10)[0].cpu().numpy()
print('img 0 max |repo - mine| %.3e (repo max %.3e)' % (np.abs(repo[0] - g_mine).max(), np.abs(repo[0]).max()))

print('===== convergence of sum(avg grad * x) toward f(x) - f(0)')
conv = {}
for i in range(10):
    x = X[i:i + 1]; c = labels[i].item()
    target = rows[i][4]
    out = []
    for rule, n in [('left', 10), ('left', 50), ('left', 200), ('mid', 10), ('mid', 50), ('mid', 200)]:
        s = (ig(x, c, n, rule) * x).sum().item()
        out.append((rule, n, s))
    conv[i] = out
    print(f'img {i} target {target:.3f}: ' + '; '.join(f'{r}{n} {s:.3f}' for r, n, s in out))

print('===== the path for image 0: f(alpha x), p(label), |grad| over alpha')
x = X[0:1]; c = labels[0].item()
alphas = np.linspace(0, 1, 101)
fs, ps, gs, gsx = [], [], [], []
for a in alphas:
    xa = a * x
    with torch.no_grad():
        o = model(xa)[0]
    fs.append(o[c].item()); ps.append(o.softmax(0)[c].item())
    g = grad_at(xa, c)
    gs.append(g.abs().max().item()); gsx.append((g * x).sum().item())
for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    k = int(round(a * 100))
    with torch.no_grad():
        am = model(alphas[k] * x)[0].argmax().item()
    print(f'alpha {a:.1f}: f {fs[k]:.3f} p(label) {ps[k]:.4f} argmax {am} max|grad| {gs[k]:.3e} sum(grad*x) {gsx[k]:.3f}')
first99 = next(a for a, p in zip(alphas, ps) if p >= 0.99)
first_arg = None
for a in alphas:
    with torch.no_grad():
        if model(a * x)[0].argmax().item() == c:
            first_arg = a; break
print('img 0: first alpha with p(label) >= 0.99: %.2f; first alpha where argmax == label: %.2f' % (first99, first_arg))
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
axs[0].plot(alphas, fs); axs[0].scatter([k / 10 for k in range(10)], [fs[k * 10] for k in range(10)], color='red', zorder=3)
axs[0].set_title('f(alpha * x): Bread logit, image 0', fontsize=10); axs[0].set_xlabel('alpha')
axs[1].plot(alphas, ps); axs[1].set_title('p(Bread)', fontsize=10); axs[1].set_xlabel('alpha')
axs[2].plot(alphas, gsx); axs[2].scatter([k / 10 for k in range(10)], [gsx[k * 10] for k in range(10)], color='red', zorder=3)
axs[2].axhline(0, color='gray', lw=0.5)
axs[2].set_title('sum(grad * x) at alpha (red = the 10 points the repo uses)', fontsize=10); axs[2].set_xlabel('alpha')
fig.savefig(IMG + 'ch05_path.png', bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(1, 11, figsize=(16, 2))
for k in range(11):
    axs[k].imshow((k / 10 * images[0]).permute(1, 2, 0).numpy())
    axs[k].set_title(f'alpha {k / 10:.1f}', fontsize=8); axs[k].set_xticks([]); axs[k].set_yticks([])
fig.savefig(IMG + 'ch05_alpha_images.png', bbox_inches='tight')
plt.close(fig)

print('===== all 10 images: path summary')
for i in range(10):
    x = X[i:i + 1]; c = labels[i].item()
    pa = []
    for a in np.linspace(0, 1, 101):
        with torch.no_grad():
            pa.append(model(a * x)[0].softmax(0)[c].item())
    first = next((a for a, p in zip(np.linspace(0, 1, 101), pa) if p >= 0.5), None)
    print(f'img {i}: p(label) at alpha 0 {pa[0]:.4f}, 0.5 {pa[50]:.4f}, 0.9 {pa[90]:.4f}, 1 {pa[100]:.4f}; first alpha with p >= 0.5: {first if first is None else round(first, 2)}')

print('===== display: normalize keeps the sign')
for i in (0, 3):
    g = repo[i]
    lo, hi = g.min(), g.max()
    print(f'img {i}: repo IG min {lo:.3e} max {hi:.3e}; value 0 maps to {(0 - lo) / (hi - lo):.3f} after normalize; frac of values < 0 {(g < 0).mean():.3f}')

print('===== figure: repo vs x multiplied vs abs')
times_x = [repo[i] * images[i].numpy() for i in range(10)]
mid200 = [(ig(X[i:i + 1], labels[i].item(), 200, 'mid') * X[i:i + 1])[0].cpu().numpy() for i in range(10)]
for i in range(10):
    a, b = E.normalize(repo[i]).ravel(), E.normalize(times_x[i]).ravel()
    print(f'img {i}: corr(normalize(repo), normalize(repo*x)) {np.corrcoef(a, b)[0, 1]:.3f}; corr(repo*x, mid200*x) {np.corrcoef(times_x[i].ravel(), mid200[i].ravel())[0, 1]:.3f}')
to_hwc = lambda a: np.moveaxis(E.normalize(a), 0, -1)
fig, axs = plt.subplots(5, 10, figsize=(15, 8.2))
grid = [(images.permute(0, 2, 3, 1).numpy(), 'image'), ([to_hwc(g) for g in repo], 'repo: avg grad (10 steps)'),
        ([to_hwc(g) for g in times_x], 'avg grad * x (10 steps)'), ([to_hwc(g) for g in mid200], 'avg grad * x (midpoint, 200)'),
        ([E.normalize(np.abs(g).max(0)) for g in mid200], '|IG| max over RGB (hot)')]
for r, (imgs, title) in enumerate(grid):
    for c in range(10):
        axs[r][c].imshow(imgs[c], cmap=plt.cm.hot if r == 4 else None)
        axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
    axs[r][0].set_ylabel(title, fontsize=7)
fig.savefig(IMG + 'ch05_ig_variants.png', bbox_inches='tight')
plt.close(fig)

print('===== flexible baseline (slide p.11): black / gray 0.5 / blurred / uniform noise; midpoint 200 steps')
torch.manual_seed(0)
bases = {
    'black': lambda x: torch.zeros_like(x),
    'gray 0.5': lambda x: torch.full_like(x, 0.5),
    'blur': lambda x: F.avg_pool2d(F.pad(x, (15, 15, 15, 15), mode='replicate'), 31, stride=1),
    'noise': lambda x: torch.rand_like(x),
}
base_maps = {k: [] for k in bases}
for i in range(10):
    x = X[i:i + 1]; c = labels[i].item()
    parts = []
    for k, fn in bases.items():
        b = fn(x)
        attr = ig(x, c, 200, 'mid', baseline=b) * (x - b)
        base_maps[k].append(attr[0].cpu().numpy())
        parts.append(f'{k}: f(b) {f(b, c).item():.3f} diff {f(x, c).item() - f(b, c).item():.3f} sum {attr.sum().item():.3f}')
    print(f'img {i}: ' + ' | '.join(parts))
for i in range(10):
    ref = np.abs(base_maps['black'][i]).max(0).ravel()
    print(f'img {i}: corr of |attr| (max over RGB) with black baseline: ' + ', '.join(f'{k} {np.corrcoef(ref, np.abs(base_maps[k][i]).max(0).ravel())[0, 1]:.3f}' for k in bases if k != 'black'))
fig, axs = plt.subplots(1 + 2 * len(bases), 10, figsize=(15, 13))
for c in range(10):
    axs[0][c].imshow(images[c].permute(1, 2, 0).numpy())
r = 1
for k, fn in bases.items():
    for c in range(10):
        axs[r][c].imshow(np.clip(fn(X[c:c + 1])[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
        axs[r + 1][c].imshow(E.normalize(np.abs(base_maps[k][c]).max(0)), cmap=plt.cm.hot)
    axs[r][0].set_ylabel(f'baseline: {k}', fontsize=7); axs[r + 1][0].set_ylabel(f'|IG| ({k})', fontsize=7)
    r += 2
axs[0][0].set_ylabel('image', fontsize=7)
for row in axs:
    for ax in row:
        ax.set_xticks([]); ax.set_yticks([])
fig.savefig(IMG + 'ch05_baselines.png', bbox_inches='tight')
plt.close(fig)
print('figures written')
