"""Numbers and extra figures for docs/HW09 ch04 (Filter explanation).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch04_filter.py
Figures are written to ../docs/HW09/img/ch04_*.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

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
NAMES = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

print('===== layers around cnn[6] and cnn[23]')
for i in list(range(3, 10)) + list(range(20, 27)):
    print(i, model.cnn[i])

captured = {}
def grab(name):
    def hook(m, inp, out):
        captured[name] = out.detach()
    return hook
hs = [model.cnn[k].register_forward_hook(grab(k)) for k in (6, 7, 8, 23, 24, 25)]
with torch.no_grad():
    model(images.cuda())
for h in hs:
    h.remove()
print('===== filter 0 activation: conv output vs after BN vs after ReLU (10 original images)')
for k in (6, 7, 8, 23, 24, 25):
    a = captured[k][:, 0]
    print(f'cnn[{k}] {type(model.cnn[k]).__name__}: shape {tuple(captured[k].shape)} filter0 min {a.min():.3f} max {a.max():.3f} mean {a.mean():.3f} frac<0 {(a < 0).float().mean():.3f} frac==0 {(a == 0).float().mean():.3f}')
for k in (6, 23):
    a = captured[k][:, 0]
    print(f'cnn[{k}] filter0 per-image sum:', [round(v) for v in a.sum((1, 2)).tolist()])
    print(f'cnn[{k}] filter0 per-image min/max:', [(round(lo, 2), round(hi, 2)) for lo, hi in zip(a.amin((1, 2)).tolist(), a.amax((1, 2)).tolist())])
w6 = model.cnn[6].weight
print('cnn[6] weight shape', tuple(w6.shape), 'filter 0 weight shape', tuple(w6[0].shape), 'bias[0] %.4f' % model.cnn[6].bias[0].item())
print('cnn[23] weight shape', tuple(model.cnn[23].weight.shape), 'bias[0] %.4f' % model.cnn[23].bias[0].item())

print('===== hooks: registration count and what the global holds')
E.layer_activations = None
fa, fv = E.filter_explanation(images, model, cnnid=6, filterid=0, iteration=1, lr=0.1)
print('after filter_explanation: hooks on cnn[6]', len(model.cnn[6]._forward_hooks), '| global layer_activations shape', tuple(E.layer_activations.shape), 'requires_grad', E.layer_activations.requires_grad, 'device', E.layer_activations.device)
h = model.cnn[6].register_forward_hook(lambda m, i, o: None)
h2 = model.cnn[6].register_forward_hook(lambda m, i, o: None)
print('register twice without remove: hooks on cnn[6]', len(model.cnn[6]._forward_hooks))
h.remove(); h2.remove()
print('after remove:', len(model.cnn[6]._forward_hooks))
x = images.cuda()
y = x.cuda()
print('x.cuda() on a cuda tensor returns the same object:', x is y, '| images.cuda() returns a new tensor each call:', images.cuda() is not images.cuda())


def ascend(x0, cnnid, iteration=100, lr=0.1, clamp=False, log=None):
    """Same loop as filter_explanation, with optional clamp and logging."""
    act = {}
    hh = model.cnn[cnnid].register_forward_hook(lambda m, i, o: act.__setitem__('a', o))
    x = x0.clone().cuda().requires_grad_()
    opt = Adam([x], lr=lr)
    for it in range(iteration):
        opt.zero_grad()
        model(x)
        obj = -act['a'][:, 0].sum()
        if log is not None:
            log.append((it, -obj.item(), x.detach().min().item(), x.detach().max().item(), act['a'][:, 0].sum((1, 2)).detach().cpu()))
        obj.backward()
        opt.step()
        if clamp:
            with torch.no_grad():
                x.clamp_(0, 1)
    model(x)
    final = act['a'][:, 0].sum((1, 2)).detach().cpu()
    hh.remove()
    return x.detach().cpu(), final


print('===== repo run, logged (cnn[6] and cnn[23])')
logs = {}
res = {}
for cid in (6, 23):
    log = []
    xv, final = ascend(images, cid, log=log)
    logs[cid] = log
    res[cid] = (xv, final)
    for it in (0, 1, 2, 5, 9, 19, 49, 99):
        _, s, lo, hi, _ = log[it]
        print(f'cnn[{cid}] step {it + 1}: sum {s:,.0f} x range {lo:.2f}..{hi:.2f}')
    print(f'cnn[{cid}] after 100 steps (one more forward): sum {final.sum():,.0f}; per image', [round(v) for v in final.tolist()])
    print(f'cnn[{cid}] final x range {xv.min():.2f}..{xv.max():.2f}; frac outside [0,1] {((xv < 0) | (xv > 1)).float().mean():.3f}; mean |x - image| {(xv - images).abs().mean():.3f}')
    with torch.no_grad():
        pred = model(xv.cuda()).argmax(1).cpu()
    print(f'cnn[{cid}] model prediction on the visualizations:', [NAMES[p] for p in pred.tolist()])

print('===== Adam step 1: every pixel moves by about lr')
x = images.clone().cuda().requires_grad_()
act = {}
hh = model.cnn[6].register_forward_hook(lambda m, i, o: act.__setitem__('a', o))
opt = Adam([x], lr=0.1)
model(x)
(-act['a'][:, 0].sum()).backward()
g = x.grad.detach().abs()
before = x.detach().clone()
opt.step()
d = (x.detach() - before).abs()
hh.remove()
print(f'cnn[6] step-1 |grad| min {g.min():.3e} median {g.median():.3e} max {g.max():.3e}; zero-grad pixels {(g == 0).float().mean():.4f}')
print(f'step-1 |Δx| min {d.min():.4f} median {d.median():.4f} max {d.max():.4f}; frac within 1% of 0.1: {((d - 0.1).abs() < 1e-3).float().mean():.4f}')

print('===== batch vs single image (BN in eval: images independent?)')
x_single, _ = ascend(images[0:1], 6)
print('cnn[6] image 0 optimized alone vs in batch: max |diff| %.3e' % (x_single[0] - res[6][0][0]).abs().max())

print('===== receptive field (gradient of one centre activation w.r.t. the input)')
for cid in (6, 23):
    x = images[0:1].clone().cuda().requires_grad_()
    act = {}
    hh = model.cnn[cid].register_forward_hook(lambda m, i, o: act.__setitem__('a', o))
    model(x)
    a = act['a']
    c = a.shape[-1] // 2
    a[0, 0, c, c].backward()
    nz = (x.grad[0].abs().sum(0) > 0).nonzero()
    hh.remove()
    print(f'cnn[{cid}] activation ({c},{c}) of {a.shape[-1]}x{a.shape[-1]} depends on input rows {nz[:, 0].min().item()}..{nz[:, 0].max().item()}, cols {nz[:, 1].min().item()}..{nz[:, 1].max().item()} -> {nz[:, 0].max().item() - nz[:, 0].min().item() + 1}x{nz[:, 1].max().item() - nz[:, 1].min().item() + 1}')

print('===== variants: from white noise (slide p.10), clamp to [0,1]')
torch.manual_seed(0)
noise = torch.rand(10, 3, 128, 128)
variants = {}
for cid in (6, 23):
    variants[(cid, 'noise')] = ascend(noise, cid)
    variants[(cid, 'clamp')] = ascend(images, cid, clamp=True)
    variants[(cid, 'noise_clamp')] = ascend(noise, cid, clamp=True)
    for k in ('noise', 'clamp', 'noise_clamp'):
        xv, final = variants[(cid, k)]
        print(f'cnn[{cid}] {k}: final sum {final.sum():,.0f}; x range {xv.min():.2f}..{xv.max():.2f}; mean |x - start| {(xv - (noise if "noise" in k else images)).abs().mean():.3f}')
    print(f'cnn[{cid}] repo: final sum {res[cid][1].sum():,.0f}')
    print(f'cnn[{cid}] noise-start, pairwise mean |x_i - x_j| of the 10 results %.3f (start noise pairwise %.3f)' % (
        torch.stack([(variants[(cid, "noise")][0][i] - variants[(cid, "noise")][0][j]).abs().mean() for i in range(10) for j in range(i + 1, 10)]).mean(),
        torch.stack([(noise[i] - noise[j]).abs().mean() for i in range(10) for j in range(i + 1, 10)]).mean()))
with torch.no_grad():
    s0 = model.cnn[:7](noise.cuda())[:, 0].sum().item()
print('noise start: cnn[6] filter0 sum before optimizing %.0f' % s0)

print('===== lr=1 (the function default)')
for cid in (6, 23):
    xv, final = ascend(images, cid, lr=1)
    print(f'cnn[{cid}] lr=1: final sum {final.sum():,.0f}; x range {xv.min():.2f}..{xv.max():.2f}')

print('===== normalize: joint over RGB per image; how much of [0,1] the original content gets')
for cid in (6, 23):
    xv = res[cid][0]
    for i in (0, 5):
        v = xv[i]
        lo, hi = v.min().item(), v.max().item()
        print(f'cnn[{cid}] img {i}: x range {lo:.2f}..{hi:.2f}; a pixel value 0 maps to {(0 - lo) / (hi - lo):.3f}, value 1 maps to {(1 - lo) / (hi - lo):.3f}; frac of pixels within [0,1] {((v >= 0) & (v <= 1)).float().mean():.3f}')

to_img = lambda t: E.normalize(t.permute(1, 2, 0)).numpy()

# figure 1: activation sum trajectory
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, cid in zip(axs, (6, 23)):
    ax.plot([l[0] + 1 for l in logs[cid]], [l[1] for l in logs[cid]])
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'cnn[{cid}] filter 0: sum of activation (10 images)', fontsize=10)
    ax.set_xlabel('step')
fig.savefig(IMG + 'ch04_trajectory.png', bbox_inches='tight')
plt.close(fig)

# figure 2: x range over steps
fig, ax = plt.subplots(figsize=(6, 4))
for cid in (6, 23):
    ax.plot([l[0] + 1 for l in logs[cid]], [l[3] for l in logs[cid]], label=f'cnn[{cid}] max')
    ax.plot([l[0] + 1 for l in logs[cid]], [l[2] for l in logs[cid]], '--', label=f'cnn[{cid}] min')
ax.axhspan(0, 1, color='gray', alpha=0.2)
ax.set_xlabel('step'); ax.legend(fontsize=8); ax.set_title('pixel value range of x', fontsize=10)
fig.savefig(IMG + 'ch04_xrange.png', bbox_inches='tight')
plt.close(fig)

# figure 3/4: variants per layer
for cid in (6, 23):
    rows = [(images, 'image'), (res[cid][0], 'repo: from image'), (variants[(cid, 'clamp')][0], 'from image, clamp [0,1]'),
            (noise, 'white noise (start)'), (variants[(cid, 'noise')][0], 'from noise'), (variants[(cid, 'noise_clamp')][0], 'from noise, clamp [0,1]')]
    fig, axs = plt.subplots(len(rows), 10, figsize=(15, 9.5))
    for r, (t, title) in enumerate(rows):
        for c in range(10):
            axs[r][c].imshow(to_img(t[c]))
            axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
        axs[r][0].set_ylabel(title, fontsize=8)
    fig.savefig(IMG + f'ch04_variants_cnn{cid}.png', bbox_inches='tight')
    plt.close(fig)

# figure 5: crop of image 0 to see texture
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, (t, title) in zip(axs, [(images[0], 'image 0'), (res[6][0][0], 'cnn[6] repo'), (res[23][0][0], 'cnn[23] repo'), (variants[(6, 'noise')][0][0], 'cnn[6] from noise')]):
    ax.imshow(to_img(t)[32:96, 32:96])
    ax.set_title(title + ' (crop 32:96)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(IMG + 'ch04_crop.png', bbox_inches='tight')
plt.close(fig)
print('figures written')
