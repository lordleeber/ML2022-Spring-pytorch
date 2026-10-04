"""Fill the TODO(本機實測) markers of docs/HW09/ch04.html (review of PR #14).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch04_review.py
Writes ../docs/HW09/img/ch04_hook_bn_relu.png.
Code changes to explain_cnn.py are simulated by editing its source text and
exec-ing it into a fresh namespace; the file itself is not modified.
"""
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels

IMG = '../docs/HW09/img/'
SRC = open('explain_cnn.py').read()
main_at = SRC.index("if __name__ == '__main__':") if "if __name__ == '__main__':" in SRC else SRC.index('if __name__')
LIB = SRC[:main_at]


def load_module(src):
    ns = {'__name__': 'explain_variant'}
    exec(compile(src, 'explain_cnn.py', 'exec'), ns)
    return ns


def fresh_model():
    m = Classifier().cuda()
    m.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
    m.eval()
    return m


paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
E = load_module(LIB)
normalize = E['normalize']

print('===== TODO 1: hook on cnn[6] (Conv, repo) vs cnn[7] (BN) vs cnn[8] (ReLU)')
model = fresh_model()


def ascend(layer, iteration=100, lr=0.1):
    act = {}
    conv = {}
    h = model.cnn[layer].register_forward_hook(lambda m, i, o: act.__setitem__('a', o))
    h6 = model.cnn[6].register_forward_hook(lambda m, i, o: conv.__setitem__('a', o.detach()))
    x = images.clone().cuda().requires_grad_()
    opt = Adam([x], lr=lr)
    for _ in range(iteration):
        opt.zero_grad()
        model(x)
        (-act['a'][:, 0].sum()).backward()
        opt.step()
    with torch.no_grad():
        model(x)
    h.remove(); h6.remove()
    return x.detach().cpu(), act['a'][:, 0].sum().item(), conv['a'][:, 0].sum().item(), (act['a'][:, 0] == 0).float().mean().item()


vis = {}
for layer in (6, 7, 8):
    xv, own, conv_sum, zero = ascend(layer)
    vis[layer] = xv
    print(f'hook cnn[{layer}] {type(model.cnn[layer]).__name__}: own filter-0 sum after 100 updates {own:,.1f}; cnn[6] conv filter-0 sum {conv_sum:,.0f}; frac==0 of hooked output {zero:.3f}; x range {xv.min():.2f}..{xv.max():.2f}')
for a, b in ((6, 7), (6, 8), (7, 8)):
    na = torch.stack([normalize(v.permute(1, 2, 0)) for v in vis[a]])
    nb = torch.stack([normalize(v.permute(1, 2, 0)) for v in vis[b]])
    print(f'cnn[{a}] vs cnn[{b}] visualizations: raw mean |diff| {(vis[a] - vis[b]).abs().mean():.3f}; normalized mean |diff| {(na - nb).abs().mean():.4f}')
fig, axs = plt.subplots(4, 10, figsize=(15, 6.6))
rows = [(images, 'image'), (vis[6], 'hook cnn[6] Conv (repo)'), (vis[7], 'hook cnn[7] BN'), (vis[8], 'hook cnn[8] ReLU')]
for r, (t, title) in enumerate(rows):
    for c in range(10):
        axs[r][c].imshow(t[c].permute(1, 2, 0).numpy() if r == 0 else normalize(t[c].permute(1, 2, 0)).numpy())
        axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
    axs[r][0].set_ylabel(title, fontsize=8)
fig.savefig(IMG + 'ch04_hook_bn_relu.png', bbox_inches='tight')
plt.close(fig)

print('===== TODO 2: remove `global layer_activations` (line 172)')
src = LIB.replace('        global layer_activations\n', '')
assert src != LIB
E2 = load_module(src)
E2['output_dir'] = '../docs/tools/__scratch_out/'
model = fresh_model()
try:
    E2['filter_explain'](model, images, cnnid=6)
except Exception as e:
    tb = traceback.format_exc().strip().splitlines()
    print('\n'.join(tb[-4:]))

print('===== TODO 3: line 189 -> Adam(model.parameters(), lr=lr)')
src = LIB.replace('optimizer = Adam([x], lr=lr)', 'optimizer = Adam(model.parameters(), lr=lr)')
assert src != LIB
E3 = load_module(src)
ref_model = fresh_model()
model = fresh_model()
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
fa, fv = E3['filter_explanation'](images, model, cnnid=6, filterid=0, iteration=100, lr=0.1)
print('cnn[6] call: visualization == original images exactly:', torch.equal(fv, images), '| max |diff|', (fv - images).abs().max().item())
changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
print('state_dict entries changed after cnn[6] call:', changed)
for k in changed:
    d = (model.state_dict()[k] - before[k]).abs()
    print(f'  {k}: max |change| {d.max():.4f} mean {d.mean():.4f} (weights mean |w| {before[k].abs().mean():.4f})')
fa23, fv23 = E3['filter_explanation'](images, model, cnnid=23, filterid=0, iteration=100, lr=0.1)
changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
print('after cnn[23] call too, changed entries:', changed)
with torch.no_grad():
    lo = model(images.cuda()); lr_ = ref_model(images.cuda())
print('predictions after both calls:', lo.argmax(1).tolist(), '| original:', lr_.argmax(1).tolist(), '| labels:', labels.tolist())
print('p(label) after both calls:', [round(v, 4) for v in lo.softmax(1).gather(1, labels.cuda().view(-1, 1)).squeeze().tolist()])
IG_ref = E['IntegratedGradients'](ref_model)
IG_mod = E3['IntegratedGradients'](model)
for i in range(10):
    img = images[i:i + 1].cuda()
    a = IG_ref.generate_integrated_gradients(img.clone(), labels[i], 10)
    b = IG_mod.generate_integrated_gradients(img.clone(), labels[i], 10)
    na, nb = normalize(a), normalize(b)
    print(f'img {i}: IG max |orig| {np.abs(a).max():.3e} |modified| {np.abs(b).max():.3e}; corr of normalized maps {np.corrcoef(na.ravel(), nb.ravel())[0, 1]:.3f}')

print('===== TODO 4: delete line 203 hook_handle.remove()')
src = LIB.replace('    hook_handle.remove()\n', '')
assert src != LIB
E4 = load_module(src)
model = fresh_model()
E4['filter_explanation'](images, model, cnnid=6, filterid=0, iteration=100, lr=0.1)
print('after cnn[6] call: hooks on cnn[6]', len(model.cnn[6]._forward_hooks), 'cnn[23]', len(model.cnn[23]._forward_hooks))
fa_kept, fv_kept = E4['filter_explanation'](images, model, cnnid=23, filterid=0, iteration=100, lr=0.1)
print('after cnn[23] call: hooks on cnn[6]', len(model.cnn[6]._forward_hooks), 'cnn[23]', len(model.cnn[23]._forward_hooks))
print('global layer_activations shape now', tuple(E4['layer_activations'].shape))
model2 = fresh_model()
fa_ok, fv_ok = E['filter_explanation'](images, model2, cnnid=23, filterid=0, iteration=100, lr=0.1)
fa_ok2, fv_ok2 = E['filter_explanation'](images, model2, cnnid=23, filterid=0, iteration=100, lr=0.1)
nrm = lambda t: torch.stack([normalize(v.permute(1, 2, 0)) for v in t])
print('cnn[23] activation row: kept-hook vs repo max |diff| %.3e' % (fa_kept - fa_ok).abs().max())
print('cnn[23] visualization, kept-hook vs repo: raw mean |diff| %.3f, normalized mean |diff| %.4f' % ((fv_kept - fv_ok).abs().mean(), (nrm(fv_kept) - nrm(fv_ok)).abs().mean()))
print('cnn[23] visualization, repo vs repo (run-to-run): raw mean |diff| %.3f, normalized mean |diff| %.4f' % ((fv_ok2 - fv_ok).abs().mean(), (nrm(fv_ok2) - nrm(fv_ok)).abs().mean()))

print('===== extra: cnn[6] img 5 x range (repo setting, for the ch04 4.8 table)')
model = fresh_model()
_, fv6 = E['filter_explanation'](images, model, cnnid=6, filterid=0, iteration=100, lr=0.1)
for i in (0, 5):
    v = fv6[i]
    lo, hi = v.min().item(), v.max().item()
    print(f'img {i}: range {lo:.2f}..{hi:.2f}; 0 -> {-lo / (hi - lo):.3f}, 1 -> {(1 - lo) / (hi - lo):.3f}')
