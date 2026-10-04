"""Numbers and extra figures for docs/HW09 ch08 (the five CNN methods side by side; exercise answers).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch08_compare.py
Figures are written to ../docs/HW09/img/ch08_*.png. explain_cnn.py is imported, not modified.
"""
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.segmentation import slic
from lime import lime_image

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels
import explain_cnn as E

IMG = '../docs/HW09/img/'
CLASSES = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
model = Classifier().cuda()
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
X = images.cuda()
with torch.no_grad():
    LOGITS = model(X)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# ---------- the five methods, exactly as explain_cnn.py computes them ----------
t0 = sync()


def predict(input):
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    return model(input.cuda()).detach().cpu().numpy()


def segmentation(input):
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)


np.random.seed(16)
lime_img, lime_mask, lime_w = [], [], []   # repo picture, repo mask (+1/-1/0), per-pixel segment weight
for image, label in zip(images.permute(0, 2, 3, 1).numpy(), labels):
    ex = lime_image.LimeImageExplainer().explain_instance(image=image.astype(np.double), classifier_fn=predict,
                                                          segmentation_fn=segmentation)
    im, m = ex.get_image_and_mask(label=label.item(), positive_only=False, hide_rest=False, num_features=11, min_weight=0.05)
    w = np.zeros(ex.segments.shape)
    for seg, val in ex.local_exp[label.item()]:
        w[ex.segments == seg] = val
    lime_img.append(im); lime_mask.append(m); lime_w.append(w)
    if len(lime_w) == 1:
        print('LIME img 0 top 3 (segment, weight):', [(int(s), round(float(v), 4)) for s, v in ex.local_exp[label.item()][:3]])
t1 = sync()
sal = E.compute_saliency_maps(images, labels, model).numpy()                       # (10,128,128), 0..1
t2 = sync()
torch.manual_seed(0)
smooth = np.stack([E.smooth_grad(i, l, model, 500, 0.4)[0] for i, l in zip(images, labels)])   # (10,3,128,128)
t3 = sync()
acts = {}
for cnnid in (6, 23):
    h = model.cnn[cnnid].register_forward_hook(lambda m, i, o: acts.__setitem__(cnnid, o.detach()))
    with torch.no_grad():
        model(X)
    h.remove()
    acts[cnnid] = acts[cnnid][:, 0].cpu().numpy()                                   # filter 0 only
t4 = sync()
IG = E.IntegratedGradients(model)
ig = np.stack([IG.generate_integrated_gradients(X[i:i + 1].clone(), labels[i], 10) for i in range(10)])   # (10,3,128,128)
t5 = sync()
igx = ig * images.numpy()                                                            # IG x (x - 0): the paper's version
print(f'times (s, this process): LIME {t1 - t0:.1f}, saliency {t2 - t1:.2f}, SmoothGrad {t3 - t2:.1f}, filter activations (both layers, no optimisation) {t4 - t3:.2f}, IG {t5 - t4:.2f}')

# ---------- one 128x128 map per method, for comparing ----------
def up(a):
    return F.interpolate(torch.tensor(a)[None, None].float(), size=(128, 128), mode='nearest')[0, 0].numpy()


MAPS = {
    'LIME |w|': [np.abs(w) for w in lime_w],
    'Saliency': list(sal),
    'SmoothGrad': [s.max(0) for s in smooth],
    'cnn6 act': [np.abs(a) for a in acts[6]],
    'cnn23 act': [up(np.abs(a)) for a in acts[23]],
    'IG (repo)': [np.abs(g).max(0) for g in ig],
    'IG x x': [np.abs(g).max(0) for g in igx],
}
names = list(MAPS)
print('map shapes: lime', lime_w[0].shape, 'sal', sal.shape, 'smooth', smooth.shape, 'act6', acts[6].shape, 'act23', acts[23].shape, 'ig', ig.shape)


def top_mask(a, q=0.9):
    return a >= np.quantile(a, q)


print('===== Spearman rank correlation between methods, mean over 10 images (min..max)')
R = np.zeros((len(names), len(names)))
J = np.zeros_like(R)
for a in range(len(names)):
    for b in range(len(names)):
        rs = [spearmanr(MAPS[names[a]][i].ravel(), MAPS[names[b]][i].ravel())[0] for i in range(10)]
        js = []
        for i in range(10):
            ma, mb = top_mask(MAPS[names[a]][i]), top_mask(MAPS[names[b]][i])
            js.append((ma & mb).sum() / (ma | mb).sum())
        R[a, b], J[a, b] = np.mean(rs), np.mean(js)
        if a < b:
            print(f'  {names[a]:>10} vs {names[b]:<10}: rho {np.mean(rs):+.3f} ({min(rs):+.3f}..{max(rs):+.3f}) | top-10% IoU {np.mean(js):.3f} ({min(js):.3f}..{max(js):.3f})')
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print('rho matrix (order', names, ')\n', R)
print('top-10% IoU matrix\n', J)
print('chance IoU of two random top-10% masks ~', round(0.1 * 0.1 / (0.1 + 0.1 - 0.01), 3))

print('===== per image: Spearman Saliency vs SmoothGrad, Saliency vs IG(repo), SmoothGrad vs IG(repo), LIME vs IG x x')
for i in range(10):
    r = lambda a, b: spearmanr(MAPS[a][i].ravel(), MAPS[b][i].ravel())[0]
    print(f'  img {i} ({CLASSES[labels[i]]}): {r("Saliency", "SmoothGrad"):+.3f} {r("Saliency", "IG (repo)"):+.3f} {r("SmoothGrad", "IG (repo)"):+.3f} {r("LIME |w|", "IG x x"):+.3f}')

print('===== how concentrated each map is: share of the total held by the top 1% / top 10% pixels (mean over 10 images)')
for n in names:
    s1 = np.mean([np.sort(m.ravel())[::-1][:164].sum() / m.sum() for m in MAPS[n]])
    s10 = np.mean([np.sort(m.ravel())[::-1][:1638].sum() / m.sum() for m in MAPS[n]])
    print(f'  {n:>10}: top1% {s1:.3f}  top10% {s10:.3f}')

print('===== where the mass sits: share of the map inside the central 64x64 square (32:96), mean over 10 (uniform = 0.25)')
for n in names:
    print(f'  {n:>10}: {np.mean([m[32:96, 32:96].sum() / m.sum() for m in MAPS[n]]):.3f}')

print('===== LIME: positive / negative segments shown by the repo mask (num_features=11, min_weight=0.05)')
for i in range(10):
    m = lime_mask[i]
    print(f'  img {i}: positive pixels {(m == 1).mean():.3f}, negative {(m == -1).mean():.3f} of the image; segments shown +{len(np.unique(lime_w[i][m == 1]))}/-{len(np.unique(lime_w[i][m == -1]))}')

print('===== what each method uses: needs the label? needs gradients? output shape')
print('  LIME: label yes (which column of logits), gradients no, one weight per superpixel (', [len(np.unique(slic(images[i].permute(1, 2, 0).numpy().astype(np.double), n_segments=200, compactness=1, sigma=1, start_label=1))) for i in range(10)], 'segments)')
print('  Saliency/SmoothGrad: label yes (CE target); IG: label yes (one-hot on the logit); filter activation: label no')

# ---------- sanity checks (exercise answers) ----------
def sal_for(m, target):
    x = X.clone().requires_grad_()
    torch.nn.CrossEntropyLoss()(m(x), target.cuda()).backward()
    return x.grad.abs().amax(1).cpu().numpy()


def ig_for(m, i, c, steps=10):
    acc = torch.zeros_like(X[i:i + 1])
    for k in range(steps):
        x = (X[i:i + 1] * k / steps).clone().requires_grad_()
        m(x)[0, c].backward()
        acc += x.grad / steps
    return acc[0].abs().amax(0).cpu().numpy()


second = LOGITS.clone()
second[range(10), labels.cuda()] = -1e9
second = second.argmax(1).cpu()
print('===== class check: the same method, target = second-ranked class instead of the label')
print('  second class per image:', [CLASSES[c] for c in second])
s_lab, s_2nd = sal_for(model, labels), sal_for(model, second)
for i in range(10):
    ig2 = ig_for(model, i, int(second[i]))
    print(f'  img {i}: saliency rho(label, 2nd) {spearmanr(s_lab[i].ravel(), s_2nd[i].ravel())[0]:+.3f} | IG(repo) rho(label, 2nd) {spearmanr(MAPS["IG (repo)"][i].ravel(), ig2.ravel())[0]:+.3f}')
# why saliency barely changes: with CE, the gradient for target y is sum_j (p_j - 1[j=y]) dz_j/dx
p = F.softmax(LOGITS.double(), 1)   # float64: in float32 1 - p(label) rounds to 0 for some images


def raw_grad(target):
    x = X.clone().requires_grad_()
    torch.nn.CrossEntropyLoss(reduction='sum')(model(x), target.cuda()).backward()
    return x.grad.flatten(1)


g_lab, g_2nd = raw_grad(labels), raw_grad(second)
cos = F.cosine_similarity(g_lab, g_2nd, dim=1)
for i in range(10):
    rest = p[i][torch.arange(11, device=p.device) != labels[i].item()].sum()   # sum of the other 10 probabilities (1 - p underflows for img 6 even in float64)
    rest32 = 1 - F.softmax(LOGITS, 1)[i, labels[i]]
    gap = (LOGITS[i, labels[i]] - LOGITS[i, second[i]]).item()
    print(f'  img {i}: logit gap label - 2nd {gap:.2f}, 1 - p(label) {rest.item():.2e} (float32: {rest32.item():.2e}), max |raw grad| {g_lab[i].abs().max().item():.2e}, p(2nd) / (1 - p(label)) {(p[i, second[i]] / rest).item():.4f} | cosine(raw CE grad for label, for 2nd) {cos[i].item():+.4f}')

print('===== model check: same code on a randomly initialised Classifier (torch.manual_seed(0), eval mode)')
torch.manual_seed(0)
rnd = Classifier().cuda().eval()
with torch.no_grad():
    rl = rnd(X)
print('  random model predictions', rl.argmax(1).tolist(), '| max softmax', [f'{v:.3f}' for v in F.softmax(rl, 1).max(1).values.tolist()])
s_rnd = sal_for(rnd, labels)
rows = []
for i in range(10):
    ig_r = ig_for(rnd, i, labels[i].item())
    rows.append((spearmanr(s_lab[i].ravel(), s_rnd[i].ravel())[0], spearmanr(MAPS['IG (repo)'][i].ravel(), ig_r.ravel())[0],
                 spearmanr(np.abs(images[i].numpy()).max(0).ravel(), s_rnd[i].ravel())[0]))
    print(f'  img {i}: rho(trained, random) saliency {rows[-1][0]:+.3f}, IG(repo) {rows[-1][1]:+.3f} | rho(random-model saliency, image brightness max over RGB) {rows[-1][2]:+.3f}')
print('  mean', np.mean(rows, 0).round(3))
ig_rnd0 = ig_for(rnd, 0, labels[0].item())

print('===== edge check: plain image gradient magnitude (Sobel on grayscale) vs each map, Spearman mean over 10')
gray = images.mean(1, keepdim=True)
kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)[None, None]
edge = torch.sqrt(F.conv2d(gray, kx, padding=1) ** 2 + F.conv2d(gray, kx.transpose(2, 3), padding=1) ** 2)[:, 0].numpy()
for n in names:
    print(f'  {n:>10}: {np.mean([spearmanr(edge[i].ravel(), MAPS[n][i].ravel())[0] for i in range(10)]):+.3f}')
print(f'  random-model saliency: {np.mean([spearmanr(edge[i].ravel(), s_rnd[i].ravel())[0] for i in range(10)]):+.3f}')

# ---------- figures ----------
plt.rcParams['font.size'] = 9
COLS = ['image', 'LIME', 'Saliency', 'SmoothGrad', 'cnn[6] filter 0', 'cnn[23] filter 0', 'IG (repo)', 'IG x input']


def show_row(axs, i):
    axs[0].imshow(images[i].permute(1, 2, 0))
    axs[1].imshow(lime_img[i])
    axs[2].imshow(sal[i], cmap=plt.cm.hot)
    axs[3].imshow(np.transpose(smooth[i], (1, 2, 0)))
    axs[4].imshow(E.normalize(torch.tensor(acts[6][i])))
    axs[5].imshow(E.normalize(torch.tensor(acts[23][i])))
    axs[6].imshow(np.moveaxis(E.normalize(ig[i]), 0, -1))
    axs[7].imshow(np.moveaxis(E.normalize(igx[i]), 0, -1))


for part, idx in (('a', range(0, 5)), ('b', range(5, 10))):
    fig, axs = plt.subplots(len(idx), len(COLS), figsize=(16, 2.2 * len(idx)))
    for r, i in enumerate(idx):
        show_row(axs[r], i)
        axs[r][0].set_ylabel(f'{i} {CLASSES[labels[i]]}')
    for c, t in enumerate(COLS):
        axs[0][c].set_title(t)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(IMG + f'ch08_side_by_side_{part}.png', bbox_inches='tight')
    plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(14, 3))
for a, (t, m) in zip(axs, [('image 0', images[0].permute(1, 2, 0)), ('saliency, trained', s_lab[0]), ('saliency, random weights', s_rnd[0]),
                           ('IG (repo), trained', MAPS['IG (repo)'][0]), ('IG (repo), random weights', ig_rnd0)]):
    a.imshow(m if t == 'image 0' else E.normalize(torch.tensor(m)), cmap=None if t == 'image 0' else plt.cm.hot)
    a.set_title(t); a.set_xticks([]); a.set_yticks([])
fig.savefig(IMG + 'ch08_random_model.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(R, vmin=-0.2, vmax=1, cmap='viridis')
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha='right')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for a in range(len(names)):
    for b in range(len(names)):
        ax.text(b, a, f'{R[a, b]:.2f}', ha='center', va='center', color='white' if R[a, b] < 0.6 else 'black', fontsize=8)
ax.set_title('Spearman correlation between maps (mean of 10 images)')
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(IMG + 'ch08_agreement.png', bbox_inches='tight')
plt.close(fig)
print('figures written')
