"""Numbers and extra figures for docs/HW09 ch02 (LIME).

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch02_lime.py
Figures are written to ../docs/HW09/img/ch02_*.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels

IMG = '../docs/HW09/img/'

model = Classifier().cuda()
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
hwc = images.permute(0, 2, 3, 1).numpy()


def predict(input):
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    with torch.no_grad():
        return model(input.cuda()).cpu().numpy()


def predict_prob(input):
    return torch.tensor(predict(input)).softmax(1).numpy()


def segmentation(input):
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)


def segmentation0(input):
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=0)


def explain(i, fn=predict, seg=segmentation, seed=16):
    np.random.seed(seed)
    return lime_image.LimeImageExplainer().explain_instance(
        image=hwc[i].astype(np.double), classifier_fn=fn, segmentation_fn=seg)


def mask_img(exp, label, **kw):
    args = dict(positive_only=False, hide_rest=False, num_features=11, min_weight=0.05)
    args.update(kw)
    return exp.get_image_and_mask(label=label, **args)


print('===== segments')
segs = [segmentation(hwc[i].astype(np.double)) for i in range(10)]
for i, s in enumerate(segs):
    sizes = np.bincount(s.ravel())[1:]
    print(f'img {i}: n {len(sizes)} ids {s.min()}..{s.max()} pixels/segment min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}')
fig, axs = plt.subplots(1, 10, figsize=(15, 8))
for i, s in enumerate(segs):
    axs[i].imshow(mark_boundaries(hwc[i], s))
    axs[i].set_title(f'{len(np.unique(s))}')
fig.savefig(IMG + 'ch02_segments.png', bbox_inches='tight')
plt.close(fig)
s0 = segmentation0(hwc[0].astype(np.double))
print('img 0 start_label=0: ids', s0.min(), '..', s0.max(), 'same partition as start_label=1:', np.array_equal(s0 + 1, segs[0]))

print('===== perturbation (img 0, replays the sampling of explain_instance)')
x = hwc[0].astype(np.double)
seg = segs[0]
n = len(np.unique(seg))
fudged = x.copy()
for sid in np.unique(seg):
    fudged[seg == sid] = np.mean(x[seg == sid], axis=0)
np.random.seed(16)
rs = np.random.mtrand._rand
_ = rs.randint(0, high=1000)   # explain_instance draws a random_seed first
data = rs.randint(0, 2, 1000 * n).reshape((1000, n))
data[0, :] = 1
print('samples', data.shape, 'features off per sample: min', (data[1:] == 0).sum(1).min(), 'mean %.1f' % (data[1:] == 0).sum(1).mean(), 'max', (data[1:] == 0).sum(1).max())
print('column 0 (no pixels) off in', (data[:, 0] == 0).sum(), 'samples; segment id', n, 'never masked')
imgs = []
for row in data:
    temp = x.copy()
    mask = np.zeros(seg.shape, bool)
    for z in np.where(row == 0)[0]:
        mask[seg == z] = True
    temp[mask] = fudged[mask]
    imgs.append(temp)
imgs = np.array(imgs)
out = np.concatenate([predict(imgs[k:k + 100]) for k in range(0, 1000, 100)])
logit0 = out[:, 0]
prob0 = torch.tensor(out).softmax(1).numpy()[:, 0]
print('label logit over 1000 samples: original %.3f min %.3f median %.3f max %.3f' % (logit0[0], logit0.min(), np.median(logit0), logit0.max()))
print('p(label) over 1000 samples: min %.4f median %.4f; fraction >= 0.99: %.3f; fraction argmax==0: %.3f' % (prob0.min(), np.median(prob0), (prob0 >= 0.99).mean(), (out.argmax(1) == 0).mean()))
d = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()
w = np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
print('cosine distance to original: min %.3f median %.3f max %.3f; kernel weight min %.3f median %.3f max %.3f' % (d[1:].min(), np.median(d[1:]), d[1:].max(), w[1:].min(), np.median(w[1:]), w[1:].max()))
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
show = [(x, 'original'), (fudged, 'all segments = mean colour')] + [(imgs[k], f'sample {k}: logit {logit0[k]:.1f}') for k in (1, 2, 3)]
for ax, (im, t) in zip(axs, show):
    ax.imshow(np.clip(im, 0, 1))
    ax.set_title(t, fontsize=9)
fig.savefig(IMG + 'ch02_perturb.png', bbox_inches='tight')
plt.close(fig)
print('samples 1-3 features off:', [(data[k] == 0).sum() for k in (1, 2, 3)], 'label logit', [round(float(logit0[k]), 3) for k in (1, 2, 3)])

print('===== explain img 0 (logits)')
seen = []
def predict_capture(input):
    seen.append(input.copy())
    return predict(input)
e = explain(0, fn=predict_capture)
replayed = np.concatenate(seen)
print('replay check: samples passed to predict match the replay:', replayed.shape, bool(np.allclose(replayed, imgs)))
print('top_labels', [int(t) for t in e.top_labels], 'intercept %.3f score %.4f local_pred %.3f' % (e.intercept[0], e.score, float(np.ravel(e.local_pred)[0])))
print('top5', [(int(a), round(float(b), 4)) for a, b in e.local_exp[0][:5]])
print('n weights', len(e.local_exp[0]), 'sum of weights %.3f' % sum(b for _, b in e.local_exp[0]))
img_logit, m_logit = mask_img(e, 0)
for kw in [dict(), dict(positive_only=True), dict(num_features=5), dict(min_weight=0.0), dict(num_features=200)]:
    _, m = mask_img(e, 0, **kw)
    print('get_image_and_mask', kw or 'default', 'green segs', len(np.unique(segs[0][m == 1])), 'red segs', len(np.unique(segs[0][m == -1])), 'pixels green', int((m == 1).sum()), 'red', int((m == -1).sum()))
print('colour: max(image) %.4f; green pixels G channel all = max:' % hwc[0].max(), bool(np.all(img_logit[m_logit == 1][:, 1] == hwc[0].max())))

print('===== explain img 0 (softmax) and start_label=0')
ep = explain(0, fn=predict_prob)
print('softmax: intercept %.4f score %.4f top5 %s; |w|>=0.05 %d' % (ep.intercept[0], ep.score, [(int(a), round(float(b), 4)) for a, b in ep.local_exp[0][:5]], sum(abs(b) >= 0.05 for _, b in ep.local_exp[0])))
img_prob, m_prob = mask_img(ep, 0)
e0 = explain(0, seg=segmentation0)
img_s0, _ = mask_img(e0, 0)
print('start_label=0: score %.4f top5 %s' % (e0.score, [(int(a), round(float(b), 4)) for a, b in e0.local_exp[0][:5]]))
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, (im, t) in zip(axs, [(img_logit, 'logits (repo)'), (img_prob, 'softmax probability'), (img_s0, 'logits, start_label=0')]):
    ax.imshow(im)
    ax.set_title(t, fontsize=10)
fig.savefig(IMG + 'ch02_img0_compare.png', bbox_inches='tight')
plt.close(fig)

print('===== softmax version for all 10 images')
np.random.seed(16)
fig, axs = plt.subplots(1, 10, figsize=(15, 8))
for i in range(10):
    ex = lime_image.LimeImageExplainer().explain_instance(image=hwc[i].astype(np.double), classifier_fn=predict_prob, segmentation_fn=segmentation)
    im, m = mask_img(ex, labels[i].item())
    big = [(a, b) for a, b in ex.local_exp[labels[i].item()] if abs(b) >= 0.05]
    print(f'img {i}: score {ex.score:.3f} |w|>=0.05 {len(big)} (pos {sum(b > 0 for _, b in big)}, neg {sum(b < 0 for _, b in big)}) green segs {len(np.unique(segs[i][m == 1]))} red segs {len(np.unique(segs[i][m == -1]))}')
    axs[i].imshow(im)
fig.savefig(IMG + 'ch02_lime_softmax.png', bbox_inches='tight')
plt.close(fig)

print('===== seed dependence')
alone = explain(3)
np.random.seed(16)
loop = None
for i in range(4):
    ex = lime_image.LimeImageExplainer().explain_instance(image=hwc[i].astype(np.double), classifier_fn=predict, segmentation_fn=segmentation)
    if i == 3:
        loop = ex
top = lambda ex, k: [int(a) for a, _ in ex.local_exp[labels[3].item()][:k]]
print('img 3 alone (seed 16) top5', top(alone, 5), '| in loop (4th image) top5', top(loop, 5), '| overlap of top 11:', len(set(top(alone, 11)) & set(top(loop, 11))))
other = explain(0, seed=0)
print('img 0 seed 0 top5', [(int(a), round(float(b), 3)) for a, b in other.local_exp[0][:5]], 'score %.4f' % other.score, '| overlap of top 11 with seed 16:', len(set(int(a) for a, _ in other.local_exp[0][:11]) & set(int(a) for a, _ in e.local_exp[0][:11])))
