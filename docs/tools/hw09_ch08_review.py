"""Review measurements for docs/HW09 ch08: the random-weights sanity check for LIME, SmoothGrad and filter activation.

Run from HW09/ (needs the GPU, checkpoint.pth and food/):
    ../.venv/bin/python ../docs/tools/hw09_ch08_review.py
Same settings as hw09_ch08_compare.py (random model = torch.manual_seed(0) then Classifier(), eval mode).
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.segmentation import slic
from lime import lime_image

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels
import explain_cnn as E

trained = Classifier().cuda()
trained.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
trained.eval()
torch.manual_seed(0)
rnd = Classifier().cuda().eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
X = images.cuda()


def segmentation(input):
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)


def lime_maps(model):
    def predict(input):
        input = torch.FloatTensor(input).permute(0, 3, 1, 2)
        return model(input.cuda()).detach().cpu().numpy()
    np.random.seed(16)
    maps, info, segw = [], [], []
    for image, label in zip(images.permute(0, 2, 3, 1).numpy(), labels):
        # explain_instance defaults to top_labels=5: only the model's 5 highest classes get a local_exp entry.
        # Asking for the label explicitly gives the same samples (same random draws) and works for any model.
        ex = lime_image.LimeImageExplainer().explain_instance(image=image.astype(np.double), classifier_fn=predict, segmentation_fn=segmentation,
                                                              labels=(label.item(),), top_labels=None)
        with torch.no_grad():
            top5 = model(torch.tensor(image).permute(2, 0, 1)[None].cuda())[0].topk(5).indices.tolist()
        info_top5 = (label.item() in top5, top5)
        w = np.zeros(ex.segments.shape)
        for seg, val in ex.local_exp[label.item()]:
            w[ex.segments == seg] = val
        ws = np.array([v for _, v in ex.local_exp[label.item()]])
        info.append((ex.score, np.abs(ws).max(), int((np.abs(ws) >= 0.05).sum()), [int(s) for s, _ in ex.local_exp[label.item()][:5]], info_top5))
        maps.append(np.abs(w))
        segw.append(dict(ex.local_exp[label.item()]))
    return maps, info, segw


def smooth_maps(model):
    torch.manual_seed(0)
    return [E.smooth_grad(i, l, model, 500, 0.4)[0].max(0) for i, l in zip(images, labels)]


def act_maps(model, cnnid):
    out = {}
    h = model.cnn[cnnid].register_forward_hook(lambda m, i, o: out.__setitem__('a', o.detach()))
    with torch.no_grad():
        model(X)
    h.remove()
    return list(np.abs(out['a'][:, 0].cpu().numpy()))


gray = images.mean(1, keepdim=True)
kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)[None, None]
edge = torch.sqrt(F.conv2d(gray, kx, padding=1) ** 2 + F.conv2d(gray, kx.transpose(2, 3), padding=1) ** 2)[:, 0].numpy()
edge32 = F.interpolate(torch.tensor(edge)[:, None], size=(32, 32), mode='area')[:, 0].numpy()


def report(name, a, b, e):
    rs = [spearmanr(a[i].ravel(), b[i].ravel())[0] for i in range(10)]
    es = [spearmanr(e[i].ravel(), b[i].ravel())[0] for i in range(10)]
    print(f'{name}: rho(trained, random) per image {" ".join(f"{r:+.3f}" for r in rs)} | mean {np.mean(rs):+.3f}')
    print(f'{" " * len(name)}  rho(Sobel edge, random-model map) mean {np.mean(es):+.3f}')


lt, it, wt = lime_maps(trained)
lr, ir, wr = lime_maps(rnd)
report('LIME |w|', lt, lr, edge)
for i in range(10):
    print(f'  img {i}: trained R2 {it[i][0]:.3f} max|w| {it[i][1]:.4f} n|w|>=0.05 {it[i][2]} top5 {it[i][3]} | random R2 {ir[i][0]:.3f} max|w| {ir[i][1]:.2e} n|w|>=0.05 {ir[i][2]} top5 {ir[i][3]} | label in random model top-5 classes {ir[i][4]}')
seg_rho = [spearmanr([abs(wt[i][k]) for k in sorted(wt[i])], [abs(wr[i][k]) for k in sorted(wt[i])])[0] for i in range(10)]
print('LIME per segment (one value per superpixel, not per pixel): rho(trained |w|, random |w|)', ' '.join(f'{r:+.3f}' for r in seg_rho), '| mean', f'{np.mean(seg_rho):+.3f}')

segs = [segmentation(images[i].permute(1, 2, 0).numpy().astype(np.double)) for i in range(10)]
for tag, W in (('trained', wt), ('random', wr)):
    r = [spearmanr([(segs[i] == k).sum() for k in sorted(W[i])], [abs(W[i][k]) for k in sorted(W[i])])[0] for i in range(10)]
    print(f'LIME {tag}: rho(superpixel area, |w|) per image', ' '.join(f'{v:+.3f}' for v in r), '| mean', f'{np.mean(r):+.3f}')


def sal_maps(model):
    x = X.clone().requires_grad_()
    torch.nn.CrossEntropyLoss()(model(x), labels.cuda()).backward()
    return list(x.grad.abs().amax(1).cpu().numpy())


def ig_maps(model):
    out = []
    for i in range(10):
        acc = torch.zeros_like(X[i:i + 1])
        for k in range(10):
            x = (X[i:i + 1] * k / 10).clone().requires_grad_()
            model(x)[0, labels[i]].backward()
            acc += x.grad / 10
        out.append(acc[0].abs().amax(0).cpu().numpy())
    return out


st, sr = smooth_maps(trained), smooth_maps(rnd)
report('SmoothGrad', st, sr, edge)
torch.manual_seed(0)
with torch.no_grad():
    for name, m in (('trained', trained), ('random', rnd)):
        xm = X + torch.randn_like(X) * (0.4 / (X.max() - X.min())) ** 2
        p = F.softmax(m(xm), 1)
        print(f'  one noisy copy per image, {name} model: argmax {p.argmax(1).tolist()}, max prob {[round(v, 3) for v in p.max(1).values.tolist()]}')
for cnnid, e in ((6, edge), (23, edge32)):
    report(f'cnn{cnnid} act', act_maps(trained, cnnid), act_maps(rnd, cnnid), e)

print('===== same comparison with an 8-pixel border cropped off (pixels 8..119), mean over 10')
pairs = {'Saliency': (sal_maps(trained), sal_maps(rnd)), 'IG (repo)': (ig_maps(trained), ig_maps(rnd)), 'SmoothGrad': (st, sr), 'LIME |w|': (lt, lr)}
for name, (a, b) in pairs.items():
    full = np.mean([spearmanr(a[i].ravel(), b[i].ravel())[0] for i in range(10)])
    crop = np.mean([spearmanr(a[i][8:120, 8:120].ravel(), b[i][8:120, 8:120].ravel())[0] for i in range(10)])
    border = np.mean([1 - b[i][8:120, 8:120].sum() / b[i].sum() for i in range(10)])
    border_t = np.mean([1 - a[i][8:120, 8:120].sum() / a[i].sum() for i in range(10)])
    print(f'  {name:>10}: full {full:+.3f} | cropped {crop:+.3f} | share of the map in the 8-px border (area 23.4%): trained {border_t:.3f}, random {border:.3f}')
