"""Measures the facts behind docs/HW02/FACTS.md (everything except training runs).

Run from HW02/:
    cd HW02 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw02_facts.py [env data split concat memory shapes model ckpt]
With no arguments every section runs. Read-only: nothing in HW02/ is written.
`ckpt` evaluates ./model.ckpt (or the path in HW02_CKPT) on the validation set.
"""
import collections, os, sys, importlib.metadata as md
import numpy as np, torch
from utils import concat_feat, shift, preprocess_data
from config import *

secs = sys.argv[1:] or ['env', 'data', 'split', 'concat', 'memory', 'shapes', 'model', 'ckpt']
P = './libriphone'
split_ids = [l.strip('\n') for l in open(f'{P}/train_split.txt').readlines()]
test_ids = [l.strip('\n') for l in open(f'{P}/test_split.txt').readlines()]
labels = {}
for line in open(f'{P}/train_labels.txt').readlines():
    line = line.strip('\n').split(' ')
    labels[line[0]] = [int(p) for p in line[1:]]


def train_val_ids(ratio=train_ratio):
    import random
    u = open(f'{P}/train_split.txt').readlines()
    random.seed(1337); random.shuffle(u)
    k = int(len(u) * ratio)
    return [x.strip('\n') for x in u[:k]], [x.strip('\n') for x in u[k:]]


def hdr(s): print(f'\n===== {s} =====')


if 'env' in secs:
    hdr('env')
    print('python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', np.__version__,
          '| tqdm', md.version('tqdm'), '| cudnn', torch.backends.cudnn.version())
    print('gpu', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')

if 'data' in secs:
    hdr('data')
    for f in ['train_split.txt', 'train_labels.txt', 'test_split.txt']:
        print(f, os.path.getsize(f'{P}/{f}'), 'bytes')
    print('train .pt files', len(os.listdir(f'{P}/feat/train')), '| test .pt files', len(os.listdir(f'{P}/feat/test')))
    print('train_split lines', len(split_ids), '| labels lines', len(labels), '| test_split lines', len(test_ids))
    print('first 3 train ids', split_ids[:3], '| first 3 test ids', test_ids[:3])
    lens, means, stds = [], [], []
    for u in split_ids:
        f = torch.load(f'{P}/feat/train/{u}.pt')
        assert len(f) == len(labels[u])
        lens.append(len(f)); means.append(f.mean(0)); stds.append(f.std(0))
    lens = np.array(lens)
    tl = np.array([len(torch.load(f'{P}/feat/test/{u}.pt')) for u in test_ids])
    print('train frames', lens.sum(), '| T min/p10/median/mean/p90/max', lens.min(), int(np.percentile(lens, 10)),
          int(np.median(lens)), round(lens.mean(), 1), int(np.percentile(lens, 90)), lens.max())
    print('test frames', tl.sum(), '| T min/median/max', tl.min(), int(np.median(tl)), tl.max())
    print('every utterance: |mean| max over dims', float(torch.stack(means).abs().max()),
          '| std range', float(torch.stack(stds).min()), float(torch.stack(stds).max()))
    f = torch.load(f'{P}/feat/train/{split_ids[0]}.pt')
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)
    print(split_ids[0], 'dtype', f.dtype, 'shape', tuple(f.shape), 'file bytes', os.path.getsize(f'{P}/feat/train/{split_ids[0]}.pt'))
    print('first frame, first 8 dims', f[0, :8])
    print('labels of', split_ids[0], 'first 40:', labels[split_ids[0]][:40])
    c = collections.Counter(x for v in labels.values() for x in v); tot = sum(c.values())
    print('classes', len(c), 'ids', min(c), '..', max(c))
    print('class counts (id: count, share):')
    for k, v in sorted(c.items(), key=lambda t: -t[1]):
        print(f'  {k:2d}: {v:7d} {v / tot:.4f}')
    runs = []
    for v in labels.values():
        r = 1
        for x, y in zip(v, v[1:]):
            if x == y: r += 1
            else: runs.append(r); r = 1
        runs.append(r)
    runs = np.array(runs)
    print('segments', len(runs), '| frames per segment mean/median/p90/max', round(runs.mean(), 2), int(np.median(runs)),
          int(np.percentile(runs, 90)), runs.max(), '| share of segments shorter than 11 frames', round((runs < 11).mean(), 4))
    print('utterances starting with class 0', sum(v[0] == 0 for v in labels.values()), '| ending with class 0',
          sum(v[-1] == 0 for v in labels.values()))

if 'split' in secs:
    hdr('split')
    tr, va = train_val_ids()
    L = {u: len(labels[u]) for u in split_ids}
    print('train utts', len(tr), 'frames', sum(L[u] for u in tr), '| val utts', len(va), 'frames', sum(L[u] for u in va))
    print('first 3 val ids', va[:3])
    spk = lambda ids: set(u.split('-')[0] for u in ids)
    print('speakers: train', len(spk(tr)), 'val', len(spk(va)), 'test', len(spk(test_ids)),
          '| val∩train', len(spk(va) & spk(tr)), '| test∩(train∪val)', len(spk(test_ids) & spk(split_ids)))
    vm = collections.Counter(labels[u][i] for u in va for i in range(L[u]))
    print('val majority class', vm.most_common(1), 'share', round(vm.most_common(1)[0][1] / sum(vm.values()), 6))
    tr8, va8 = train_val_ids(0.8)
    print('ratio 0.8 (official sample): train utts', len(tr8), 'frames', sum(L[u] for u in tr8),
          '| val utts', len(va8), 'frames', sum(L[u] for u in va8))

if 'concat' in secs:
    hdr('concat')
    x = torch.arange(1, 9, dtype=torch.float32).view(4, 2)   # T=4 frames, 2 dims
    print('x (T=4, dim=2):'); print(x)
    print('shift(x, 1):'); print(shift(x, 1))
    print('shift(x, -1):'); print(shift(x, -1))
    print('concat_feat(x, 3):'); print(concat_feat(x.clone(), 3))
    print('concat_feat(x, 5):'); print(concat_feat(x.clone(), 5))
    lab = torch.LongTensor(labels[split_ids[0]]).view(-1, 1)
    cl = concat_feat(lab, 11)
    print('labels of', split_ids[0], 'concat 11, rows 44..50:'); print(cl[44:51])
    f = torch.load(f'{P}/feat/train/{split_ids[0]}.pt')
    cf = concat_feat(f, 11)
    print('feature concat shape', tuple(f.shape), '->', tuple(cf.shape),
          '| row 0 block 0..4 == frame 0:', all(torch.equal(cf[0, 39 * k:39 * k + 39], f[0]) for k in range(6)),
          '| row 10 block 10 == frame 15:', torch.equal(cf[10, 390:429], f[15]))

if 'memory' in secs:
    hdr('memory')
    X = torch.empty(3000000, 39 * concat_nframes)
    print('torch.empty(3000000, 429) float32 bytes', X.untyped_storage().nbytes(), '=', round(X.untyped_storage().nbytes() / 2**30, 3), 'GiB')
    Y = torch.empty(3000000, concat_nframes, dtype=torch.long)
    print('label buffer (3000000, 11) int64 bytes', Y.untyped_storage().nbytes(), '=', round(Y.untyped_storage().nbytes() / 2**30, 3), 'GiB')
    s = X[:264570, :]
    print('X[:264570] keeps storage bytes', s.untyped_storage().nbytes(), '| actually needed', s.numel() * 4)
    t = torch.LongTensor(Y[:10])
    print('torch.LongTensor(y) shares memory with y:', t.data_ptr() == Y.data_ptr())
    del X, Y, s, t

if 'shapes' in secs:
    hdr('shapes')
    from torch.utils.data import DataLoader
    from data_loader import LibriDataset
    import math
    print('batches per epoch: train', math.ceil(2379588 / batch_size), '(last batch', 2379588 % batch_size, ')',
          '| val', math.ceil(264570 / batch_size), '(last batch', 264570 % batch_size, ')',
          '| test', math.ceil(646268 / batch_size), '(last batch', 646268 % batch_size, ')')
    Xs = torch.randn(130, 429); ys = torch.randint(0, 41, (130, 11))
    b = next(iter(DataLoader(LibriDataset(Xs, ys), batch_size=batch_size)))
    print('batch features', tuple(b[0].shape), b[0].dtype, '| labels', tuple(b[1].shape), b[1].dtype)
    print('features.view(-1, 11, 39)', tuple(b[0].view(-1, concat_nframes, input_dim_lstm).shape))

if 'model' in secs:
    hdr('model')
    from model import Classifier
    import model_dnn
    m = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
    print(m)
    for n, p in m.named_parameters():
        if n.endswith('l0') or n.endswith('l1') or n.startswith('out'):
            print(' ', n, tuple(p.shape), p.numel())
    print('total params', sum(p.numel() for p in m.parameters()),
          '| layer 0', sum(p.numel() for n, p in m.named_parameters() if n.endswith('_l0')),
          '| each of layers 1-9', sum(p.numel() for n, p in m.named_parameters() if n.endswith('_l1')),
          '| out', sum(p.numel() for n, p in m.named_parameters() if n.startswith('out')))
    x = torch.randn(64, 11, 39)
    lo, (h, c) = m.lstm(x, None)
    print('lstm_out', tuple(lo.shape), '| h_n', tuple(h.shape), '| c_n', tuple(c.shape), '| out', tuple(m(x).shape),
          '| lstm_out[:, -1] == h_n[-1]:', torch.allclose(lo[:, -1], h[-1]))
    d = model_dnn.Classifier(input_dim=39, hidden_layers=1, hidden_dim=256)
    print('official sample DNN (concat 1, hidden 256, 1 hidden layer) params', sum(p.numel() for p in d.parameters()))
    d = model_dnn.Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
    print('model_dnn with this config (429, 1, 512) params', sum(p.numel() for p in d.parameters()))
    # causality: does changing future frames change the middle output?
    m.eval()
    with torch.no_grad():
        x2 = x.clone(); x2[:, 6:] = torch.randn(64, 5, 39)
        print('middle output unchanged when frames 6..10 change:', torch.equal(m(x)[:, 5], m(x2)[:, 5]),
              '| last output changes:', not torch.equal(m(x)[:, 10], m(x2)[:, 10]))

if 'ckpt' in secs:
    hdr('ckpt')
    from model import Classifier
    path = os.environ.get('HW02_CKPT', model_path)
    X, y = preprocess_data(split='val', feat_dir=f'{P}/feat', phone_path=P, concat_nframes=concat_nframes, train_ratio=train_ratio)
    m = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).cuda()
    m.load_state_dict(torch.load(path)); m.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            preds.append(m(X[i:i + 4096].cuda().view(-1, concat_nframes, input_dim_lstm)).argmax(-1).cpu())
    pred = torch.cat(preds)
    yt, pm = y[:, concat_nframes // 2], pred[:, concat_nframes // 2]
    print(path, '| full-set middle-frame acc', int((pm == yt).sum()), '/', len(yt), '=', round(float((pm == yt).float().mean()), 6))
    print('per-position acc', [round(float(v), 4) for v in (pred == y).float().mean(0)])
    print('per-position acc on interior frames only (label rows not touched by edge padding):')
    # frame index within its utterance, to exclude the first/last 5 frames
    _, va = train_val_ids()
    pos_in = torch.cat([torch.arange(len(labels[u])) for u in va]); Ls = torch.cat([torch.full((len(labels[u]),), len(labels[u])) for u in va])
    inner = (pos_in >= 5) & (pos_in < Ls - 5)
    print('  ', int(inner.sum()), 'rows', [round(float(v), 4) for v in (pred[inner] == y[inner]).float().mean(0)])
    cls = collections.Counter(yt.tolist())
    acc_c = {k: float(((pm == yt) & (yt == k)).sum()) / v for k, v in cls.items()}
    print('per-class acc, 5 best:', [(k, round(v, 3), cls[k]) for k, v in sorted(acc_c.items(), key=lambda t: -t[1])[:5]])
    print('per-class acc, 5 worst:', [(k, round(v, 3), cls[k]) for k, v in sorted(acc_c.items(), key=lambda t: t[1])[:5]])
    conf = collections.Counter((int(a), int(b)) for a, b in zip(yt, pm) if a != b)
    print('top confusions (true, pred): count', conf.most_common(8))
    print('predicted class 0 share', round(float((pm == 0).float().mean()), 4), '| true class 0 share', round(float((yt == 0).float().mean()), 4))
