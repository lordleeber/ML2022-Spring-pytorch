"""Facts for the HW04 textbook (docs/HW04/FACTS.md). Run from HW04/:
    cd HW04 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw04_facts.py <section> [out_dir]
Sections:
  data     metadata/testdata/mapping statistics (json only, no .pt reads)
  mel      reads every training/test .pt: shapes, value range, padding value, clamp floor
  split    train.py's random_split under set_seed(87): sizes, speakers per side, short utterances
  model    parameters per module, tensor shapes per layer, warnings for 1 vs 2 layers
  sched    learning rate of get_cosine_schedule_with_warmup at chosen steps; steps/epoch
  workers  what each DataLoader worker's python random is seeded with
  figs     PNG figures for the book into out_dir (mel heatmaps, length histogram)
"""
import json, os, sys, random, warnings, collections
import numpy as np
import torch

D = './Dataset'
sec = sys.argv[1]


def meta():
    return json.load(open(os.path.join(D, 'metadata.json')))


if sec == 'data':
    m = meta()
    sp = m['speakers']
    t = json.load(open(os.path.join(D, 'testdata.json')))
    mp = json.load(open(os.path.join(D, 'mapping.json')))
    L = np.array([u['mel_len'] for s in sp.values() for u in s])
    TL = np.array([u['mel_len'] for u in t['utterances']])
    c = np.array([len(s) for s in sp.values()])
    print('n_mels', m['n_mels'], 'speakers', len(sp), 'utterances', len(L), 'test', len(TL))
    print('file sizes: metadata', os.path.getsize(os.path.join(D, 'metadata.json')),
          'testdata', os.path.getsize(os.path.join(D, 'testdata.json')),
          'mapping', os.path.getsize(os.path.join(D, 'mapping.json')))
    for name, x in [('train', L), ('test', TL)]:
        q = np.percentile(x, [5, 25, 50, 75, 95])
        print(f'{name} mel_len min {x.min()} max {x.max()} mean {x.mean():.1f} median {np.median(x)} '
              f'p5/25/50/75/95 {q.tolist()} sum {x.sum()} hours {x.sum() / 100 / 3600:.2f}')
        print(f'  <=128: {(x <= 128).sum()}  <128: {(x < 128).sum()}  >1000: {(x > 1000).sum()}  >2000: {(x > 2000).sum()}')
        h, e = np.histogram(x, bins=[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 8192])
        print('  hist', list(zip(e[:-1].tolist(), h.tolist())))
    print('short utterances', [(k, u) for k, s in sp.items() for u in s if u['mel_len'] <= 128])
    print('mean fraction seen by a 128-frame crop', float(np.mean(np.minimum(128, L) / L)))
    print('per speaker: min', c.min(), 'max', c.max(), 'mean', c.mean(), 'median', np.median(c),
          'hist', sorted(collections.Counter((c // 5 * 5).tolist()).items()))
    keys = list(sp.keys())
    print('first speakers in metadata order', keys[:5], 'sorted?', keys == sorted(keys))
    s2i = mp['speaker2id']
    print('speaker2id first', list(s2i.items())[:5], 'last', list(s2i.items())[-2:])
    print('speaker2id order == metadata order?', list(s2i.keys()) == keys,
          'ids 0..599?', sorted(s2i.values()) == list(range(600)))
    print('id2speaker inverse ok?', all(mp['id2speaker'][str(v)] == k for k, v in s2i.items()))
    ids = sorted(int(k[2:]) for k in keys)
    print('speaker id range', ids[0], ids[-1])
    print('metadata first entry', keys[0], sp[keys[0]][0])
    print('testdata first entry', t['utterances'][0], 'keys', list(t.keys()), 'n_mels', t['n_mels'])
    paths = [u['feature_path'] for s in sp.values() for u in s] + [u['feature_path'] for u in t['utterances']]
    print('distinct paths', len(set(paths)), 'files on disk', len([f for f in os.listdir(D) if f.startswith('uttr-')]))

elif sec == 'mel':
    m = meta()
    sp = m['speakers']
    mins, maxs, sums, n, bad, dt = [], [], 0.0, 0, 0, collections.Counter()
    for s in sp.values():
        for u in s:
            x = torch.load(os.path.join(D, u['feature_path']))
            dt[(str(x.dtype), x.dim(), x.shape[1])] += 1
            bad += int(x.shape[0] != u['mel_len'])
            mins.append(x.min().item()); maxs.append(x.max().item()); sums += x.sum().item(); n += x.numel()
    print('train dtype/dim/n_mels', dict(dt), 'mel_len mismatches', bad)
    print('train value min', min(mins), 'max', max(maxs), 'mean', sums / n,
          'files whose min equals the clamp floor', sum(abs(v - np.log(1e-9)) < 1e-4 for v in mins))
    print('log(1e-9) =', float(np.log(1e-9)), ' padding value -20 vs floor')
    t = json.load(open(os.path.join(D, 'testdata.json')))['utterances']
    tm = [torch.load(os.path.join(D, u['feature_path'])) for u in t]
    print('test dtype', collections.Counter(str(x.dtype) for x in tm), 'mismatch',
          sum(x.shape[0] != u['mel_len'] for x, u in zip(tm, t)),
          'min', min(x.min().item() for x in tm), 'max', max(x.max().item() for x in tm))
    x = torch.load(os.path.join(D, sp[next(iter(sp))][0]['feature_path']))
    print('first file', x.shape, x[:2, :6])

elif sec == 'split':
    import train  # set_seed(87)
    from dataset import myDataset
    from torch.utils.data import random_split
    ds = myDataset(D)
    tl = int(0.9 * len(ds))
    tr, va = random_split(ds, [tl, len(ds) - tl])
    print('sizes', len(ds), len(tr), len(va), 'first valid indices', va.indices[:5])
    vs = collections.Counter(ds.data[i][1] for i in va.indices)
    ts = collections.Counter(ds.data[i][1] for i in tr.indices)
    print('speakers in valid', len(vs), 'in train', len(ts),
          'valid per speaker min/max/mean', min(vs.values()), max(vs.values()), len(va) / len(vs))
    m = meta()['speakers']
    lens = {u['feature_path']: u['mel_len'] for s in m.values() for u in s}
    short = [(i, ds.data[i]) for i in range(len(ds)) if lens[ds.data[i][0]] <= 128]
    print('short utterances', [(i, p, lens[p], 'valid' if i in set(va.indices) else 'train') for i, (p, _) in short])
    vl = np.array([lens[ds.data[i][0]] for i in va.indices])
    print('valid mel_len mean', vl.mean(), 'median', np.median(vl), 'min', vl.min())
    print('steps/epoch (drop_last)', len(tr) // 32, 'left out per epoch', len(tr) % 32,
          'valid batches', len(va) // 32, 'valid dropped', len(va) % 32,
          '70000 steps =', 70000 / (len(tr) // 32), 'epochs')
    # the dataset item of utterance 0, twice: random crop differs each call
    random.seed(0)
    a, _ = ds[0]
    b, _ = ds[0]
    print('item 0 shapes', a.shape, b.shape, 'same crop twice?', torch.equal(a, b))

elif sec == 'model':
    from classifier import Classifier
    torch.manual_seed(0)
    for layers in (1, 2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = Classifier(n_spks=600)
            if layers == 2:
                m.encoder = torch.nn.TransformerEncoder(m.encoder_layer, num_layers=2)
            print('layers', layers, 'warnings:', [str(x.message)[:200] for x in w])
    m = Classifier(n_spks=600)
    groups = collections.OrderedDict()
    for n, p in m.named_parameters():
        g = n.split('.')[0] if not n.startswith('encoder_layer') else '.'.join(n.split('.')[:2])
        groups[g] = groups.get(g, 0) + p.numel()
        print(f'  {n:45s} {str(tuple(p.shape)):12s} {p.numel()}')
    print('groups', dict(groups), 'total', sum(groups.values()))
    print(m)
    el = m.encoder_layer
    print('norm_first', el.norm_first, 'batch_first', el.self_attn.batch_first, 'activation', el.activation,
          'dropout p', el.dropout.p, 'head dim', el.self_attn.head_dim, 'num_heads', el.self_attn.num_heads)
    shapes = []
    hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: shapes.append((n, tuple(o.shape) if torch.is_tensor(o) else [tuple(t.shape) for t in o if torch.is_tensor(t)])))
             for n, mod in m.named_modules() if n in ('prenet', 'encoder_layer', 'pred_layer', 'pred_layer.0', 'pred_layer.2')]
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(32, 128, 40))
    print('shapes', shapes, 'out', tuple(out.shape))
    with torch.no_grad():
        o1 = m(torch.randn(1, 4940, 40))
    print('one long test utterance (1, 4940, 40) ->', tuple(o1.shape))
    # padding: mean pooling includes the -20 frames
    x = torch.randn(1, 91, 40)
    xp = torch.cat([x, torch.full((1, 37, 40), -20.0)], 1)
    with torch.no_grad():
        print('91-frame utterance: argmax alone', m(x).argmax().item(), 'padded to 128', m(xp).argmax().item(),
              'max |logit diff|', (m(x) - m(xp)).abs().max().item())
    print('attention mask used? no src_key_padding_mask argument in classifier.py forward')

elif sec == 'sched':
    import train
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sc = train.get_cosine_schedule_with_warmup(opt, 1000, 70000)
    want = {0, 1, 500, 999, 1000, 1001, 10000, 35500, 50000, 60000, 68000, 69999, 70000}
    lr = []
    for s in range(70001):
        if s in want:
            print('lr at scheduler step', s, opt.param_groups[0]['lr'])
        lr.append(opt.param_groups[0]['lr'])
        opt.step(); sc.step()
    print('lr used by the update of step index 0 (first batch):', lr[0], ' max lr', max(lr), 'at', int(np.argmax(lr)))
    print('lr at the 35 validations (after step 2000k):', [f'{lr[k]:.3e}' for k in range(2000, 70001, 2000)][:5], '...')

elif sec == 'workers':
    from torch.utils.data import DataLoader, Dataset

    class P(Dataset):
        def __len__(self): return 16
        def __getitem__(self, i):
            w = torch.utils.data.get_worker_info()
            return torch.tensor([w.id if w else -1, random.randint(0, 10 ** 9), w.seed % 2 ** 32 if w else -1])

    torch.manual_seed(87); random.seed(87)
    for ep in range(2):
        b = next(iter(DataLoader(P(), batch_size=16, num_workers=4)))
        print('iterator', ep, b.tolist()[:4])
    print('each worker: python random seeded with (base_seed + worker_id) by torch; new iterator -> new base_seed')

elif sec == 'figs':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = meta()['speakers']
    keys = list(m.keys())
    plt.rcParams.update({'figure.facecolor': '#161c24', 'axes.facecolor': '#161c24', 'savefig.facecolor': '#161c24',
                         'text.color': '#dde5ec', 'axes.labelcolor': '#dde5ec', 'xtick.color': '#93a1af',
                         'ytick.color': '#93a1af', 'axes.edgecolor': '#2b3540', 'font.size': 10})
    # two utterances of one speaker, one of another, on the same colour scale
    picks = [(keys[0], 0), (keys[0], 1), (keys[1], 0)]
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.2), constrained_layout=True)
    for ax, (k, j) in zip(axs, picks):
        u = m[k][j]
        x = torch.load(os.path.join(D, u['feature_path'])).numpy()
        im = ax.imshow(x.T, origin='lower', aspect='auto', cmap='magma', vmin=-12, vmax=6, interpolation='nearest')
        ax.set_title(f"{k}  {u['feature_path'][:13]}…  {x.shape[0]} frames ({x.shape[0] / 100:.2f} s)", fontsize=9, loc='left')
        ax.set_ylabel('mel bin')
        ax.axvspan(0, 128, color='#58a6ff', alpha=0.0)
        ax.add_patch(plt.Rectangle((0, -0.5), 128, 40, fill=False, ec='#58a6ff', lw=1.2, ls='--'))
    axs[-1].set_xlabel('frame (10 ms each); dashed box = 128 frames')
    fig.colorbar(im, ax=axs, shrink=0.8, label='log mel energy')
    fig.savefig(os.path.join(out, 'mel3.png'), dpi=110)
    print('wrote mel3.png', picks)
