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
  ch04     training-loop facts: AdamW defaults, the first update (lr 0), state_dict() is a reference,
           and full-length / fixed-crop accuracy of every validation snapshot (needs <snap_dir>)
  ch03     encoder internals: hand-written forward == module, attention shapes, no positional
           encoding (frame order), padding, unused dropout argument (needs <ckpt>)
  ch02     DataLoader facts: a real batch, padding, worker seeds, loader throughput,
           and how much the validation accuracy moves with the random crops (needs <ckpt>)
  ch05     test.py facts: torch.stack with 2 utterances, id2speaker keys, eval() vs train(), live vs best
           predictions, accuracy by length, padding with/without a mask, inference time (needs <live> <best>)
  ch06     Conformer block of hw04_exp.py (classes exec'd from its source, CPU only): parameters per part,
           shapes, a same-size pre-norm Transformer, frame-order sensitivity at random init
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

elif sec == 'ch02':
    import time
    import train  # set_seed(87)
    from torch.utils.data import DataLoader, Dataset
    tl, vl, n = train.get_dataloader(D, 32, 8)
    it = iter(tl)
    mels, labels = next(it)
    print('first batch', tuple(mels.shape), mels.dtype, tuple(labels.shape), labels.dtype, labels[:8].tolist(),
          'padded frames (value -20 rows):', int((mels == -20).all(-1).sum()))
    print('train batches', len(tl), 'valid batches', len(vl), 'train sampler', type(tl.sampler).__name__,
          'valid sampler', type(vl.sampler).__name__, 'pin_memory', tl.pin_memory, 'prefetch', tl.prefetch_factor)
    # a batch containing the 91-frame utterance (index 20902 of the full dataset)
    full = tl.dataset.dataset
    items = [full[20902]] + [full[i] for i in (0, 1, 2)]
    m, l = train.collate_batch(items)
    print('collate with the 91-frame utterance:', [tuple(x[0].shape) for x in items], '->', tuple(m.shape),
          'rows of -20 in item 0:', int((m[0] == -20).all(-1).sum()), 'labels', l.tolist())
    # worker seeds
    class W(Dataset):
        def __len__(self): return 8
        def __getitem__(self, i):
            w = torch.utils.data.get_worker_info()
            return torch.tensor([w.id, w.seed, int(random.random() * 1e9)])
    torch.manual_seed(87)
    for k in range(2):
        b = torch.cat([x for x in DataLoader(W(), batch_size=1, num_workers=8)])
        rows = sorted(set(tuple(r) for r in b.tolist()))
        ok = all(int(random.Random(s).random() * 1e9) == v for _, s, v in rows)
        print(f'iterator {k}: (worker id, seed, first python random*1e9):', rows[:3], '... python random == Random(seed)?', ok)
    # throughput of the train loader alone (no model)
    for nw in (0, 8):
        tl2, _, _ = train.get_dataloader(D, 32, nw)
        t0 = time.time(); it2 = iter(tl2)
        for _ in range(500):
            next(it2)
        print(f'num_workers={nw}: 500 batches in {time.time() - t0:.2f} s')
    # validation noise from random crops
    ck = sys.argv[2]
    from classifier import Classifier
    model = Classifier(n_spks=600).cuda()
    model.load_state_dict(torch.load(ck)); model.eval()
    accs = []
    for k in range(10):
        torch.manual_seed(1000 + k)
        a = 0.0
        with torch.no_grad():
            for mm, ll in vl:
                a += (model(mm.cuda()).argmax(1) == ll.cuda()).float().mean().item()
        accs.append(a / len(vl))
    print('valid() accuracy of', os.path.basename(ck), 'under 10 different crop seeds:', [round(x, 4) for x in accs],
          'min', round(min(accs), 4), 'max', round(max(accs), 4), 'mean', round(sum(accs) / 10, 4))

elif sec == 'ch03':
    import math
    import torch.nn.functional as F
    from classifier import Classifier
    ck = sys.argv[2]
    torch.manual_seed(0)
    m = Classifier(n_spks=600)
    m.load_state_dict(torch.load(ck)); m.eval()
    el = m.encoder_layer
    x = torch.randn(128, 4, 80)   # (length, batch, d_model)
    with torch.no_grad():
        ref = el(x)
        # hand-written post-norm layer
        W, b = el.self_attn.in_proj_weight, el.self_attn.in_proj_bias
        q, k, v = (x @ W[i*80:(i+1)*80].T + b[i*80:(i+1)*80] for i in range(3))   # (L, B, 80) each
        def heads(t): return t.reshape(128, 4, 2, 40).permute(1, 2, 0, 3)       # (B, h, L, 40)
        qh, kh, vh = heads(q), heads(k), heads(v)
        att = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(40), dim=-1)   # (B, h, L, L)
        ctx = (att @ vh).permute(2, 0, 1, 3).reshape(128, 4, 80)
        sa = ctx @ el.self_attn.out_proj.weight.T + el.self_attn.out_proj.bias
        h1 = F.layer_norm(x + sa, (80,), el.norm1.weight, el.norm1.bias, el.norm1.eps)
        ff = F.relu(h1 @ el.linear1.weight.T + el.linear1.bias) @ el.linear2.weight.T + el.linear2.bias
        out = F.layer_norm(h1 + ff, (80,), el.norm2.weight, el.norm2.bias, el.norm2.eps)
        print('hand-written == encoder_layer (eval):', torch.allclose(out, ref, atol=1e-5), 'max diff', (out - ref).abs().max().item())
        print('attention', tuple(att.shape), 'row sums', att.sum(-1).min().item(), att.sum(-1).max().item(), 'scale 1/sqrt(40) =', 1 / math.sqrt(40))
        _, w = el.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        print('self_attn need_weights shape', tuple(w.shape), 'equals hand-written', torch.allclose(w, att, atol=1e-6))
        print('eps', el.norm1.eps, 'activation', el.activation.__name__)
        # train mode: dropout changes the output
        el.train(); a1 = el(x); a2 = el(x); el.eval()
        print('train mode twice differ?', not torch.allclose(a1, a2), 'max diff', (a1 - a2).abs().max().item())
    # unused dropout argument
    print('Classifier(dropout=0.5): encoder dropout p =', Classifier(dropout=0.5).encoder_layer.dropout.p)
    # frame order: the trained model on real validation utterances
    import train
    from dataset import myDataset
    from torch.utils.data import random_split
    ds = myDataset(D)
    tl = int(0.9 * len(ds))
    _, va = random_split(ds, [tl, len(ds) - tl])
    m = m.cuda()
    g = torch.Generator().manual_seed(0)
    ok = {'orig': 0, 'reversed': 0, 'shuffled': 0}
    maxd = 0.0
    with torch.no_grad():
        for i in range(len(va)):
            p, spk = ds.data[va.indices[i]]
            mel = torch.load(os.path.join(D, p)).cuda()
            perm = torch.randperm(len(mel), generator=g).cuda()
            outs = {'orig': m(mel[None]), 'reversed': m(mel.flip(0)[None]), 'shuffled': m(mel[perm][None])}
            for kk, o in outs.items():
                ok[kk] += int(o.argmax().item() == spk)
            maxd = max(maxd, (outs['orig'] - outs['shuffled']).abs().max().item())
    n = len(va)
    print('full-length valid acc, frames in order / reversed / shuffled:', {kk: round(vv / n, 5) for kk, vv in ok.items()},
          'max |logit diff| orig vs shuffled', maxd)
    # padding with the trained model: the 91-frame utterance (index 20902, speaker 534)
    p, spk = ds.data[20902]
    mel = torch.load(os.path.join(D, p)).cuda()
    padded = torch.cat([mel, torch.full((37, 40), -20.0, device='cuda')])
    with torch.no_grad():
        a, b2 = m(mel[None]), m(padded[None])
    print('91-frame utterance, label', spk, ': alone argmax', a.argmax().item(), 'p(label)', round(a.softmax(-1)[0, spk].item(), 4),
          '| padded to 128 argmax', b2.argmax().item(), 'p(label)', round(b2.softmax(-1)[0, spk].item(), 4))
    L = 4940
    print('attention matrix for the longest test utterance: 2 heads x', L, 'x', L, 'float32 =', 2 * L * L * 4 / 2**20, 'MiB')

elif sec == 'ch04':
    import copy, glob
    import train
    from classifier import Classifier
    from dataset import myDataset
    from torch.utils.data import random_split
    # the first update: lr 0
    torch.manual_seed(0)
    m = Classifier(n_spks=600)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    print('AdamW defaults:', {k: opt.defaults[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')})
    sch = train.get_cosine_schedule_with_warmup(opt, 1000, 70000)
    before = copy.deepcopy(m.state_dict())
    x, y = torch.randn(32, 128, 40), torch.randint(0, 600, (32,))
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward()
    print('lr used by the 1st optimizer.step():', opt.param_groups[0]['lr'])
    opt.step(); sch.step(); opt.zero_grad()
    print('parameters unchanged after the 1st step?', all(torch.equal(before[k], v) for k, v in m.state_dict().items()),
          '| Adam state exp_avg nonzero?', any(s['exp_avg'].abs().sum() > 0 for s in opt.state.values()))
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward()
    print('lr used by the 2nd step:', opt.param_groups[0]['lr']); opt.step()
    print('changed after the 2nd step?', not all(torch.equal(before[k], v) for k, v in m.state_dict().items()))
    # state_dict() returns references
    sd = m.state_dict()
    w0 = sd['prenet.weight'].clone()
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step()
    print('sd = model.state_dict(); after optimizer.step(): sd tensor changed?', not torch.equal(sd['prenet.weight'], w0),
          '| same storage as the parameter?', sd['prenet.weight'].data_ptr() == m.prenet.weight.data_ptr())
    # every validation snapshot: full-length and fixed-crop accuracy on all 5,667 validation utterances
    snap = sys.argv[2]
    train.set_seed(87)   # the code above used the global RNG; re-seed so random_split matches train.py
    ds = myDataset(D)
    tl = int(0.9 * len(ds))
    _, va = random_split(ds, [tl, len(ds) - tl])
    assert va.indices[:5] == [43097, 46091, 22324, 24456, 274], va.indices[:5]
    mels = [torch.load(os.path.join(D, ds.data[i][0])) for i in va.indices]
    ys = torch.tensor([ds.data[i][1] for i in va.indices])
    rng = random.Random(0)
    crops = []
    for mel in mels:
        if len(mel) > 128:
            s = rng.randint(0, len(mel) - 128); crops.append(mel[s:s + 128])
        else:
            crops.append(mel)
    crop_batch = torch.stack(crops).cuda()   # all validation utterances are >= 152 frames
    m = Classifier(n_spks=600).cuda().eval()
    rows = []
    for f in sorted(glob.glob(os.path.join(snap, 'step_*.pt')), key=lambda s: int(s.split('_')[-1][:-3])):
        m.load_state_dict(torch.load(f))
        with torch.no_grad():
            full = sum(int(m(mel[None].cuda()).argmax().item() == y) for mel, y in zip(mels, ys.tolist()))
            crop = (m(crop_batch).argmax(1).cpu() == ys).sum().item()
        rows.append((int(f.split('_')[-1][:-3]), round(full / len(mels), 5), round(crop / len(mels), 5)))
        print('step', rows[-1][0], 'full', rows[-1][1], 'fixed-crop', rows[-1][2], flush=True)
    bf = max(rows, key=lambda r: r[1]); bc = max(rows, key=lambda r: r[2])
    print('best full at step', bf, '| best fixed-crop at step', bc)

elif sec == 'ch05':
    # test.py facts: torch.stack with batch 2, id2speaker keys, eval() vs train(), deepcopy bug on the
    # test predictions, accuracy by utterance length, padding with / without a mask, inference time.
    import csv, hashlib, io, time
    import train  # set_seed(87)
    import test as T
    from dataset import myDataset
    from classifier import Classifier
    from torch.utils.data import random_split
    live_ck, best_ck = sys.argv[2], sys.argv[3]
    tds = T.InferenceDataset(D)
    a, b = tds[0], tds[1]
    print('test[0]', a[0], tuple(a[1].shape), '| test[1]', b[0], tuple(b[1].shape))
    try:
        T.inference_collate_batch([a, b])
    except RuntimeError as e:
        print('stack of 2 ->', type(e).__name__ + ':', e)
    paths, x = T.inference_collate_batch([a])
    print('batch of 1 ->', type(paths).__name__, len(paths), tuple(x.shape))
    mapping = json.load(open(os.path.join(D, 'mapping.json')))
    m = Classifier(n_spks=600).cuda()
    m.load_state_dict(torch.load(live_ck)); m.eval()
    with torch.no_grad():
        pred = m(x.cuda()).argmax(1).cpu().numpy()
    p0 = pred[0]
    print('pred type', type(p0).__name__, repr(p0), '| str(pred) ->', repr(str(p0)),
          '| id2speaker[str]', mapping['id2speaker'][str(p0)])
    try:
        mapping['id2speaker'][p0]
    except KeyError as e:
        print('id2speaker[pred] ->', 'KeyError:', e)
    print('speaker2id[id2speaker[str(p)]] == p for all 600?',
          all(mapping['speaker2id'][mapping['id2speaker'][str(i)]] == i for i in range(600)))

    # load everything once
    test_paths = [u['feature_path'] for u in tds.data]
    test_mels = [torch.load(os.path.join(D, p)) for p in test_paths]
    train.set_seed(87)
    ds = myDataset(D)
    tl = int(0.9 * len(ds))
    _, va = random_split(ds, [tl, len(ds) - tl])
    assert va.indices[:5] == [43097, 46091, 22324, 24456, 274], va.indices[:5]
    v_mels = [torch.load(os.path.join(D, ds.data[i][0])) for i in va.indices]
    v_y = np.array([ds.data[i][1] for i in va.indices])

    def preds_b1(model, mels, times=None):
        out = []
        with torch.no_grad():
            for mel in mels:
                if times is not None:
                    torch.cuda.synchronize(); t = time.perf_counter()
                p = model(mel[None].cuda()).argmax(1).cpu().numpy()
                if times is not None:
                    times.append(time.perf_counter() - t)
                out.append(int(p[0]))
        return np.array(out)

    def csv_md5(preds):
        f = io.StringIO(newline='')
        w = csv.writer(f)
        w.writerows([['Id', 'Category']] + [[p, mapping['id2speaker'][str(q)]] for p, q in zip(test_paths, preds)])
        s = f.getvalue().encode()
        return hashlib.md5(s).hexdigest(), s.count(b'\n')

    preds_b1(m, test_mels[:200])  # warm-up
    t0 = time.perf_counter(); live_test = preds_b1(m, test_mels); t_b1 = time.perf_counter() - t0
    print('live: test batch-1 model loop %.2f s (mels already in RAM)' % t_b1, '| csv md5/lines', csv_md5(live_test))
    live_val = preds_b1(m, v_mels)
    print('live: valid full acc', round((live_val == v_y).mean(), 5))

    mb = Classifier(n_spks=600).cuda(); mb.load_state_dict(torch.load(best_ck)); mb.eval()
    best_test = preds_b1(mb, test_mels)
    best_val = preds_b1(mb, v_mels)
    print('best(68k) valid full acc', round((best_val == v_y).mean(), 5),
          '| test preds differing live vs best:', int((best_test != live_test).sum()),
          '| valid preds differing:', int((best_val != live_val).sum()))

    # forgetting model.eval(): dropout stays on
    m.train(); torch.manual_seed(0)
    tr_val = preds_b1(m, v_mels); tr_test = preds_b1(m, test_mels)
    torch.manual_seed(1); tr_test2 = preds_b1(m, test_mels)
    m.eval()
    print('train() mode: valid full acc', round((tr_val == v_y).mean(), 5),
          '| test preds differing from eval():', int((tr_test != live_test).sum()),
          '| two train()-mode runs differ on', int((tr_test != tr_test2).sum()))

    # accuracy by utterance length (validation), full vs fixed 128-frame crop; test lengths in the same bins
    rng = random.Random(0)
    crops = []
    for mel in v_mels:
        s = rng.randint(0, len(mel) - 128); crops.append(mel[s:s + 128])
    with torch.no_grad():
        crop_pred = m(torch.stack(crops).cuda()).argmax(1).cpu().numpy()
    vl = np.array([len(x) for x in v_mels]); tlen = np.array([len(x) for x in test_mels])
    edges = [0, 400, 500, 650, 900, 1300, 10 ** 9]
    print('valid crop acc (fixed) overall', round((crop_pred == v_y).mean(), 5))
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (vl >= lo) & (vl < hi)
        print(f'len [{lo},{hi}): valid n={k.sum()} full={(live_val[k] == v_y[k]).mean():.4f} '
              f'crop={(crop_pred[k] == v_y[k]).mean():.4f} | test n={((tlen >= lo) & (tlen < hi)).sum()}')
    print('valid lengths min/median/max', vl.min(), int(np.median(vl)), vl.max())

    # batching the test set: sort by length, pad with -20 (like train.py), with and without a mask
    def fwd_masked(model, x, lens):
        out = model.prenet(x).permute(1, 0, 2)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] >= lens[:, None]  # True = padding
        out = model.encoder_layer(out, src_key_padding_mask=mask).transpose(0, 1)
        keep = (~mask).unsqueeze(-1).float()
        return model.pred_layer((out * keep).sum(1) / keep.sum(1))

    order = np.argsort(tlen, kind='stable')
    def batched(masked, bs=32):
        pred = np.zeros(len(test_mels), dtype=int); maxdiff = 0.0
        with torch.no_grad():
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                xs = [test_mels[j] for j in idx]
                lens = torch.tensor([len(t) for t in xs], device='cuda')
                x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-20).cuda()
                o = fwd_masked(m, x, lens) if masked else m(x)
                pred[idx] = o.argmax(1).cpu().numpy()
        return pred
    for masked in (False, True):
        batched(masked)  # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter(); p = batched(masked); torch.cuda.synchronize()
        print(f'sorted batches of 32, pad -20, mask={masked}: {time.perf_counter() - t0:.2f} s, '
              f'preds differing from batch-1: {int((p != live_test).sum())}')
    # unsorted (test.py order) batches without mask: how much padding
    pads = []
    for i in range(0, len(tlen), 32):
        L = tlen[i:i + 32]; pads.append((L.max() - L).sum() / (L.max() * len(L)))
    pads_s = []
    for i in range(0, len(order), 32):
        L = tlen[order[i:i + 32]]; pads_s.append((L.max() - L).sum() / (L.max() * len(L)))
    print('padding fraction, batches of 32: file order %.3f, sorted by length %.4f' % (np.mean(pads), np.mean(pads_s)))
    # masked logits vs batch-1 logits for one batch
    with torch.no_grad():
        idx = order[4000:4032]; xs = [test_mels[j] for j in idx]
        lens = torch.tensor([len(t) for t in xs], device='cuda')
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-20).cuda()
        ob = fwd_masked(m, x, lens); on = m(x)
        o1 = torch.cat([m(t[None].cuda()) for t in xs])
        print('one batch (lens %d-%d): max |logit diff| masked vs batch-1 %.2e, unmasked vs batch-1 %.2e'
              % (lens.min(), lens.max(), (ob - o1).abs().max(), (on - o1).abs().max()))

    # per-utterance time (batch 1, model only) by length
    times = []
    preds_b1(m, test_mels, times)
    times = np.array(times) * 1000
    for lo, hi in [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000)]:
        k = (tlen >= lo) & (tlen < hi)
        print(f'time len [{lo},{hi}): n={k.sum()} median {np.median(times[k]):.2f} ms max {times[k].max():.2f} ms')
    print('time total %.2f s, longest utterance %d frames: %.2f ms' % (times.sum() / 1000, tlen.max(), times[tlen.argmax()]))

elif sec == 'ch06':
    # hw04_exp.py parses argv and trains at import time, so exec only its model classes (ConvModule..ConformerBlock)
    import re as _re
    import torch.nn as nn
    import torch.nn.functional as F
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hw04_exp.py')).read()
    code = src[src.index('class ConvModule'):src.index('class Net')]
    ns = dict(torch=torch, nn=nn, F=F)
    exec(code, ns)
    torch.manual_seed(0)
    cnt = lambda mod: sum(p.numel() for p in mod.parameters())
    blk = ns['ConformerBlock'](160, 4, 640, 31, 0.1)
    print('ConformerBlock(d=160, heads=4, ffn=640, kernel=31): total', cnt(blk))
    for n, mod in blk.named_children():
        print('  ', n, type(mod).__name__, cnt(mod))
    for n, mod in blk.conv.named_children():
        print('     conv.', n, type(mod).__name__, cnt(mod), tuple(getattr(mod, 'weight', torch.empty(0)).shape))
    tl = nn.TransformerEncoderLayer(160, 4, 640)
    print('TransformerEncoderLayer(160, 4, 640):', cnt(tl))
    head = 40 * 160 + 160 + 160 * 160 + 160 + 160 * 600 + 600
    print('prenet + pred_layer (d=160, 600 speakers):', head,
          '| conf160 (2 blocks):', head + 2 * cnt(blk), '| tf160x4 (4 layers):', head + 4 * cnt(tl),
          '| med160 (2 layers, ffn 512):', head + 2 * cnt(nn.TransformerEncoderLayer(160, 4, 512)))
    # shapes through the block and the conv module
    blk.eval()
    x = torch.randn(2, 128, 160)
    c = blk.conv
    y = c.ln(x).transpose(1, 2); print('conv: in', tuple(x.shape), '-> LN+transpose', tuple(y.shape), end=' ')
    y = c.pw1(y); print('-> pointwise1', tuple(y.shape), end=' ')
    y = F.glu(y, dim=1); print('-> GLU', tuple(y.shape), end=' ')
    y = c.dw(y); print('-> depthwise', tuple(y.shape), end=' ')
    y = c.pw2(F.silu(c.bn(y))); print('-> BN, Swish, pointwise2', tuple(y.transpose(1, 2).shape))
    with torch.no_grad():
        print('block: in', tuple(x.shape), 'out', tuple(blk(x).shape), '| T=4940:', tuple(blk(torch.randn(1, 4940, 160)).shape))
    # frame-order sensitivity at random init (eval mode): mean-pooled output, original vs reversed vs shuffled
    perm = torch.randperm(128)
    tl.eval()
    with torch.no_grad():
        xt = torch.randn(128, 2, 160)  # (T, B, d) for the post-norm layer
        a0, ar, ap_ = (tl(xt).mean(0), tl(xt.flip(0)).mean(0), tl(xt[perm]).mean(0))
        b0, br, bp = (blk(x).mean(1), blk(x.flip(1)).mean(1), blk(x[:, perm]).mean(1))
    print('random init, mean-pooled output max |diff|: Transformer layer reversed %.2e shuffled %.2e | '
          'Conformer block reversed %.2e shuffled %.2e (output std %.2f)'
          % ((a0 - ar).abs().max(), (a0 - ap_).abs().max(), (b0 - br).abs().max(), (b0 - bp).abs().max(), b0.std()))
    # how far the depthwise conv sees
    print('depthwise kernel 31 frames = 310 ms; 2 blocks -> receptive field of the convs alone', 2 * 30 + 1, 'frames')
