"""Numbers and extra figures for docs/HW09 ch07 (bert_embedding.py, Q28-30).

Run from HW09/ (CPU; bert-base-chinese must be in the HF cache, or it is downloaded):
    ../.venv/bin/python ../docs/tools/hw09_ch07_embedding.py
Figures are written to ../docs/HW09/img/ch07_*.png. Variants of the script are made by
editing its source text and exec-ing it; bert_embedding.py itself is not modified.
"""
import io
import sys
import warnings
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from sklearn.metrics import pairwise_distances

sys.path.insert(0, '.')
import bert_embedding as BE
from transformers import BertModel, BertTokenizerFast

IMG = '../docs/HW09/img/'
SRC = open('bert_embedding.py').read()
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True).eval()
tok = BertTokenizerFast.from_pretrained('bert-base-chinese')
S = BE.sentences
toks = [tok(s, return_tensors='pt') for s in S]
with torch.no_grad():
    outs = [model(**t) for t in toks]

print('===== model / tokenizer')
cfg = model.config
print('BertModel layers', cfg.num_hidden_layers, 'hidden', cfg.hidden_size, 'vocab', cfg.vocab_size, 'params', sum(p.numel() for p in model.parameters()))
print('hidden_states per sentence:', len(outs[0].hidden_states), tuple(outs[0].hidden_states[0].shape))

print('===== words vs characters vs tokens')
for i, (s, t) in enumerate(zip(S, toks)):
    w = t.word_ids()
    print(f'{i} {s} | chars {len(s)} | tokens {len(w)} | n words {max(x for x in w if x is not None) + 1} | tokens {tok.convert_ids_to_tokens(t["input_ids"][0])}')
s1 = toks[1]
print('sentence 1 word_ids:', s1.word_ids())
for k in (2, 13, 15, 16, 17, 18, 19):
    span = s1.word_to_tokens(k)
    tk = tok.convert_ids_to_tokens(int(s1['input_ids'][0][span.start])) if span is not None else None
    print(f'  sentence 1 index {k}: char {S[1][k]!r}, char_to_token -> {s1.char_to_token(k)}; word_to_tokens -> {span} token {tk!r}')
print('select 蘋:', BE.select_word_index, '-> tokens', [tok.convert_ids_to_tokens(int(toks[i]['input_ids'][0][toks[i].word_to_tokens(BE.select_word_index[i]).start])) for i in range(10)],
      'token idx', [toks[i].word_to_tokens(BE.select_word_index[i]).start for i in range(10)])
guo = [5, 3, 1, 9, 3, 1, 1, 5, 1, 1]
print('select 果:', guo, '-> tokens', [tok.convert_ids_to_tokens(int(toks[i]['input_ids'][0][toks[i].word_to_tokens(guo[i]).start])) for i in range(10)])
print('sentence 4 has 果 at chars', [k for k, c in enumerate(S[4]) if c == '果'])
print('token_type_ids all zero:', all(int(t['token_type_ids'].sum()) == 0 for t in toks))


def emb(layer, idx):
    return [outs[i].hidden_states[layer][0][toks[i].word_to_tokens(idx[i]).start].numpy() for i in range(10)]


np.set_printoptions(precision=2, suppress=True, linewidth=140)
print('===== layer 0 (embedding output), 蘋, euclidean')
M0 = pairwise_distances(emb(0, BE.select_word_index), metric=BE.euclidean_distance)
print(M0)
print('蘋 token position per sentence:', [toks[i].word_to_tokens(BE.select_word_index[i]).start for i in range(10)])
same = [(a, b) for a in range(10) for b in range(a + 1, 10) if M0[a, b] == 0]
print('pairs with distance exactly 0 at layer 0:', same)

print('===== layer 12, 蘋: euclidean (= bert_embedding.png) and its symmetry / text placement')
E12 = pairwise_distances(emb(12, BE.select_word_index), metric=BE.euclidean_distance)
print(E12)
print('dtype', E12.dtype, '| max |M - M.T|', np.abs(E12 - E12.T).max(), '| diagonal', np.diag(E12))
C12 = pairwise_distances(emb(12, BE.select_word_index), metric=BE.cosine_similarity)
print('cosine diagonal', np.diag(C12), '| max |C - C.T|', np.abs(C12 - C12.T).max())
for a, b in [(0, 7), (2, 6), (2, 1), (7, 5)]:
    print(f'  pair {a}-{b}: euclid {E12[a, b]:.2f} cosine {C12[a, b]:.2f}')
order = np.argsort(E12, axis=1)
for i in range(10):
    print(f'  sentence {i} nearest (euclid): {[int(j) for j in order[i][1:4]]} | nearest (cosine): {[int(j) for j in np.argsort(-C12[i])[1:4]]}')

print('===== layer sweep, 蘋 and 果: within fruit (0-4), within company (5-9), between; euclid and cosine')
F, Cc = range(5), range(5, 10)


def groups(M):
    w1 = np.mean([M[a, b] for a in F for b in F if a < b])
    w2 = np.mean([M[a, b] for a in Cc for b in Cc if a < b])
    bt = np.mean([M[a, b] for a in F for b in Cc])
    return w1, w2, bt


sweep = {}
for name, idx in (('蘋', BE.select_word_index), ('果', guo)):
    for metric, fn in (('euclid', BE.euclidean_distance), ('cosine', BE.cosine_similarity)):
        rows = []
        for L in range(13):
            rows.append(groups(pairwise_distances(emb(L, idx), metric=fn)))
        sweep[(name, metric)] = rows
        print(f'{name} {metric}: ' + ' | '.join(f'L{L} {w1:.3f}/{w2:.3f}/{bt:.3f}' for L, (w1, w2, bt) in enumerate(rows)))

print('===== simple separability per layer: nearest-neighbour (cosine) agrees with the fruit/company group')
for name, idx in (('蘋', BE.select_word_index), ('果', guo)):
    acc = []
    for L in range(13):
        C = pairwise_distances(emb(L, idx), metric=BE.cosine_similarity)
        np.fill_diagonal(C, -9)
        acc.append(int(sum((C[i].argmax() < 5) == (i < 5) for i in range(10))))
    print(f'{name}: correct out of 10, layer 0..12: {acc}')

FP = BE.FONT_PATH
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=FP).get_name()]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axs, ('euclid', 'cosine')):
    rows = np.array(sweep[('蘋', metric)])
    ax.plot(range(13), rows[:, 0], marker='o', label='within fruit (0-4)')
    ax.plot(range(13), rows[:, 1], marker='o', label='within company (5-9)')
    ax.plot(range(13), rows[:, 2], marker='o', label='between groups')
    ax.set_xticks(range(13)); ax.set_xlabel('layer'); ax.set_title(f'"蘋", mean {"euclidean distance" if metric == "euclid" else "cosine similarity"}', fontsize=10)
    ax.legend(fontsize=8)
fig.savefig(IMG + 'ch07_layer_sweep.png', bbox_inches='tight')
plt.close(fig)


def run_variant(name, edits):
    src = SRC
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    src = src.replace("path = os.path.join(output_dir, 'bert_embedding.png')", f"path = os.path.join('{IMG}', '{name}')")
    ns = {'__name__': '__main__'}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, 'bert_embedding.py', 'exec'), ns)
    glyph = [str(x.message) for x in w if 'Glyph' in str(x.message)]
    print(f'{name}: stdout {buf.getvalue().strip()!r}; glyph warnings {len(glyph)}' + (f' e.g. {glyph[0]!r}' if glyph else ''))
    return ns


print('===== variants of bert_embedding.py (figures)')
run_variant('ch07_cosine.png', [('METRIC = euclidean_distance', 'METRIC = cosine_similarity')])
run_variant('ch07_guo.png', [('select_word_index = [4, 2, 0, 8, 2, 0, 0, 4, 0, 0]\n# select_word_index = [5, 3, 1, 9, 3, 1, 1, 5, 1, 1]',
                              '# select_word_index = [4, 2, 0, 8, 2, 0, 0, 4, 0, 0]\nselect_word_index = [5, 3, 1, 9, 3, 1, 1, 5, 1, 1]')])
run_variant('ch07_layer0.png', [('LAYER = 12', 'LAYER = 0')])
run_variant('ch07_layer8.png', [('LAYER = 12', 'LAYER = 8')])
ns = run_variant('ch07_unimplemented.png', [('    return np.linalg.norm(a - b)', '    return 0')])
print('unimplemented euclid: matrix all zero?', np.all(ns['similarity_matrix'] == 0))
run_variant('ch07_font_droid_only.png', [("plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=FONT_PATH).get_name()]",
                                          "plt.rcParams['font.family'] = [font_manager.FontProperties(fname=FONT_PATH).get_name()]")])
plt.rcParams['font.family'] = ['DejaVu Sans']
run_variant('ch07_font_none.png', [("FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'", "FONT_PATH = '/nonexistent.ttf'")])
print('font file exists:', __import__('os').path.exists(FP), '| family name', font_manager.FontProperties(fname=FP).get_name())
print('figures written')
