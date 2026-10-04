"""Fill the TODO(本機實測) markers of docs/HW09/ch06.html (review of PR #16).

Run from HW09/ (CPU; the HF model must be in the cache, or it is downloaded):
    ../.venv/bin/python ../docs/tools/hw09_ch06_review.py
Nothing is written to docs/; the quiz-6 variant saves its 36 figures to ./output/ch06_quiz6/.
"""
import os
import sys
import torch
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA

sys.path.insert(0, '.')
import bert_hidden_states as B
from transformers import BertForQuestionAnswering, BertTokenizerFast

tok = BertTokenizerFast.from_pretrained(B.qa_model_name)
model = BertForQuestionAnswering.from_pretrained(B.qa_model_name, output_hidden_states=True).eval()

print('===== which PCA solver sklearn picks for (seq_len, 768)')
for q in range(3):
    inputs = tok(B.questions[q], B.contexts[q], return_tensors='pt')
    with torch.no_grad():
        hs = model(**inputs).hidden_states
    p = PCA(n_components=2, random_state=0).fit(hs[12][0])
    a = PCA(n_components=2, random_state=1).fit_transform(hs[12][0])
    b = PCA(n_components=2, random_state=0).fit_transform(hs[12][0])
    print(f'group {q + 1}: shape {tuple(hs[12][0].shape)} solver {p._fit_svd_solver}; random_state 0 vs 1 max |diff| {abs(a - b).max():.3e}')

print('===== group 2, layer 12: the special tokens in PCA coordinates')
inputs = tok(B.questions[1], B.contexts[1], return_tensors='pt')
with torch.no_grad():
    hs = model(**inputs).hidden_states
ids = inputs['input_ids'][0].tolist()
r = PCA(n_components=2, random_state=0).fit_transform(hs[12][0])
for i, t in enumerate(ids):
    if t in (101, 102):
        print(f'position {i} {tok.convert_ids_to_tokens(t)}: ({r[i, 0]:.1f}, {r[i, 1]:.1f})')
drawn = [i for i, t in enumerate(ids) if t not in (101, 102)]
print('drawn tokens: x %.1f..%.1f y %.1f..%.1f' % (r[drawn, 0].min(), r[drawn, 0].max(), r[drawn, 1].min(), r[drawn, 1].max()))
for L in (1, 4, 8, 12):
    r = PCA(n_components=2, random_state=0).fit_transform(hs[L][0])
    print(f'layer {L}: ' + ', '.join(f'{tok.convert_ids_to_tokens(ids[i])}@{i} ({r[i, 0]:.1f}, {r[i, 1]:.1f})' for i in (0, 12, 119)))

print('===== quiz 6: line 103 -> word.lower(); count the blue points per group')
src = open('bert_hidden_states.py').read()
old = '            if word in answers[QUESTION-1].split():  # Check if word in answer\n'
assert src.count(old) == 1
ns = {'__name__': 'variant'}
exec(compile(src.replace(old, old.replace('word in', 'word.lower() in')), 'bert_hidden_states.py', 'exec'), ns)
ns['output_dir'] = './output/ch06_quiz6/'
os.makedirs(ns['output_dir'], exist_ok=True)
import matplotlib.pyplot as plt
blue = {}
orig_scatter = plt.scatter
def counting_scatter(x, y, color=None, **kw):
    if color == 'blue':
        blue[cur] = blue.get(cur, 0) + 1
    return orig_scatter(x, y, color=color, **kw)
ns['plt'].scatter = counting_scatter
for cur in (1, 2, 3):
    ns['visualize'](tok, model, cur)
ns['plt'].scatter = orig_scatter
print('blue points per group, summed over 12 layers:', blue, '-> per layer:', {k: v // 12 for k, v in blue.items()})
