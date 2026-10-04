"""Numbers and extra figures for docs/HW09 ch06 (BERT hidden states, Q21-27).

Run from HW09/ (CPU is enough; the HF model must be in the cache, or it is downloaded):
    ../.venv/bin/python ../docs/tools/hw09_ch06_bert.py
Figures are written to ../docs/HW09/img/ch06_*.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, '.')
import bert_hidden_states as B
from transformers import BertForQuestionAnswering, BertTokenizerFast

IMG = '../docs/HW09/img/'
tok = BertTokenizerFast.from_pretrained(B.qa_model_name)
model = BertForQuestionAnswering.from_pretrained(B.qa_model_name, output_hidden_states=True).eval()
cfg = model.config

print('===== model / tokenizer')
print('class', type(model).__name__, '| layers', cfg.num_hidden_layers, 'hidden', cfg.hidden_size, 'heads', cfg.num_attention_heads,
      'intermediate', cfg.intermediate_size, 'max_position', cfg.max_position_embeddings, 'vocab', cfg.vocab_size)
print('params total', sum(p.numel() for p in model.parameters()), '| qa_outputs', model.qa_outputs)
print('special ids: [CLS]', tok.cls_token_id, '[SEP]', tok.sep_token_id, '[PAD]', tok.pad_token_id, '[UNK]', tok.unk_token_id, '| do_lower_case', getattr(tok, 'do_lower_case', None))

print('===== the context strings (backslash continuation keeps the indentation)')
for q in range(3):
    c = B.contexts[q]
    k = c.find('  ')
    print(f'Q{q + 1}: len {len(c)} chars; first run of spaces at {k}: {c[k - 12:k + 16]!r}; runs of 12 spaces: {c.count(" " * 12)}')

data = {}
for q in range(3):
    QUESTION = q + 1
    inputs = tok(B.questions[q], B.contexts[q], return_tensors='pt')
    ids = inputs['input_ids'][0].tolist()
    qs, qe = 1, ids.index(102) - 1
    cs, ce = qe + 2, len(ids) - 2
    toks = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs)
    hs = out.hidden_states
    data[QUESTION] = dict(inputs=inputs, ids=ids, toks=toks, qs=qs, qe=qe, cs=cs, ce=ce, hs=hs, out=out)
    print(f'===== Q{QUESTION}: {B.questions[q]!r} answer {B.answers[q]!r}')
    print('seq len', len(ids), '| question', qs, '..', qe, '| context', cs, '..', ce, '| [SEP] at', [i for i, t in enumerate(ids) if t == 102])
    print('token_type_ids: 0 for', int((inputs['token_type_ids'][0] == 0).sum()), 'tokens, 1 for', int((inputs['token_type_ids'][0] == 1).sum()))
    print('tokens:', toks)
    decoded = [tok.decode(t) for t in inputs['input_ids'][0]]
    blue = [i for i, w in enumerate(decoded) if w in B.answers[q].split()]
    print('blue (answer) positions', blue, [decoded[i] for i in blue])
    print('decode vs convert_ids_to_tokens differ at', [(i, toks[i], decoded[i]) for i in range(len(ids)) if toks[i] != decoded[i]][:8])
    print('hidden_states: len', len(hs), 'shape', tuple(hs[0].shape))
    emb = model.bert.embeddings(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    print('hidden_states[0] == model.bert.embeddings(...):', torch.allclose(emb, hs[0], atol=1e-6))
    print('hidden_states[12] == last_hidden_state used by qa_outputs:', torch.allclose(model.qa_outputs(hs[12]), torch.stack([out.start_logits, out.end_logits], -1), atol=1e-5))
    norms = [round(h[0].norm(dim=-1).mean().item(), 2) for h in hs]
    print('mean token vector norm, layer 0..12:', norms)
    # QA prediction
    s, e = out.start_logits[0], out.end_logits[0]
    null = (s[0] + e[0]).item()
    best = None
    for i in range(cs, ce + 1):
        for j in range(i, min(i + 30, ce + 1)):
            sc = (s[i] + e[j]).item()
            if best is None or sc > best[0]:
                best = (sc, i, j)
    print(f'argmax start {s.argmax().item()} ({toks[s.argmax()]}) end {e.argmax().item()} ({toks[e.argmax()]}) | null score [CLS] {null:.3f} | best non-null span {best[1]}..{best[2]} {tok.decode(ids[best[1]:best[2] + 1])!r} score {best[0]:.3f}')
    top_s = torch.topk(s, 3)
    print('top-3 start', [(toks[i], round(v, 2)) for v, i in zip(top_s.values.tolist(), top_s.indices.tolist())])
    # PCA
    evr = []
    for L in range(13):
        p = PCA(n_components=2, random_state=0).fit(hs[L][0].numpy())
        evr.append(round(float(p.explained_variance_ratio_.sum()), 3))
    print('PCA 2-comp explained variance, layer 0..12:', evr)
    r = PCA(n_components=2, random_state=0).fit_transform(hs[12][0])
    print('layer 12 PCA coordinate range x %.1f..%.1f y %.1f..%.1f' % (r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()))
    r1 = PCA(n_components=2, random_state=0).fit_transform(hs[1][0])
    print('layer 1 PCA coordinate range x %.1f..%.1f y %.1f..%.1f' % (r1[:, 0].min(), r1[:, 0].max(), r1[:, 1].min(), r1[:, 1].max()))

print('===== per layer, 768-D cosine: question vs context, and the nearest neighbours of the answer token')


def cos_matrix(h):
    h = torch.nn.functional.normalize(h, dim=-1)
    return h @ h.T


for QUESTION, ans_pos in [(1, 32), (2, 14), (3, 12)]:
    d = data[QUESTION]
    qidx = list(range(d['qs'], d['qe'] + 1))
    cidx = list(range(d['cs'], d['ce'] + 1))
    print(f'--- Q{QUESTION} answer token {ans_pos} {d["toks"][ans_pos]!r}')
    for L in range(13):
        C = cos_matrix(d['hs'][L][0])
        qq = C[qidx][:, qidx][~torch.eye(len(qidx), dtype=bool)].mean().item()
        cc = C[cidx][:, cidx][~torch.eye(len(cidx), dtype=bool)].mean().item()
        qc = C[qidx][:, cidx].mean().item()
        a2q = C[ans_pos, qidx].mean().item()
        others = [i for i in cidx if i != ans_pos]
        c2q = torch.stack([C[i, qidx].mean() for i in others])
        rank = int((c2q > a2q).sum().item()) + 1
        row = C[ans_pos].clone(); row[ans_pos] = -2
        nn = [d['toks'][i] for i in torch.topk(row, 5).indices.tolist()]
        print(f'layer {L:2d}: mean cos q-q {qq:.3f} c-c {cc:.3f} q-c {qc:.3f} | answer->question mean cos {a2q:.3f}, rank among {len(cidx)} context tokens {rank} | answer NN {nn}')

print('===== Q3: animal nouns vs others, and the answer-span tokens')
d = data[3]
animals = [i for i, t in enumerate(d['toks']) if t.lower() in ('wolves', 'cats', 'wolf', 'sheep', 'mouse') or t in ('She', '##ep', 'Mi', '##ce')]
print('animal-ish token positions', [(i, d['toks'][i]) for i in animals])
names = [i for i, t in enumerate(d['toks']) if t in ('Emily', 'Gertrude', 'Jessica', 'Win', '##ona')]
dots = [i for i, t in enumerate(d['toks']) if t == '.']
for L in (1, 6, 12):
    C = cos_matrix(d['hs'][L][0])
    def m(a, b):
        M = C[a][:, b]
        if a == b:
            M = M[~torch.eye(len(a), dtype=bool)]
        return M.mean().item()
    print(f'layer {L}: animals-animals {m(animals, animals):.3f} dots-dots {m(dots, dots):.3f} names-names {m(names, names):.3f} animals-dots {m(animals, dots):.3f}')

print('===== composite figures from the existing PNGs')
for QUESTION in (1, 2, 3):
    fig, axs = plt.subplots(2, 2, figsize=(16, 13.5))
    for ax, L in zip(axs.ravel(), (1, 4, 8, 12)):
        ax.imshow(plt.imread(IMG + f'bert_q{QUESTION}_layer{L}.png'))
        ax.axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(IMG + f'ch06_q{QUESTION}_layers.png', bbox_inches='tight', dpi=90)
    plt.close(fig)

print('===== explained variance plot')
fig, ax = plt.subplots(figsize=(6, 3.6))
for QUESTION in (1, 2, 3):
    hs = data[QUESTION]['hs']
    ev = [PCA(n_components=2, random_state=0).fit(hs[L][0].numpy()).explained_variance_ratio_.sum() for L in range(13)]
    ax.plot(range(13), ev, marker='o', label=f'question set {QUESTION}')
ax.set_xlabel('layer (0 = embedding output, not drawn by the script)'); ax.set_ylabel('variance kept by 2 PCs')
ax.set_xticks(range(13)); ax.legend(fontsize=8); ax.set_ylim(0, 0.55)
fig.savefig(IMG + 'ch06_pca_variance.png', bbox_inches='tight')
plt.close(fig)
print('figures written')
