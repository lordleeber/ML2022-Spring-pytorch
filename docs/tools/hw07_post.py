# Post-processing and evaluation-stride experiments on one checkpoint (no training).
# Runs the model once per stride over all dev windows, keeps the logits, then applies several rules:
#   sample            HW07/postprocess.py evaluate (max start, max end per window, sum, decode)
#   sample_off        same choice, text cut from the paragraph with offsets ('' if end < start, like decode)
#   valid             per window the best pair i <= j, both inside the paragraph part; decode
#   valid_len         valid + answer at most --maxlen tokens; decode
#   valid_len_off     valid_len, text from offsets
#   valid_len_off_lp  valid_len_off, windows compared by log-softmax(start)+log-softmax(end) instead of raw logits
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_post.py --ckpt <dir> --tag <name> [--tokenizer bert-base-chinese]
#        [--lower default|true|false] [--strides 150,100,75,50,32] [--jsonl ../docs/tools/hw07_post.jsonl] [--dump preds.jsonl]
import argparse
import json
import torch
from transformers import AutoModelForQuestionAnswering, BertTokenizerFast
from dataset import read_data

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--tokenizer', default='bert-base-chinese')
p.add_argument('--lower', choices=['default', 'true', 'false'], default='default')
p.add_argument('--strides', default='150,100,75,50,32')
p.add_argument('--maxlen', type=int, default=30)
p.add_argument('--jsonl', default=None)
p.add_argument('--dump', default=None, help='write per-question predictions (stride 150 and the best stride) here')
args = p.parse_args()

kw = {} if args.lower == 'default' else {'do_lower_case': args.lower == 'true'}
tok = BertTokenizerFast.from_pretrained(args.tokenizer, **kw)
model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt).to('cuda').eval()
qs, ps = read_data('hw7_dev.json')
qt = tok([q['question_text'] for q in qs], add_special_tokens=False)
pt = tok(ps, add_special_tokens=False)
MAXQ, MAXP, L = 40, 150, 193

def windows(i, stride):
  q = qs[i]; para = pt[q['paragraph_id']]
  qids = [101] + qt[i].ids[:MAXQ] + [102]
  out = []
  for s in range(0, len(para), stride):
    pids = para.ids[s:s + MAXP] + [102]
    pad = L - len(qids) - len(pids)
    ids = qids + pids + [0] * pad
    tt = [0] * len(qids) + [1] * len(pids) + [0] * pad
    am = [1] * (len(qids) + len(pids)) + [0] * pad
    out.append((ids, tt, am, s, len(qids), len(pids) - 1))   # paragraph part = [len(qids), len(qids)+n)
  return out

def run(stride):
  allw = []; owner = []
  for i in range(len(qs)):
    for w in windows(i, stride):
      allw.append(w); owner.append(i)
  S = []; E = []
  with torch.no_grad():
    for b in range(0, len(allw), 256):
      chunk = allw[b:b + 256]
      ids = torch.tensor([w[0] for w in chunk], device='cuda'); tt = torch.tensor([w[1] for w in chunk], device='cuda')
      am = torch.tensor([w[2] for w in chunk], device='cuda')
      o = model(input_ids=ids, token_type_ids=tt, attention_mask=am)
      S.append(o.start_logits.float().cpu()); E.append(o.end_logits.float().cpu())
  return allw, owner, torch.cat(S), torch.cat(E)

def text_off(i, w, a, b):
  # a, b are positions in the window; map to paragraph tokens, then to characters
  q = qs[i]; para = pt[q['paragraph_id']]; s0, qlen = w[3], w[4]
  ta, tb = s0 + a - qlen, s0 + b - qlen
  return ps[q['paragraph_id']][para.offsets[ta][0]:para.offsets[tb][1]]

TRI = torch.triu(torch.ones(L, L, dtype=torch.bool, device='cuda'))
LEN = TRI & ~torch.triu(torch.ones(L, L, dtype=torch.bool, device='cuda'), diagonal=args.maxlen)
POS = torch.arange(L, device='cuda')
results = {}
dump = {}
strides = [int(x) for x in args.strides.split(',')]
for stride in strides:
  allw, owner, S, E = run(stride)
  rules = ['sample', 'sample_off', 'valid', 'valid_len', 'valid_len_off', 'valid_len_off_lp']
  # per window: (score, a, b) for each rule, computed in batches on the GPU
  picks = {r: [] for r in ['sample', 'valid', 'valid_len', 'valid_len_lp']}
  for b0 in range(0, len(allw), 512):
    s = S[b0:b0 + 512].cuda(); e = E[b0:b0 + 512].cuda()
    qlen = torch.tensor([w[4] for w in allw[b0:b0 + 512]], device='cuda')
    n = torch.tensor([w[5] for w in allw[b0:b0 + 512]], device='cuda')
    sp, si = s.max(1); ep, ei = e.max(1)
    picks['sample'] += list(zip((sp + ep).tolist(), si.tolist(), ei.tolist()))
    inside = (POS[None, :] >= qlen[:, None]) & (POS[None, :] < (qlen + n)[:, None])
    m = inside[:, :, None] & inside[:, None, :]
    pair = s[:, :, None] + e[:, None, :]
    lp = torch.log_softmax(s, 1)[:, :, None] + torch.log_softmax(e, 1)[:, None, :]
    for rule, mm, mat in [('valid', m & TRI, pair), ('valid_len', m & LEN, pair), ('valid_len_lp', m & LEN, lp)]:
      x = mat.masked_fill(~mm, float('-inf')).flatten(1)
      v, idx = x.max(1)
      picks[rule] += list(zip(v.tolist(), (idx // L).tolist(), (idx % L).tolist()))
  best = {r: {} for r in rules}   # question -> (score, text)
  for k, (w, i) in enumerate(zip(allw, owner)):
    sc, si, ei = picks['sample'][k]
    if i not in best['sample'] or sc > best['sample'][i][0]:
      best['sample'][i] = (sc, tok.decode(torch.tensor(w[0][si:ei + 1])).replace(' ', ''))
      ok = w[4] <= si <= ei < w[4] + w[5]
      best['sample_off'][i] = (sc, text_off(i, w, si, ei) if ok else best['sample'][i][1])
    sc, a, b = picks['valid'][k]
    if i not in best['valid'] or sc > best['valid'][i][0]:
      best['valid'][i] = (sc, tok.decode(torch.tensor(w[0][a:b + 1])).replace(' ', ''))
    sc, a, b = picks['valid_len'][k]
    if i not in best['valid_len'] or sc > best['valid_len'][i][0]:
      best['valid_len'][i] = (sc, tok.decode(torch.tensor(w[0][a:b + 1])).replace(' ', ''))
      best['valid_len_off'][i] = (sc, text_off(i, w, a, b))
    sc, a, b = picks['valid_len_lp'][k]
    if i not in best['valid_len_off_lp'] or sc > best['valid_len_off_lp'][i][0]:
      best['valid_len_off_lp'][i] = (sc, text_off(i, w, a, b))
  for r in rules:
    em = sum(best[r][i][1] == qs[i]['answer_text'] for i in range(len(qs))) / len(qs)
    empty = sum(best[r][i][1] == '' for i in range(len(qs)))
    results[f'{stride}/{r}'] = round(em, 5)
    results[f'{stride}/{r}/empty'] = empty
    if args.dump:
      dump.setdefault(r, {})[stride] = [best[r][i][1] for i in range(len(qs))]
  results[f'{stride}/windows'] = len(allw)
  print(stride, {r: results[f'{stride}/{r}'] for r in rules}, 'windows', len(allw), flush=True)

rec = dict(tag=args.tag, ckpt=args.ckpt, tokenizer=args.tokenizer, lower=args.lower, do_lower_case=tok.do_lower_case, maxlen=args.maxlen, results=results)
print('RESULT', json.dumps(rec))
if args.jsonl:
  with open(args.jsonl, 'a') as f: f.write(json.dumps(rec) + '\n')
if args.dump:
  with open(args.dump, 'w') as f: json.dump(dump, f, ensure_ascii=False)
