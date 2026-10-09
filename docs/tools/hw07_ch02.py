# ch02 facts: training windows (where the answer sits), dev windows (how many, how full),
# answers cut by window borders for several strides, and one example item.
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_ch02.py   (CPU only)
import collections
import numpy as np
from transformers import BertTokenizerFast
from dataset import read_data, QA_Dataset

tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
tq, tp = read_data('hw7_train.json')
dq, dp = read_data('hw7_dev.json')
tqt = tok([q['question_text'] for q in tq], add_special_tokens=False); tpt = tok(tp, add_special_tokens=False)
dqt = tok([q['question_text'] for q in dq], add_special_tokens=False); dpt = tok(dp, add_special_tokens=False)
tr = QA_Dataset('train', tq, tqt, tpt)
dv = QA_Dataset('dev', dq, dqt, dpt)
print('max_seq_len', tr.max_seq_len)

# one training item
ids, tt, am, s, e = tr[1]
print('train[1] question:', tq[1]['question_text'], '| answer:', tq[1]['answer_text'])
print('  shapes', tuple(ids.shape), tuple(tt.shape), tuple(am.shape), 'start/end', s, e)
print('  answer tokens', tok.convert_ids_to_tokens(ids[s:e + 1].tolist()))
qlen = 2 + min(40, len(tqt[1].ids))
print('  question part length (with CLS/SEP)', qlen, '| token_type 0s', int((tt == 0).sum()), '1s', int(tt.sum()), '| attention 1s', int(am.sum()), 'padding', int((am == 0).sum()))
print('  first 12 tokens', tok.convert_ids_to_tokens(ids[:12].tolist()))

# where does the answer sit in the training window (position inside the 150-token paragraph part)
rel_start, rel_mid, clipped_left, clipped_right, short = [], [], 0, 0, 0
for i in range(len(tr)):
  q = tq[i]; para = tpt[q['paragraph_id']]
  a = para.char_to_token(q['answer_start']); b = para.char_to_token(q['answer_end'])
  mid = (a + b) // 2
  start = max(0, min(mid - 75, len(para) - 150))
  if len(para) < 150: short += 1
  if mid - 75 < 0: clipped_left += 1
  elif mid - 75 > len(para) - 150: clipped_right += 1
  rel_start.append(a - start); rel_mid.append(mid - start)
rel_mid = np.array(rel_mid); rel_start = np.array(rel_start)
print(f'train windows: answer mid at paragraph position 75 exactly: {(rel_mid == 75).mean():.4f}; within 70..80: {((rel_mid >= 70) & (rel_mid <= 80)).mean():.4f}')
print(f'  window pushed by paragraph start: {clipped_left} ({clipped_left/len(tr):.4f}), by paragraph end: {clipped_right} ({clipped_right/len(tr):.4f}), paragraph shorter than 150: {short}')
hist = np.histogram(rel_mid, bins=[0, 15, 30, 45, 60, 75, 76, 90, 105, 120, 135, 150])[0]
print('  answer-mid position histogram [0,15,30,45,60,75,76,90,105,120,135,150):', hist.tolist())

# dev windows
nw = np.array([dv[i][0].shape[0] for i in range(len(dv))])
print(f'dev windows per question: mean {nw.mean():.2f}, counts {sorted(collections.Counter(nw.tolist()).items())}, total {nw.sum()}')
real = pad = 0; last_sizes = []
for i in range(len(dv)):
  ids, tt, am = dv[i]
  real += int(am.sum()); pad += int((am == 0).sum())
  plen = len(dpt[dq[i]['paragraph_id']])
  last_sizes.append(plen - (len(range(0, plen, 150)) - 1) * 150)
last_sizes = np.array(last_sizes)
print(f'dev padding share {pad/(real+pad):.4f}; last window paragraph tokens: mean {last_sizes.mean():.1f}, <=20: {(last_sizes <= 20).mean():.4f}, <=10: {(last_sizes <= 10).sum()}')
# which window holds the answer (stride 150), and position of the answer inside that window
which = []; pos = []
for q in dq:
  para = dpt[q['paragraph_id']]
  a = para.char_to_token(q['answer_start']); b = para.char_to_token(q['answer_end'])
  if a // 150 == b // 150:
    which.append(a // 150); pos.append((a + b) // 2 - (a // 150) * 150)
print('dev answer window index counts (stride 150, uncut only):', sorted(collections.Counter(which).items()))
print('dev answer-mid position in its window histogram [0,30,60,90,120,150):', np.histogram(pos, bins=[0, 30, 60, 90, 120, 150])[0].tolist())
# cut answers vs stride, with dev windows counted
for stride in [150, 128, 100, 75, 50]:
  cut = 0; total = 0
  for q in dq:
    para = dpt[q['paragraph_id']]
    a = para.char_to_token(q['answer_start']); b = para.char_to_token(q['answer_end'])
    starts = range(0, len(para), stride); total += len(starts)
    if not any(i <= a and b < i + 150 for i in starts): cut += 1
  print(f'stride {stride}: dev windows {total}, answers not inside any window {cut}')
