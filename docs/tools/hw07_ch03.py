# ch03 facts: the training metric (one window centred on the answer) measured on dev and on train,
# and where dev EM is lost (answer cut / wrong window / right window but wrong span / text recovery).
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_ch03.py <saved_model>   (GPU, eval mode, no grad)
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
from dataset import read_data, QA_Dataset
from postprocess import evaluate

tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
model = BertForQuestionAnswering.from_pretrained(sys.argv[1]).to('cuda').eval()

def train_style(split, n=None):
  qs, ps = read_data(f'hw7_{split}.json')
  if n:
    rng = random.Random(0); idx = sorted(rng.sample(range(len(qs)), n)); qs = [qs[i] for i in idx]
  qt = tok([q['question_text'] for q in qs], add_special_tokens=False); pt = tok(ps, add_special_tokens=False)
  ds = QA_Dataset('train', qs, qt, pt)   # 'train' split = one window centred on the answer
  dl = DataLoader(ds, batch_size=64, shuffle=False)
  s_ok = e_ok = both = 0; loss = 0.0; nb = 0
  with torch.no_grad():
    for d in dl:
      d = [x.cuda() for x in d]
      o = model(input_ids=d[0], token_type_ids=d[1], attention_mask=d[2], start_positions=d[3], end_positions=d[4])
      si, ei = o.start_logits.argmax(1), o.end_logits.argmax(1)
      s_ok += int((si == d[3]).sum()); e_ok += int((ei == d[4]).sum()); both += int(((si == d[3]) & (ei == d[4])).sum())
      loss += float(o.loss) * len(d[0]); nb += len(d[0])
  print(f'{split} ({nb} questions), one centred window, eval mode: start acc {s_ok/nb:.4f}, end acc {e_ok/nb:.4f}, both {both/nb:.4f}, loss {loss/nb:.4f}')

train_style('dev')
train_style('train', 4131)

# dev with the real evaluation windows
qs, ps = read_data('hw7_dev.json')
qt = tok([q['question_text'] for q in qs], add_special_tokens=False); pt = tok(ps, add_special_tokens=False)
ds = QA_Dataset('dev', qs, qt, pt)
cnt = dict(em=0, cut=0, wrong_window=0, right_window_wrong_span=0, right_span_text_mismatch=0, empty=0, end_before_start=0, span_in_question=0)
with torch.no_grad():
  for i in range(len(ds)):
    d = [x.unsqueeze(0) for x in ds[i]]
    o = model(input_ids=d[0][0].cuda(), token_type_ids=d[1][0].cuda(), attention_mask=d[2][0].cuda())
    ans = evaluate(d, o, tok)
    q = qs[i]; para = pt[q['paragraph_id']]
    a = para.char_to_token(q['answer_start']); b = para.char_to_token(q['answer_end'])
    qlen = 2 + min(40, len(qt[i].ids))
    # replay evaluate's choice
    best, k_best, si_b, ei_b = -1e9, 0, 0, 0
    for k in range(d[0].shape[1]):
      sp, si = torch.max(o.start_logits[k], 0); ep, ei = torch.max(o.end_logits[k], 0)
      if sp + ep > best: best, k_best, si_b, ei_b = float(sp + ep), k, int(si), int(ei)
    if ans == '': cnt['empty'] += 1
    if ei_b < si_b: cnt['end_before_start'] += 1
    if si_b < qlen and ei_b >= si_b: cnt['span_in_question'] += 1
    if ans == q['answer_text']: cnt['em'] += 1; continue
    if a // 150 != b // 150: cnt['cut'] += 1; continue
    gold_k = a // 150
    if k_best != gold_k: cnt['wrong_window'] += 1; continue
    gs, ge = a - gold_k * 150 + qlen, b - gold_k * 150 + qlen
    if (si_b, ei_b) != (gs, ge): cnt['right_window_wrong_span'] += 1
    else: cnt['right_span_text_mismatch'] += 1
print('dev with evaluation windows (stride 150):', cnt, 'total', len(ds))
