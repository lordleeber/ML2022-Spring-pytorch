# Where does the model put its answers inside a window? (tests the "answer at position 75" shortcut)
# For every dev window (stride s): argmax start position relative to the paragraph part, and for questions answered
# correctly by the sample rule, where the gold answer sits inside the chosen window.
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_pos.py --ckpt <dir> --tag <name> [--strides 150,32] [--lower ...]
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, BertTokenizerFast
from dataset import read_data

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True); p.add_argument('--tag', required=True)
p.add_argument('--tokenizer', default='bert-base-chinese'); p.add_argument('--lower', default='default')
p.add_argument('--strides', default='150,32'); p.add_argument('--jsonl', default=None)
args = p.parse_args()
kw = {} if args.lower == 'default' else {'do_lower_case': args.lower == 'true'}
tok = BertTokenizerFast.from_pretrained(args.tokenizer, **kw)
model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt).to('cuda').eval()
qs, ps = read_data('hw7_dev.json')
qt = tok([q['question_text'] for q in qs], add_special_tokens=False); pt = tok(ps, add_special_tokens=False)
BINS = [0, 15, 30, 45, 60, 70, 81, 90, 105, 120, 135, 150]
out = dict(tag=args.tag, bins=BINS)
for stride in [int(x) for x in args.strides.split(',')]:
  pred_mid = []; gold_mid_in_window = []; win_has_answer = []; win_best = []
  chosen_rel = []
  for i, q in enumerate(qs):
    para = pt[q['paragraph_id']]
    a = para.char_to_token(q['answer_start']); b = para.char_to_token(q['answer_end'])
    qids = [101] + qt[i].ids[:40] + [102]; ql = len(qids)
    ids, tts, ams, starts = [], [], [], []
    for s in range(0, len(para), stride):
      pids = para.ids[s:s + 150] + [102]; pad = 193 - ql - len(pids)
      ids.append(qids + pids + [0] * pad); tts.append([0] * ql + [1] * len(pids) + [0] * pad); ams.append([1] * (ql + len(pids)) + [0] * pad); starts.append(s)
    with torch.no_grad():
      o = model(input_ids=torch.tensor(ids, device='cuda'), token_type_ids=torch.tensor(tts, device='cuda'), attention_mask=torch.tensor(ams, device='cuda'))
    sp, si = o.start_logits.max(1); ep, ei = o.end_logits.max(1)
    sc = (sp + ep).tolist(); k = int(np.argmax(sc))
    for kk, s in enumerate(starts):
      has = s <= a and b < s + 150
      pm = (int(si[kk]) + int(ei[kk])) // 2 - ql
      if 0 <= pm < 150: pred_mid.append(pm)
      if has:
        win_has_answer.append(sc[kk])
      else:
        win_best.append(sc[kk])
    s = starts[k]
    if s <= a and b < s + 150:
      chosen_rel.append((a + b) // 2 - s)
  h = lambda x: np.histogram(x, bins=BINS)[0].tolist()
  out[stride] = dict(pred_mid_hist=h(pred_mid), chosen_window_gold_mid_hist=h(chosen_rel), chosen_has_answer=len(chosen_rel),
                     score_window_with_answer_mean=round(float(np.mean(win_has_answer)), 3), score_window_without_mean=round(float(np.mean(win_best)), 3))
  print(stride, out[stride], flush=True)
if args.jsonl:
  with open(args.jsonl, 'a') as f: f.write(json.dumps(out) + '\n')
