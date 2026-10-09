# Data facts for the HW07 book (CPU only): sizes, token lengths, windows, answers vs windows,
# and how often tokenizer.decode cannot give back the answer text.
# usage: cd HW07 && python ../docs/tools/hw07_data_facts.py
import json
import numpy as np
from transformers import BertTokenizerFast

tok = BertTokenizerFast.from_pretrained("bert-base-chinese")

def pct(a, qs=(50, 90, 99, 100)):
  return ' '.join(f'p{q}={int(np.percentile(a, q))}' for q in qs)

for split in ['train', 'dev', 'test']:
  d = json.load(open(f'hw7_{split}.json', encoding='utf-8'))
  qs, ps = d['questions'], d['paragraphs']
  print(f'== {split}: paragraphs {len(ps)}, questions {len(qs)}, keys {sorted(qs[0].keys())}')
  pt = tok(ps, add_special_tokens=False)
  qt = tok([q['question_text'] for q in qs], add_special_tokens=False)
  plen = np.array([len(x) for x in pt['input_ids']])
  qlen = np.array([len(x) for x in qt['input_ids']])
  pchar = np.array([len(p) for p in ps])
  print(f'paragraph chars mean {pchar.mean():.1f} {pct(pchar)}')
  print(f'paragraph tokens mean {plen.mean():.1f} {pct(plen)}; >150: {(plen > 150).mean():.4f}; >512: {(plen > 512).sum()}')
  print(f'question tokens mean {qlen.mean():.1f} {pct(qlen)}; >40 (truncated): {(qlen > 40).sum()}')
  qpl = np.array([plen[q['paragraph_id']] for q in qs])
  for stride in [150, 100, 75, 50, 32]:
    nw = np.array([len(range(0, n, stride)) for n in qpl])
    print(f'  stride {stride}: windows/question mean {nw.mean():.2f} max {nw.max()} total {nw.sum()}')
  unk_q = sum(1 for x in qt['input_ids'] if tok.unk_token_id in x)
  unk_p = sum(1 for x in pt['input_ids'] if tok.unk_token_id in x)
  print(f'[UNK] in question: {unk_q}, in paragraph: {unk_p}')
  if 'answer_text' not in qs[0] or qs[0]['answer_text'] is None:
    continue
  alen = np.array([q['answer_end'] - q['answer_start'] + 1 for q in qs])
  print(f'answer chars mean {alen.mean():.2f} {pct(alen)}')
  bad_span = sum(1 for q in qs if ps[q['paragraph_id']][q['answer_start']:q['answer_end'] + 1] != q['answer_text'])
  print(f'answer_text != paragraph[start:end+1]: {bad_span}')
  none_tok = 0; atok = []; dec_bad = 0; dec_unk = 0; cross = {150: 0, 100: 0, 75: 0, 50: 0, 32: 0}
  ex_bad = []
  for q in qs:
    e = pt[q['paragraph_id']]
    s, t = e.char_to_token(q['answer_start']), e.char_to_token(q['answer_end'])
    if s is None or t is None:
      none_tok += 1; continue
    atok.append(t - s + 1)
    dec = tok.decode(e.ids[s:t + 1]).replace(' ', '')
    if dec != q['answer_text']:
      dec_bad += 1
      if '[UNK]' in dec: dec_unk += 1
      if len(ex_bad) < 12: ex_bad.append((q['answer_text'], dec))
    for stride in cross:
      # answer fully inside at least one eval window [i, i+150)
      if not any(i <= s and t < i + 150 for i in range(0, len(e.ids), stride)):
        cross[stride] += 1
  atok = np.array(atok)
  print(f'char_to_token None: {none_tok}; answer tokens mean {atok.mean():.2f} {pct(atok)}; >150: {(atok > 150).sum()}')
  print(f'decode(answer tokens) != answer_text: {dec_bad} ({dec_bad / len(qs):.4f}), with [UNK]: {dec_unk}')
  for a, b in ex_bad: print(f'   {a!r} -> {b!r}')
  print('answer not inside any eval window: ' + ', '.join(f'stride {k}: {v}' for k, v in cross.items()))
