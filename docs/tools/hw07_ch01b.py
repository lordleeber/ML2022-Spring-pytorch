# ch01 facts, part 2: effect of lowercasing on [UNK] and on answers that decode cannot give back.
# usage: cd HW07 && python ../docs/tools/hw07_ch01b.py   (CPU only)
import json
from transformers import BertTokenizerFast

for lower in [False, True]:
  tok = BertTokenizerFast.from_pretrained("bert-base-chinese", do_lower_case=lower)
  print(f'== do_lower_case={lower}: HTTP -> {tok.tokenize("HTTP")}, Duff Roblin -> {tok.tokenize("Duff Roblin")}, GDP -> {tok.tokenize("GDP")}')
  for split in ['train', 'dev']:
    d = json.load(open(f'hw7_{split}.json', encoding='utf-8'))
    ps, qs = d['paragraphs'], d['questions']
    pt = tok(ps, add_special_tokens=False)
    unk = sum(x.count(tok.unk_token_id) for x in pt['input_ids'])
    bad = unkbad = 0
    for q in qs:
      e = pt[q['paragraph_id']]
      s, t = e.char_to_token(q['answer_start']), e.char_to_token(q['answer_end'])
      dec = tok.decode(e.ids[s:t + 1]).replace(' ', '')
      if dec != q['answer_text']:
        bad += 1; unkbad += '[UNK]' in dec
    # answers rebuilt from offsets instead of decode
    obad = 0
    for q in qs:
      e = pt[q['paragraph_id']]
      s, t = e.char_to_token(q['answer_start']), e.char_to_token(q['answer_end'])
      a, b = e.offsets[s][0], e.offsets[t][1]
      obad += ps[q['paragraph_id']][a:b] != q['answer_text']
    print(f'  {split}: paragraph [UNK] tokens {unk}; decode != answer {bad} (with [UNK] {unkbad}); offsets != answer {obad}')
