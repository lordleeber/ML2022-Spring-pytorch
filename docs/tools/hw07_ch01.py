# ch01 facts: what BertTokenizerFast (bert-base-chinese) does to Chinese, English, digits;
# vocabulary make-up; [UNK]; offsets and char_to_token; why decode does not give back the text.
# usage: cd HW07 && python ../docs/tools/hw07_ch01.py   (CPU only)
import collections
import json
import re
import time
import unicodedata
from transformers import BertTokenizerFast

tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
print('vocab size', len(tok), '| special', {t: tok.convert_tokens_to_ids(t) for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']})
vocab = tok.get_vocab()
cjk = [t for t in vocab if len(t) == 1 and '一' <= t <= '鿿']
sub = [t for t in vocab if t.startswith('##')]
unused = [t for t in vocab if t.startswith('[unused')]
ascii_words = [t for t in vocab if re.fullmatch(r'[a-z]+', t)]
digits = [t for t in vocab if re.fullmatch(r'\d+', t)]
print(f'single CJK chars {len(cjk)}, ##pieces {len(sub)} (of which ##CJK {sum(1 for t in sub if len(t)==3 and "一"<=t[2]<="鿿")}), [unusedN] {len(unused)}, lowercase ascii words {len(ascii_words)}, digit strings {len(digits)}')
print('some digit tokens:', sorted(digits, key=lambda x: (len(x), x))[:5], '...', [d for d in digits if len(d) == 4][:12])
print('tokenizer lowercases:', tok.backend_tokenizer.normalizer.__class__.__name__, getattr(tok, 'do_lower_case', None))

def show(s):
  e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
  toks = tok.convert_ids_to_tokens(e['input_ids'])
  print(f'{s!r}\n   tokens {toks}\n   ids {e["input_ids"]}\n   offsets {e["offset_mapping"]}\n   decode {tok.decode(e["input_ids"])!r}')

for s in ['李宏毅教授2022機器學習', '1338年白蓮教徒起義的結果是?', 'Duff Roblin', 'HTTP', 'PAL制', 'iPhone 13', '128所', '64所', '張騫', '朱允炆', '「HD-Ready」', '１２３ＡＢＣ', '台灣臺灣', 'J·R·R·托爾金']:
  show(s)

e = tok('天神地區', add_special_tokens=True)
print('with special tokens:', tok.convert_ids_to_tokens(e['input_ids']), e['input_ids'], e['token_type_ids'])
pair = tok('哪一地區?', '天神地區', add_special_tokens=True)
print('pair:', tok.convert_ids_to_tokens(pair['input_ids']), pair['token_type_ids'])

# traditional vs simplified coverage on the data
d = json.load(open('hw7_train.json', encoding='utf-8'))
text = ''.join(d['paragraphs']) + ''.join(q['question_text'] for q in d['questions'])
chars = collections.Counter(c for c in text if '一' <= c <= '鿿')
oov = {c: n for c, n in chars.items() if c not in vocab}
print(f'train distinct CJK chars {len(chars)}, not in vocab {len(oov)}, their occurrences {sum(oov.values())} of {sum(chars.values())}')
print('most frequent OOV CJK chars:', sorted(oov.items(), key=lambda x: -x[1])[:20])

# what becomes [UNK] in the dev paragraphs (by original text span)
dv = json.load(open('hw7_dev.json', encoding='utf-8'))
unk_src = collections.Counter()
enc = tok(dv['paragraphs'], add_special_tokens=False, return_offsets_mapping=True)
for p, ids, offs in zip(dv['paragraphs'], enc['input_ids'], enc['offset_mapping']):
  for i, (a, b) in zip(ids, offs):
    if i == tok.unk_token_id:
      unk_src[p[a:b]] += 1
print(f'dev [UNK] tokens {sum(unk_src.values())}, distinct sources {len(unk_src)}; top:', unk_src.most_common(25))
cat = collections.Counter()
for s, n in unk_src.items():
  c = 'cjk' if all('一' <= ch <= '鿿' or '㐀' <= ch <= '䶿' for ch in s) else ('latin' if re.search(r'[A-Za-z]', s) else 'other')
  cat[c] += n
print('dev [UNK] by kind:', dict(cat))

# speed
t0 = time.time(); tok(d['paragraphs'], add_special_tokens=False); print(f'tokenize 10,524 train paragraphs: {time.time()-t0:.2f} s')
