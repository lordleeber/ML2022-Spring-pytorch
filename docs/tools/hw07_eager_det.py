# Is training deterministic with the eager (2022-style) attention instead of SDPA? 100 steps, run twice by the caller.
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_eager_det.py <eager|sdpa>
import sys
import numpy as np, random, torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
from legacy_adamw import AdamW
from dataset import read_data, QA_Dataset
torch.manual_seed(0); torch.cuda.manual_seed_all(0); np.random.seed(0); random.seed(0)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese", attn_implementation=sys.argv[1]).cuda()
tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
qs, ps = read_data('hw7_train.json')
ds = QA_Dataset('train', qs, tok([q['question_text'] for q in qs], add_special_tokens=False), tok(ps, add_special_tokens=False))
dl = DataLoader(ds, batch_size=32, shuffle=True)
opt = AdamW(model.parameters(), lr=1e-4); model.train()
for step, d in enumerate(dl, 1):
  d = [x.cuda() for x in d]
  o = model(input_ids=d[0], token_type_ids=d[1], attention_mask=d[2], start_positions=d[3], end_positions=d[4])
  o.loss.backward(); opt.step(); opt.zero_grad()
  if step == 100: break
print(sys.argv[1], model.config._attn_implementation, 'W', sum(p.double().sum().item() for p in model.parameters()))
