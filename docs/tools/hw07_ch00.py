# ch00 facts: parameter count by part, tensor shapes, and the baseline model's answers on a few dev questions.
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_ch00.py <saved_model dir>   (runs on CPU)
import sys
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from dataset import read_data, QA_Dataset
from postprocess import evaluate

torch.manual_seed(0)
model = BertForQuestionAnswering.from_pretrained(sys.argv[1]).eval()
tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
cfg = model.config
print('config:', {k: getattr(cfg, k) for k in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'max_position_embeddings', 'type_vocab_size']})
parts = {}
for n, p in model.named_parameters():
  key = n.split('.')[1] if n.startswith('bert.') else n.split('.')[0]
  if n.startswith('bert.encoder.layer.'):
    key = 'encoder.layer[*]'
  parts[key] = parts.get(key, 0) + p.numel()
for k, v in parts.items(): print(f'{k:20s} {v:,}')
print('total', f'{sum(p.numel() for p in model.parameters()):,}')
print('layer 0:')
for n, p in model.bert.encoder.layer[0].named_parameters(): print(f'   {n:45s} {tuple(p.shape)} {p.numel():,}')
print('qa_outputs', model.qa_outputs)

qs, ps = read_data('hw7_dev.json')
qt = tok([q['question_text'] for q in qs], add_special_tokens=False)
pt = tok(ps, add_special_tokens=False)
ds = QA_Dataset('dev', qs, qt, pt)
for i in [0, 1, 2, 3, 4, 5, 6, 7]:
  data = [x.unsqueeze(0) for x in ds[i]]
  with torch.no_grad():
    out = model(input_ids=data[0].squeeze(0), token_type_ids=data[1].squeeze(0), attention_mask=data[2].squeeze(0))
  ans = evaluate(data, out, tok)
  print(f'--- dev {i}: windows {tuple(data[0].shape)} start_logits {tuple(out.start_logits.shape)}')
  print('   Q:', qs[i]['question_text'])
  print('   gold:', qs[i]['answer_text'], '| pred:', ans, '| EM', ans == qs[i]['answer_text'])
  if i == 0:
    print('   paragraph:', ps[qs[i]['paragraph_id']][:200])
