# Build the HW07 "reference" script: the code cells of HW07.ipynb joined as-is,
# with only the changes needed to run as plain Python on this machine.
# usage: python docs/tools/hw07_make_ref.py ~/poyi/GitHubPublic/ML2022-Spring/HW07/HW07.ipynb <out.py>
# Run the output with HW07/ on PYTHONPATH (for legacy_adamw) and hw7_*.json in the working directory.
import json
import sys

nb = json.load(open(sys.argv[1]))
out = []
for i, cell in enumerate(nb['cells']):
  if cell['cell_type'] != 'code':
    continue
  src = ''.join(cell['source'])
  lines = []
  for line in src.split('\n'):
    if line.lstrip().startswith('!'):
      # shell commands (gdown / unzip / pip / nvidia-smi): the data is already local
      lines.append('# [ref] ' + line)
    elif line == 'from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast':
      # transformers 5.x removed AdamW; use the verbatim copy of the old class
      lines.append('# [ref] ' + line)
      lines.append('from legacy_adamw import AdamW')
      lines.append('from transformers import BertForQuestionAnswering, BertTokenizerFast')
    else:
      lines.append(line)
  out.append(f'# ---- cell [{i}] ----\n' + '\n'.join(lines) + '\n')

open(sys.argv[2], 'w').write('\n'.join(out))
