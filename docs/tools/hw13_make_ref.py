# Build the HW13 "reference" script: the code cells of HW13/HW13.ipynb joined as-is,
# with only the changes needed to run as plain Python on this machine.
# usage: python docs/tools/hw13_make_ref.py HW13/HW13.ipynb <out.py>
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
      # shell commands (wget / tar): the data is already extracted locally
      lines.append('# [ref] ' + line)
    elif "torch.hub.load('pytorch/vision:v0.10.0', 'resnet18'" in line:
      # torch.hub downloads torchvision v0.10.0 source from GitHub; use the installed torchvision
      lines.append('# [ref] ' + line)
      lines.append("import torchvision; teacher_model = torchvision.models.resnet18(weights=None, num_classes=11)")
    else:
      lines.append(line)
  out.append(f'# ---- cell [{i}] ----\n' + '\n'.join(lines) + '\n')

open(sys.argv[2], 'w').write('\n'.join(out))
