# Build the HW10 "reference" script: the code cells of HW10/HW10.ipynb joined as-is,
# with only the changes needed to run as plain Python on this machine.
# usage: python docs/tools/hw10_make_ref.py HW10/HW10.ipynb <out.py>
import json
import sys

nb = json.load(open(sys.argv[1]))
out = ["import matplotlib; matplotlib.use('Agg')  # [ref] no display"]
for i, cell in enumerate(nb['cells']):
  if cell['cell_type'] != 'code':
    continue
  src = ''.join(cell['source'])
  if i in (24, 26):
    # ensembleNet.forward is an unfinished TODO (a syntax error); cell [26] uses it
    out.append(f'# ---- cell [{i}] skipped (unfinished TODO) ----\n')
    continue
  lines = []
  for line in src.split('\n'):
    s = line.lstrip()
    if s.startswith('!'):
      if s.startswith('!tar'):
        lines.append("import subprocess; subprocess.run('tar zcvf ../' + os.path.basename(os.getcwd()) + '.tgz *', shell=True)  # [ref] " + s)
      else:
        # pip / wget / unzip / rm: packages are installed and the data is already extracted locally
        lines.append('# [ref] ' + line)
    elif s.startswith('%cd'):
      lines.append(f"os.chdir('{s.split()[1]}')  # [ref] {s}")
    else:
      lines.append(line)
  out.append(f'# ---- cell [{i}] ----\n' + '\n'.join(lines) + '\n')

open(sys.argv[2], 'w').write('\n'.join(out))
