"""Build the HW07 book's charts from docs/tools/hw07_runs.jsonl and put each one into the page
at its marker <!-- CHART:name --> ... <!-- /CHART:name --> (the marker pair is kept, so the script can be re-run).
usage: python docs/tools/hw07_book/make_charts.py <repo root> <chart name>..."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import charts

root = sys.argv[1]
R = {}
for line in open(os.path.join(root, 'docs/tools/hw07_runs.jsonl')):
  r = json.loads(line)
  R[r['tag']] = r
C = {'base': '#8b949e', 'decay': '#3987e5', 'window': '#d95926', 'm1': '#199e70', 'm2': '#9085e9', 'm3': '#d55181', 'm4': '#c98500'}


def mean(xs):
  return sum(xs) / len(xs)


def ch05_loss():
  # epoch-1 training loss every 100 steps, mean of seeds 0-2: no decay vs linear decay
  def curve(prefix):
    hs = [[x[2] for x in R[f'{prefix}_s{s}']['history'] if x[0] == 1] for s in range(3)]
    return [round(mean(v), 4) for v in zip(*hs)]
  series = [dict(name='nd', label='不衰減', values=curve('nd3'), color=C['base']),
            dict(name='lin', label='線性衰減', values=curve('lin1'), color=C['decay'])]
  svg = charts.line_chart(series, 9, 0.4, 1.6, [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], xlabel='訓練步數（第 1 個 epoch）', ylabel='最近 100 步的平均 loss',
                          aria='第 1 個 epoch 的訓練 loss，三個種子的平均：不衰減與線性衰減在前 200 步接近，之後線性衰減一路較低，最後 100 步約 0.46，不衰減約 0.58',
                          xticks=[(i, str(100 * (i + 1))) for i in range(9)], fmt='{:.3f}', xname=lambda i: f'第 {100 * (i + 1)} 步')
  return charts.legend(series) + '\n' + svg


def ch05_em():
  # final dev EM (sample evaluation, stride 150) of every training setting, three seeds each
  def final(tag):
    return R[tag]['dev_by_epoch'][-1]
  rows = [
    dict(label='不衰減 1 epoch', color=C['base'], values=[R[f'nd3_s{s}']['dev_by_epoch'][0] for s in range(3)]),
    dict(label='不衰減 2 epoch', color=C['base'], values=[R[f'nd3_s{s}']['dev_by_epoch'][1] for s in range(3)]),
    dict(label='不衰減 3 epoch', color=C['base'], values=[R[f'nd3_s{s}']['dev_by_epoch'][2] for s in range(3)]),
  ]
  if all(f'lr5_s{s}' in R for s in range(3)):
    rows.append(dict(label='固定 lr 5e-5', color=C['m4'], values=[final(f'lr5_s{s}') for s in range(3)]))
  rows += [dict(label=f'線性衰減 {e} epoch', color=C['decay'], values=[final(f'lin{e}_s{s}') for s in range(3)]) for e in (1, 2, 3)]
  for r in rows:
    r['mean'] = mean(r['values'])
  return charts.dot_rows(rows, 0.40, 0.60, [0.40, 0.45, 0.50, 0.55, 0.60], xlabel='dev EM（範例的評估，stride 150）', vfmt='{:.3f}', label_w=170,
                         aria='各訓練設定的 dev EM，每個設定 3 個種子各一點、直線是平均：不衰減 1、2、3 epoch 平均 0.446、0.461、0.499；線性衰減 1、2、3 epoch 平均 0.552、0.550、0.546')


def ch05_epochs():
  # dev EM after each epoch, mean of seeds: 3 epochs without decay vs 3 epochs with linear decay
  nd = [round(mean([R[f'nd3_s{s}']['dev_by_epoch'][e] for s in range(3)]), 4) for e in range(3)]
  li = [round(mean([R[f'lin3_s{s}']['dev_by_epoch'][e] for s in range(3)]), 4) for e in range(3)]
  series = [dict(name='nd3', label='不衰減', values=nd, color=C['base']), dict(name='lin3', label='線性衰減（3 epoch）', values=li, color=C['decay'])]
  svg = charts.line_chart(series, 3, 0.40, 0.60, [0.40, 0.45, 0.50, 0.55, 0.60], xlabel='epoch', ylabel='dev EM（3 個種子平均）',
                          aria='每個 epoch 結束時的 dev EM：不衰減 0.446、0.461、0.499；3 個 epoch 的線性衰減 0.502、0.547、0.546',
                          xticks=[(i, str(i + 1)) for i in range(3)], fmt='{:.3f}', height=300)
  return charts.legend(series) + '\n' + svg


def put(page, name, html):
  path = os.path.join(root, 'docs/HW07', page)
  s = open(path).read()
  pat = re.compile(r'<!-- CHART:%s -->.*?<!-- /CHART:%s -->' % (name, name), re.S)
  new = f'<!-- CHART:{name} -->\n{html}\n<!-- /CHART:{name} -->'
  if pat.search(s):
    s = pat.sub(lambda m: new, s)
  else:
    s = s.replace(f'<!-- CHART:{name} -->', new, 1)
  open(path, 'w').write(s)
  print('chart', name, '->', page)


TABLE = {'ch05_loss': ('ch05.html', ch05_loss), 'ch05_em': ('ch05.html', ch05_em), 'ch05_epochs': ('ch05.html', ch05_epochs)}
for name in sys.argv[2:]:
  page, fn = TABLE[name]
  put(page, name, fn())
