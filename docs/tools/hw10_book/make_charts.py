"""Build the HW10 book's charts from docs/tools/hw10_runs.jsonl (and the per-chapter *.txt facts) and put each one
into the page at its marker <!-- CHART:name --> ... <!-- /CHART:name --> (the marker pair is kept, so the script can be re-run).
usage: python docs/tools/hw10_book/make_charts.py <repo root> <chart name>..."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import charts

root = sys.argv[1]
R = {}
for line in open(os.path.join(root, 'docs/tools/hw10_runs.jsonl')):
  r = json.loads(line)
  R[r['tag']] = r
C = {'clean': '#8b949e', 'fgsm': '#3987e5', 'ifgsm': '#d95926', 'mi': '#199e70', 'dim': '#9085e9', 'ens': '#d55181', 'jpeg': '#c98500'}
V = ['wrn28_10_cifar10', 'wrn40_8_cifar10', 'pyramidnet110_a48_cifar10', 'resnext29_32x4d_cifar10', 'ror3_110_cifar10',
     'rir_cifar10', 'shakeshakeresnet26_2x32d_cifar10', 'diaresnet56_cifar10']


def mean(xs):
  return sum(xs) / len(xs)


def vavg(tag, suffix=''):
  return round(mean([R[tag]['acc'][v + suffix] for v in V]), 4)


def white(tag):
  return round(R[tag]['acc'][R[tag]['surrogates'][0]], 4) if len(R[tag]['surrogates']) == 1 else None


def ch02_eps():
  # FGSM with resnet110: white-box (PNG) and mean of the 8 victims vs epsilon; random +-eps noise from hw10_ch02.txt
  tags = ['clean', 'fgsm_eps1', 'fgsm_eps2', 'fgsm_eps4', 'fgsm_eps8', 'fgsm_eps16']
  rnd = [0.950, 0.950, 0.945, 0.935, 0.845, 0.560]   # docs/tools/hw10_ch02.txt, rand_acc column (float images)
  series = [dict(name='w', label='白箱 resnet110', values=[white(t) if t != 'clean' else R[t]['acc']['resnet110_cifar10'] for t in tags], color=C['fgsm']),
            dict(name='b', label='8 個受害者平均', values=[vavg(t) for t in tags], color=C['fgsm'], dash='6 4'),
            dict(name='r', label='隨機 ±ε（白箱）', values=rnd, color=C['clean'])]
  svg = charts.line_chart(series, 6, 0.2, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], xlabel='ε（0–255 的尺度；橫軸不是等距）', ylabel='準確率',
                          aria='FGSM 的 ε 掃描，代理是 resnet110。白箱準確率：ε 0、1、2、4、8、16 依序 0.95、0.735、0.665、0.66、0.59、0.315。8 個受害者平均：0.963、0.891、0.843、0.778、0.626、0.267。隨機正負 ε 的雜訊在白箱上：0.95、0.95、0.945、0.935、0.845、0.56',
                          xticks=[(i, str(e)) for i, e in enumerate([0, 1, 2, 4, 8, 16])], fmt='{:.3f}', height=320,
                          xname=lambda i: f'ε = {[0, 1, 2, 4, 8, 16][i]}')
  return charts.legend(series) + '\n' + svg


def put(page, name, html):
  path = os.path.join(root, 'docs/HW10', page)
  s = open(path).read()
  pat = re.compile(r'<!-- CHART:%s -->.*?<!-- /CHART:%s -->' % (name, name), re.S)
  new = f'<!-- CHART:{name} -->\n{html}\n<!-- /CHART:{name} -->'
  if pat.search(s):
    s = pat.sub(lambda m: new, s)
  else:
    s = s.replace(f'<!-- CHART:{name} -->', new, 1)
  open(path, 'w').write(s)
  print('chart', name, '->', page)


TABLE = {'ch02_eps': ('ch02.html', ch02_eps)}
for name in sys.argv[2:]:
  page, fn = TABLE[name]
  put(page, name, fn())
