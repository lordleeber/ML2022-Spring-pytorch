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


def scatter(points, x0, x1, y0, y1, xticks, yticks, *, xlabel, ylabel, aria, height=420, diag=False):
  """points: list of {x, y, label, color, show (bool: print the label next to the dot)}"""
  W, ML, MR, MT, MB = 900, 64, 40, 20, 46
  pw, ph = W - ML - MR, height - MT - MB
  X = lambda v: ML + pw * (v - x0) / (x1 - x0)
  Y = lambda v: MT + ph * (1 - (v - y0) / (y1 - y0))
  out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{charts._esc(aria)}">']
  for t in yticks:
    out.append(f'<line x1="{ML}" y1="{Y(t):.1f}" x2="{ML + pw}" y2="{Y(t):.1f}" stroke="#2a343f" stroke-width="1"/>')
    out.append(f'<text class="s-mono" x="{ML - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
  for t in xticks:
    out.append(f'<line x1="{X(t):.1f}" y1="{MT}" x2="{X(t):.1f}" y2="{MT + ph}" stroke="#2a343f" stroke-width="1"/>')
    out.append(f'<text class="s-mono" x="{X(t):.1f}" y="{MT + ph + 18}" text-anchor="middle">{t}</text>')
  if diag:
    a, b = max(x0, y0), min(x1, y1)
    out.append(f'<line x1="{X(a):.1f}" y1="{Y(a):.1f}" x2="{X(b):.1f}" y2="{Y(b):.1f}" stroke="#4a5562" stroke-width="1" stroke-dasharray="4 4"/>')
  out.append(f'<text class="s-sm" x="{ML + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">{charts._esc(xlabel)}</text>')
  out.append(f'<text class="s-sm" x="16" y="{MT + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {MT + ph / 2:.1f})">{charts._esc(ylabel)}</text>')
  for p in points:
    out.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="5" fill="{p["color"]}" stroke="#181e25" stroke-width="1.5" data-tip="{charts._esc(p["label"])}：{p["x"]:.3f}／{p["y"]:.3f}"/>')
    if p.get('show'):
      if p.get('left'):
        out.append(f'<text class="s-lbl" x="{X(p["x"]) - 9:.1f}" y="{Y(p["y"]) + 4:.1f}" text-anchor="end">{charts._esc(p["label"])}</text>')
      else:
        out.append(f'<text class="s-lbl" x="{X(p["x"]) + 9:.1f}" y="{Y(p["y"]) + 4:.1f}">{charts._esc(p["label"])}</text>')
  out.append('</svg>')
  return '\n'.join(out)


def ch03_single():
  # every surrogate in the pool: FGSM vs I-FGSM, mean accuracy of the 8 victims
  pts = []
  for t in R:
    if t.startswith('single_fgsm_'):
      s = t[len('single_fgsm_'):]
      show = s in ('nin', 'resnet110', 'densenet40_k12_bc', 'resnet20', 'sepreresnet110', 'resnet1001')
      pts.append(dict(x=vavg(t), y=vavg('single_ifgsm_' + s), label=s, show=show, left=s in ('sepreresnet110', 'resnet1001'),
                      color=C['ifgsm'] if show else C['clean']))
  return scatter(pts, 0.55, 0.75, 0.30, 0.75, [0.55, 0.60, 0.65, 0.70, 0.75], [0.3, 0.4, 0.5, 0.6, 0.7],
                 xlabel='FGSM：8 個受害者的平均準確率', ylabel='I-FGSM：8 個受害者的平均準確率',
                 aria='40 個代理模型各自的轉移結果。橫軸是 FGSM、縱軸是 I-FGSM 時 8 個受害者的平均準確率，越低越好。點散得很開，兩者的相關只有 0.30。nin 在 I-FGSM 最低 0.362；densenet40_k12_bc 最高 0.681；resnet110 是 0.626 與 0.485')


def ch04_traj():
  # docs/tools/hw10_ch04.txt: resnet110 I-FGSM (step 0.8), PNG-equivalent images after each recorded step
  rows = [l.split() for l in open(os.path.join(root, 'docs/tools/hw10_ch04.txt')) if l[:1].isdigit()]
  steps = [int(r[0]) for r in rows]
  series = [dict(name='w', label='白箱 resnet110', values=[float(r[1]) for r in rows], color=C['ifgsm']),
            dict(name='v', label='8 個受害者平均', values=[float(r[2]) for r in rows], color=C['ifgsm'], dash='6 4'),
            dict(name='e', label='受害者 ensemble', values=[float(r[3]) for r in rows], color=C['ens'], dash='2 3')]
  svg = charts.line_chart(series, len(steps), 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel='I-FGSM 的步數（步長 0.8；橫軸不是等距）', ylabel='準確率',
                          aria='I-FGSM 的軌跡，代理 resnet110。白箱準確率在第 10 步降到 0.03、第 50 步降到 0；8 個受害者的平均在第 10 步 0.589、第 20 步 0.486、第 50 步 0.446、第 100 步 0.441，之後幾乎不再下降；受害者 ensemble 與受害者平均相近',
                          xticks=[(i, str(t)) for i, t in enumerate(steps)], fmt='{:.3f}', height=320, xname=lambda i: f'第 {steps[i]} 步')
  return charts.legend(series) + '\n' + svg


def ch05_epochs():
  # paper B: I-FGSM with our own checkpoints as surrogates, mean of the 8 victims, mean of 3 seeds
  ep = [1, 2, 3, 5, 10, 15, 20, 30, 40, 45, 60]
  cur = lambda a: [round(mean([vavg(f'u_{a}_s{s}_e{e}') for s in range(3)]), 4) for e in ep]
  series = [dict(name='r20', label='自訓 resnet20', values=cur('resnet20'), color=C['ifgsm']),
            dict(name='r56', label='自訓 resnet56', values=cur('resnet56'), color=C['ifgsm'], dash='6 4'),
            dict(name='p20', label='預訓 resnet20', values=[vavg('single_ifgsm_resnet20')] * len(ep), color=C['clean']),
            dict(name='p56', label='預訓 resnet56', values=[vavg('single_ifgsm_resnet56')] * len(ep), color=C['clean'], dash='6 4')]
  svg = charts.line_chart(series, len(ep), 0.4, 1.0, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xlabel='代理訓練到第幾個 epoch（第 30、45 個 epoch 之後學習率各乘 0.1；橫軸不是等距）',
                          ylabel='8 個受害者平均準確率', fmt='{:.3f}', height=340, xticks=[(i, str(e)) for i, e in enumerate(ep)], xname=lambda i: f'第 {ep[i]} 個 epoch',
                          aria='用自己訓練的 checkpoint 當代理做 I-FGSM，8 個受害者的平均準確率（3 個種子平均）。resnet20：第 1 個 epoch 0.879，一路下降到第 30 個 epoch 0.539，第 40 個 epoch 最低 0.435，之後回升到第 60 個 epoch 0.485。resnet56：0.924 降到第 40 個 epoch 0.438，第 60 個 epoch 回升到 0.588。pytorchcv 預訓練的 resnet20 是 0.535、resnet56 是 0.502')
  return charts.legend(series) + '\n' + svg


def ch07_quality():
  # docs/tools/hw10_jpeg.jsonl: victims behind JPEG of decreasing quality, mean of the 8 victims
  J = {}
  for l in open(os.path.join(root, 'docs/tools/hw10_jpeg.jsonl')):
    r = json.loads(l)
    J[(r['tag'], r['rate'])] = r['victims_mean']
  rates = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
  qs = ['無', '90', '80', '71', '61', '51', '41', '31', '22', '12']
  tags = [('clean', '原圖', C['clean'], None), ('fgsm_eps8', 'FGSM', C['fgsm'], None), ('ifgsm_it20', 'I-FGSM', C['ifgsm'], None),
          ('single_ifgsm_nin', 'I-FGSM（nin）', C['ifgsm'], '6 4')]
  series = [dict(name=t, label=lab, values=[round(J[(t, r)], 4) for r in rates], color=c, dash=d) for t, lab, c, d in tags]
  svg = charts.line_chart(series, len(rates), 0.3, 1.0, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xlabel='受害者前面的 JPEG 品質（越往右壓縮越重；31 = 作業的壓縮率 70）',
                          ylabel='8 個受害者平均準確率', fmt='{:.3f}', height=340, xticks=list(enumerate(qs)), xname=lambda i: f'JPEG 品質 {qs[i]}',
                          aria='受害者先做 JPEG 再分類。原圖：無 JPEG 0.962，品質 90 0.926、80 0.901、71 0.876、51 0.803、31 0.664、12 0.403。I-FGSM（resnet110）：0.485，品質 90 0.792、80 0.841、71 0.806、31 0.651。FGSM：0.626，品質 80 0.731，31 0.641。nin 的 I-FGSM：0.362，品質 80 0.603，31 0.563。品質 80 左右防禦效果最好、原圖代價又小')
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


TABLE = {'ch02_eps': ('ch02.html', ch02_eps), 'ch03_single': ('ch03.html', ch03_single), 'ch04_traj': ('ch04.html', ch04_traj), 'ch05_epochs': ('ch05.html', ch05_epochs), 'ch07_quality': ('ch07.html', ch07_quality)}
for name in sys.argv[2:]:
  page, fn = TABLE[name]
  put(page, name, fn())
