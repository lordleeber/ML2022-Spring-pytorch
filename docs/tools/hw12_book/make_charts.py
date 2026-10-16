"""Build the HW12 book's charts from docs/tools/hw12_runs.jsonl (and the other hw12 records) and put each one into
its page at the marker pair <!-- CHART:name --> ... <!-- /CHART:name --> (the markers stay, so the script can be re-run).
usage: python docs/tools/hw12_book/make_charts.py <repo root> <chart name>...   (no names = all)"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import charts

root = sys.argv[1]
RUNS = [json.loads(l) for l in open(os.path.join(root, 'docs/tools/hw12_runs.jsonl'))]
C = {'sample': '#8b949e', 'rtg': '#3987e5', 'base': '#d95926', 'a2c': '#199e70', 'ppo': '#9085e9',
     'dqn': '#d55181', 'judge': '#c98500', 'extra': '#008300'}
HEURISTIC = 225.5   # docs/tools/hw12_facts.txt: Gymnasium's heuristic controller on the 100 evaluation seeds
RANDOM = -184.5     # same file: uniform random policy


def runs(tag):
    return [r for r in RUNS if r['tag'] == tag]


def snap_values(tag, snap, key):
    out = []
    for r in runs(tag):
        s = r['snapshots'].get(snap)
        if s is None:
            continue
        out.append(s['judge_mean'] if key == 'judge' else s['closed_' + key]['mean'])
    return out


def score_rows(items, snap, key, x0=-250, x1=300, aria='', xlabel=''):
    """items: list of (label, tag, color). One row per setting, one dot per training seed, white tick = mean."""
    rows = []
    for label, tag, color in items:
        v = snap_values(tag, snap, key)
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v)) if v else None})
    ticks = [t for t in (-200, -100, 0, 100, 200, 300) if x0 <= t <= x1]
    return charts.dot_rows(rows, x0, x1, ticks, xlabel=xlabel, aria=aria, label_w=230, vfmt='{:.1f}')


def ch00_curve():
    log = json.load(open(os.path.join(root, 'docs/tools/hw12_logs/ch00/train_log.json')))
    tot = log['avg_total_rewards']
    roll = [float(np.mean(tot[max(0, i - 19):i + 1])) for i in range(len(tot))]
    series = [dict(name='b', label='每批 5 局的平均', values=[round(v, 1) for v in tot], color='#3a4552'),
              dict(name='m', label='最近 20 批的平均', values=[round(v, 1) for v in roll], color='#c9d1d9')]
    n = len(tot)
    svg = charts.line_chart(series, n, -500, 300, [-400, -200, 0, 200], xlabel='第幾批（每批 5 局）', ylabel='平均總獎勵',
                            aria=f'範例（種子 543）400 批的訓練曲線。每批 5 局的平均總獎勵從第 1 批的 {tot[0]:.1f} 開始，'
                                 f'最近 20 批的平均在第 400 批是 {roll[-1]:.1f}；最後一批是 {tot[-1]:.1f}。',
                            xticks=[(i, str(i + 1)) for i in (0, 99, 199, 299, 399)], fmt='{:.1f}', height=340,
                            xname=lambda i: f'第 {i + 1} 批')
    return charts.legend(series) + '\n' + svg


def ch00_seeds():
    # the sample on 21 training seeds, three ways of scoring the same policies; references from hw12_facts.txt
    rows = []
    for label, snap, key, color in [('judge（固定 5 局）', 'ep', 'judge', C['judge']),
                                     ('閉迴路 stochastic（100 局）', 'ep', 'stochastic', C['sample']),
                                     ('閉迴路 greedy（100 局）', 'ep', 'greedy', C['sample']),
                                     ('訓練到 100 萬步，stochastic', 'steps', 'stochastic', C['sample'])]:
        v = snap_values('A_sample', snap, key)
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v))})
    rows.append({'label': '參考：隨機亂按', 'color': '#4a5562', 'values': [RANDOM]})
    rows.append({'label': '參考：手寫控制器', 'color': C['extra'], 'values': [HEURISTIC]})
    m = [r['mean'] for r in rows[:4]]
    return charts.dot_rows(rows, -250, 300, [-200, -100, 0, 100, 200, 300], label_w=230, vfmt='{:.1f}',
                           xlabel='平均總獎勵（每個點是一個訓練種子；白色短線是平均）',
                           aria=f'範例在 21 個訓練種子上的分數。訓練 2,000 局後：judge 平均 {m[0]:.1f}，閉迴路 stochastic 平均 {m[1]:.1f}，'
                                f'greedy 平均 {m[2]:.1f}；訓練到 100 萬步後 stochastic 平均 {m[3]:.1f}。參考線：隨機亂按 {RANDOM}、'
                                f'Gymnasium 的手寫控制器 {HEURISTIC}。')


def mean_curve(tag, bins=100, seeds=None):
    # mean over training seeds of the 20-episode-bin training return; the first `bins` bins = 2,000 episodes
    c = [r['curve_return'][:bins] for r in runs(tag) if seeds is None or r['seed'] in seeds]
    return [round(float(v), 1) for v in np.mean(c, 0)], len(c)


def ch03_curves():
    items = [('範例', 'A_sample', C['sample'], None), ('不標準化', 'A_nonorm', C['rtg'], None),
             ('學習率 ×0.1', 'A_lr1e-4', C['base'], None), ('學習率 ×10', 'A_lr1e-2', C['dqn'], None)]
    series = []
    for label, tag, color, dash in items:
        v, n = mean_curve(tag)
        series.append(dict(name=tag, label=label, values=v, color=color))
    aria = '訓練曲線，每 20 局平均總獎勵再對種子取平均，前 2,000 局。' + '；'.join(
        f'{s["label"]}：第 1 個 20 局 {s["values"][0]}、第 1,000 局附近 {s["values"][49]}、最後 {s["values"][-1]}' for s in series)
    svg = charts.line_chart(series, 100, -550, 0, [-500, -400, -300, -200, -100, 0], xlabel='訓練局數', ylabel='平均總獎勵',
                            aria=aria, xticks=[(i, str((i + 1) * 20)) for i in (0, 24, 49, 74, 99)], fmt='{:.1f}', height=340,
                            xname=lambda i: f'第 {i * 20 + 1}–{(i + 1) * 20} 局')
    return charts.legend(series) + '\n' + svg


def ch03_lr():
    rows = []
    for label, tag, color in [('學習率 ×0.1（1e-4）', 'A_lr1e-4', C['base']), ('範例（1e-3）', 'A_sample', C['sample']),
                              ('學習率 ×10（1e-2）', 'A_lr1e-2', C['dqn']), ('範例但不標準化', 'A_nonorm', C['rtg'])]:
        v = [r['snapshots']['ep']['closed_stochastic']['mean'] for r in runs(tag) if r['seed'] < 10]
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v))})
    rows.append({'label': '參考：永遠不動', 'color': '#4a5562', 'values': [-130.3]})
    rows.append({'label': '參考：永遠開主引擎', 'color': '#4a5562', 'values': [-883.0]})
    m = [r['mean'] for r in rows[:4]]
    return charts.dot_rows(rows, -900, 100, [-800, -600, -400, -200, 0], label_w=230, vfmt='{:.1f}',
                           xlabel='閉迴路 stochastic 平均總獎勵（訓練 2,000 局；每個點是一個種子 0–9）',
                           aria=f'學習率與標準化，各 10 個種子的閉迴路分數平均：學習率 1e-4 {m[0]:.1f}、範例 1e-3 {m[1]:.1f}、'
                                f'1e-2 {m[2]:.1f}、不標準化 {m[3]:.1f}。參考：永遠不動 −130.3、永遠開主引擎 −883.0。')


B_ITEMS = [('範例：瞬時（γ = 0）', 'A_sample', C['sample']), ('整局總報酬', 'B_total', '#4a5562'),
           ('reward-to-go γ = 0.9', 'B_rtg0.9', C['rtg']), ('γ = 0.95', 'B_rtg0.95', C['rtg']),
           ('γ = 0.99', 'B_rtg0.99', C['rtg']), ('γ = 0.999', 'B_rtg0.999', C['rtg']), ('γ = 1', 'B_rtg1.0', C['rtg'])]


def ch04_gamma():
    rows = []
    for label, tag, color in B_ITEMS:
        v = [r['snapshots']['steps']['closed_stochastic']['mean'] for r in runs(tag) if r['seed'] < 10]
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v))})
    return charts.dot_rows(rows, -300, 300, [-200, -100, 0, 100, 200, 300], label_w=230, vfmt='{:.1f}',
                           xlabel='閉迴路 stochastic 平均總獎勵（訓練到 100 萬步；每個點是一個種子 0–9）',
                           aria='訓練到 100 萬步後各種權重的閉迴路分數平均：' + '、'.join(f'{r["label"]} {r["mean"]:.1f}' for r in rows))


def stacked_outcomes(items, snap, key):
    """one horizontal bar per setting: mean landed / crashed / timeout out of 100 closed-loop episodes"""
    W2, label_w, mr, row_h, mt, mb = 900, 230, 60, 34, 34, 34
    h = mt + row_h * len(items) + mb
    pw = W2 - label_w - mr
    cols = [('降落', '#199e70'), ('墜毀', '#d55181'), ('截斷（撐滿 1,000 步）', '#c98500')]
    out = [f'<svg viewBox="0 0 {W2} {h}" role="img" aria-label="ARIA">']
    x = label_w
    for name, col in cols:
        out.append(f'<rect x="{x}" y="8" width="12" height="12" rx="2" fill="{col}"/>')
        out.append(f'<text class="s-sm" x="{x + 18}" y="18">{name}</text>')
        x += 190
    aria = []
    for i, (label, tag, _) in enumerate(items):
        ss = [r['snapshots'][snap]['closed_' + key] for r in runs(tag) if r['seed'] < 10]
        vals = [float(np.mean([s[k] for s in ss])) for k in ('landed', 'crashed', 'timeout')]
        y = mt + row_h * i
        out.append(f'<text class="s-lbl" x="{label_w - 12}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end">{charts._esc(label)}</text>')
        x = label_w
        for v, (name, col) in zip(vals, cols):
            wdt = pw * v / 100
            out.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{wdt:.1f}" height="{row_h - 12}" fill="{col}" '
                       f'data-tip="{charts._esc(label)}：{name} {v:.1f} 局"/>')
            if wdt > 34:
                out.append(f'<text class="s-mono" x="{x + wdt / 2:.1f}" y="{y + row_h / 2 + 4:.1f}" text-anchor="middle" style="fill:#0f1316">{v:.0f}</text>')
            x += wdt
        aria.append(f'{label} 降落 {vals[0]:.1f}、墜毀 {vals[1]:.1f}、截斷 {vals[2]:.1f}')
    out.append(f'<text class="s-sm" x="{label_w + pw / 2:.1f}" y="{h - 8}" text-anchor="middle">100 局閉迴路裡的結局（10 個種子的平均）</text>')
    out.append('</svg>')
    return '\n'.join(out).replace('ARIA', charts._esc('；'.join(aria)))


def ch04_outcomes():
    return stacked_outcomes(B_ITEMS, 'steps', 'greedy')


C_ITEMS = [('無 baseline（Adam）', 'C_rtg_adam', C['rtg']), ('MC baseline（p.5）', 'C_mc', C['base']),
           ('MC，critic 只練 5 步', 'C_mc_ci5', C['base']), ('MC，範例的 SGD', 'C_mc_sgd', C['base']),
           ('A2C TD(0)', 'C_td0', C['a2c']), ('A2C 5-step', 'C_n5', C['a2c']), ('A2C 20-step', 'C_n20', C['a2c']),
           ('A2C GAE λ 0.95', 'C_gae0.95', C['a2c']), ('GAE 0.95，critic 只練 5 步', 'C_gae0.95_ci5', C['a2c']),
           ('A2C GAE λ 0.99', 'C_gae0.99', C['a2c'])]


def ch05_scores():
    rows = []
    for label, tag, color in C_ITEMS:
        v = [r['snapshots']['steps']['closed_stochastic']['mean'] for r in runs(tag) if r['seed'] < 10]
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v))})
    rows.append({'label': '參考：手寫控制器', 'color': C['extra'], 'values': [HEURISTIC]})
    return charts.dot_rows(rows, -300, 300, [-200, -100, 0, 100, 200, 300], label_w=250, vfmt='{:.1f}',
                           xlabel='閉迴路 stochastic 平均總獎勵（訓練到 100 萬步；每個點是一個種子 0–9）',
                           aria='C 組訓練到 100 萬步後的閉迴路分數平均：' + '、'.join(f'{r["label"]} {r["mean"]:.1f}' for r in rows[:-1]))


def ch05_outcomes():
    return stacked_outcomes(C_ITEMS, 'steps', 'greedy')


CHARTS = {'ch00_curve': ('ch00.html', ch00_curve), 'ch00_seeds': ('ch00.html', ch00_seeds),
          'ch05_scores': ('ch05.html', ch05_scores), 'ch05_outcomes': ('ch05.html', ch05_outcomes),
          'ch04_gamma': ('ch04.html', ch04_gamma), 'ch04_outcomes': ('ch04.html', ch04_outcomes),
          'ch03_curves': ('ch03.html', ch03_curves), 'ch03_lr': ('ch03.html', ch03_lr)}


def put(page, name, content):
    path = os.path.join(root, 'docs/HW12', page)
    s = open(path).read()
    pat = re.compile(r'(<!-- CHART:%s -->)(.*?)(<!-- /CHART:%s -->)' % (name, name), re.S)
    if not pat.search(s):
        print('marker missing:', name, 'in', page)
        return
    s = pat.sub(lambda m: m.group(1) + '\n' + content + '\n' + m.group(3), s)
    open(path, 'w').write(s)
    print('put', name, '->', page)


for name in (sys.argv[2:] or CHARTS):
    page, fn = CHARTS[name]
    put(page, name, fn())
