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
    series = [dict(name='b', label='每批 5 局的平均', values=[round(v, 1) for v in tot], color='#4a5562'),
              dict(name='m', label='最近 20 批的平均', values=[round(v, 1) for v in roll], color=C['sample'])]
    n = len(tot)
    svg = charts.line_chart(series, n, -500, 300, [-400, -200, 0, 200], xlabel='第幾批（每批 5 局）', ylabel='平均總獎勵',
                            aria=f'範例（種子 543）400 批的訓練曲線。每批 5 局的平均總獎勵從第 1 批的 {tot[0]:.1f} 開始，'
                                 f'最近 20 批的平均在第 400 批是 {roll[-1]:.1f}；最後一批是 {tot[-1]:.1f}。',
                            xticks=[(i, str(i + 1)) for i in (0, 99, 199, 299, 399)], fmt='{:.1f}', height=340,
                            xname=lambda i: f'第 {i + 1} 批')
    return charts.legend(series) + '\n' + svg


CHARTS = {'ch00_curve': ('ch00.html', ch00_curve)}


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
