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
# reference policies on the 100 evaluation seeds, read from docs/tools/hw12_facts.txt (written by hw12_facts.py)
REF = {}
for line in open(os.path.join(root, 'docs/tools/hw12_facts.txt')):
    m = re.match(r'(random|heuristic)\s+mean (\S+) .*>=200 (\S+) .*timeout (\d+)', line)
    if m:
        REF[m.group(1)] = {'mean': float(m.group(2)), 'solved': float(m.group(3)), 'timeout': int(m.group(4))}
HEURISTIC, RANDOM = REF['heuristic']['mean'], REF['random']['mean']


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


F_ITEMS = [('A2C GAE λ 0.99（第 5 章）', 'C_gae0.99', C['a2c']), ('＋隱藏層 64', 'D_h64', C['a2c']),
           ('＋熵 0.01', 'D_h64_ent', C['a2c']), ('＋梯度裁切 1', 'D_h64_ent_clip', C['a2c']),
           ('＋每批 10 局', 'D_h64_ent_clip_b10', C['a2c']), ('PPO 隱藏層 16', 'E_ppo_h16', C['ppo']),
           ('PPO 隱藏層 64', 'E_ppo', C['ppo']), ('PPO 64 ＋熵 0.01', 'E_ppo_ent', C['ppo']), ('DQN 隱藏層 64', 'E_dqn', C['dqn'])]


def ch06_scores():
    rows = []
    for label, tag, color in F_ITEMS:
        v = [r['snapshots']['steps']['closed_stochastic']['mean'] for r in runs(tag) if r['seed'] < 10]
        rows.append({'label': label, 'color': color, 'values': v, 'mean': float(np.mean(v))})
    rows.append({'label': '參考：手寫控制器', 'color': C['extra'], 'values': [HEURISTIC]})
    return charts.dot_rows(rows, -300, 300, [-200, -100, 0, 100, 200, 300], label_w=250, vfmt='{:.1f}',
                           xlabel='閉迴路 stochastic 平均總獎勵（訓練到 100 萬步；DQN 兩種量法相同；每個點是一個種子 0–9）',
                           aria='D、E 組訓練到 100 萬步後的閉迴路分數平均：' + '、'.join(f'{r["label"]} {r["mean"]:.1f}' for r in rows[:-1]))


def ch06_outcomes():
    return stacked_outcomes(F_ITEMS, 'steps', 'greedy')


def curve_on_steps(tag, grid):
    # each run's 20-episode-bin returns placed at the env step where the bin ends, interpolated onto `grid`, then averaged
    ys = []
    for r in runs(tag):
        if r['seed'] >= 10:
            continue
        xs, vs = np.array(r['curve_steps']), np.array(r['curve_return'])
        ys.append(np.interp(grid, xs, vs, left=np.nan, right=np.nan))
    return [None if np.isnan(v) else round(float(v), 1) for v in np.nanmean(ys, 0)]


def ch06_curves():
    grid = np.arange(50_000, 1_000_001, 50_000)
    items = [('範例', 'A_sample', C['sample']), ('reward-to-go', 'B_rtg0.99', C['rtg']),
             ('MC baseline', 'C_mc', C['base']), ('A2C GAE 0.99', 'C_gae0.99', C['a2c']),
             ('PPO＋熵', 'E_ppo_ent', C['ppo']), ('DQN', 'E_dqn', C['dqn'])]
    series = [dict(name=t, label=l, values=curve_on_steps(t, grid), color=c) for l, t, c in items]
    svg = charts.line_chart(series, len(grid), -250, 250, [-200, -100, 0, 100, 200], xlabel='環境步數', ylabel='訓練中每 20 局的平均總獎勵',
                            aria='訓練曲線（以環境步數為橫軸，種子 0–9 平均）。100 萬步時：' + '、'.join(
                                f'{s["label"]} {s["values"][-1]}' for s in series),
                            xticks=[(i, f'{(i + 1) * 5} 萬') for i in (1, 5, 9, 13, 17, 19)],
                            fmt='{:.1f}', height=360, xname=lambda i: f'{(i + 1) * 50_000:,} 步')
    return charts.legend(series) + '\n' + svg


def ch07_scatter():
    # every snapshot of every training run: x = closed-loop stochastic (100 unseen seeds), y = judge (5 episodes, seed 543)
    W2, H2, ML, MR, MT, MB = 900, 520, 64, 150, 20, 46
    x0, x1, y0, y1 = -450, 300, -450, 300
    pw, ph = W2 - ML - MR, H2 - MT - MB
    X = lambda v: ML + pw * (min(max(v, x0), x1) - x0) / (x1 - x0)
    Y = lambda v: MT + ph * (1 - (min(max(v, y0), y1) - y0) / (y1 - y0))
    groups = [('A 範例與旋鈕', 'A_', C['sample']), ('B 回報', 'B_', C['rtg']), ('C baseline／critic', 'C_', C['base']),
              ('D 技巧', 'D_', C['a2c']), ('E PPO', 'E_ppo', C['ppo']), ('E DQN', 'E_dqn', C['dqn'])]
    out = [f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="ARIA">']
    for t in (-400, -200, 0, 200):
        out.append(f'<line x1="{ML}" y1="{Y(t):.1f}" x2="{ML + pw}" y2="{Y(t):.1f}" stroke="#2a343f" stroke-width="1"/>')
        out.append(f'<text class="s-mono" x="{ML - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        out.append(f'<line x1="{X(t):.1f}" y1="{MT}" x2="{X(t):.1f}" y2="{MT + ph}" stroke="#2a343f" stroke-width="1"/>')
        out.append(f'<text class="s-mono" x="{X(t):.1f}" y="{MT + ph + 18}" text-anchor="middle">{t}</text>')
    out.append(f'<line x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" y2="{Y(y1):.1f}" stroke="#4a5562" stroke-width="1" stroke-dasharray="4 4"/>')
    out.append(f'<text class="s-sm" x="{ML + pw / 2:.1f}" y="{H2 - 6}" text-anchor="middle">閉迴路 stochastic（100 個沒看過的地形）</text>')
    out.append(f'<text class="s-sm" x="16" y="{MT + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {MT + ph / 2:.1f})">judge（種子 543，5 局）</text>')
    n = 0
    for gi, (label, prefix, color) in enumerate(groups):
        for r in RUNS:
            if not r['tag'].startswith(prefix):
                continue
            for k, s in r['snapshots'].items():
                xv, yv = s['closed_stochastic']['mean'], s['judge_mean']
                out.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="3" fill="{color}" opacity="0.75" '
                           f'data-tip="{charts._esc(r["tag"])} 種子 {r["seed"]}（{"2,000 局" if k == "ep" else "100 萬步"}）：閉迴路 {xv:.1f}／judge {yv:.1f}"/>')
                n += 1
        ly = MT + 16 + gi * 20
        out.append(f'<circle cx="{ML + pw + 18}" cy="{ly - 4}" r="4" fill="{color}"/>')
        out.append(f'<text class="s-lbl" x="{ML + pw + 28}" y="{ly}">{charts._esc(label)}</text>')
    out.append('</svg>')
    js = [s['judge_mean'] for r in RUNS if r['tag'][:2] in ('A_', 'B_', 'C_', 'D_', 'E_') for s in r['snapshots'].values()]
    cs = [s['closed_stochastic']['mean'] for r in RUNS if r['tag'][:2] in ('A_', 'B_', 'C_', 'D_', 'E_') for s in r['snapshots'].values()]
    aria = (f'全部 {n} 個快照的 judge 分數對閉迴路分數的散佈圖；虛線是兩者相等。兩者整體相關係數 {np.corrcoef(js, cs)[0, 1]:.3f}，'
            f'但個別點偏離對角線可達 {np.max(np.abs(np.array(js) - np.array(cs))):.0f} 分。超出 −450 到 300 的點畫在邊界上。')
    return '\n'.join(out).replace('ARIA', charts._esc(aria))


ALL_ROWS = [
    ('範例（瞬時獎勵、SGD 1e-3）', 'A_sample', 3), ('範例，不標準化', 'A_nonorm', 3), ('範例，學習率 1e-4', 'A_lr1e-4', 3),
    ('範例，學習率 1e-2', 'A_lr1e-2', 3), ('整局總報酬', 'B_total', 4), ('reward-to-go γ 0.9', 'B_rtg0.9', 4),
    ('reward-to-go γ 0.95', 'B_rtg0.95', 4), ('reward-to-go γ 0.99', 'B_rtg0.99', 4), ('reward-to-go γ 0.999', 'B_rtg0.999', 4),
    ('reward-to-go γ 1', 'B_rtg1.0', 4), ('reward-to-go 0.99，不標準化', 'B_rtg0.99_nonorm', 4),
    ('reward-to-go 0.99，Adam 3e-3', 'C_rtg_adam', 5), ('MC baseline（p.5）', 'C_mc', 5), ('MC，critic 5 步', 'C_mc_ci5', 5),
    ('MC，SGD 1e-3', 'C_mc_sgd', 5), ('A2C TD(0)', 'C_td0', 5), ('A2C 5-step', 'C_n5', 5), ('A2C 20-step', 'C_n20', 5),
    ('A2C GAE 0.95', 'C_gae0.95', 5), ('GAE 0.95，critic 5 步', 'C_gae0.95_ci5', 5), ('A2C GAE 0.99', 'C_gae0.99', 5),
    ('GAE 0.99 ＋隱藏層 64', 'D_h64', 6), ('＋熵 0.01', 'D_h64_ent', 6), ('＋梯度裁切 1', 'D_h64_ent_clip', 6),
    ('＋每批 10 局', 'D_h64_ent_clip_b10', 6), ('PPO 隱藏層 16', 'E_ppo_h16', 6), ('PPO 隱藏層 64', 'E_ppo', 6),
    ('PPO 64 ＋熵 0.01', 'E_ppo_ent', 6), ('DQN 隱藏層 64', 'E_dqn', 6)]


def ch08_table():
    head = ('<div class="tablewrap"><table>\n<thead><tr><th rowspan="2">設定（種子 0–9）</th><th rowspan="2">章</th>'
            '<th colspan="2">2,000 局</th><th colspan="6">100 萬環境步</th></tr>\n'
            '<tr><th class="num">步數（萬）</th><th class="num">stoch.</th><th class="num">局數</th><th class="num">judge</th>'
            '<th class="num">stochastic</th><th class="num">greedy</th><th class="num">greedy 截斷</th><th class="num">≥ 200</th></tr></thead>\n<tbody>\n')
    rows = []
    for label, tag, ch in ALL_ROWS:
        rs = [r for r in runs(tag) if r['seed'] < 10]
        e = [r['snapshots']['ep'] for r in rs]
        s = [r['snapshots']['steps'] for r in rs]
        m = lambda xs: float(np.mean(xs))
        sto = [x['closed_stochastic']['mean'] for x in s]
        b = (lambda t: f'<strong>{t}</strong>') if tag == 'E_ppo_ent' else (lambda t: t)
        rows.append(f'<tr><td>{b(label)}</td><td>{ch}</td>'
                    f'<td class="num">{m([x["env_steps"] for x in e]) / 1e4:.1f}</td>'
                    f'<td class="num">{m([x["closed_stochastic"]["mean"] for x in e]):.1f}</td>'
                    f'<td class="num">{m([x["episodes"] for x in s]):,.0f}</td>'
                    f'<td class="num">{m([x["judge_mean"] for x in s]):.1f}</td>'
                    f'<td class="num">{b(f"{m(sto):.1f} ± {np.std(sto):.1f}")}</td>'
                    f'<td class="num">{m([x["closed_greedy"]["mean"] for x in s]):.1f}</td>'
                    f'<td class="num">{m([x["closed_greedy"]["timeout"] for x in s]):.1f}</td>'
                    f'<td class="num">{m([x["closed_stochastic"]["solved_rate"] for x in s]) * 100:.0f}%</td></tr>')
    rnd = runs('A_random')[0]['snapshots']['ep']
    f = lambda v: f'{v:.1f}'.replace('-', '−')
    rows.append(f'<tr><td>參考：均勻隨機</td><td>1</td><td class="num">—</td><td class="num">{f(rnd["closed_stochastic"]["mean"])}</td>'
                f'<td class="num">—</td><td class="num">{f(rnd["judge_mean"])}</td><td class="num">{f(rnd["closed_stochastic"]["mean"])}</td>'
                f'<td class="num">—</td><td class="num">{rnd["closed_greedy"]["timeout"]}</td><td class="num">{rnd["closed_stochastic"]["solved_rate"] * 100:.0f}%</td></tr>')
    h = REF['heuristic']
    rows.append(f'<tr><td>參考：手寫控制器 <code>heuristic</code>（<code>hw12_facts.txt</code>）</td><td>0</td><td class="num">—</td><td class="num">—</td>'
                f'<td class="num">—</td><td class="num">—</td><td class="num">{f(h["mean"])}</td><td class="num">—</td>'
                f'<td class="num">{h["timeout"]}</td><td class="num">{h["solved"] * 100:.0f}%</td></tr>')
    return head + '\n'.join(rows) + '\n</tbody></table></div>'


CHARTS = {'ch00_curve': ('ch00.html', ch00_curve), 'ch00_seeds': ('ch00.html', ch00_seeds),
          'ch08_table': ('ch08.html', ch08_table),
          'ch07_scatter': ('ch07.html', ch07_scatter),
          'ch06_scores': ('ch06.html', ch06_scores), 'ch06_outcomes': ('ch06.html', ch06_outcomes),
          'ch06_curves': ('ch06.html', ch06_curves),
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
