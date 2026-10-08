"""SVG charts for the HW14 book, generated from docs/tools/hw14_runs.jsonl.

Follows the dataviz skill: thin 2px lines, recessive grid, one y axis, direct labels at the
line ends plus a legend, and a hover layer (each x has an invisible band whose data-tip lists
every series' value; book.js shows it). Text uses the page's text classes, never series colors.
"""
import html

W, H = 900, 360
ML, MR, MT, MB = 56, 120, 20, 46   # margins: room for direct labels on the right


def _esc(s):
  return html.escape(str(s), quote=True)


def line_chart(series, n, y0, y1, yticks, *, xlabel, ylabel, aria, task_lines=True,
               xticks=None, height=H, fmt='{:.1f}', xname=lambda i: f'epoch {i + 1}'):
  """series: list of dicts {name, values (len n, None allowed), color, dash (optional), label (optional)}"""
  h = height
  pw, ph = W - ML - MR, h - MT - MB

  def X(i):
    return ML + pw * (i / (n - 1) if n > 1 else 0)

  def Y(v):
    return MT + ph * (1 - (v - y0) / (y1 - y0))

  out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{_esc(aria)}">']
  # grid + y ticks
  for t in yticks:
    y = Y(t)
    out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" stroke="#2a343f" stroke-width="1"/>')
    out.append(f'<text class="s-mono" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{_esc(t)}</text>')
  # task boundaries
  if task_lines:
    for k in range(1, 5):
      x = (X(10 * k - 1) + X(10 * k)) / 2
      out.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + ph}" stroke="#3a4552" stroke-width="1" stroke-dasharray="4 4"/>')
    for k in range(5):
      xc = (X(10 * k) + X(10 * k + 9)) / 2
      out.append(f'<text class="s-sm" x="{xc:.1f}" y="{MT + ph + 18}" text-anchor="middle">任務 {k + 1}</text>')
  if xticks:
    for i, lab in xticks:
      out.append(f'<text class="s-mono" x="{X(i):.1f}" y="{MT + ph + 18}" text-anchor="middle">{_esc(lab)}</text>')
  # axes
  out.append(f'<line x1="{ML}" y1="{MT + ph}" x2="{ML + pw}" y2="{MT + ph}" stroke="#4a5562" stroke-width="1"/>')
  out.append(f'<text class="s-sm" x="{ML + pw / 2:.1f}" y="{h - 6}" text-anchor="middle">{_esc(xlabel)}</text>')
  out.append(f'<text class="s-sm" x="14" y="{MT + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {MT + ph / 2:.1f})">{_esc(ylabel)}</text>')
  # lines
  ends = []
  for s in series:
    pts, seg = [], []
    for i, v in enumerate(s['values']):
      if v is None:
        if seg:
          pts.append(seg)
        seg = []
      else:
        seg.append(f'{X(i):.1f},{Y(v):.1f}')
    if seg:
      pts.append(seg)
    dash = f' stroke-dasharray="{s["dash"]}"' if s.get('dash') else ''
    for seg in pts:
      out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round"{dash}/>')
    last = max(i for i, v in enumerate(s['values']) if v is not None)
    ends.append([Y(s['values'][last]), s.get('label', s['name']), s['color'], X(last)])
  # direct labels, nudged apart
  ends.sort()
  for i in range(1, len(ends)):
    if ends[i][0] - ends[i - 1][0] < 14:
      ends[i][0] = ends[i - 1][0] + 14
  for y, lab, color, x in ends:
    out.append(f'<rect x="{x + 6:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="2" fill="{color}"/>')
    out.append(f'<text class="s-lbl" x="{x + 18:.1f}" y="{y + 4:.1f}">{_esc(lab)}</text>')
  # hover bands
  bw = pw / max(n - 1, 1)
  for i in range(n):
    vals = '　'.join(f'{s.get("label", s["name"])} {fmt.format(s["values"][i])}' for s in series if s['values'][i] is not None)
    out.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{MT}" width="{bw:.1f}" height="{ph}" fill="transparent" data-tip="{_esc(xname(i) + "：" + vals)}"/>')
  out.append('</svg>')
  return '\n'.join(out)


def legend(series):
  items = ''.join(f'<span><i style="background:{s["color"]}"></i>{_esc(s.get("label", s["name"]))}</span>' for s in series)
  return f'<div class="legend">{items}</div>'
