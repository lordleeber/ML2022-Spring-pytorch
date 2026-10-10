"""ch06 §6.5 table: combinations of surrogates and attacks, victims without defence and behind JPEG (hw10_jpeg.jsonl).
usage: python docs/tools/hw10_book/c2_table.py <repo root>  (prints HTML)"""
import json
import os
import statistics as st
import sys

root = sys.argv[1]
J = {}
for l in open(os.path.join(root, 'docs/tools/hw10_jpeg.jsonl')):
  r = json.loads(l)
  J[(r['tag'], r['rate'])] = r
R = {json.loads(l)['tag']: json.loads(l) for l in open(os.path.join(root, 'docs/tools/hw10_runs.jsonl'))}
rows = [('resnet110', 'I-FGSM', ['ifgsm_it20']),
        ('resnet110', 'DIM-MI-FGSM', [f'dim_mi_p0.5_s{s}' for s in range(3)]),
        ('自訓 6 個（第 40 epoch）', 'I-FGSM', ['c2_u6_ifgsm']),
        ('自訓 6 個（第 40 epoch）', 'MI-FGSM', ['c2_u6_mi']),
        ('自訓 6 個（第 40 epoch）', 'DIM-MI-FGSM', [f'c2_u6_dimmi_s{s}' for s in range(3)]),
        ('預訓練 8 個', 'I-FGSM', ['c2_k8_mean_ifgsm']),
        ('預訓練 8 個', 'MI-FGSM', ['c2_k8_mi']),
        ('預訓練 8 個', 'DIM-MI-FGSM', [f'c2_k8_dimmi_s{s}' for s in range(3)]),
        ('自訓 6 ＋ 預訓練 8', 'I-FGSM', ['c2_mix_ifgsm']),
        ('自訓 6 ＋ 預訓練 8', 'MI-FGSM', ['c2_mix_mi']),
        ('自訓 6 ＋ 預訓練 8', 'DIM-MI-FGSM', [f'c2_mix_dimmi_s{s}' for s in range(3)])]


def cell(tags, rate, key='victims_mean'):
  if not all((t, rate) in J for t in tags):
    return '<td class="num">—</td>'
  v = [J[(t, rate)][key] for t in tags]
  txt = f'{st.mean(v):.3f}' + (f'<br><small>{min(v):.3f}–{max(v):.3f}</small>' if len(v) > 1 else '')
  return f'<td class="num">{txt}</td>'


def white(tags):
  if not all(t in R for t in tags):
    return '<td class="num">—</td>'
  return f'<td class="num">{st.mean(st.mean(R[t]["acc"][s] for s in R[t]["surrogates"]) for t in tags):.3f}</td>'


out = ['<div class="tablewrap"><table>',
       '<thead><tr><th>代理</th><th>攻擊</th><th class="num">白箱（代理平均）</th><th class="num">無防禦：8 個受害者</th><th class="num">無防禦：受害者 ensemble</th>'
       '<th class="num">JPEG 品質 90</th><th class="num">JPEG 品質 80</th><th class="num">JPEG 品質 31（作業的壓縮率 70）</th></tr></thead>', '<tbody>',
       '<tr><td>（原圖，不攻擊）</td><td>—</td><td class="num">—</td>' + cell(['clean'], 0.0) + cell(['clean'], 0.0, 'victim_ens') + cell(['clean'], 10.0) + cell(['clean'], 20.0) + cell(['clean'], 70.0) + '</tr>']
for name, atk, tags in rows:
  out.append(f'<tr><td>{name}</td><td>{atk}</td>' + white(tags) + cell(tags, 0.0) + cell(tags, 0.0, 'victim_ens') + cell(tags, 10.0) + cell(tags, 20.0) + cell(tags, 70.0) + '</tr>')
out += ['</tbody>', '</table></div>']
print('\n'.join(out))
