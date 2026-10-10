"""ch08 §8.1: the book's main results in one table (victims without defence, behind JPEG quality 80 and 31).
usage: python docs/tools/hw10_book/ladder.py <repo root>  (prints HTML)"""
import json
import os
import statistics as st
import sys

root = sys.argv[1]
J = {}
for l in open(os.path.join(root, 'docs/tools/hw10_jpeg.jsonl')):
  r = json.loads(l)
  J[(r['tag'], r['rate'])] = r
ROWS = [('（原圖，不攻擊）', '—', ['clean'], '—'),
        ('resnet110（範例）', 'FGSM', ['fgsm_eps8'], '2'),
        ('resnet110（範例）', 'I-FGSM（範例）', ['ifgsm_it20'], '4'),
        ('resnet110', 'I-FGSM 100 步', ['ifgsm_it100'], '4'),
        ('nin（40 個裡最好的單一代理）', 'I-FGSM', ['single_ifgsm_nin'], '3'),
        ('自訓 resnet20 第 40 epoch', 'I-FGSM', [f'u_resnet20_s{s}_e40' for s in range(3)], '5'),
        ('resnet110', 'MI-FGSM', ['mi_decay1.0'], '6'),
        ('resnet110', 'DIM-I-FGSM（p = 0.5）', [f'dim_ifgsm_p0.5_s{s}' for s in range(3)], '6'),
        ('resnet110', 'DIM-MI-FGSM', [f'dim_mi_p0.5_s{s}' for s in range(3)], '6'),
        ('notebook 的 3 個（logits 平均）', 'I-FGSM', ['ens_trio_logits_mean'], '5'),
        ('隨機 8 個預訓練（3 次抽樣）', 'I-FGSM', [f'ens_k8_d{d}' for d in range(3)], '5'),
        ('隨機 16 個預訓練（logits 平均）', 'I-FGSM', [f'ens_k16_d{d}_mean' for d in range(3)], '5'),
        ('自訓 6 個', 'DIM-MI-FGSM', [f'c2_u6_dimmi_s{s}' for s in range(3)], '6'),
        ('預訓練 8 個', 'DIM-MI-FGSM', [f'c2_k8_dimmi_s{s}' for s in range(3)], '6'),
        ('自訓 6 ＋ 預訓練 8', 'I-FGSM', ['c2_mix_ifgsm'], '6'),
        ('自訓 6 ＋ 預訓練 8', 'MI-FGSM', ['c2_mix_mi'], '6'),
        ('自訓 6 ＋ 預訓練 8', 'DIM-MI-FGSM', [f'c2_mix_dimmi_s{s}' for s in range(3)], '6'),
        ('JPEG 品質 80 ＋ resnet110（BPDA）', 'I-FGSM', ['d_bpda20_resnet110'], '7')]


def cell(tags, rate, key='victims_mean'):
  v = [J[(t, rate)][key] for t in tags]
  txt = f'{st.mean(v):.3f}' + (f'<br><small>{min(v):.3f}–{max(v):.3f}</small>' if len(v) > 1 else '')
  return f'<td class="num">{txt}</td>'


out = ['<div class="tablewrap"><table>',
       '<thead><tr><th>代理</th><th>攻擊</th><th class="num">無防禦：8 個受害者</th><th class="num">受害者 ensemble</th>'
       '<th class="num">JPEG 品質 80</th><th class="num">JPEG 品質 31</th><th class="num">章</th></tr></thead>', '<tbody>']
for name, atk, tags, ch in ROWS:
  out.append(f'<tr><td>{name}</td><td>{atk}</td>' + cell(tags, 0.0) + cell(tags, 0.0, 'victim_ens') + cell(tags, 20.0) + cell(tags, 70.0)
             + f'<td class="num">{ch}</td></tr>')
out += ['</tbody>', '</table></div>']
print('\n'.join(out))
