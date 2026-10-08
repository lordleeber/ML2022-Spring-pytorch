# Plot function of HW14.ipynb: average accuracy per epoch of every method -> acc_summary.png
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_acc(acc_list, label_list, path='acc_summary.png'):
  for acc, label in zip(acc_list, label_list):
    # the notebook passes lineStyle='--', which Matplotlib 3.11 rejects (keyword is linestyle)
    plt.plot(acc, marker='o', linestyle='--', linewidth=2, markersize=4, label=label)
    plt.legend()
  plt.savefig(path)


if __name__ == '__main__':
  results = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'output/acc.json'))
  draw_acc(list(results.values()), list(results.keys()))
