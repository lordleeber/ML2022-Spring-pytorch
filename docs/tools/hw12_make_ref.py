# Build the HW12 "reference" script: the code cells of HW12/HW12_EN.ipynb (2021 sample) joined as-is,
# with only the changes needed to run as plain Python on this machine. Each changed line keeps the
# original as a "# [ref]" comment. At the end it dumps what the split scripts must reproduce.
# usage: python docs/tools/hw12_make_ref.py HW12/HW12_EN.ipynb <out.py>
#        cd HW12 && python <out.py> <dump.pt>
import json
import sys

# exact line -> replacement (None = drop the line)
REPLACE = {
    # gym 0.18 is not installable on Python 3.12; HW12/gym_compat.py emulates its API on Gymnasium
    'import gym': 'import gym_compat as gym',
    # removed from torch; the 1.8 docs name use_deterministic_algorithms as its replacement
    '  torch.set_deterministic(True)': '  torch.use_deterministic_algorithms(True)',
    # NumPy 2 raises on ragged lists and on tensors that require grad; NumPy 1.19 built object arrays
    '    print(f"log_probs looks like ", np.shape(log_probs))     ':
        '    print(f"log_probs looks like ", (len(log_probs),))     ',
    'print("Action list\'s shape looks like ", np.shape(action_list))':
        'print("Action list\'s shape looks like ", np.shape(np.array(action_list, dtype=object)))',
    'np.save(PATH ,np.array(action_list)) ': 'np.save(PATH ,np.array(action_list, dtype=object)) ',
    # Colab only
    'from google.colab import files': None,
    'files.download(PATH)': None,
    'virtual_display = Display(visible=0, size=(1400, 900))': None,
    'virtual_display.start()': None,
    'from pyvirtualdisplay import Display': None,
    '%%capture': None,
    '%matplotlib inline': 'import matplotlib; matplotlib.use("Agg")',
    # the script has no notebook frontend
    'from tqdm.notebook import tqdm': 'from tqdm.auto import tqdm',
}


def rendering(line):
    # drawing frames does not touch the environment or any RNG; see gym_compat.py for render_mode
    s = line.strip()
    return 'render(' in s or s.startswith('display.') or s.startswith('img.set_data')


nb = json.load(open(sys.argv[1]))
out = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    lines = []
    for line in ''.join(cell['source']).split('\n'):
        if line.lstrip().startswith('!') or rendering(line):
            lines.append('# [ref] ' + line)
        elif line in REPLACE:
            lines.append('# [ref] ' + line)
            if REPLACE[line] is not None:
                lines.append(REPLACE[line])
        else:
            lines.append(line)
    out.append(f'# ---- cell [{i}] ----\n' + '\n'.join(lines) + '\n')

out.append('''# ---- [ref] dump ----
import sys as _sys
torch.save({
    'avg_total_rewards': avg_total_rewards,
    'avg_final_rewards': avg_final_rewards,
    'network': network.state_dict(),
    'action_list': [list(a) for a in action_list],
    'judge_rewards': test_total_reward,
}, _sys.argv[1])
''')
open(sys.argv[2], 'w').write('\n'.join(out))
