# Check that HW12/train.py + test.py + judge.py reproduce the reference script bit for bit.
# usage: python docs/tools/hw12_check_split.py <ref_dump.pt> <HW12 dir that ran train/test/judge> <judge stdout>
import json
import os
import re
import sys

import numpy as np
import torch

ref = torch.load(sys.argv[1], weights_only=False)
d = sys.argv[2]
log = json.load(open(os.path.join(d, 'outputs', 'train_log.json')))
net = torch.load(os.path.join(d, 'outputs', 'network.pt'))
actions = [list(a) for a in np.load(os.path.join(d, 'Action_List.npy'), allow_pickle=True)]
judge = re.findall(r'Your reward is : (-?[\d.]+)', open(sys.argv[3]).read())

checks = {
    'avg_total_rewards': log['avg_total_rewards'] == ref['avg_total_rewards'],
    'avg_final_rewards': log['avg_final_rewards'] == ref['avg_final_rewards'],
    'network': net.keys() == ref['network'].keys() and all(torch.equal(net[k], ref['network'][k]) for k in net),
    'action_list': actions == ref['action_list'],
    'judge_rewards (2 dp)': judge == ['%.2f' % r for r in ref['judge_rewards']],
}
for k, ok in checks.items():
    print(f'{k:22s} {"OK" if ok else "DIFFERENT"}')
print('episodes per test:', [len(a) for a in actions])
sys.exit(0 if all(checks.values()) else 1)
