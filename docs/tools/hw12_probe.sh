#!/bin/bash
# HW12 Phase 0 probe: the TA's "medium" hint (discounted reward-to-go, gamma 0.99) and the whole-episode
# return, 10 training seeds each, against the sample's 21 seeds; plus a re-check that the default path
# still matches the reference script.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
P=.venv/bin/python
S=${HW12_SCRATCH:?}
{
  echo "--seed 543 --tag p0_verify2 --dump $S/exp_dump2.pt"
  for s in $(seq 0 9); do echo "--seed $s --returns rtg --gamma 0.99 --tag p0_probe"; done
  for s in $(seq 0 9); do echo "--seed $s --returns total --tag p0_probe"; done
} | xargs -P 21 -L 1 sh -c "$P docs/tools/hw12_exp.py --out docs/tools/hw12_p0_runs.jsonl \$0 \"\$@\" >> docs/tools/hw12_logs/probe.log 2>&1"
echo DONE >> docs/tools/hw12_logs/probe.log
