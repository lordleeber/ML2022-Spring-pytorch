#!/bin/bash
# NOTE: written with the Phase 0 version of hw12_exp.py (flat record format); kept as the Phase 0 record.
# HW12 Phase 0: sample on 21 training seeds, the no-exploration variant, and the random policy floor.
# Runs 20 CPU processes at a time; one JSON line per run in docs/tools/hw12_p0_runs.jsonl.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
P=.venv/bin/python
{
  echo "--algo random --seed 543 --tag p0_random"
  echo "--seed 543 --no_explore --tag p0_no_explore"
  for s in 543 $(seq 0 19); do echo "--seed $s --tag p0_seeds"; done
} | xargs -P 20 -L 1 sh -c "$P docs/tools/hw12_exp.py --out docs/tools/hw12_p0_runs.jsonl \$0 \"\$@\" >> docs/tools/hw12_logs/phase0.log 2>&1"
echo DONE >> docs/tools/hw12_logs/phase0.log
