#!/bin/bash
# HW14 experiment grid, one run at a time (user decision 2026-10-08: A+B+C+D, sequential).
# Run from HW14/:  bash ../docs/tools/hw14_run_grid.sh <logdir>
# Each run trains ONE method in its own process, so seed 0 here differs from the
# notebook-order run (tag "ref"), where all six methods share one random stream.
set -u
LOG=${1:-logs}
mkdir -p "$LOG"
PY=../.venv/bin/python
EXP=../docs/tools/hw14_exp.py
OUT=../docs/tools/hw14_runs.jsonl
METHODS="baseline EWC MAS SI RWALK SCP"

run() {  # tag, args...
  local tag=$1; shift
  if grep -q "\"tag\": \"$tag\"" "$OUT" 2>/dev/null; then echo "skip $tag"; return; fi
  echo "$(date +%T) start $tag $*"
  $PY -u $EXP --tag "$tag" --out "$OUT" "$@" > "$LOG/$tag.out" 2> "$LOG/$tag.err" \
    || { echo "$(date +%T) FAILED $tag"; tail -5 "$LOG/$tag.err"; exit 1; }
  echo "$(date +%T) done $tag $(tail -1 "$LOG/$tag.out" | grep -o '[0-9.]*\]$')"
}

# A: 5 seeds x 6 methods
for s in 0 1 2 3 4; do for m in $METHODS; do run "A_${m}_s$s" --methods $m --seed $s; done; done
# B: lambda sweep, seed 0 (default lambda point = A_*_s0)
for l in 10 1000 10000; do run "B_EWC_l$l" --methods EWC --lam $l; done
for l in 0.001 0.01 1; do run "B_MAS_l$l" --methods MAS --lam $l; done
for l in 0.1 10 100; do run "B_SI_l$l" --methods SI --lam $l; done
for l in 10 1000 10000; do run "B_RWALK_l$l" --methods RWALK --lam $l; done
for l in 10 1000 10000; do run "B_SCP_l$l" --methods SCP --lam $l; done
# C: each task's importance counted once
for m in EWC MAS RWALK SCP; do run "C_${m}_latest" --methods $m --guards latest; done
# D: SCP with every slice squared separately
run "D_SCP_per_slice" --methods SCP --scp_per_slice
echo "$(date +%T) ALL DONE"
