#!/bin/bash
# HW13 grid B2 (added with the user's approval 2026-10-09 15:25): the sample student under grid B's recipe,
# Run from HW13/:  bash ../docs/tools/hw13_run_grid_B2.sh <ckpt_dir> <T> <alpha>
# to separate the effect of the architecture from the training recipe. 200 epochs, HW03 augmentation, 16 workers.
set -u
CK=${1:?checkpoint dir}; T=${2:?T}; A=${3:?alpha}
LOG=../docs/tools/hw13_logs
mkdir -p "$LOG" "$CK"
PY=../.venv/bin/python
EXP=../docs/tools/hw13_exp.py
OUT=../docs/tools/hw13_runs.jsonl

run() {  # name, args...
  local name=$1; shift
  if grep -q "\"name\": \"$name\"" "$OUT" 2>/dev/null; then echo "skip $name"; return; fi
  echo "$(date +%T) start $name $*"
  $PY -u $EXP --name "$name" --nw 16 --epochs 200 --aug hw03 --save "$CK/$name" --jsonl "$OUT" "$@" > "$LOG/$name.out" 2> "$LOG/$name.err" \
    || { echo "$(date +%T) FAILED $name"; tail -5 "$LOG/$name.err"; exit 1; }
  echo "$(date +%T) done $(tail -1 "$LOG/$name.out")"
}

run "B200_sample_kd" --student sample --loss KD --T $T --alpha $A
run "B200_sample_ce" --student sample
echo "$(date +%T) ALL DONE"
