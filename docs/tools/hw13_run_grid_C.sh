#!/bin/bash
# HW13 grid C (ch06, added with the user's approval 2026-10-09): structured pruning of the sample student,
# fine-tuned vs the same narrow architecture trained from scratch. 20 epochs, CE, 16 workers.
# Run from HW13/:  bash ../docs/tools/hw13_run_grid_C.sh <ckpt_dir>
set -u
CK=${1:?checkpoint dir}
LOG=../docs/tools/hw13_logs
PY=../.venv/bin/python
EXP=../docs/tools/hw13_exp.py
OUT=../docs/tools/hw13_runs.jsonl
INIT=outputs/simple_baseline/student_best.ckpt

run() {  # name, args...
  local name=$1; shift
  if grep -q "\"name\": \"$name\"" "$OUT" 2>/dev/null; then echo "skip $name"; return; fi
  echo "$(date +%T) start $name $*"
  $PY -u $EXP --name "$name" --nw 16 --epochs 20 --save "$CK/$name" --jsonl "$OUT" "$@" > "$LOG/$name.out" 2> "$LOG/$name.err" \
    || { echo "$(date +%T) FAILED $name"; tail -5 "$LOG/$name.err"; exit 1; }
  echo "$(date +%T) done $(tail -1 "$LOG/$name.out")"
}

run C20_ft_keep75 --keep 0.75 --init_ckpt $INIT
run C20_ft_keep50 --keep 0.5 --init_ckpt $INIT
run C20_scratch_keep75 --student narrow75
run C20_scratch_keep50 --student narrow50
echo "$(date +%T) ALL DONE"
