#!/bin/bash
# HW13 experiment grid B (architecture), one run at a time (user decision 2026-10-09).
# Run from HW13/:  bash ../docs/tools/hw13_run_grid_B.sh <ckpt_dir> <T> <alpha>
# T and alpha = the best KD setting of grid A (50 epochs). 200 epochs, HW03 augmentation, 16 workers.
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

run "B200_dw_kd" --student dw --loss KD --T $T --alpha $A
run "B200_dw_ce" --student dw
run "B200_mbv2_kd" --student mbv2 --loss KD --T $T --alpha $A
run "B200_mbv2_ce" --student mbv2
run "B200_plain_kd" --student plain --loss KD --T $T --alpha $A
echo "$(date +%T) ALL DONE"
