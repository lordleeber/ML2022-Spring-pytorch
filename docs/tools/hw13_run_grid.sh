#!/bin/bash
# HW13 experiment grid A (knowledge distillation), one run at a time (user decision 2026-10-09).
# Run from HW13/:  bash ../docs/tools/hw13_run_grid.sh <ckpt_dir>
# Every run uses 16 DataLoader workers, so even the 10-epoch CE run differs from train.py
# (workers change the random stream); compare runs inside this grid only.
set -u
CK=${1:?checkpoint dir}
LOG=../docs/tools/hw13_logs
mkdir -p "$LOG" "$CK"
PY=../.venv/bin/python
EXP=../docs/tools/hw13_exp.py
OUT=../docs/tools/hw13_runs.jsonl

run() {  # name, args...
  local name=$1; shift
  if grep -q "\"name\": \"$name\"" "$OUT" 2>/dev/null; then echo "skip $name"; return; fi
  echo "$(date +%T) start $name $*"
  $PY -u $EXP --name "$name" --nw 16 --save "$CK/$name" --jsonl "$OUT" "$@" > "$LOG/$name.out" 2> "$LOG/$name.err" \
    || { echo "$(date +%T) FAILED $name"; tail -5 "$LOG/$name.err"; exit 1; }
  echo "$(date +%T) done $(tail -1 "$LOG/$name.out")"
}

# A: 10 epochs, CE vs KD (alpha 0.5, T 1 = the report's setting)
run A10_ce --epochs 10
run A10_kd_T1_a0.5 --epochs 10 --loss KD --T 1 --alpha 0.5
# A: 50 epochs, CE and KD over T x alpha
run A50_ce --epochs 50
for a in 0.5 0.9; do for T in 1 2 4 8; do
  run "A50_kd_T${T}_a$a" --epochs 50 --loss KD --T $T --alpha $a
done; done
echo "$(date +%T) ALL DONE"
