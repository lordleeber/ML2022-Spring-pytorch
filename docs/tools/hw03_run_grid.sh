#!/bin/bash
# HW03 ch08 runs, one at a time. Run from the repo root:  bash docs/tools/hw03_run_grid.sh <ckpt_dir>
# Each run appends its JSON line to docs/tools/hw03_ch08_runs.jsonl and its stdout to <ckpt_dir>/<name>.log.
# Step 0 checks that hw03_exp.py with default arguments prints exactly what train.py printed.
set -e
CK=${1:?ckpt dir}; mkdir -p "$CK"
cd HW03
run() {  # name, args...
  n=$1; shift
  PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw03_exp.py --name "$n" --save "$CK/$n.ckpt" --dump "$CK/$n.npz" "$@" > "$CK/$n.log" 2>&1
  tail -1 "$CK/$n.log" >> ../docs/tools/hw03_ch08_runs.jsonl
}
if [ -n "$BASELINE" ]; then
  run verify5
  grep -E '^\[ (Train|Valid)|^Best' "$CK/verify5.log" > "$CK/verify5.lines"
  diff "$BASELINE" "$CK/verify5.lines" && echo "verify5: identical to train.py"
fi
run base40   --epochs 40
run augA40   --epochs 40 --aug A
run res40    --epochs 40 --aug A --arch res
run resplit40 --epochs 40 --aug A --resplit 1
run ls40     --epochs 40 --aug A --ls 0.1
run long200  --epochs 200 --aug A
