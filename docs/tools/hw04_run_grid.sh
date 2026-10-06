#!/usr/bin/env bash
# HW04 ch08 experiments: one run at a time, all at train.py's schedule (70,000 steps)
# except long_sap_am (210,000 steps, run separately: bash ... <ckpt_dir> long_sap_am).
# med256_lr3e4 / med256_pre (why med256 fails) and tf160x4_pre (a pre-norm Transformer with as many
# parameters as conf160, ch06) are also run by name.
# Usage (repo root): bash docs/tools/hw04_run_grid.sh <ckpt_dir> [name ...]
# Appends one JSON line per run to docs/tools/hw04_ch08_runs.jsonl and the stdout to hw04_ch08_runs.txt.
set -u
OUT=${1:?ckpt dir}; shift
mkdir -p "$OUT"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
declare -A V=(
  [orig]=""
  [layers2]="--layers 2"
  [med160]="--d_model 160 --nhead 4 --ffn 512 --layers 2"
  [med256]="--d_model 256 --nhead 8 --ffn 1024 --layers 3"
  [conf160]="--arch conformer --d_model 160 --nhead 4 --ffn 640 --layers 2"
  [conf256]="--arch conformer --d_model 256 --nhead 4 --ffn 1024 --layers 3"
  [conf160_sap]="--arch conformer --d_model 160 --nhead 4 --ffn 640 --layers 2 --pool sap"
  [conf160_sap_am]="--arch conformer --d_model 160 --nhead 4 --ffn 640 --layers 2 --pool sap --loss amsm"
  [seg256]="--seg 256"
  [med256_lr3e4]="--d_model 256 --nhead 8 --ffn 1024 --layers 3 --lr 3e-4"
  [med256_pre]="--d_model 256 --nhead 8 --ffn 1024 --layers 3 --norm_first 1"
  [tf160x4_pre]="--d_model 160 --nhead 4 --ffn 640 --layers 4 --norm_first 1"
  [long_sap_am]="--arch conformer --d_model 160 --nhead 4 --ffn 640 --layers 2 --pool sap --loss amsm --steps 210000"
)
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(orig layers2 med160 med256 conf160 conf256 conf160_sap conf160_sap_am seg256)
cd "$ROOT/HW04"
for n in "${NAMES[@]}"; do
  echo "=== $n ${V[$n]} ($(date +%T))" | tee -a "$ROOT/docs/tools/hw04_ch08_runs.txt"
  PYTHONPATH=. "$ROOT/.venv/bin/python" "$ROOT/docs/tools/hw04_exp.py" --name "$n" ${V[$n]} \
      --save "$OUT/$n.ckpt" --dump "$OUT/$n.npz" > "$OUT/$n.out" 2> "$OUT/$n.err" || { echo "FAILED $n"; tail -5 "$OUT/$n.err"; exit 1; }
  grep -v '^{' "$OUT/$n.out" >> "$ROOT/docs/tools/hw04_ch08_runs.txt"
  tail -1 "$OUT/$n.out" | sed "s#$OUT#<ckpt_dir>#g" >> "$ROOT/docs/tools/hw04_ch08_runs.jsonl"
  echo "done $n ($(date +%T))"
done
