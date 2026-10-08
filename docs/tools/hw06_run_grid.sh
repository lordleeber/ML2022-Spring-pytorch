#!/usr/bin/env bash
# HW06 experiments: one run at a time, each 100 epochs (train.py's config), then FID/AFD of every
# saved generator (epoch 1, 5, 10, ..., 100) with hw06_eval.py.
# Usage (repo root): bash docs/tools/hw06_run_grid.sh <out_dir> <pylib> <cascade.xml> [name ...]
#   <out_dir>/stats must already hold real64.npz / real96.npz (hw06_eval.py real).
# Appends one JSON line per run to docs/tools/hw06_runs.jsonl, and one per checkpoint to
# docs/tools/hw06_eval.jsonl; the per-step log of each run is copied to docs/tools/hw06_logs/<name>.jsonl.
set -u
OUT=${1:?out dir}; PYLIB=${2:?pylib}; CASCADE=${3:?cascade xml}; shift 3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
declare -A V=(
  [gan]=""
  [wgan]="--model_type WGAN --opt rmsprop --lr 5e-5 --clip 0.01 --n_critic 5"
  [wgangp]="--model_type WGANGP --norm in --lr 1e-4 --beta1 0 --beta2 0.9 --gp_lambda 10 --n_critic 5"
  [wgan_sigmoid]="--model_type WGAN --sigmoid 1 --clip 0"
  [wgan_c1]="--model_type WGAN --opt rmsprop --lr 5e-5 --clip 0.01 --n_critic 1"
)
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(gan wgan wgangp wgan_sigmoid)
mkdir -p "$ROOT/docs/tools/hw06_logs"
cd "$ROOT/HW06"
for n in "${NAMES[@]}"; do
  echo "=== $n ${V[$n]} ($(date +%T))"
  PYTHONPATH=. "$ROOT/.venv/bin/python" "$ROOT/docs/tools/hw06_exp.py" --name "$n" --out "$OUT/$n" ${V[$n]} \
      > "$OUT/$n.out" 2> "$OUT/$n.err" || { echo "FAILED $n"; tail -5 "$OUT/$n.err"; exit 1; }
  tail -1 "$OUT/$n.out" | sed "s#$OUT#<out_dir>#g" >> "$ROOT/docs/tools/hw06_runs.jsonl"
  cp "$OUT/$n/log.jsonl" "$ROOT/docs/tools/hw06_logs/$n.jsonl"
  echo "trained $n ($(date +%T))"
  PYTHONPATH=.:"$PYLIB" "$ROOT/.venv/bin/python" "$ROOT/docs/tools/hw06_eval.py" gen --stats "$OUT/stats" \
      --cascade "$CASCADE" $(ls -v "$OUT/$n"/G_*.pth) > "$OUT/$n.eval" 2> "$OUT/$n.eval.err" \
      || { echo "EVAL FAILED $n"; tail -5 "$OUT/$n.eval.err"; exit 1; }
  sed "s#$OUT/##g" "$OUT/$n.eval" >> "$ROOT/docs/tools/hw06_eval.jsonl"
  echo "evaluated $n ($(date +%T))"
done
