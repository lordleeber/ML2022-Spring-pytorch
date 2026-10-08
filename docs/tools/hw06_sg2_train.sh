#!/usr/bin/env bash
# HW06 StyleGAN2 run (Boss baseline suggestion): lucidrains stylegan2-pytorch 1.9.0 trained from
# scratch on HW06/faces at 64x64, batch 32, no gradient accumulation, 50,000 steps (~4.4 h on the
# RTX PRO 4000), a checkpoint (models/<name>/model_N.pt, N = step / 2500) and a sample grid every
# 2,500 steps. Everything else is the package default (network_capacity 16, lr 2e-4, ttur 1.5,
# mixed_prob 0.9, seed 42).
# Usage (repo root): bash docs/tools/hw06_sg2_train.sh <work_dir> <sg2lib> [name]
#   <sg2lib> holds stylegan2-pytorch and its deps installed with --no-deps, plus an empty
#   aim/__init__.py (the package imports aim at module level but only uses it with --log).
set -u
WORK=${1:?work dir}; LIB=${2:?sg2lib}; NAME=${3:-sg2}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK" && cd "$WORK"
PYTHONPATH="$LIB" "$ROOT/.venv/bin/python" -c "from stylegan2_pytorch.cli import main; main()" \
  --data "$ROOT/HW06/faces" --name "$NAME" --results_dir ./results --models_dir ./models \
  --image_size 64 --batch_size 32 --gradient_accumulate_every 1 \
  --num_train_steps 50000 --save_every 2500 --evaluate_every 2500
