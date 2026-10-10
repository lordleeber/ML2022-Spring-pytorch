#!/bin/bash
# Controls for the full stack (ch06): own + pretrained surrogates with plain I-FGSM and MI-FGSM, after grid D.
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
cd $R
while ! grep -q ALLDONE docs/tools/hw10_logs/hw10_grid_D.progress 2>/dev/null; do sleep 20; done
docs/tools/hw10_run_grid.sh docs/tools/hw10_grid_C3.txt docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
