#!/bin/bash
# Clean timings on an otherwise idle GPU (run last): hw10.py (normal and deterministic mode) and a few attacks.
# Writes docs/tools/hw10_timing.txt. Nothing else may use the GPU meanwhile.
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
OUT=$R/docs/tools/hw10_timing.txt
T=/tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/timing; rm -rf $T; mkdir -p $T; cp $R/HW10/*.py $T/; ln -s $R/HW10/data $T/data
cd $T
echo "== $(date) gpu apps before: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" > $OUT
for i in 1 2; do
  rm -rf fgsm ifgsm *.tgz; s=$(date +%s.%N); $R/.venv/bin/python hw10.py > /dev/null 2>&1; e=$(date +%s.%N)
  echo "hw10.py normal mode run $i: $(echo "$e - $s" | bc) s" >> $OUT
  rm -rf fgsm ifgsm *.tgz; s=$(date +%s.%N); CUBLAS_WORKSPACE_CONFIG=:4096:8 $R/.venv/bin/python $R/docs/tools/hw10_det.py hw10.py > /dev/null 2>&1; e=$(date +%s.%N)
  echo "hw10.py deterministic run $i: $(echo "$e - $s" | bc) s" >> $OUT
done
for tag in ifgsm_it20 dim_mi_p0.5_s0 c2_u6_dimmi_s0 c2_k8_dimmi_s0 c2_mix_dimmi_s0; do
  args=$(grep -h "^$tag	" $R/docs/tools/hw10_grid_*.txt | head -1 | cut -f2)
  [ -z "$args" ] && args="--attack ifgsm"
  rm -rf $T/o_$tag
  $R/.venv/bin/python $R/docs/tools/hw10_exp.py --tag $tag $args --out_dir $T/o_$tag --jsonl $T/t.jsonl > /dev/null 2>&1
  echo "$tag attack_s (normal mode): $(tail -1 $T/t.jsonl | python3 -c 'import sys,json;print(json.load(sys.stdin)["attack_s"])')" >> $OUT
done
echo "== $(date) gpu apps after: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $OUT
echo DONE >> $OUT
