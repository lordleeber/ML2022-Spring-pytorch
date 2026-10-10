#!/bin/bash
# paper B: after grid C finishes, train resnet20 / resnet56 from scratch (3 seeds, 60 epochs, checkpoints kept in HW10/surrogates/),
# then attack with every checkpoint (I-FGSM) through the normal grid runner.
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
cd $R
while ! grep -q ALLDONE docs/tools/hw10_logs/hw10_grid_C.progress 2>/dev/null; do sleep 20; done
P=docs/tools/hw10_logs/hw10_train.progress
for arch in resnet20_cifar10 resnet56_cifar10; do
  for seed in 0 1 2; do
    if grep -q "\"arch\": \"$arch\", \"seed\": $seed, \"epoch\": 60" docs/tools/hw10_train.jsonl 2>/dev/null; then continue; fi
    echo "$(date +%T) START $arch s$seed" >> $P
    (cd HW10 && ../.venv/bin/python ../docs/tools/hw10_train_surrogate.py --arch $arch --seed $seed --epochs 60 \
       --out_dir surrogates --jsonl ../docs/tools/hw10_train.jsonl < /dev/null > /dev/null 2>> ../docs/tools/hw10_logs/hw10_train.err) \
       && echo "$(date +%T) OK $arch s$seed" >> $P || echo "$(date +%T) FAILED $arch s$seed" >> $P
  done
done
echo "$(date +%T) ALLDONE" >> $P
: > docs/tools/hw10_grid_U.txt
for arch in resnet20_cifar10 resnet56_cifar10; do
  a=${arch%_cifar10}
  for seed in 0 1 2; do
    for e in 1 2 3 5 10 15 20 30 40 45 60; do
      printf "u_${a}_s${seed}_e${e}\t--attack ifgsm --surrogates ${arch}@$R/HW10/surrogates/${arch}_s${seed}_e${e}.pth\n" >> docs/tools/hw10_grid_U.txt
    done
  done
done
docs/tools/hw10_run_grid.sh docs/tools/hw10_grid_U.txt docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
