#!/bin/bash
# Run every line "tag<TAB>args" of a spec file through hw10_exp.py (deterministic mode), one at a time.
# Tags already in the jsonl are skipped, so the queue can be restarted. Output PNGs go to the scratch dir.
# usage: setsid nohup docs/tools/hw10_run_grid.sh <spec> <jsonl> <scratch_dir> &
SPEC=$1; JSONL=$2; OUT=$3
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
V=wrn28_10_cifar10,wrn40_8_cifar10,pyramidnet110_a48_cifar10,resnext29_32x4d_cifar10,ror3_110_cifar10,rir_cifar10,shakeshakeresnet26_2x32d_cifar10,diaresnet56_cifar10
SPEC=$(realpath $SPEC); JSONL=$(realpath -m $JSONL)
cd $R/HW10
LOG=$R/docs/tools/hw10_logs/$(basename $SPEC .txt)
while IFS=$'\t' read -r tag args; do
  [ -z "$tag" ] && continue
  [[ "$tag" == \#* ]] && continue
  if [ -f "$JSONL" ] && grep -q "\"tag\": \"$tag\"" "$JSONL"; then continue; fi
  echo "$(date +%T) START $tag" >> $LOG.progress
  rm -rf $OUT/$tag
  if CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.venv/bin/python ../docs/tools/hw10_det.py ../docs/tools/hw10_exp.py \
      --tag $tag $args --victims $V --jpeg 70 --out_dir $OUT/$tag --jsonl $JSONL < /dev/null > /dev/null 2>> $LOG.err; then
    echo "$(date +%T) OK $tag" >> $LOG.progress
  else
    echo "$(date +%T) FAILED $tag" >> $LOG.progress
  fi
done < $SPEC
echo "$(date +%T) ALLDONE" >> $LOG.progress
