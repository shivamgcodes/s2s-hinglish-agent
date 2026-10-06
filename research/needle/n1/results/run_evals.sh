#!/bin/bash
# N1 §5 eval on pod2: one chain per model, pinned to disjoint CPU slices (quota ~31 CPUs).
set -uo pipefail
N1_ROOT=${N1_ROOT:-/root/needle}; NEEDLE_VENV=${NEEDLE_VENV:-/root/venv-needle}
RC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # research/needle/n1 (code); the data/work dir is $N1_ROOT
export N1_WORKDIR=${N1_WORKDIR:-$N1_ROOT}
cd $N1_ROOT && source $NEEDLE_VENV/bin/activate
export CUDA_VISIBLE_DEVICES="" NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
tag=$1; cpus=$2; w=$3; shift 3
for t in "$@"; do
  if [ "$w" = base ]; then taskset -c $cpus python $RC/evaluate.py run --test $t --tag $tag > results/logs/${tag}_$t.log 2>&1
  else taskset -c $cpus python $RC/evaluate.py run --test $t --tag $tag --weights $w > results/logs/${tag}_$t.log 2>&1; fi
  echo "$t rc=$?" >> results/logs/${tag}_done.txt
done
