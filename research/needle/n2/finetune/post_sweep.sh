#!/bin/bash
# After run_sweep.sh: build a 20-layer .cact per adapter (GPU lock), engine-decode val per .cact (CPU, one process),
# then the JAX plain-vs-think diagnostic on each adapter (GPU lock).
set -euo pipefail
N2_ROOT=${N2_ROOT:-/root/n2}; GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
FC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/finetune (code); work dir = $N2_ROOT/finetune
C=${NEEDLE_COMMON:-$FC/../../common}               # timed.py, tstamp.py, val_loss.py (were in $N2_ROOT/finetune)
source $FC/../setup/env_needle.sh
F=$N2_ROOT/finetune; S=$F/sweep; D=$N2_ROOT/data/rows
cd $S
for T in ${TAGS-A_e3 A_e6 A_e10 B_e3 B_e6 B_e10}; do
  if [ ! -s full_$T.cact ]; then
    flock $GPU_LOCK python $C/timed.py needle build $CKPT --lora adapter_$T.safetensors --out $S/full_$T.cact > build_$T.log 2>&1
  fi
  echo "BUILT $T $(date -Is)" >> post_progress.txt
done
mkdir -p $F/evals
for T in ${TAGS-A_e3 A_e6 A_e10 B_e3 B_e6 B_e10}; do
  [ -s $F/evals/engine_val_$T.json ] || python $C/timed.py python $FC/eval_engine.py --weights $S/full_$T.cact --rows $D/val --out $F/evals/engine_val_$T.json > $F/evals/engine_val_$T.log 2>&1
  echo "EVAL $T $(date -Is)" >> post_progress.txt
done
for T in ${DIAG_TAGS-A_e10 B_e10}; do
  [ -s $F/evals/diag_think_$T.json ] || flock $GPU_LOCK python $C/timed.py python $FC/diag_think_v2.py --checkpoint $CKPT --adapter adapter_$T.safetensors --out $F/evals/diag_think_$T.json > $F/evals/diag_think_$T.log 2>&1
  echo "DIAG $T $(date -Is)" >> post_progress.txt
done
echo "post finished $(date -Is)" >> post_progress.txt
