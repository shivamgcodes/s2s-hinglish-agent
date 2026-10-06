#!/bin/bash
# Think-block variant R: train_grounded rows + a short reasoning line (add_reasoning.py), so the target is
# <think>\n{reasoning}\n</think>\n<tool_call>..., the path the engine decodes. Same recipe otherwise; epochs 3/6/10.
set -euo pipefail
N2_ROOT=${N2_ROOT:-/root/n2}; GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
FC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/finetune (code); work dir = $N2_ROOT/finetune
C=${NEEDLE_COMMON:-$FC/../../common}               # timed.py, tstamp.py, val_loss.py (were in $N2_ROOT/finetune)
source $FC/../setup/env_needle.sh
F=$N2_ROOT/finetune; S=$F/sweep; D=$F/rows_r
cd $S
for N in 3 6 10; do
  T=R_e$N
  [ -f done_$T ] && continue
  echo "START $T $(date -Is)" >> progress.txt
  flock $GPU_LOCK python $C/timed.py needle finetune $D/train_grounded_r.jsonl --checkpoint $CKPT --epochs $N --max-len 1024 \
    --val-split 0 --seed 0 --checkpoint-dir ckpt_$T --out $S/adapter_$T.safetensors 2>&1 | python $C/tstamp.py > train_$T.log
  test -s adapter_$T.safetensors
  echo "DONE $T $(date -Is)" >> progress.txt; touch done_$T
done
for V in val val_clean val_opus; do
  flock $GPU_LOCK python $C/timed.py python $C/val_loss.py --checkpoint $CKPT --data $D/${V}_r.jsonl --base \
    --adapter adapter_R_e3.safetensors adapter_R_e6.safetensors adapter_R_e10.safetensors \
    --out val_losses_R_$V.json 2>&1 | python $C/tstamp.py > val_loss_R_$V.log
done
echo "R sweep finished $(date -Is)" >> progress.txt
