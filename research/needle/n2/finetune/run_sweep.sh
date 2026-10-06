#!/bin/bash
# N2 fine-tune sweep (runpod2 RTX 4090): N1 recipe (needle finetune, base needle3.safetensors, LoRA defaults,
# --val-split 0, seed 0) on two train sets (A = train.jsonl 902 rows, B = train_grounded.jsonl 852 rows), epochs 3/6/10.
set -euo pipefail
N2_ROOT=${N2_ROOT:-/root/n2}; GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
FC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/finetune (code); work dir = $N2_ROOT/finetune
C=${NEEDLE_COMMON:-$FC/../../common}               # timed.py, tstamp.py, val_loss.py (were in $N2_ROOT/finetune)
source $FC/../setup/env_needle.sh
F=$N2_ROOT/finetune; S=$F/sweep; D=$N2_ROOT/data/rows
cd $S
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 2000 > smi_sweep.csv &
SMI=$!
trap "kill $SMI 2>/dev/null || true" EXIT
for SET in A B; do
  if [ $SET = A ]; then DATA=$D/train.jsonl; else DATA=$D/train_grounded.jsonl; fi
  for N in 3 6 10; do
    T=${SET}_e$N
    [ -f done_$T ] && continue
    echo "START $T $(date -Is)" >> progress.txt
    flock $GPU_LOCK python $C/timed.py needle finetune $DATA --checkpoint $CKPT --epochs $N --max-len 1024 \
      --val-split 0 --seed 0 --checkpoint-dir ckpt_$T --out $S/adapter_$T.safetensors 2>&1 | python $C/tstamp.py > train_$T.log
    test -s adapter_$T.safetensors
    echo "DONE $T $(date -Is)" >> progress.txt; touch done_$T
  done
done
echo "START val $(date -Is)" >> progress.txt
for V in val val_clean val_opus; do
  flock $GPU_LOCK python $C/timed.py python $C/val_loss.py --checkpoint $CKPT --data $D/$V.jsonl --base \
    --adapter adapter_A_e3.safetensors adapter_A_e6.safetensors adapter_A_e10.safetensors \
              adapter_B_e3.safetensors adapter_B_e6.safetensors adapter_B_e10.safetensors \
    --out val_losses_$V.json 2>&1 | python $C/tstamp.py > val_loss_$V.log
done
touch done_val; echo "sweep finished $(date -Is)" >> progress.txt
