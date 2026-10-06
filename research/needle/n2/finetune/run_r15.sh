#!/bin/bash
# R_e15: extends the R sweep (R val loss and correct still improving at e10); nvidia-smi sampled for VRAM.
set -euo pipefail
N2_ROOT=${N2_ROOT:-/root/n2}; GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
FC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/finetune (code); work dir = $N2_ROOT/finetune
C=${NEEDLE_COMMON:-$FC/../../common}               # timed.py, tstamp.py, val_loss.py (were in $N2_ROOT/finetune)
source $FC/../setup/env_needle.sh
F=$N2_ROOT/finetune; S=$F/sweep; D=$F/rows_r; T=R_e15
cd $S
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 2000 > smi_$T.csv &
SMI=$!; trap "kill $SMI 2>/dev/null || true" EXIT
echo "START $T $(date -Is)" >> progress.txt
flock $GPU_LOCK python $C/timed.py needle finetune $D/train_grounded_r.jsonl --checkpoint $CKPT --epochs 15 --max-len 1024 \
  --val-split 0 --seed 0 --checkpoint-dir ckpt_$T --out $S/adapter_$T.safetensors 2>&1 | python $C/tstamp.py > train_$T.log
test -s adapter_$T.safetensors; echo "DONE $T $(date -Is)" >> progress.txt
flock $GPU_LOCK python $C/timed.py python $C/val_loss.py --checkpoint $CKPT --data $D/val_r.jsonl \
  --adapter adapter_R_e10.safetensors adapter_$T.safetensors --out val_losses_R_val_e15.json 2>&1 | python $C/tstamp.py > val_loss_R_val_e15.log
TAGS="$T" DIAG_TAGS="" bash $FC/post_sweep.sh
echo "R15 finished $(date -Is)" >> progress.txt
