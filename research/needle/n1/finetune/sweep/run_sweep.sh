#!/bin/bash
# N1 spec §4 sweep on pod2 GPU: separate --epochs 3/6/10 runs, then val_loss on val.jsonl, then builds of the best adapter.
set -euo pipefail
N1_ROOT=${N1_ROOT:-/root/needle}; NEEDLE_VENV=${NEEDLE_VENV:-/root/venv-needle}
SH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n1/finetune/sweep (code); work dir = $N1_ROOT/finetune/sweep
C=${NEEDLE_COMMON:-$SH/../../../common}            # timed.py, tstamp.py, val_loss.py (were in $N1_ROOT/finetune)
cd $N1_ROOT/finetune
source $NEEDLE_VENV/bin/activate
source $SH/../env_pod2.sh
S=$N1_ROOT/finetune/sweep
cd $S
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 2000 > smi_sweep.csv &
SMI=$!
trap "kill $SMI 2>/dev/null || true" EXIT
for N in 3 6 10; do
  echo "START e$N $(date -Is)" >> progress.txt
  python $C/timed.py needle finetune $N1_ROOT/train.jsonl --checkpoint $CKPT --epochs $N --max-len 1536 \
     --val-split 0 --seed 0 --checkpoint-dir ckpt_e$N --out $S/adapter_e$N.safetensors 2>&1 | python $C/tstamp.py > train_e$N.log
  test -s adapter_e$N.safetensors
  echo "DONE e$N $(date -Is)" >> progress.txt; touch done_e$N
done
echo "START val $(date -Is)" >> progress.txt
python $C/timed.py python $C/val_loss.py --checkpoint $CKPT --data $N1_ROOT/val.jsonl --base \
   --adapter adapter_e3.safetensors adapter_e6.safetensors adapter_e10.safetensors --out val_losses.json 2>&1 | python $C/tstamp.py > val_loss.log
test -s val_losses.json
echo "DONE val $(date -Is)" >> progress.txt; touch done_val
echo "sweep finished $(date -Is)" >> progress.txt
