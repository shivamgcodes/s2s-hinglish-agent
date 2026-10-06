#!/bin/bash
set -euo pipefail
N1_ROOT=${N1_ROOT:-/root/needle}; NEEDLE_VENV=${NEEDLE_VENV:-/root/venv-needle}
SH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n1/finetune/sweep (code); work dir = $N1_ROOT/finetune/sweep
C=${NEEDLE_COMMON:-$SH/../../../common}            # timed.py, tstamp.py, val_loss.py (were in $N1_ROOT/finetune)
cd $N1_ROOT/finetune && source $NEEDLE_VENV/bin/activate && source $SH/../env_pod2.sh
cd sweep
python $C/timed.py needle build $CKPT --lora adapter_e10.safetensors --layers 8 --out tuned_l8.cact 2>&1 | python $C/tstamp.py > build_l8.log
python $C/timed.py needle build $CKPT --lora adapter_e10.safetensors --out tuned_full.cact 2>&1 | python $C/tstamp.py > build_full.log
md5sum tuned_l8.cact tuned_full.cact adapter_e*.safetensors > md5s.txt
touch done_build
