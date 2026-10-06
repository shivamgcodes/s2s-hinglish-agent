#!/bin/bash
# usage: run_driver_lora.sh <spec.json> <out_dir> <log> [ADAPTER|none]
# GPU job: call it under  flock /workspace/hinglish/gpu.lock
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
source /workspace/venv-pp/bin/activate
cd /workspace/personaplex
python -u /workspace/hinglish/infer/driver_lora.py "$1" "$2" "${4:-none}" 2>&1 | tee "$3"
echo "DRIVER EXIT ${PIPESTATUS[0]}" | tee -a "$3"
