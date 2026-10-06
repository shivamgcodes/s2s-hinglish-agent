#!/bin/bash
# CPU fallback for the remaining stages of a run (used when the GPU lock is held long by another job).
set -euo pipefail
CALLS=$1; ROOT=$2
A=/workspace/hinglish/audio
export HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
/workspace/venv-tts/bin/python $A/synth.py plan "$CALLS" "$ROOT"
for K in 0 1 2; do
  /workspace/venv-tts/bin/python $A/synth.py tts "$ROOT" --try $K
  /workspace/venv-asr/bin/python $A/asr.py "$ROOT" --device cpu --cpu-threads 48
done
/workspace/venv-tts/bin/python $A/synth.py build "$ROOT"
(cd /workspace/personaplex && /workspace/venv-pp/bin/python $A/align.py "$ROOT")
/workspace/venv-tts/bin/python $A/assemble.py "$CALLS" "$ROOT"
/workspace/venv-tts/bin/python $A/qc.py plots "$ROOT" 10
(cd /workspace/personaplex && /workspace/venv-pp/bin/python $A/qc.py mimi "$ROOT" 4)
echo "VARIANT DONE $ROOT"
