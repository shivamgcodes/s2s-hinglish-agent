#!/bin/bash
# F5 script/duration test (S1-S8 x 4 voices x deva/mixed); each GPU stage under gpu.lock.
set -uo pipefail
A=/workspace/hinglish/audio
export HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p $A/f5_test
cd $A
echo "$(date -u +%FT%T) waiting for lock" > $A/f5_test/status
flock /workspace/hinglish/gpu.lock bash -c "echo \$(date -u +%FT%T) render START >> $A/f5_test/status; /workspace/venv-f5/bin/python $A/f5_scripttest.py render > $A/f5_test/render.log 2>&1; echo \$(date -u +%FT%T) render rc=\$? >> $A/f5_test/status; /workspace/venv-asr/bin/python $A/f5_scripttest.py score > $A/f5_test/score.log 2>&1; echo \$(date -u +%FT%T) score rc=\$? >> $A/f5_test/status"
echo "$(date -u +%FT%T) TEST DONE" >> $A/f5_test/status
