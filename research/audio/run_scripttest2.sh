#!/bin/bash
A=/workspace/hinglish/audio
export HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for TAG in b1r46 b1r50; do
  R=${TAG#b1r}; export F5_TEST_TAG=$TAG F5_RATE_MAX=${R:0:1}.${R:1:1}
  mkdir -p $A/f5_test_$TAG
  flock /workspace/hinglish/gpu.lock bash -c "/workspace/venv-f5/bin/python $A/f5_scripttest.py render > $A/f5_test_$TAG/render.log 2>&1; /workspace/venv-asr/bin/python $A/f5_scripttest.py score > $A/f5_test_$TAG/score.log 2>&1"
  echo "$(date -u +%FT%T) $TAG done" >> $A/f5_test/status2
done
echo "TEST2 DONE" >> $A/f5_test/status2
