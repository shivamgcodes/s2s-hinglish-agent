#!/bin/bash
# Full audio pipeline for one variant. Resumable: re-run the same command after any interruption.
#   bash /workspace/hinglish/audio/run_variant.sh CALLS.jsonl ROOT [ASR_SHARDS=4]
# e.g. bash run_variant.sh /workspace/hinglish/data/V1/calls.jsonl /workspace/hinglish/data/V1
# Each GPU stage takes the GPU lock separately (short holds). Run inside tmux, log to ROOT/work/run.log.
# HOLDOUT=<holdout json> (D2 review): passed to assemble.py --holdout; unset = assemble default data/holdout.json (V1/V3/CONTROL).
#   V4 MUST use HOLDOUT=/workspace/hinglish/data/V4/holdout.json (assemble.py refuses V4 calls without val_scenarios).
# TTS backend: TTS_BACKEND=kokoro (default, V1) | f5cs (IndicF5 code-switch, venv-f5) | kokoro_en (English control).
#   e.g. TTS_BACKEND=f5cs bash run_variant.sh /workspace/hinglish/data/V3/calls.jsonl /workspace/hinglish/data/V3
#   The backend is fixed into ROOT/work/plan.json at plan time; changing it re-plans and wipes that call's chunks.
set -euo pipefail
CALLS=$1; ROOT=$2; SHARDS=${3:-4}
A=${AUDIO_DIR:-/workspace/hinglish/audio}
LOCK=/workspace/hinglish/gpu.lock
export HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/work"
export TTS_BACKEND=${TTS_BACKEND:-kokoro}
case "$TTS_BACKEND" in f5cs) TTS_PY=/workspace/venv-f5/bin/python ;; *) TTS_PY=/workspace/venv-tts/bin/python ;; esac
/workspace/venv-tts/bin/python $A/synth.py plan "$CALLS" "$ROOT"
for K in 0 1 2; do
  flock $LOCK $TTS_PY $A/synth.py tts "$ROOT" --try $K
  # review fix: bare `wait` returned 0 even when a shard crashed; wait on each pid and fail the stage
  flock $LOCK bash -c "pids=; for i in \$(seq 0 $((SHARDS-1))); do /workspace/venv-asr/bin/python $A/asr.py '$ROOT' --shard \$i --nshards $SHARDS & pids=\"\$pids \$!\"; done; rc=0; for p in \$pids; do wait \$p || rc=1; done; exit \$rc"
done
/workspace/venv-tts/bin/python $A/synth.py build "$ROOT"
(cd /workspace/personaplex && flock $LOCK /workspace/venv-pp/bin/python $A/align.py "$ROOT")
/workspace/venv-tts/bin/python $A/assemble.py "$CALLS" "$ROOT" ${INTERRUPT_MODE:+--interrupt-mode $INTERRUPT_MODE} ${HOLDOUT:+--holdout "$HOLDOUT"}
/workspace/venv-tts/bin/python $A/qc.py plots "$ROOT" 10
(cd /workspace/personaplex && flock $LOCK /workspace/venv-pp/bin/python $A/qc.py mimi "$ROOT" 4)
echo "VARIANT DONE $ROOT"
