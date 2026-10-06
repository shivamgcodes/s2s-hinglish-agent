#!/bin/bash
# D2 spec section 8 post-processing for the given tags (default: all 4 section-8 tags):
#   1) Trelis Whisper-Hinglish ASR, silence-trimmed -> <run>.trelis.json   (tests/trelis_pass.py, venv-tts, gpu.lock)
#   2) Gemma 4 31B, one load: r2 (.judge.json) + naturalness all 3 seeds (.nat.json) + call judge (.call.json)
#      (tests/judge.py --suite, venv-vllm, gpu.lock)
#   3) CPU scoring: tests/v4_eval.py -> out/<tag>/v4_scores.{json,csv}, v4_runs.*; tests/V4_EVAL.md (only when all 4 tags)
# Resumable: every step skips runs whose output file exists.
set -u
H=${HINGLISH_ROOT:-/workspace/hinglish}; T=$H/tests; L=$H/queue_v4_eval_logs; mkdir -p $L
TAGS="${*:-base_V4 V3_A200_V4 V4_A V4_A2}"; K=$(echo $TAGS | tr ' ' '+')
s(){ echo "$(date -u +%FT%H:%M:%SZ) $*" | tee -a $H/queue_v4_eval.status; }
S=$(date -u +%FT%H:%M); s "START trelis $TAGS"
cd $T && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 flock $H/gpu.lock /workspace/venv-tts/bin/python -u trelis_pass.py $TAGS >> $L/trelis_$K.log 2>&1
s "END trelis rc=$?"; echo "v4eval_trelis_$K,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
S=$(date -u +%FT%H:%M); s "START judge suite $TAGS"
cd $T && VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf flock $H/gpu.lock /workspace/venv-vllm/bin/python -u judge.py --suite --variant V4 --seed 1001,1002,1003 $TAGS >> $L/judge_$K.log 2>&1
s "END judge rc=$?"; echo "v4eval_judge_$K,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
MD=(); [ "$TAGS" = "base_V4 V3_A200_V4 V4_A V4_A2" ] && MD=(--md $T/V4_EVAL.md)
cd $T && python3 v4_eval.py $TAGS "${MD[@]}" >> $L/v4_eval_$K.log 2>&1
s "END v4_eval rc=$?"
