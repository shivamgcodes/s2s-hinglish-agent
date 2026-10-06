#!/bin/bash
# D2 spec section 7 re-rank: PersonaPlex tests (12-call reduced V4 subset, seed 1001, no gate0) for the top-3
# val-loss checkpoints of V4_A and V4_A2, then the Gemma naturalness judge. run_tests.sh / flock take gpu.lock.
set -u
H=${HINGLISH_ROOT:-/workspace/hinglish}; T=$H/tests; L=$H/queue_v4_rerank_logs; mkdir -p $L
SUBV4="air_11_g2,air_26_g3,bank_07_g1,bank_20_g4,cab_03_g1,cab_26_g4,ecom_11_g3,ecom_23_g1,food_03_g2,food_07_g3,sub_11_g4,tel_03_g2"
s(){ echo "$(date -u +%FT%H:%M:%SZ) $*" | tee -a $H/queue_v4_rerank.status; }
s "QUEUE START"
TAGS=""
for rc in V4_A:400 V4_A:350 V4_A:300 V4_A2:600 V4_A2:500 V4_A2:450; do
  r=${rc%%:*}; c=${rc##*:}; tag=rr4_${r}_s$c; TAGS="$TAGS $tag"
  AD=/workspace/runs/$r/checkpoints/checkpoint_$(printf %06d $c)/consolidated
  S=$(date -u +%FT%H:%M); s "START tests $tag adapter=$AD"
  bash $T/run_tests.sh --tag $tag --variants V4 -K 1 --seeds 1001 --calls $SUBV4 --no-gate0 --adapter $AD > $L/tests_$tag.log 2>&1
  rc2=$?; s "END tests $tag rc=$rc2"; echo "rr4_tests_$tag,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
done
S=$(date -u +%FT%H:%M); s "START judge naturalness$TAGS"
cd $T && VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf flock $H/gpu.lock /workspace/venv-vllm/bin/python judge.py --naturalness --variant V4 --seed 1001 $TAGS > $L/judge_nat.log 2>&1
s "END judge rc=$?"; echo "rr4_judge_nat,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
s "QUEUE DONE"
