#!/bin/bash
# V3 tests, reduced set (user 2026-10-04 ~02:00 IST): base_V3 and V3_A@best(step 200), 12 calls x seed 1001 + gate0 x1.
set -uo pipefail
H=${HINGLISH_ROOT:-/workspace/hinglish}; T=$H/tests; ST=$H/queue_v3.status; LOGD=$H/queue_v3_logs; mkdir -p $LOGD
SUB=air_03_g1,air_16_g4,cab_07_g3,cab_11_g4,cab_16_g2,ecom_07_g4,ecom_11_g1,food_07_g1,food_11_g2,food_12_g3,sub_07_g2,sub_11_g3
s() { echo "$(date -u +%FT%T) $*" >> $ST; }
s "QUEUE START"
s "START tests base_V3"; bash $T/run_tests.sh --tag base_V3 --variants V3 -K 1 --seeds 1001 --calls $SUB > $LOGD/tests_base_V3.log 2>&1; s "END tests base_V3 rc=$?"
AD=$(grep -m1 '^path=' /workspace/runs/V3_A/BEST_CKPT | cut -d= -f2-)
s "START tests V3_A adapter=$AD"; bash $T/run_tests.sh --tag V3_A --variants V3 -K 1 --seeds 1001 --calls $SUB --adapter $AD > $LOGD/tests_V3_A.log 2>&1; s "END tests V3_A rc=$?"
s "START asr"; ( cd $T && flock $H/gpu.lock /workspace/venv-asr/bin/python asr_pass.py base_V3 V3_A ) > $LOGD/asr.log 2>&1; s "END asr rc=$?"
s "START judge"; ( cd $T && flock $H/gpu.lock /workspace/venv-vllm/bin/python judge.py base_V3 V3_A ) > $LOGD/judge.log 2>&1; s "END judge rc=$?"
s "START score"; ( cd $T && /workspace/venv-pp/bin/python score.py base_V3 V3_A ) > $LOGD/score.log 2>&1; s "END score rc=$?"
s "QUEUE DONE"
