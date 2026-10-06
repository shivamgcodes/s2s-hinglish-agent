#!/bin/bash
# User 2026-10-05 ~10:30 IST: "just do single seeds from the next time ... and do single seeds now too".
# V4_A2 3-seed run stopped at 04:59Z (rc 143); finish V4_A2 with seed 1001 only (run_tests resumes: existing s1001 runs
# are skipped), then the same post-processing chain as queue_v4_eval_chain.sh. base_V4 / V3_A200_V4 / V4_A already have
# 3 seeds; comparisons across all four tags must use seed 1001 (3-seed numbers only as extra for those three).
H=${HINGLISH_ROOT:-/workspace/hinglish}; T=$H/tests; L=$H/queue_v4_eval_logs
s(){ echo "$(date -u +%FT%H:%M:%SZ) $*" | tee -a $H/queue_v4_eval.status; }
AD=$(sed -n "s/^path=//p" /workspace/runs/V4_A2/FINAL_CKPT)
S=$(date -u +%FT%H:%M); s "START tests V4_A2 SINGLE SEED 1001 (user) adapter=$AD"
bash $T/run_tests.sh --tag V4_A2 --variants V4 -K 1 --seeds 1001 --adapter "$AD" > $L/tests_V4_A2_1seed.log 2>&1
rc=$?; s "END tests V4_A2 seed1001 rc=$rc"; echo "v4eval_tests_V4_A2_1seed,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
s "QUEUE DONE (1seed)"
bash $H/queue_v4_post.sh V3_A200_V4 V4_A V4_A2
cd $T && python3 v4_eval.py base_V4 V3_A200_V4 V4_A V4_A2 --md $T/V4_EVAL.md >> $L/v4_eval_all.log 2>&1
echo "$(date -u +%FT%H:%M:%SZ) CHAIN DONE rc=$?" >> $H/queue_v4_eval.status
