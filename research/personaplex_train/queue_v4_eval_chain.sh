#!/bin/bash
# waits for queue_v4_eval.sh to finish, then post-processes the 3 adapter tags and writes tests/V4_EVAL.md for all 4.
H=${HINGLISH_ROOT:-/workspace/hinglish}
until grep -q "QUEUE DONE" $H/queue_v4_eval.status; do sleep 30; done
bash $H/queue_v4_post.sh V3_A200_V4 V4_A V4_A2
cd $H/tests && python3 v4_eval.py base_V4 V3_A200_V4 V4_A V4_A2 --md $H/tests/V4_EVAL.md >> $H/queue_v4_eval_logs/v4_eval_all.log 2>&1
echo "$(date -u +%FT%H:%M:%SZ) CHAIN DONE rc=$?" >> $H/queue_v4_eval.status
