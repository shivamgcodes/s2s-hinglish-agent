#!/bin/bash
# D2 spec section 8: full V4 test set (30 holdout test calls) x 3 seeds (1001,1002,1003) + gate0 clips (2 clips x 3
# seeds) for base, V3_A@200 (old adapter on the new V4 inputs), V4_A@FINAL_CKPT, V4_A2@FINAL_CKPT.
# run_tests.sh takes gpu.lock around each tag. Between tags: if $H/queue_v4_eval.hold exists, wait until it is removed
# (lets a short GPU test slot in deterministically). Resumable: run_tests.py skips runs whose .meta.json exists.
set -u
H=${HINGLISH_ROOT:-/workspace/hinglish}; T=$H/tests; L=$H/queue_v4_eval_logs; mkdir -p $L
s(){ echo "$(date -u +%FT%H:%M:%SZ) $*" | tee -a $H/queue_v4_eval.status; }
fin(){ sed -n 's/^path=//p' /workspace/runs/$1/FINAL_CKPT; }
s "QUEUE START"
for spec in "base_V4:" "V3_A200_V4:/workspace/runs/V3_A/checkpoints/checkpoint_000200/consolidated" \
            "V4_A:FINAL" "V4_A2:FINAL"; do
  tag=${spec%%:*}; AD=${spec#*:}
  [ "$AD" = FINAL ] && AD=$(fin $tag)  # read FINAL_CKPT when the tag starts (user may still change the V4_A pick)
  while [ -e $H/queue_v4_eval.hold ]; do sleep 20; done
  ADA=(); [ -n "$AD" ] && ADA=(--adapter "$AD")
  S=$(date -u +%FT%H:%M); s "START tests $tag adapter=${AD:-none}"
  bash $T/run_tests.sh --tag $tag --variants V4 -K 1 --seeds 1001,1002,1003 "${ADA[@]}" > $L/tests_$tag.log 2>&1
  rc=$?; s "END tests $tag rc=$rc"; echo "v4eval_tests_$tag,$S,$(date -u +%FT%H:%M)," >> $H/gpu_minutes.csv
done
s "QUEUE DONE"
