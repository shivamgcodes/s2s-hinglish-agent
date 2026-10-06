#!/bin/bash
# A250 listening pass: base / V1_A ckpt250 / V1_A final, seed 1001, 8 V1 test calls + 2 gate0 substitutes.
# 3 driver processes in parallel under ONE hold of gpu.lock (aggregate throughput flat in K, loads overlap).
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/workspace/.pycache-tests
T=/workspace/hinglish/tests
CALLS=air_03_g1,air_16_g4,cab_07_g3,cab_11_g4,ecom_07_g4,ecom_11_g1,food_12_g3,sub_07_g2
L=/workspace/hinglish/tests/out/listen_A250.log
flock /workspace/hinglish/gpu.lock bash -c "
echo START \$(date -u +%FT%T) >> $L
cd $T
P=/workspace/venv-pp/bin/python
\$P -u run_tests.py --tag listen_base --variants V1 --calls $CALLS --seeds 1001 > out/listen_base.log 2>&1 &
\$P -u run_tests.py --tag listen_A250 --variants V1 --calls $CALLS --seeds 1001 --adapter /workspace/runs/V1_A/checkpoints/checkpoint_000250/consolidated > out/listen_A250.run.log 2>&1 &
\$P -u run_tests.py --tag listen_A1500 --variants V1 --calls $CALLS --seeds 1001 --adapter /workspace/runs/V1_A > out/listen_A1500.log 2>&1 &
wait
echo END \$(date -u +%FT%T) >> $L
"
