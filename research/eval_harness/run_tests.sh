#!/bin/bash
# PersonaPlex test runs for one model tag, all K driver processes under ONE hold of the GPU lock.
#   bash /workspace/hinglish/tests/run_tests.sh --tag base [-K 1] [--variants V1] [--adapter DIR] ...
# Run inside tmux; log goes to tests/out/<tag>/logs/run_tests.log. Arguments go to run_tests.py.
set -uo pipefail
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/workspace/.pycache-tests
T=/workspace/hinglish/tests
TAG=$(python3 -c 'import sys;a=sys.argv;print(a[a.index("--tag")+1])' "$@")
# User decision 2026-10-03 ~18:45 IST: reduced test set for V1_B and V1_C (1 seed, 12 balanced calls; gate0 clips 1 seed).
SUB12="air_03_g1,air_16_g4,cab_07_g3,cab_11_g4,cab_16_g2,ecom_07_g4,ecom_11_g1,food_07_g1,food_11_g2,food_12_g3,sub_07_g2,sub_11_g3"
EXTRA=()
case "$TAG" in V1_A|V1_B|V1_C) EXTRA=(--seeds 1001 --calls "$SUB12");; esac
mkdir -p $T/out/$TAG/logs
flock /workspace/hinglish/gpu.lock /workspace/venv-pp/bin/python -u $T/run_tests.py "$@" "${EXTRA[@]}" 2>&1 | tee -a $T/out/$TAG/logs/run_tests.log
echo "RUN_TESTS EXIT ${PIPESTATUS[0]}" | tee -a $T/out/$TAG/logs/run_tests.log
