#!/bin/bash
# run_eval.sh <logfile> [eval_heldout.py args...]   -- same env as the run's launch.sh, under gpu.lock
LOG=$1; shift
cd /workspace/moshi-finetune && export CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 NO_TORCH_COMPILE=1 \
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPYCACHEPREFIX=/workspace/.pycache-ft
echo "[run_eval] $(date -u +%FT%T) waiting for gpu.lock" | tee -a "$LOG"
flock /workspace/hinglish/gpu.lock bash -c "echo [run_eval] \$(date -u +%FT%T) got gpu.lock; \
  /workspace/venv-ft/bin/torchrun --nproc-per-node 1 --master_port $((29500 + RANDOM % 1000)) \
  /workspace/hinglish/trainer/eval_heldout.py $*; echo EVAL EXIT \$? \$(date -u +%FT%T)" 2>&1 | tee -a "$LOG"
