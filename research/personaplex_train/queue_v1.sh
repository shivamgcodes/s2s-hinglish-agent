#!/bin/bash
# V1 GPU queue: strictly sequential, run inside ONE tmux session 'q_v1':
#   tmux new-session -d -s q_v1 'bash /workspace/hinglish/queue_v1.sh'
# Steps: train V1_A, V1_B, V1_C (train_run.sh FG=1; launch.sh takes gpu.lock itself) ->
#        pick_best.py (CPU) -> <run>/BEST_CKPT for V1_A/B/C ->
#        run_tests.sh base_V1, V1_A@best, V1_B@best, V1_C@best (takes gpu.lock itself) ->
#        asr_pass.py (flock) -> judge.py (flock) -> score.py (CPU).
# No outer flock around train_run.sh / run_tests.sh (they flock internally; nesting would deadlock).
# A failing step is recorded in queue_v1.status and the queue continues.
# Re-running skips training steps whose run dir finished (TRAIN EXIT 0 + lora.safetensors); a partial run dir
# is moved aside to <dir>.partial_<ts>. Test/ASR/judge steps are resumable by themselves.
set -uo pipefail
# ---- resolved hyper-parameters (NOTES 'Orchestrator decisions after Gate 1'; step-250 rule may scale both) ----
LR_A=1.5e-5          # config A and B
LR_C=7.5e-6          # config C
BATCH=8
# 2026-10-03 shortened B/C (user decision after V1-A held-out loss was best at step 250 and rose after; NOTES):
STEPS_B=600          # spec 1500
STEPS_C=1200         # spec 3000
CKPT_FREQ=50         # spec 250; so the best-held-out checkpoint is loadable
export EVAL_FREQ=50  # in-loop held-out eval (train_run.sh)
export EVAL_DATA=/workspace/hinglish/data/V1/heldout/heldout_all.jsonl
# ---------------------------------------------------------------------------------------------------------------
H=${HINGLISH_ROOT:-/workspace/hinglish}
T=$H/tests
ST=$H/queue_v1.status
LOGD=$H/queue_v1_logs
DATA=$H/data/V1/train/train.jsonl
mkdir -p $LOGD
now() { date -u +%Y-%m-%dT%H:%M:%S; }
status() { echo "$(now) $*" | tee -a $ST; }
gpumin() { # stage start_epoch end_epoch
  echo "$1,$(date -u -d @$2 +%Y-%m-%dT%H:%M),$(date -u -d @$3 +%Y-%m-%dT%H:%M),$(( ($3-$2+30)/60 ))" >> $H/gpu_minutes.csv; }

train() { # name cfg overrides...
  local NAME=$1 CFG=$2; shift 2
  local RUN=/workspace/runs/$NAME
  if [ -f $RUN/lora.safetensors ] && grep -q "TRAIN EXIT 0" $RUN/run.log 2>/dev/null; then
    status "SKIP $NAME (already finished)"; return; fi
  if [ -d $RUN ]; then mv $RUN $RUN.partial_$(date -u +%H%M%S); status "moved partial $RUN aside"; fi
  local t0=$(date +%s)
  status "START $NAME cfg=$CFG $*"
  DATA=$DATA BATCH=$BATCH FG=1 RUN_NAME=$NAME bash $H/trainer/train_run.sh V1 $CFG "$@" > $LOGD/$NAME.log 2>&1
  local t1=$(date +%s)
  local ex=$(grep -o 'TRAIN EXIT [0-9]*' $RUN/run.log 2>/dev/null | tail -1 | awk '{print $3}')
  local lr=$(grep -m1 -E '^\s+lr:' $RUN/args.yaml 2>/dev/null | awk '{print $2}')
  local ms=$(grep -m1 -E '^max_steps:' $RUN/args.yaml 2>/dev/null | awk '{print $2}')
  local fe=$(grep -m1 -E '^\s+ft_embed:' $RUN/args.yaml 2>/dev/null | awk '{print $2}')
  if [ "$ex" = "0" ] && [ -f $RUN/lora.safetensors ]; then r=OK; else r="FAIL(exit=${ex:-none})"; fi
  status "END $NAME $r min=$(( (t1-t0)/60 )) lr=$lr max_steps=$ms ft_embed=$fe"
  gpumin "train_$NAME" $t0 $t1
}

tests() { # tag [adapter_dir]
  local TAG=$1 AD=${2:-}
  local t0=$(date +%s)
  status "START tests $TAG ${AD:+adapter=$AD}"
  if [ -n "$AD" ] && [ ! -f $AD/lora.safetensors ]; then status "END tests $TAG FAIL(no adapter $AD)"; return; fi
  bash $T/run_tests.sh --tag $TAG --variants V1 -K 1 ${AD:+--adapter $AD} > $LOGD/tests_$TAG.log 2>&1
  local t1=$(date +%s)
  local ex=$(grep -o 'RUN_TESTS EXIT [0-9]*' $T/out/$TAG/logs/run_tests.log 2>/dev/null | tail -1 | awk '{print $3}')
  local n=$(find $T/out/$TAG -name '*.meta.json' 2>/dev/null | wc -l)
  [ "$ex" = "0" ] && r=OK || r="FAIL(exit=${ex:-none})"
  status "END tests $TAG $r min=$(( (t1-t0)/60 )) run_metas=$n"
  gpumin "tests_$TAG" $t0 $t1
}

best() { # run name -> <run>/BEST_CKPT (lowest held-out total_pooled; V1_A from val_metrics.csv)
  local RUN=/workspace/runs/$1
  status "START pick_best $1"
  /workspace/venv-ft/bin/python $H/pick_best.py $RUN >> $LOGD/pick_best.log 2>&1
  local ex=$?
  status "END pick_best $1 exit=$ex $(tr '\n' ' ' < $RUN/BEST_CKPT 2>/dev/null | cut -c1-400)"
}
bestpath() { grep -m1 '^path=' /workspace/runs/$1/BEST_CKPT 2>/dev/null | cut -d= -f2-; }

TAGS="base_V1 V1_A V1_B V1_C"
status "QUEUE START LR_A=$LR_A LR_C=$LR_C BATCH=$BATCH STEPS_B=$STEPS_B STEPS_C=$STEPS_C CKPT_FREQ=$CKPT_FREQ EVAL_FREQ=$EVAL_FREQ"
train V1_A A optim.lr=$LR_A
train V1_B B optim.lr=$LR_A lora.ft_embed=true max_steps=$STEPS_B ckpt_freq=$CKPT_FREQ
train V1_C C optim.lr=$LR_C max_steps=$STEPS_C ckpt_freq=$CKPT_FREQ
best V1_A; best V1_B; best V1_C
tests base_V1
for R in V1_A V1_B V1_C; do
  P=$(bestpath $R)
  if [ -z "$P" ]; then status "END tests $R FAIL(no BEST_CKPT)"; else tests $R "$P"; fi
done

t0=$(date +%s); status "START asr_pass $TAGS"
( cd $T && flock $H/gpu.lock /workspace/venv-asr/bin/python asr_pass.py $TAGS ) > $LOGD/asr_pass.log 2>&1
ex=$?; t1=$(date +%s); status "END asr_pass exit=$ex min=$(( (t1-t0)/60 ))"; gpumin asr_V1 $t0 $t1

t0=$(date +%s); status "START judge $TAGS"
( cd $T && flock $H/gpu.lock /workspace/venv-vllm/bin/python judge.py $TAGS ) > $LOGD/judge.log 2>&1
ex=$?; t1=$(date +%s); status "END judge exit=$ex min=$(( (t1-t0)/60 ))"; gpumin judge_V1 $t0 $t1

status "START score (CPU)"
( cd $T && /workspace/venv-pp/bin/python score.py $TAGS ) > $LOGD/score.log 2>&1
status "END score exit=$?"
status "QUEUE DONE"
