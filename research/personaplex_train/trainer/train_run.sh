#!/bin/bash
# PersonaPlex LoRA run (patched moshi-finetune).
#   train_run.sh <variant> <A|B|C> [yaml.key=value ...]
# Env: DATA (manifest jsonl; default /workspace/hinglish/data/<V>/train/train.jsonl), BATCH (default 16),
#      RUN_NAME (default <V>_<cfg>), FG=1 run in the foreground instead of tmux.
#      EVAL_FREQ (default 100; 0 = no eval), EVAL_DATA (default /workspace/hinglish/data/<V>/heldout/heldout_all.jsonl):
#      in-loop held-out loss (training loss, no_grad) at step 0, every EVAL_FREQ steps and the last step ->
#      <run>/eval_metrics.csv + dashed curves on loss.png. A missing DEFAULT eval manifest disables eval with a
#      warning; a missing explicit EVAL_DATA fails. key=val args (do_eval=, eval_freq=, data.eval_data=) still win.
#      Several named eval sets: EVAL_DATA="val=/path/a.jsonl,control=/path/b.jsonl" -> each set evaluated at every
#      eval step with the same loss code, one eval_metrics.csv row per set (split=<name>); loss.png val dashed,
#      control dotted. pick_best.py selects on split=val only (control = observation, e.g. forgetting).
#      A plain path is one set named "eval" (as before).
# Output: /workspace/runs/<RUN_NAME>/ {run.log, args.yaml (resolved config), metrics.csv, loss.png,
#         checkpoints/checkpoint_XXXXXX/consolidated/{lora.safetensors,config.json}, lora.safetensors, config.json}
# The GPU is taken with flock /workspace/hinglish/gpu.lock for the whole run.
# Example overrides: max_steps=300 optim.lr=1e-4 overwrite_run_dir=true
set -euo pipefail
V=$1; CFG=$2; shift 2
T=${TRAINER_DIR:-/workspace/hinglish/trainer}
NAME=${RUN_NAME:-${V}_${CFG}}
RUN=${RUNS_ROOT:-/workspace/runs}/$NAME
DATA=${DATA:-/workspace/hinglish/data/$V/train/train.jsonl}
[ -f "$DATA" ] || { echo "no manifest $DATA"; exit 1; }
EVAL_FREQ=${EVAL_FREQ:-100}
# eval_paths SPEC -> one manifest path per line ("name=path,..." or a plain path)
eval_paths() { local IFS=,; for it in $1; do echo "${it#*=}"; done; }
if [ -n "${EVAL_DATA:-}" ]; then
  if [[ "$EVAL_DATA" == *=* ]]; then
    [[ ",$EVAL_DATA" == *",val="* ]] || echo "[train_run] WARNING: named EVAL_DATA has no 'val' set; pick_best.py needs split val (or eval)"
    while read -r P; do [ -f "$P" ] || { echo "no eval manifest $P"; exit 1; }; done < <(eval_paths "$EVAL_DATA")
  else
    [ -f "$EVAL_DATA" ] || { echo "no eval manifest $EVAL_DATA"; exit 1; }
  fi
else
  EVAL_DATA=/workspace/hinglish/data/$V/heldout/heldout_all.jsonl
  if [ "$EVAL_FREQ" != "0" ] && [ ! -f "$EVAL_DATA" ]; then
    echo "[train_run] WARNING: no default eval manifest $EVAL_DATA -> in-loop eval disabled"; EVAL_FREQ=0
  fi
fi
if [ "$EVAL_FREQ" = "0" ]; then EVAL_ARGS=(do_eval=false); else
  EVAL_ARGS=(do_eval=true eval_freq="$EVAL_FREQ" data.eval_data="$EVAL_DATA"); fi
mkdir -p "$RUN"
/workspace/venv-ft/bin/python $T/make_config.py $T/configs/$CFG.yaml "$RUN/config_input.yaml" \
  data.train_data="$DATA" run_dir="$RUN" batch_size="${BATCH:-16}" "${EVAL_ARGS[@]}" "$@"
read -r DUR DO_EVAL EVAL_DATA EVAL_FREQ < <(/workspace/venv-ft/bin/python -c "import yaml;c=yaml.safe_load(open('$RUN/config_input.yaml'));print(c['duration_sec'],int(bool(c.get('do_eval'))),c['data'].get('eval_data') or '-',c.get('eval_freq',0))")
HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 /workspace/venv-ft/bin/python $T/preflight.py "$DATA" "$DUR" | tee -a "$RUN/run.log"
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 1; }
if [ "$DO_EVAL" = "1" ]; then
  echo "[train_run] in-loop eval: every $EVAL_FREQ steps on $EVAL_DATA" | tee -a "$RUN/run.log"
  while read -r P; do
    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 /workspace/venv-ft/bin/python $T/preflight.py "$P" "$DUR" < /dev/null | tee -a "$RUN/run.log"
    [ "${PIPESTATUS[0]}" = "0" ] || { echo "eval preflight failed ($P); not starting"; exit 1; }
  done < <(eval_paths "$EVAL_DATA")
else
  echo "[train_run] in-loop eval: off" | tee -a "$RUN/run.log"
fi
PORT=$((29500 + RANDOM % 1000))
CMD="cd /workspace/moshi-finetune && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 NO_TORCH_COMPILE=1 \
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false && \
echo \"[train_run] \$(date) waiting for GPU lock\" | tee -a '$RUN/run.log' && \
flock /workspace/hinglish/gpu.lock bash -c \"echo [train_run] \\\$(date) got GPU lock; \
/workspace/venv-ft/bin/torchrun --nproc-per-node 1 --master_port $PORT -m train '$RUN/config_input.yaml'\" 2>&1 \
| tee -a '$RUN/run.log'; echo \"TRAIN EXIT \${PIPESTATUS[0]} \$(date)\" | tee -a '$RUN/run.log'"
printf '%s\n' "$CMD" > "$RUN/launch.sh"
if [ "${FG:-0}" = "1" ]; then
  bash "$RUN/launch.sh"
else
  tmux new-session -d -s "train_$NAME" "bash $RUN/launch.sh"
  echo "started tmux session train_$NAME; log: $RUN/run.log"
fi
