#!/bin/bash
# D2 / V4 training queue (handoff D2 §6-7). NOT started by the D2 trainer worker; start it once data/V4 exists:
#   tmux new-session -d -s queue_v4 "bash /workspace/hinglish/queue_v4_train.sh"
# Runs, sequentially (each train_run.sh FG=1 takes /workspace/hinglish/gpu.lock for its whole run):
#   V4_A : config A (rank 64, scaling 2.0, ft_embed false, loss weights unchanged), lr 1.5e-5, batch 8, max_steps 1200
#   V4_A2: same, lr 7.5e-6, max_steps 1500
# Both: keep_and_shift=true (no agent token dropped by the interleaver), duration_sec=$DUR,
#       in-loop eval at step 0 and every 50 steps on val (15 V4 val scenarios x g1-g4 = 60 calls) + control
#       (data/CONTROL heldout, English forgetting probe, never used for selection), checkpoint every 50 steps,
#       then pick_best.py (selects on split val) -> /workspace/runs/<run>/BEST_CKPT.
# Env overrides (paths are produced by the V4 data workers; defaults are the agreed layout):
#   V4_TRAIN   train manifest      (default /workspace/hinglish/data/V4/train/train.jsonl)
#   V4_VAL     val manifest        (default /workspace/hinglish/data/V4/val/val.jsonl)
#   V4_CONTROL control manifest    (default /workspace/hinglish/data/CONTROL/heldout/heldout_all.jsonl)
#   V4_HOLDOUT holdout json        (default /workspace/hinglish/data/V4/holdout.json; needs test_scenarios + val_scenarios)
#   DUR        window seconds      (default auto: 140, or min(ceil(max train/val call s), 160) if max > 140)
#   ONLY       "V4_A" or "V4_A2" to run one of them (default both)
#   VAL_N      expected val calls  (default 60)
set -uo pipefail
H=${HINGLISH_ROOT:-/workspace/hinglish}
V4_TRAIN=${V4_TRAIN:-$H/data/V4/train/train.jsonl}
V4_VAL=${V4_VAL:-$H/data/V4/val/val.jsonl}
V4_CONTROL=${V4_CONTROL:-$H/data/CONTROL/heldout/heldout_all.jsonl}
V4_HOLDOUT=${V4_HOLDOUT:-$H/data/V4/holdout.json}
DUR_ENV=${DUR:-}
# user 2026-10-04: window = 140 s, or min(ceil(max call duration in train+val), 160) if the longest call is > 140 s
if [ -z "$DUR_ENV" ]; then
  DUR=$(python3 -c 'import json,math,sys; m=max(json.loads(l)["duration"] for p in sys.argv[1:3] for l in open(p) if l.strip()); print(140 if m<=140 else min(math.ceil(m),160))' "$V4_TRAIN" "$V4_VAL") || { echo "DUR auto failed"; exit 1; }
else
  DUR=$DUR_ENV
fi
VAL_N=${VAL_N:-60}
ONLY=${ONLY:-}
ST=$H/queue_v4.status
LOGS=$H/queue_v4_logs
mkdir -p $LOGS
say() { echo "$(date -u +%FT%T) $*" | tee -a $ST; }

if [ -f $H/data/V4/TOPUP_PENDING ]; then say "REFUSED: data/V4/TOPUP_PENDING exists - the V4 top-up (tmux d2_v4_topup, data/V4/topup_go.sh) is still regenerating/synthesising dropped calls; start the queue after it removes the file"; exit 1; fi
# ---- fail-fast checks on the V4 split (CPU only) ----
python3 - "$V4_TRAIN" "$V4_VAL" "$V4_CONTROL" "$V4_HOLDOUT" "$VAL_N" "$DUR" <<'EOF' || { say "PRECHECK FAILED"; exit 1; }
import json, os, re, sys
tr, va, ct, ho, val_n, dur = sys.argv[1:7]
val_n, dur = int(val_n), float(dur)
for p in (tr, va, ct, ho):
    assert os.path.isfile(p), f"missing {p}"
def calls(m):
    out = []
    for l in open(m):
        if l.strip():
            r = json.loads(l); out.append((os.path.basename(r["path"])[:-4], r["duration"]))
    return out
scen = lambda c: re.sub(r"_g[1-4]$", "", c)
T, V, C = calls(tr), calls(va), calls(ct)
h = json.load(open(ho))
test_s, val_s = set(h["test_scenarios"]), set(h["val_scenarios"])
Ts, Vs = {scen(c) for c, _ in T}, {scen(c) for c, _ in V}
assert len(V) == val_n, f"val manifest has {len(V)} calls, expected {val_n}"
assert Vs == val_s, f"val manifest scenarios != holdout val_scenarios: {sorted(Vs ^ val_s)[:10]}"
assert not (Vs & test_s), f"val contains test scenarios {sorted(Vs & test_s)}"
assert not (Ts & test_s), f"train contains test scenarios {sorted(Ts & test_s)}"
assert not (Ts & Vs), f"train and val share scenarios {sorted(Ts & Vs)}"
assert len(C) == 100, f"control has {len(C)} calls"
over = [c for c, d in T + V if d > dur]
print(f"[queue_v4] train {len(T)} calls / {sum(d for _, d in T)/60:.1f} min ({len(Ts)} scenarios), "
      f"val {len(V)} ({len(Vs)} scenarios), control {len(C)}, test scenarios {len(test_s)}; "
      f"calls > {dur:.0f} s (2nd window with its own prefix): {len(over)} {over[:10]}")
EOF

run() {  # run NAME LR STEPS
  local N=$1 LR=$2 STEPS=$3 R=/workspace/runs/$1
  [ -n "$ONLY" ] && [ "$ONLY" != "$N" ] && return 0
  if [ -e "$R" ]; then say "$N: run dir $R exists -> not touching it"; return 1; fi
  say "$N: start lr=$LR max_steps=$STEPS dur=$DUR"
  local T0; T0=$(date -u +%Y-%m-%dT%H:%M)
  FG=1 RUN_NAME=$N BATCH=8 EVAL_FREQ=50 DATA="$V4_TRAIN" \
  EVAL_DATA="val=$V4_VAL,control=$V4_CONTROL" \
    bash $H/trainer/train_run.sh V4 A optim.lr=$LR max_steps=$STEPS ckpt_freq=50 \
      duration_sec=$DUR keep_and_shift=true > $LOGS/$N.log 2>&1
  local T1; T1=$(date -u +%Y-%m-%dT%H:%M)
  echo "train_$N,$T0,$T1,$(( ( $(date -u -d "$T1" +%s) - $(date -u -d "$T0" +%s) ) / 60 ))" >> $H/gpu_minutes.csv
  if ! grep -q "TRAIN EXIT 0" "$R/run.log" 2>/dev/null; then say "$N: FAILED (see $R/run.log, $LOGS/$N.log)"; return 1; fi
  python3 $H/pick_best.py "$R" >> $LOGS/$N.log 2>&1 && say "$N: done, $(head -1 $R/BEST_CKPT 2>/dev/null)" \
    || say "$N: pick_best failed"
}

say "queue_v4 start (train=$V4_TRAIN val=$V4_VAL control=$V4_CONTROL dur=$DUR)"
run V4_A 1.5e-5 1200 || exit 1
run V4_A2 7.5e-6 1500 || exit 1
say "queue_v4 DONE"
