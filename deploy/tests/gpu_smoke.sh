#!/bin/bash
# S2S serverless, Integrate stage: ONE short GPU smoke of the REAL worker outside docker on runpod2 (DESIGN 8).
# Stops the live DEP1 demo, runs the worker (worker/local/run_local.sh gpu, under /root/gpu.lock) with PersonaPlex
# resolved through the model-cache code path (S2S_MODEL_CACHE=/root/hf/hub, same hub layout as
# /runpod-volume/huggingface-cache/hub) and Trelis from a local snapshot dir (as baked in the image), with EMPTY
# torch.compile caches (fresh serverless worker), behind tests/fake_runpod.py (LB) + the real Space backend, drives
# one call with tests/gpu_call.py, stops the worker with SIGTERM, and ALWAYS restarts DEP1 (trap) and checks it.
#   setsid nohup bash /root/deploy_serverless/tests/gpu_smoke.sh > /root/deploy_serverless/results/gpu_smoke.log 2>&1 &
# NOT inside tmux: the tmux server keeps the argv of the first 'tmux new-session' (DEP1's '... bash
# /root/deploy/gpu_stack.sh ...'), so stop.sh's 'pkill -f "bash /root/deploy/gpu_stack.sh"' kills the whole tmux
# server, including a session running this script (INTEG-1).
set -u
R=/root/deploy_serverless
OUT=$R/results/gpu_smoke
mkdir -p "$OUT"
PY=/root/deploy/venv-pp/bin/python
ts() { date -u +%FT%T.%3NZ; }
say() { echo "[smoke] $(ts) $*"; }
now() { date +%s.%N; }
SECS=${SMOKE_CALL_S:-75}

touch "$OUT/marker_deploy_unchanged"
PIDS=()
restore() {
  say "cleanup: stopping worker stack, fake proxy, Space, VRAM poll"
  pkill -TERM -f "^bash $R/worker/stack.sh" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  for i in $(seq 1 60); do flock -n /root/gpu.lock true && break; sleep 1; done
  flock -n /root/gpu.lock true || { say "GPU lock still held:"; fuser -v /root/gpu.lock; pkill -KILL -f "$R/worker/"; sleep 3; }
  say "restarting DEP1: bash /root/deploy/start.sh"
  bash /root/deploy/start.sh 2>&1 | sed 's/^/[dep1] /'
  say "DEP1 health:"
  echo "  https 8998 : $(curl -sk -o /dev/null -w '%{http_code}' https://127.0.0.1:8998/)"
  echo "  metrics    : $(curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8999/metrics)"
  echo "  asr 8996   : $(curl -s 127.0.0.1:8996/health)"
  echo "  router 8995: $(curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8995/health)"
  echo "  tmux       : $(tmux ls 2>&1 | tr '\n' ' ')"
  echo "  vram       : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  say "files under /root/deploy changed during the smoke (excluding logs/):"
  find /root/deploy -newer "$OUT/marker_deploy_unchanged" -type f -not -path '*/logs/*' 2>/dev/null | head -20
  say "done"
}

# 0. DEP1 idle?
act=$(curl -s 127.0.0.1:8999/metrics | python3 -c "import json,sys;print(json.load(sys.stdin)['session']['active'])" 2>/dev/null)
say "DEP1 session active: $act"
[ "$act" = "True" ] && { say "a DEP1 call is live; not stopping it"; exit 1; }
trap restore EXIT
trap 'say "signal: exiting"; exit 1' HUP INT TERM

# 1. stop DEP1
say "bash /root/deploy/stop.sh"
bash /root/deploy/stop.sh 2>&1 | sed 's/^/[dep1] /'
flock -n /root/gpu.lock true || { say "GPU lock not free after stop.sh"; exit 1; }
sleep 2
say "VRAM after stop: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

# 2. VRAM poll, fake LB proxy, Space
(while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 1; done) > "$OUT/vram.csv" &
PIDS+=($!)
SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))")
export S2S_SESSION_SECRET=$SECRET S2S_AUDIENCE=local-gpu-smoke
$PY -u $R/tests/fake_runpod.py --lb-port 18080 --workers http://127.0.0.1:18000 --worker-ids fw-gpu --hold-s 120 \
  > "$OUT/fake_runpod.log" 2>&1 &
PIDS+=($!)
(cd $R/space && PORT=17860 HOST=127.0.0.1 S2S_MODE=lb RUNPOD_ENDPOINT_ID=ep-local RUNPOD_API_KEY=test-key \
  RUNPOD_LB_URL=http://127.0.0.1:18080 MAX_CONCURRENT_CALLS=1 RATE_PER_IP_PER_HOUR=100 KEEPALIVE_S=20 \
  WAKE_POLL_S=3 WAKE_TIMEOUT_S=900 S2S_PASSCODE= exec $R/.venv-space/bin/python -u -m app.main) > "$OUT/space.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 30); do curl -sf 127.0.0.1:17860/healthz >/dev/null && break; sleep 0.5; done

# 3. the real worker (fresh compile caches; model-cache resolver path; Trelis as a local snapshot dir)
rm -rf /tmp/s2s-gpu
T0=$(now)
say "worker launch (T0)"
S2S_FRESH_COMPILE_CACHE=1 S2S_PP_DIR= S2S_MODEL_CACHE=/root/hf/hub \
  S2S_ASR_DIR=/root/hf/hub/models--Trelis--whisper-hinglish-preview/snapshots/eab1188fd2d0e91f2584229b32b3bfe1901c896c \
  S2S_TMP=/tmp/s2s-gpu S2S_LOGS=/tmp/s2s-gpu/logs \
  bash $R/worker/local/run_local.sh gpu > "$OUT/worker_stack.log" 2>&1 &
WP=$!
# the browser-equivalent client starts at once: its wake goes through the Space + fake proxy during the cold load
$PY -u $R/tests/gpu_call.py --space http://127.0.0.1:17860 --worker http://127.0.0.1:18000 --seconds "$SECS" \
  --out "$OUT/call.json" > "$OUT/call.log" 2>&1 &
CP=$!
# /ping timeline
p_prev=""
while kill -0 $WP 2>/dev/null; do
  p=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 127.0.0.1:18000/ping)
  if [ "$p" != "$p_prev" ]; then
    echo "$(python3 -c "print(round($(now)-$T0,1))") ping=$p" >> "$OUT/ping_timeline.txt"; p_prev=$p
  fi
  [ "$p" = 200 ] && break
  [ "$p" = 500 ] && { say "worker /ping 500 (load failed)"; break; }
  sleep 0.5
done
say "worker /ping $p at T0+$(python3 -c "print(round($(now)-$T0,1))") s"
say "VRAM loaded: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
curl -s 127.0.0.1:18000/status > "$OUT/status_ready.json"
[ "${p:-}" = 200 ] || { say "worker never became ready; stopping the call client"; kill -TERM $CP 2>/dev/null; }

# 4. wait for the call client
wait $CP; crc=$?
say "call client rc=$crc: $(tail -1 "$OUT/call.log" | head -c 1500)"
curl -s 127.0.0.1:18000/status > "$OUT/status_after.json"
say "worker /status after: $(head -c 400 "$OUT/status_after.json")"

# 5. SIGTERM the worker stack; check a clean exit and no listeners
# INTEG-2: run_local.sh gpu now execs stack.sh itself, so $WP is stack.sh (the first run signalled the flock(1)
# wrapper, whose argv also contains ".../stack.sh", and it died without forwarding the signal)
say "SIGTERM stack.sh pid $WP ($(tr '\0' ' ' < /proc/$WP/cmdline 2>/dev/null | head -c 120))"
kill -TERM "$WP" 2>/dev/null
for i in $(seq 1 60); do kill -0 $WP 2>/dev/null || break; sleep 1; done
wait $WP; wrc=$?
say "worker exit rc=$wrc after SIGTERM"
sleep 1
say "listeners left on 18000/18999/18996/18995: $(ss -ltn | grep -cE ':(18000|18999|18996|18995) ')"
cp /tmp/s2s-gpu/logs/*.log "$OUT/" 2>/dev/null
cp /tmp/s2s-gpu/models.env "$OUT/" 2>/dev/null
say "VRAM peak (MiB): $(cut -d, -f2 "$OUT/vram.csv" | sort -n | tail -1)"
exit 0
