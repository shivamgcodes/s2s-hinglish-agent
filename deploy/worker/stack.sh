#!/bin/bash
# S2S serverless worker: the GPU stack inside ONE container (DESIGN.md 3.2; fork of DEP1 gpu_stack.sh + router start).
#   bash worker/stack.sh            (run by entrypoint.sh; worker/local/run_local.sh on runpod2)
# Order (kept from DEP1): worker_server (binds $PORT at once, /ping 204; resolves + loads PersonaPlex in its load task)
#   -> once /status says engine=true: asr_service (Trelis, GPU) -> once ASR /health: router_service (Needle, CPU).
# worker_server's /ping turns 200 only when engine + ASR + router are all healthy.
# If any child exits, the rest are killed and this script exits non-zero (LB mode: the container exits and RunPod
# replaces the worker). SIGTERM/SIGINT are forwarded to every child (worker_server then ends a live call with
# session_end "worker_shutdown"); exit 0. No flock: a serverless worker owns its GPU.
#
# Env (defaults = image layout; see ops/ENV.md):
#   PORT (80) S2S_INTERNAL_PORT (8999) S2S_ASR_PORT (8996) S2S_ROUTER_PORT (8995)
#   S2S_PY_PP (/opt/venv-pp/bin/python) S2S_PY_ASR (/opt/venv-asr/bin/python) S2S_PY_NEEDLE (/opt/venv-needle/bin/python)
#   ASR_BACKEND (trelis | fw-small | fw-medium | off)  ASR_DEVICE (cuda)  ROUTER (on | off)
#   S2S_ROUTER (v2 | n1: the Needle router used by router_service; default v2, n1 = rollback, D-ROUTER-V2)
#   S2S_THREADS (4)  S2S_WORKER_ARGS (extra worker_server.py args, e.g. "--mock --mock-load-s 3")
#   S2S_LOAD_TIMEOUT_S (900)  S2S_LOGS (/tmp/s2s/logs)
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=${PORT:-80}; IPORT=${S2S_INTERNAL_PORT:-8999}; APORT=${S2S_ASR_PORT:-8996}; RPORT=${S2S_ROUTER_PORT:-8995}
export PORT S2S_INTERNAL_PORT=$IPORT S2S_ASR_PORT=$APORT S2S_ROUTER_PORT=$RPORT
PY_PP=${S2S_PY_PP:-/opt/venv-pp/bin/python}; PY_ASR=${S2S_PY_ASR:-/opt/venv-asr/bin/python}
PY_NEEDLE=${S2S_PY_NEEDLE:-/opt/venv-needle/bin/python}
ASR_BACKEND=${ASR_BACKEND:-trelis}; ROUTER=${ROUTER:-on}; export ASR_BACKEND ROUTER
L=${S2S_LOGS:-/tmp/s2s/logs}; mkdir -p "$L"; export S2S_LOGS=$L
LOAD_TIMEOUT=${S2S_LOAD_TIMEOUT_S:-900}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
# DEP1 I1 / A1.3: cap BLAS/OpenMP pools of the GPU processes (serverless vCPU count is unknown, DESIGN R13)
T=${S2S_THREADS:-4}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T DEP1_TORCH_THREADS=$T
ts() { date -u +%FT%TZ; }
say() { echo "[stack] $(ts) $*"; }
PIDS=()
stop_all() {
  local sig=${1:-TERM}
  for p in "${PIDS[@]}"; do kill -"$sig" "$p" 2>/dev/null; done
  for i in $(seq 1 50); do
    local alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && return 0; sleep 0.1
  done
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done
}
on_term() { say "signal received: stopping children"; stop_all TERM; say "stopped"; exit 0; }
trap on_term TERM INT HUP

say "worker_server :$PORT (internal :$IPORT) mode=${S2S_MODE:-lb} asr=$ASR_BACKEND router=$ROUTER"
# shellcheck disable=SC2086
"$PY_PP" -u "$W/server/worker_server.py" --port "$PORT" --internal-port "$IPORT" ${S2S_WORKER_ARGS:-} \
  > >(sed -u 's/^/[server] /' | tee -a "$L/server.log") 2>&1 &
SP=$!; PIDS+=($SP)

# wait for the engine (NOT /ping: /ping needs ASR + router, which start after the engine, as in DEP1)
st=""
for i in $(seq 1 $((LOAD_TIMEOUT / 2))); do
  st=$(curl -s --max-time 3 "127.0.0.1:$PORT/status" || true)
  case "$st" in *'"engine": true'*|*'"engine":true'*) break ;; esac
  case "$st" in *'"state": "failed"'*|*'"state":"failed"'*)
    say "worker_server reports a failed load; /ping answers 500 (see [server] lines). Not starting ASR/router."
    wait "$SP"; exit 1 ;; esac
  kill -0 $SP 2>/dev/null || { say "worker_server died during load"; stop_all; exit 1; }
  sleep 2
done
case "$st" in *'"engine": true'*|*'"engine":true'*) ;; *) say "engine not loaded after ${LOAD_TIMEOUT}s"; stop_all; exit 1 ;; esac
say "engine loaded: $(echo "$st" | head -c 300)"

if [ "$ASR_BACKEND" != "off" ]; then
  say "asr_service :$APORT backend=$ASR_BACKEND device=${ASR_DEVICE:-cuda}"
  "$PY_ASR" -u "$W/router/asr_service.py" --backend "$ASR_BACKEND" --device "${ASR_DEVICE:-cuda}" --port "$APORT" \
    > >(sed -u 's/^/[asr] /' | tee -a "$L/asr_service.log") 2>&1 &
  AP=$!; PIDS+=($AP)
  for i in $(seq 1 180); do
    curl -sf --max-time 3 "127.0.0.1:$APORT/health" >/dev/null && break
    kill -0 $AP 2>/dev/null || { say "asr died during load"; stop_all; exit 1; }
    sleep 2
  done
  curl -sf --max-time 3 "127.0.0.1:$APORT/health" >/dev/null || { say "asr not healthy after 360 s"; stop_all; exit 1; }
  say "asr READY: $(curl -s 127.0.0.1:$APORT/health)"
fi
command -v nvidia-smi >/dev/null && say "VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

if [ "$ROUTER" != "off" ]; then
  say "router_service :$RPORT"
  "$PY_NEEDLE" -u "$W/router/router_service.py" --internal "http://127.0.0.1:$IPORT" --asr-url "http://127.0.0.1:$APORT" \
    --window ring30 --unbound strict --port "$RPORT" \
    > >(sed -u 's/^/[router] /' | tee -a "$L/router_service.log") 2>&1 &
  RP=$!; PIDS+=($RP)
fi

for i in $(seq 1 60); do
  curl -sf --max-time 3 "127.0.0.1:$PORT/ping" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200 && break
  sleep 1
done
say "/ping -> $(curl -s --max-time 3 -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/ping)"

wait -n "${PIDS[@]}"
say "a worker process exited; stopping the others"
stop_all TERM
exit 1
