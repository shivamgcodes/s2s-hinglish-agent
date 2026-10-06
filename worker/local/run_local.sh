#!/bin/bash
# S2S serverless worker: run the worker stack OUTSIDE docker on runpod2 (DESIGN 8), with the DEP1 venvs and assets.
#
#   bash worker/local/run_local.sh mock            CPU MockEngine (DEP1 V3 replay), ASR off, router off, no GPU.
#                                                  Safe next to the live DEP1 demo (ports 18000/18999/18996/18995).
#   bash worker/local/run_local.sh gpu             real stack: resolve -> PersonaPlex + V3 LoRA -> Trelis -> Needle.
#                                                  Takes /root/gpu.lock (fails at once while DEP1 holds it). Integrate
#                                                  stage only: bash /root/deploy/stop.sh first, start.sh afterwards.
#                                                  S2S_FRESH_COMPILE_CACHE=1: empty torch.compile caches (cold start)
#   bash worker/local/run_local.sh resolve         only run resolve_models.py (no GPU)
#   bash worker/local/run_local.sh entrypoint      the image entrypoint as is (S2S_MODE=queue needs the runpod SDK
#                                                  venv and a RunPod job source; use tests/fake_runpod.py instead)
# Any env var can be overridden from the caller (e.g. S2S_MODE=queue PORT=18100 CALL_MAX_S=60 ...).
# S2S_SESSION_SECRET: taken from the caller's env; if unset, a random local one is generated and written (mode 600)
# to $S2S_TMP/session_secret so test clients can mint tokens. S2S_AUDIENCE defaults to "local-runpod2".
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$(dirname "$HERE")
WHAT=${1:-mock}; shift || true
# defaults from runpod2.env, without overriding anything the caller already set
while IFS='=' read -r k v; do
  case "$k" in ''|\#*) continue ;; esac
  [ -z "${!k+x}" ] && export "$k=$v"
done < "${S2S_LOCAL_ENV:-$HERE/runpod2.env}"   # pod1 (2026-10-06): S2S_LOCAL_ENV=worker/local/pod1.env
mkdir -p "$S2S_TMP" "$S2S_LOGS"
if [ -z "${S2S_SESSION_SECRET:-}" ]; then
  if [ ! -s "$S2S_TMP/session_secret" ]; then
    (umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(48))" > "$S2S_TMP/session_secret")
  fi
  S2S_SESSION_SECRET=$(cat "$S2S_TMP/session_secret"); export S2S_SESSION_SECRET
  echo "[run_local] S2S_SESSION_SECRET from $S2S_TMP/session_secret (local test secret)"
fi
export S2S_AUDIENCE=${S2S_AUDIENCE:-local-runpod2} S2S_MODE=${S2S_MODE:-lb}
case "$WHAT" in
  mock)
    export CUDA_VISIBLE_DEVICES="" ASR_BACKEND=${ASR_BACKEND:-off} ROUTER=${ROUTER:-off}
    export S2S_WORKER_ARGS="--mock ${S2S_WORKER_ARGS:-}"
    exec bash "$W/stack.sh" ;;
  gpu)
    if [ "${S2S_FRESH_COMPILE_CACHE:-0}" = 1 ]; then
      # runpod2 has warm inductor/Triton caches (/tmp/torchinductor_root, ~/.triton/cache) that a fresh serverless
      # worker does not: use empty ones to measure a realistic cold start (DESIGN Amendment W2)
      rm -rf "$S2S_TMP/inductor" "$S2S_TMP/triton"
      export TORCHINDUCTOR_CACHE_DIR="$S2S_TMP/inductor" TRITON_CACHE_DIR="$S2S_TMP/triton"
    fi
    # INTEG-2: hold the lock on fd 9 and exec stack.sh itself (not "exec flock ... bash stack.sh"): with the flock(1)
    # wrapper a SIGTERM to this pid hit flock, which died without forwarding it and left stack.sh + children running.
    # Now this pid IS stack.sh (its TERM trap stops the children); the lock is released when the last fd-9 holder exits.
    LOCK=${S2S_GPU_LOCK:-/root/gpu.lock}
    exec 9>>"$LOCK"
    if ! flock -n 9; then
      echo "[run_local] $LOCK is held (the live DEP1 demo?). Stop it first: bash /root/deploy/stop.sh"; exit 1
    fi
    exec bash "$W/stack.sh" ;;
  resolve)
    exec "$S2S_PY_PP" "$W/resolve_models.py" "$@" ;;
  entrypoint)
    exec bash "$W/entrypoint.sh" ;;
  *) echo "usage: run_local.sh mock|gpu|resolve|entrypoint"; exit 2 ;;
esac
