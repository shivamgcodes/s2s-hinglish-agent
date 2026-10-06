#!/bin/bash
# S2S serverless worker entrypoint (DESIGN.md 1 and 2.2). One image, two modes:
#   S2S_MODE=lb     (default) RunPod load-balancing endpoint: stack.sh in the foreground (exec, so SIGTERM reaches it)
#                   PORT default 80 ([LB-OV]); /ping on the same port. Set PORT=80 and PORT_HEALTH=80
#                   explicitly on the endpoint (runpod/docs#853; ops/ENV.md 3.1; CRIT-5).
#   S2S_MODE=queue  RunPod queue endpoint: stack.sh in the background + rp_handler.py (runpod SDK) in the foreground
#                   shell; PORT default 8765 = the TCP port to expose ([EX-QWS]). If either exits, the other is stopped.
# Required env (never baked): S2S_SESSION_SECRET, S2S_AUDIENCE (= endpoint id; RUNPOD_ENDPOINT_ID is the fallback).
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export S2S_MODE=${S2S_MODE:-lb}
case "$S2S_MODE" in
  lb) export PORT=${PORT:-80} ;;
  queue) export PORT=${PORT:-8765} ;;
  *) echo "[entrypoint] S2S_MODE must be lb or queue (got $S2S_MODE)"; exit 2 ;;
esac
mkdir -p "${S2S_LOGS:-/tmp/s2s/logs}"
echo "[entrypoint] $(date -u +%FT%TZ) mode=$S2S_MODE PORT=$PORT worker=${RUNPOD_POD_ID:-$(hostname)} endpoint=${RUNPOD_ENDPOINT_ID:-?}"
[ -n "${S2S_SESSION_SECRET:-}" ] || echo "[entrypoint] WARNING: S2S_SESSION_SECRET is not set; /ping will answer 500"

if [ "$S2S_MODE" = lb ]; then
  exec bash "$W/stack.sh"
fi

PY_RP=${S2S_PY_RP:-/opt/venv-rp/bin/python}
bash "$W/stack.sh" &
STACK=$!
"$PY_RP" -u "$W/rp_handler.py" &
HANDLER=$!
term() { kill -TERM $STACK $HANDLER 2>/dev/null; wait $STACK $HANDLER 2>/dev/null; exit 0; }
trap term TERM INT HUP
wait -n $STACK $HANDLER
echo "[entrypoint] $(date -u +%FT%TZ) stack or handler exited; stopping the other"
kill -TERM $STACK $HANDLER 2>/dev/null
wait $STACK $HANDLER 2>/dev/null
exit 1
