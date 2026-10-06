#!/usr/bin/env bash
# Ticker-indicator browser check (2026-10-06, pod1 /root/dep2, CPU only): real worker_server --mock (MockEngine replay,
# real TurnFiller) + tests/fake_runpod.py (LB proxy) + the Space backend serving space/static (the built client), then
# Playwright (/root/e2e) runs space/tests/e2e_ticker.mjs. Ports base 19000 (worker), 19080 (fake LB), 18860 (Space).
set -uo pipefail
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # repo root (was /root/dep2 on pod1)
OUT=${1:-$R/results/e2e_ticker}
B=19000; SECRET=ticker-e2e-secret-0123456789-abcdefghij
mkdir -p "$OUT"
cd $R; set -a; . worker/local/pod1.env; set +a; export S2S_ROOT=$R
export S2S_LOCAL_ENV=$R/worker/local/pod1.env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
PY=/root/deploy/venv-pp/bin/python; SP=${SP:-$R/.venv-space/bin/python}
env PORT=$B S2S_INTERNAL_PORT=$((B+999)) S2S_ASR_PORT=$((B+996)) S2S_ROUTER_PORT=$((B+995)) S2S_TMP=/tmp/s2s-tick-$B \
  S2S_LOGS=/tmp/s2s-tick-$B/logs S2S_SESSION_SECRET=$SECRET S2S_AUDIENCE=ep-test S2S_MODE=lb ASR_BACKEND=off ROUTER=off \
  S2S_WORKER_ARGS="--mock-load-s 1 --mock-prompt-s 1" CALL_MAX_S=300 CLAIM_TTL_S=90 RUNPOD_POD_ID=mock-tick \
  bash $R/worker/local/run_local.sh mock > "$OUT/worker.log" 2>&1 &
W=$!
$PY -u $R/tests/fake_runpod.py --lb-port $((B+80)) --workers http://127.0.0.1:$B --worker-ids fw-0 --hold-s 2 > "$OUT/fake.log" 2>&1 &
F=$!
cd $R/space
env PORT=18860 HOST=127.0.0.1 S2S_MODE=lb RUNPOD_ENDPOINT_ID=ep-test RUNPOD_API_KEY=test-key S2S_SESSION_SECRET=$SECRET \
  S2S_AUDIENCE=ep-test RUNPOD_LB_URL=http://127.0.0.1:$((B+80)) RUNPOD_API_URL=http://127.0.0.1:$((B+81))/v2/ep-test \
  MAX_CONCURRENT_CALLS=2 RATE_PER_IP_PER_HOUR=100 KEEPALIVE_S=3 WAKE_POLL_S=0.5 S2S_PASSCODE= S2S_STATIC=${S2S_STATIC:-$R/client/dist} \
  $SP -u -m app.main > "$OUT/space.log" 2>&1 &
S=$!
trap 'kill $W $F $S 2>/dev/null; pkill -f "s2s-tick-$B" 2>/dev/null' EXIT
for i in $(seq 1 100); do curl -sf http://127.0.0.1:18860/healthz >/dev/null && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$B/ping | grep -q 200 && break; sleep 1; done
cd /root/e2e && PATH=/root/node/bin:$PATH node $R/space/tests/e2e_ticker.mjs http://localhost:18860 "$OUT" 40
