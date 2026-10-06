#!/usr/bin/env bash
# Browser e2e (no GPU): fake LB + fake worker (real Opus via sphn, so run under a venv with sphn: DEP1 venv-pp),
# the Space backend (.venv-space), then Playwright (/root/e2e). Ports 27080/27000/27860. Test secret only.
set -uo pipefail
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # repo root (was /root/dep2 on pod1)
OUT=${1:-$R/space/tests/out_e2e}
SECRET=test-secret-0123456789-abcdefghijklmnop
mkdir -p "$OUT"
/root/deploy/venv-pp/bin/python $R/space/tests/fake_runpod_lite.py --mode lb --secret $SECRET --aud ep-test \
  --load-s 6 --prompt-s 2 --opus > "$OUT/fake.log" 2>&1 &
FAKE=$!
cd $R/space
env S2S_MODE=lb RUNPOD_ENDPOINT_ID=ep-test RUNPOD_API_KEY=test-key S2S_SESSION_SECRET=$SECRET \
  RUNPOD_LB_URL=http://127.0.0.1:27080 WAKE_POLL_S=1 KEEPALIVE_S=5 CALL_MAX_S=15 PORT=27860 HOST=127.0.0.1 \
  $R/.venv-space/bin/python -u -m app.main > "$OUT/space.log" 2>&1 &
SPACE=$!
trap 'kill $FAKE $SPACE 2>/dev/null' EXIT
for i in $(seq 1 50); do curl -sf http://127.0.0.1:27860/healthz >/dev/null && break; sleep 0.2; done
cd /root/e2e && node $R/space/tests/e2e_browser.mjs http://localhost:27860 "$OUT" 90
