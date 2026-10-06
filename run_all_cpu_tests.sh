#!/bin/bash
# Run every CPU test suite from this checkout (no GPU, no network, no secrets; test-only fixtures). D-MONOREPO 2026-10-06.
#
#   bash run_all_cpu_tests.sh            logs -> .test-out/cpu_<date>/ (O=... to change); exit 1 if any suite fails
#
# The defaults are pod1's (the box these suites were written on): override them with env vars.
#   S2S_LOCAL_ENV  worker env file (default worker/local/pod1.env; runpod2: worker/local/runpod2.env). S2S_ROOT is
#                  always forced to this checkout, so the suites test THIS tree, not the one the env file names.
#   PP_PY          python with torch + moshi (the worker venv)         default /root/deploy/venv-pp/bin/python
#   SPACE_PY       python with the space/requirements.txt packages      default /root/dep2/.venv-space/bin/python
#   NEEDLE_PY      python with cactus-needle (router v2 suite)          default /root/dep2/.venvs/needle/bin/python
#   DEMO_CORE      the live demo's server/core.py (test_fixes compares) default /root/deploy/server/core.py
#   GATE_WAV       a real call recording for the noise-gate check      default /root/dep2/testdata/call_0851.wav
#   V2W / N1W      Needle v2 / N1 .cact weights (read in place)         defaults under /workspace/hinglish
# A suite whose prerequisite is missing is reported as SKIP (not a failure). Run one copy at a time (fixed ports).
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$R" || exit 2
ENVF=${S2S_LOCAL_ENV:-$R/worker/local/pod1.env}
[ -f "$ENVF" ] || { echo "no env file $ENVF (set S2S_LOCAL_ENV)" >&2; exit 2; }
set -a; . "$ENVF"; set +a
export S2S_ROOT=$R S2S_LOCAL_ENV=$ENVF PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
O=${O:-$R/.test-out/cpu_$(date +%Y-%m-%d_%H%M)}; mkdir -p "$O"
PP=${PP_PY:-/root/deploy/venv-pp/bin/python}; SP=${SPACE_PY:-/root/dep2/.venv-space/bin/python}
NP=${NEEDLE_PY:-/root/dep2/.venvs/needle/bin/python}
DEMO_CORE=${DEMO_CORE:-/root/deploy/server/core.py}; GATE_WAV=${GATE_WAV:-/root/dep2/testdata/call_0851.wav}
V2W=${V2W:-/workspace/hinglish/needle_v2/finetune/tuned_full.cact}; N1W=${N1W:-/workspace/hinglish/needle/tuned_full.cact}
echo "checkout $R  env $ENVF  logs $O"
PASS=0; FAIL=0; SKIP=0; FAILED=""
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing $f"; return 1; }; done; }
run() {   # run <name> <prereq files...> -- <command...>
  local n=$1; shift; local pre=(); while [ "$1" != -- ]; do pre+=("$1"); shift; done; shift
  local why; if ! why=$(need "${pre[@]}"); then echo "=== $n: SKIP ($why)"; SKIP=$((SKIP+1)); return; fi
  echo "=== $n: $*"
  timeout 1200 nice -n 10 "$@" > "$O/$n.log" 2>&1; local rc=$?
  echo "rc=$rc  $(grep -E 'checks passed|ALL OK|passed|FAIL' "$O/$n.log" | tail -2 | tr '\n' ' ')"
  if [ $rc = 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); FAILED="$FAILED $n"; fi
}
run check_dockerfile -- python3 worker/tests/check_dockerfile.py
run test_token -- python3 tests/test_token.py
run test_fixes "$PP" "$DEMO_CORE" "$GATE_WAV" -- "$PP" worker/tests/test_fixes.py --demo-core "$DEMO_CORE" --gate-wav "$GATE_WAV"
run test_worker_mock "$PP" -- "$PP" worker/tests/test_worker_mock.py
run test_crit_token_query "$PP" "$SP" -- "$PP" tests/test_crit_token_query.py --space-python "$SP"
run test_fake_runpod "$PP" "$SP" -- "$PP" tests/test_fake_runpod.py --space-python "$SP"
run test_mock_e2e "$PP" -- "$PP" tests/test_mock_e2e.py --space-python "$SP"
run test_space_flows "$SP" -- "$SP" space/tests/test_space_flows.py
run xcheck_repo_fakes "$PP" -- "$PP" space/tests/xcheck_repo_fakes.py
run test_router_v2 "$NP" "$V2W" "$N1W" -- env NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 "$NP" worker/tests/test_router_v2.py --v2-weights "$V2W" --n1-weights "$N1W" --out "$O/test_router_v2.json"
echo "SUMMARY: $PASS passed, $FAIL failed${FAILED:+ ($FAILED )}, $SKIP skipped   (logs $O)"
[ $FAIL = 0 ]
