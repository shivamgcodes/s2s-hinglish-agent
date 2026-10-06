#!/bin/bash
# CPU smoke test of research/ (no GPU, no network, no data): every .py compiles, every .sh parses, and the pure-python
# unit tests that need no data run. D-TRAINING-CODE 2026-10-07; run by deploy/run_all_cpu_tests.sh (suite research_smoke).
#   bash research/smoke_test.sh            exit 1 on any failure
# The research scripts themselves need the dev boxes' data + GPUs (see research/README.md); this only proves the code
# of record is complete, parses, and finds the shared modules in ../packages.
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$R" || exit 2
PY=${PY:-python3}
FAIL=0; N=0
export PYTHONDONTWRITEBYTECODE=1
while IFS= read -r -d '' f; do
  N=$((N+1))
  "$PY" -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read(), sys.argv[1])" "$f" 2>/tmp/research_smoke_$$.err \
    || { echo "FAIL compile $f: $(tail -1 /tmp/research_smoke_$$.err)"; FAIL=$((FAIL+1)); }
done < <(find . -name '*.py' -not -path '*/__pycache__/*' -print0)
echo "python files parsed: $N"
M=0
while IFS= read -r -d '' f; do
  M=$((M+1)); bash -n "$f" 2>/tmp/research_smoke_$$.err || { echo "FAIL bash -n $f: $(tail -1 /tmp/research_smoke_$$.err)"; FAIL=$((FAIL+1)); }
done < <(find . -name '*.sh' -print0)
echo "shell files parsed: $M"
rm -f /tmp/research_smoke_$$.err
# pure-python unit tests (stdlib only, seconds; each must exit 0)
UNIT=(${RESEARCH_UNIT_TESTS:-})
[ -f unit_tests.txt ] && while read -r t; do case "$t" in ''|\#*) ;; *) UNIT+=("$t") ;; esac; done < unit_tests.txt
for t in "${UNIT[@]}"; do
  if timeout 300 "$PY" $t > /tmp/research_unit_$$.log 2>&1; then echo "PASS unit $t"
  else echo "FAIL unit $t: $(tail -3 /tmp/research_unit_$$.log | tr '\n' ' ')"; FAIL=$((FAIL+1)); fi
done
rm -f /tmp/research_unit_$$.log
[ $FAIL = 0 ] && echo "research smoke: ALL OK" || { echo "research smoke: $FAIL FAIL"; exit 1; }
