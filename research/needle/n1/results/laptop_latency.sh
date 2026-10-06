#!/bin/bash
# N1 §5 latency on the laptop CPU: test_n0 (the fixed 60 rows), one process at a time, engine threads
# uncapped as in N0 (E2). A watchdog kills the run if MemAvailable drops below 6 GB.
RC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # research/needle/n1 (code); the data/work dir is $N1_ROOT
cd "${N1_ROOT:?set N1_ROOT to the N1 working dir (data, results/, tuned_*.cact)}"
export N1_WORKDIR=${N1_WORKDIR:-$N1_ROOT}
source "${NEEDLE_EXP0_ENV:?set NEEDLE_EXP0_ENV to needle-exp0/env.sh (defines NEEDLE_PY)}"
L=results/logs
for spec in "laptop_base:" "laptop_tuned_l8:$PWD/tuned_l8.cact" "laptop_tuned_full:$PWD/tuned_full.cact"; do
  tag=${spec%%:*}; w=${spec#*:}
  echo "$(date +%T) $tag loadavg $(cut -d' ' -f1-3 /proc/loadavg) memavail_gb $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" >> $L/laptop_latency_driver.log
  if [ -z "$w" ]; then $NEEDLE_PY $RC/evaluate.py run --test test_n0 --tag $tag > $L/$tag.log 2>&1 &
  else $NEEDLE_PY $RC/evaluate.py run --test test_n0 --tag $tag --weights $w > $L/$tag.log 2>&1 & fi
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$ma" -lt 6 ]; then kill $pid; echo "ABORT $tag memavail ${ma}GB" >> $L/laptop_latency_driver.log; exit 1; fi
    sleep 2
  done
  wait $pid; echo "$(date +%T) $tag rc=$?" >> $L/laptop_latency_driver.log
done
