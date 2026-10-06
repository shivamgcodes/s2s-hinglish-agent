#!/bin/bash
# latency: idle CPU, one engine process, after all eval runs finished
N2_ROOT=${N2_ROOT:-/root/n2}
EC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/eval/lat (code); work dir = $N2_ROOT/eval/lat
source $EC/../../setup/env_needle.sh
while [ ! -f $N2_ROOT/eval/runs/done ]; do sleep 20; done
cd $N2_ROOT/eval/lat
uptime > lat_env.txt; cat /sys/fs/cgroup/cpu.max >> lat_env.txt 2>/dev/null; lscpu | grep -E "Model name|^CPU\(s\)" >> lat_env.txt
python $EC/../eval_n1.py --weights $N2_ROOT/needle_n1/tuned_full.cact --rows $N2_ROOT/data/rows/n1_test_opus --ids ids.txt --out n1_lat.json > n1_lat.log 2>&1
python $EC/../../finetune/eval_engine.py --weights $N2_ROOT/finetune/tuned_full.cact --rows v2_lat --out v2_lat.json > v2_lat.log 2>&1
python $EC/../eval_n1.py --weights $N2_ROOT/needle_n1/tuned_full.cact --rows $N2_ROOT/data/rows/n1_test_opus --ids ids.txt --out n1_lat2.json > n1_lat2.log 2>&1
python $EC/../../finetune/eval_engine.py --weights $N2_ROOT/finetune/tuned_full.cact --rows v2_lat --out v2_lat2.json > v2_lat2.log 2>&1
echo done > lat_done
