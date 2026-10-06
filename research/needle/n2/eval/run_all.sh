#!/bin/bash
# N2 eval: sequential engine runs (one engine at a time; runpod2 CPU quota ~10.2)
N2_ROOT=${N2_ROOT:-/root/n2}
EC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # research/needle/n2/eval (code); work dir = $N2_ROOT/eval
source $EC/../setup/env_needle.sh
cd $N2_ROOT/eval
R=$N2_ROOT/data/rows
for f in test_clean test_opus test_exact; do
  python $EC/../finetune/eval_engine.py --weights $N2_ROOT/finetune/tuned_full.cact --rows $R/$f --out runs/v2_$f.json > runs/v2_$f.log 2>&1
done
for f in n1_test_clean n1_test_opus n1_test_exact n1_test_clean_nc n1_test_opus_nc n1_test_exact_nc; do
  python $EC/eval_n1.py --weights $N2_ROOT/needle_n1/tuned_full.cact --rows $R/$f --out runs/n1_$f.json > runs/n1_$f.log 2>&1
done
for f in test_clean test_opus test_exact; do
  python $EC/../finetune/eval_engine.py --weights $N2_ROOT/finetune/sweep/full_A_e10.cact --rows $R/$f --out runs/Ae10_$f.json > runs/Ae10_$f.log 2>&1
done
echo ALLDONE > runs/done
