#!/bin/bash
# Sequential: parallel engine processes thrash under the ~31-CPU cgroup quota (6-14 s/row observed).
N1_ROOT=${N1_ROOT:-/root/needle}
RC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # research/needle/n1 (code); the data/work dir is $N1_ROOT
cd $N1_ROOT
S=$N1_ROOT/finetune/sweep
$RC/results/run_evals.sh tuned_full 0-29 $S/tuned_full.cact test_heldout_a test_heldout_b
$RC/results/run_evals.sh base_pod2 0-29 base test_n0 test_heldout_a test_heldout_b
$RC/results/run_evals.sh tuned_l8 0-29 $S/tuned_l8.cact test_heldout_a test_heldout_b
$RC/results/run_evals.sh base_l8 0-29 $S/base_l8.cact test_n0
echo ALLDONE > results/logs/ALLDONE
