# V4 comparison: base / V3_A / V4_A / V4_A2 (D2 spec section 9)

Generated 2026-10-05 05:49 UTC by tests/v4_comparison.py from tests/out/<tag>/v4_scores.json (tests/v4_eval.py) and /workspace/runs/<run>/eval_metrics.csv. Facts only; decisions and definitions are in NOTES.md ('D2 / V4 ...' entries) and tests/V4_EVAL.md, tests/V4_RERANK.md, data/V4/STATS.md.

Models (adapter = checkpoints/checkpoint_000NNN/consolidated, merged at load, 253 LoRA pairs, scaling 2.0):
- **base**: PersonaPlex, no adapter (tag base_V4).
- **V3_A@200**: old V3 adapter /workspace/runs/V3_A step 200, run on the NEW V4 test inputs (tag V3_A200_V4).
- **V4_A@350**: lr 1.5e-5, 1200 steps; FINAL_CKPT step 350 (re-rank pick; val-best is step 400).
- **V4_A2@600**: lr 7.5e-6, 1500 steps; FINAL_CKPT step 600 (= its val-best).

Test set: the 30 V4 holdout test calls (one per test scenario; 13 standard / 16 long / 1 mixed), PersonaPlex seeds 1001-1003. **V4_A2 has seed 1001 only** (user, 2026-10-05 ~10:30 IST: single seeds); its 6 leftover seed-1002/1003 runs ({'1002': 3, '1003': 3}) are excluded. So the like-for-like 4-way comparison is **section B on seed 1001**; section C adds 3 seeds for the three tags that have them.

How to read the cells: `mean ± std` = per-seed mean over the band's calls, then mean and population std over seeds (std shown only when there are 3 seeds; one seed has no std). `(pooled r, n=N)` = rate pooled over every run of the band with its denominator N in the unit given in the row ([n: calls / writes / lines / ref chars / ref words]); for one seed and per-call metrics n = the calls in the column header. Rows with only a pooled value have no per-seed statistic in v4_scores.json.

## A. Headline (seed 1001, all 30 calls)

| | base | V3_A@200 | V4_A@350 | V4_A2@600 |
|---|---|---|---|---|
| task score | 0.28 | 0.55 | 0.70 | 0.72 |
| COMPLETE calls | 2/30 | 9/30 | 15/30 | 17/30 |
| facts CONSISTENT calls | 16/30 | 15/30 | 16/30 | 19/30 |
| echo (writes) | 1/34 | 11/34 | 17/34 | 11/34 |
| NATURAL lines | 0.74 | 0.59 | 0.58 | 0.57 |
| NATURAL and Hinglish | 0.00 | 0.37 | 0.44 | 0.47 |
| NONSENSE lines | 0.11 | 0.23 | 0.24 | 0.21 |
| Trelis CER m1 | 0.056 | 0.086 | 0.065 | 0.077 |

Seed-to-seed std of the task score on the three 3-seed tags: base ±0.055, V3_A ±0.016, V4_A ±0.008. The V4_A vs V4_A2 differences on seed 1001 (task 0.70 vs 0.72, CONSISTENT 16 vs 19 of 30, echo 17 vs 11 of 34) are single-seed, on 30 calls / 34 writes.

### Length band: standard (13 calls) / long (16 calls), seed 1001

| model | task score standard / long | facts CONSISTENT standard / long | judge confirm+value standard / long | echo standard / long | NATURAL standard / long | NATURAL Hinglish standard / long | NONSENSE standard / long | Trelis CER m1 standard / long |
|---|---|---|---|---|---|---|---|---|
| base | 0.27 / 0.25 | 0.62 / 0.44 | 0.27 / 0.16 | 0.07 / 0.00 | 0.74 / 0.73 | 0.00 / 0.00 | 0.07 / 0.14 | 0.07 / 0.05 |
| V3_A@200 | 0.62 / 0.47 | 0.38 / 0.62 | 0.40 / 0.37 | 0.40 / 0.26 | 0.59 / 0.59 | 0.36 / 0.38 | 0.23 / 0.24 | 0.08 / 0.09 |
| V4_A@350 | 0.69 / 0.69 | 0.62 / 0.44 | 0.67 / 0.58 | 0.47 / 0.53 | 0.59 / 0.56 | 0.47 / 0.41 | 0.24 / 0.24 | 0.07 / 0.06 |
| V4_A2@600 | 0.77 / 0.69 | 0.85 / 0.44 | 0.67 / 0.68 | 0.33 / 0.32 | 0.57 / 0.56 | 0.47 / 0.46 | 0.22 / 0.21 | 0.09 / 0.07 |

Long calls have lower fact consistency than standard calls for base and both V4 adapters (V3_A@200 is the exception on seed 1001); NATURAL rate and CER differ little by band. The mixed band is 1 call (air_26_g3) and is not interpretable.

## B. All metrics × length band, seed 1001, all four models

### B: band all (30 calls)

| metric | base (30 calls × 1 seed) | V3_A@200 (30 calls × 1 seed) | V4_A@350 (30 calls × 1 seed) | V4_A2@600 (30 calls × 1 seed) |
|---|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.283 | 0.550 | 0.700 | 0.717 |
| Judge: COMPLETE calls [n: calls] | 0.067 | 0.300 | 0.500 | 0.567 |
| Judge: FAILED calls [n: calls] | 0.500 | 0.200 | 0.100 | 0.133 |
| **Judge: facts CONSISTENT** [n: calls] | 0.533 | 0.500 | 0.533 | 0.633 |
| Judge: facts CONTRADICTS [n: calls] | 0.400 | 0.433 | 0.333 | 0.267 |
| Judge: wrong facts per call [n: calls] | 0.633 | 0.633 | 0.733 | 0.533 |
| Judge: write confirmed with the right value [n: writes] | 0.182 (pooled 0.206, n=34) | 0.424 (pooled 0.382, n=34) | 0.651 (pooled 0.618, n=34) | 0.712 (pooled 0.676, n=34) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.023 (pooled 0.029, n=34) | 0.341 (pooled 0.324, n=34) | 0.530 (pooled 0.500, n=34) | 0.341 (pooled 0.324, n=34) |
| Check-line before write (score.py m3) [n: writes] | 0.023 (pooled 0.029, n=34) | 0.424 (pooled 0.500, n=34) | 0.682 (pooled 0.735, n=34) | 0.629 (pooled 0.647, n=34) |
| **NATURAL lines** [n: lines] | 0.735 (pooled 0.737, n=676) | 0.585 (pooled 0.586, n=411) | 0.578 (pooled 0.578, n=462) | 0.566 (pooled 0.568, n=440) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=676) | 0.370 (n=411) | 0.437 (n=462) | 0.466 (n=440) |
| - NATURAL and pure English (en=true) [n: lines] | 0.737 (n=676) | 0.216 (n=411) | 0.141 (n=462) | 0.102 (n=440) |
| STIFF lines [n: lines] | 0.154 (pooled 0.151, n=676) | 0.156 (pooled 0.156, n=411) | 0.133 (pooled 0.128, n=462) | 0.162 (pooled 0.161, n=440) |
| PEPPERED lines [n: lines] | 0.000 (pooled 0.000, n=676) | 0.023 (pooled 0.024, n=411) | 0.050 (pooled 0.052, n=462) | 0.059 (pooled 0.059, n=440) |
| NONSENSE lines [n: lines] | 0.111 (pooled 0.112, n=676) | 0.236 (pooled 0.234, n=411) | 0.238 (pooled 0.242, n=462) | 0.213 (pooled 0.211, n=440) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.093 (n=676) | 0.151 (n=411) | 0.208 (n=462) | 0.168 (n=440) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=676) | 0.499 (n=411) | 0.498 (n=462) | 0.589 (n=440) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.057 (pooled 0.056, n=19420) | 0.086 (pooled 0.086, n=17565) | 0.065 (pooled 0.065, n=18516) | 0.078 (pooled 0.077, n=18755) |
| Trelis WER m1 [n: ref words] | 0.116 (pooled 0.114, n=3700) | 0.167 (pooled 0.165, n=3270) | 0.128 (pooled 0.127, n=3443) | 0.146 (pooled 0.145, n=3471) |
| Model Hindi share (score.py m1) [n: calls] | 0.000 | 0.473 | 0.546 | 0.571 |
| Read rate (m4) [n: calls] | 0.673 | 0.520 | 0.548 | 0.575 |
| Value rate (m4) [n: calls] | 0.517 | 0.384 | 0.487 | 0.553 |
| Invented-fact flag (m4) [n: calls] | 0.500 | 0.300 | 0.233 | 0.267 |
| Greeting (m5) [n: calls] | 1.000 | 1.000 | 1.000 | 1.000 |
| Degenerate (m5) [n: calls] | 0.033 | 0.000 | 0.000 | 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 | 1.000 | 1.000 | 1.000 |
| Words per call [n: calls] | 114.3 | 101.7 | 108.0 | 109.3 |

### B: band standard (13 calls)

| metric | base (13 calls × 1 seed) | V3_A@200 (13 calls × 1 seed) | V4_A@350 (13 calls × 1 seed) | V4_A2@600 (13 calls × 1 seed) |
|---|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.269 | 0.615 | 0.692 | 0.769 |
| Judge: COMPLETE calls [n: calls] | 0.000 | 0.385 | 0.462 | 0.615 |
| Judge: FAILED calls [n: calls] | 0.462 | 0.154 | 0.077 | 0.077 |
| **Judge: facts CONSISTENT** [n: calls] | 0.615 | 0.385 | 0.615 | 0.846 |
| Judge: facts CONTRADICTS [n: calls] | 0.308 | 0.462 | 0.231 | 0.154 |
| Judge: wrong facts per call [n: calls] | 0.615 | 0.615 | 0.385 | 0.231 |
| Judge: write confirmed with the right value [n: writes] | 0.200 (pooled 0.267, n=15) | 0.450 (pooled 0.400, n=15) | 0.700 (pooled 0.667, n=15) | 0.700 (pooled 0.667, n=15) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.050 (pooled 0.067, n=15) | 0.450 (pooled 0.400, n=15) | 0.500 (pooled 0.467, n=15) | 0.350 (pooled 0.333, n=15) |
| Check-line before write (score.py m3) [n: writes] | 0.000 (pooled 0.000, n=15) | 0.450 (pooled 0.533, n=15) | 0.750 (pooled 0.800, n=15) | 0.650 (pooled 0.667, n=15) |
| **NATURAL lines** [n: lines] | 0.732 (pooled 0.739, n=283) | 0.586 (pooled 0.586, n=174) | 0.594 (pooled 0.595, n=200) | 0.571 (pooled 0.568, n=185) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=283) | 0.362 (n=174) | 0.465 (n=200) | 0.470 (n=185) |
| - NATURAL and pure English (en=true) [n: lines] | 0.739 (n=283) | 0.224 (n=174) | 0.130 (n=200) | 0.097 (n=185) |
| STIFF lines [n: lines] | 0.192 (pooled 0.187, n=283) | 0.142 (pooled 0.144, n=174) | 0.124 (pooled 0.120, n=200) | 0.153 (pooled 0.157, n=185) |
| PEPPERED lines [n: lines] | 0.000 (pooled 0.000, n=283) | 0.037 (pooled 0.040, n=174) | 0.044 (pooled 0.045, n=200) | 0.053 (pooled 0.054, n=185) |
| NONSENSE lines [n: lines] | 0.076 (pooled 0.074, n=283) | 0.234 (pooled 0.230, n=174) | 0.237 (pooled 0.240, n=200) | 0.223 (pooled 0.222, n=185) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.060 (n=283) | 0.167 (n=174) | 0.215 (n=200) | 0.168 (n=185) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=283) | 0.494 (n=174) | 0.495 (n=200) | 0.584 (n=185) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.064 (pooled 0.065, n=8291) | 0.083 (pooled 0.083, n=7334) | 0.071 (pooled 0.072, n=7690) | 0.089 (pooled 0.089, n=7774) |
| Trelis WER m1 [n: ref words] | 0.131 (pooled 0.131, n=1590) | 0.166 (pooled 0.166, n=1368) | 0.131 (pooled 0.131, n=1426) | 0.156 (pooled 0.156, n=1446) |
| Model Hindi share (score.py m1) [n: calls] | 0.001 | 0.458 | 0.557 | 0.572 |
| Read rate (m4) [n: calls] | 0.632 | 0.604 | 0.511 | 0.660 |
| Value rate (m4) [n: calls] | 0.350 | 0.375 | 0.383 | 0.533 |
| Invented-fact flag (m4) [n: calls] | 0.462 | 0.231 | 0.000 | 0.385 |
| Greeting (m5) [n: calls] | 1.000 | 1.000 | 1.000 | 1.000 |
| Degenerate (m5) [n: calls] | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 | 1.000 | 1.000 | 1.000 |
| Words per call [n: calls] | 113.5 | 98.7 | 103.5 | 105.2 |

### B: band long (16 calls)

| metric | base (16 calls × 1 seed) | V3_A@200 (16 calls × 1 seed) | V4_A@350 (16 calls × 1 seed) | V4_A2@600 (16 calls × 1 seed) |
|---|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.250 | 0.469 | 0.688 | 0.688 |
| Judge: COMPLETE calls [n: calls] | 0.062 | 0.188 | 0.500 | 0.562 |
| Judge: FAILED calls [n: calls] | 0.562 | 0.250 | 0.125 | 0.188 |
| **Judge: facts CONSISTENT** [n: calls] | 0.438 | 0.625 | 0.438 | 0.438 |
| Judge: facts CONTRADICTS [n: calls] | 0.500 | 0.375 | 0.438 | 0.375 |
| Judge: wrong facts per call [n: calls] | 0.688 | 0.562 | 1.062 | 0.812 |
| Judge: write confirmed with the right value [n: writes] | 0.167 (pooled 0.158, n=19) | 0.403 (pooled 0.368, n=19) | 0.611 (pooled 0.579, n=19) | 0.722 (pooled 0.684, n=19) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.000 (pooled 0.000, n=19) | 0.250 (pooled 0.263, n=19) | 0.556 (pooled 0.526, n=19) | 0.333 (pooled 0.316, n=19) |
| Check-line before write (score.py m3) [n: writes] | 0.042 (pooled 0.053, n=19) | 0.403 (pooled 0.474, n=19) | 0.625 (pooled 0.684, n=19) | 0.611 (pooled 0.632, n=19) |
| **NATURAL lines** [n: lines] | 0.730 (pooled 0.730, n=374) | 0.585 (pooled 0.587, n=223) | 0.563 (pooled 0.562, n=249) | 0.555 (pooled 0.564, n=243) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=374) | 0.381 (n=223) | 0.410 (n=249) | 0.461 (n=243) |
| - NATURAL and pure English (en=true) [n: lines] | 0.730 (n=374) | 0.206 (n=223) | 0.153 (n=249) | 0.103 (n=243) |
| STIFF lines [n: lines] | 0.127 (pooled 0.126, n=374) | 0.163 (pooled 0.161, n=223) | 0.144 (pooled 0.137, n=249) | 0.170 (pooled 0.165, n=243) |
| PEPPERED lines [n: lines] | 0.000 (pooled 0.000, n=374) | 0.013 (pooled 0.013, n=223) | 0.058 (pooled 0.060, n=249) | 0.067 (pooled 0.066, n=243) |
| NONSENSE lines [n: lines] | 0.143 (pooled 0.144, n=374) | 0.239 (pooled 0.238, n=223) | 0.235 (pooled 0.241, n=249) | 0.208 (pooled 0.206, n=243) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.120 (n=374) | 0.139 (n=223) | 0.197 (n=249) | 0.169 (n=243) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=374) | 0.507 (n=223) | 0.494 (n=249) | 0.589 (n=243) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.053 (pooled 0.051, n=10391) | 0.090 (pooled 0.089, n=9676) | 0.062 (pooled 0.061, n=10254) | 0.071 (pooled 0.070, n=10416) |
| Trelis WER m1 [n: ref words] | 0.107 (pooled 0.103, n=1974) | 0.165 (pooled 0.163, n=1803) | 0.125 (pooled 0.123, n=1917) | 0.137 (pooled 0.136, n=1926) |
| Model Hindi share (score.py m1) [n: calls] | 0.000 | 0.484 | 0.537 | 0.573 |
| Read rate (m4) [n: calls] | 0.698 | 0.424 | 0.547 | 0.496 |
| Value rate (m4) [n: calls] | 0.644 | 0.395 | 0.531 | 0.561 |
| Invented-fact flag (m4) [n: calls] | 0.562 | 0.375 | 0.438 | 0.188 |
| Greeting (m5) [n: calls] | 1.000 | 1.000 | 1.000 | 1.000 |
| Degenerate (m5) [n: calls] | 0.062 | 0.000 | 0.000 | 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 | 1.000 | 1.000 | 1.000 |
| Words per call [n: calls] | 114.2 | 104.6 | 112.5 | 113.4 |

### B: band mixed (1 call)

| metric | base (1 call × 1 seed) | V3_A@200 (1 call × 1 seed) | V4_A@350 (1 call × 1 seed) | V4_A2@600 (1 call × 1 seed) |
|---|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 1.000 | 1.000 | 1.000 | 0.500 |
| Judge: COMPLETE calls [n: calls] | 1.000 | 1.000 | 1.000 | 0.000 |
| Judge: FAILED calls [n: calls] | 0.000 | 0.000 | 0.000 | 0.000 |
| **Judge: facts CONSISTENT** [n: calls] | 1.000 | 0.000 | 1.000 | 1.000 |
| Judge: facts CONTRADICTS [n: calls] | 0.000 | 1.000 | 0.000 | 0.000 |
| Judge: wrong facts per call [n: calls] | 0.000 | 2.000 | 0.000 | 0.000 |
| Judge: write confirmed with the right value [n: writes] | - | - | - | - |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | - | - | - | - |
| Check-line before write (score.py m3) [n: writes] | - | - | - | - |
| **NATURAL lines** [n: lines] | 0.842 (pooled 0.842, n=19) | 0.571 (pooled 0.571, n=14) | 0.615 (pooled 0.615, n=13) | 0.667 (pooled 0.667, n=12) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=19) | 0.286 (n=14) | 0.538 (n=13) | 0.500 (n=12) |
| - NATURAL and pure English (en=true) [n: lines] | 0.842 (n=19) | 0.286 (n=14) | 0.077 (n=13) | 0.167 (n=12) |
| STIFF lines [n: lines] | 0.105 (pooled 0.105, n=19) | 0.214 (pooled 0.214, n=14) | 0.077 (pooled 0.077, n=13) | 0.167 (pooled 0.167, n=12) |
| PEPPERED lines [n: lines] | 0.000 (pooled 0.000, n=19) | 0.000 (pooled 0.000, n=14) | 0.000 (pooled 0.000, n=13) | 0.000 (pooled 0.000, n=12) |
| NONSENSE lines [n: lines] | 0.053 (pooled 0.053, n=19) | 0.214 (pooled 0.214, n=14) | 0.308 (pooled 0.308, n=13) | 0.167 (pooled 0.167, n=12) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.053 (n=19) | 0.143 (n=14) | 0.308 (n=13) | 0.167 (n=12) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=19) | 0.429 (n=14) | 0.615 (n=13) | 0.667 (n=12) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.031 (pooled 0.031, n=738) | 0.086 (pooled 0.086, n=555) | 0.045 (pooled 0.045, n=572) | 0.060 (pooled 0.060, n=565) |
| Trelis WER m1 [n: ref words] | 0.066 (pooled 0.066, n=136) | 0.202 (pooled 0.202, n=99) | 0.140 (pooled 0.140, n=100) | 0.151 (pooled 0.151, n=99) |
| Model Hindi share (score.py m1) [n: calls] | 0.000 | 0.494 | 0.549 | 0.523 |
| Read rate (m4) [n: calls] | 0.875 | 0.675 | 1.000 | 0.500 |
| Value rate (m4) [n: calls] | 0.667 | 0.333 | 1.000 | 0.667 |
| Invented-fact flag (m4) [n: calls] | 0.000 | 0.000 | 0.000 | 0.000 |
| Greeting (m5) [n: calls] | 1.000 | 1.000 | 1.000 | 1.000 |
| Degenerate (m5) [n: calls] | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 | 1.000 | 1.000 | 1.000 |
| Words per call [n: calls] | 126.0 | 96.0 | 94.0 | 96.0 |

## C. All metrics × length band, 3 seeds (1001-1003), base / V3_A@200 / V4_A@350

V4_A2 is absent here (seed 1001 only).

### C: band all (30 calls)

| metric | base (30 calls × 3 seeds) | V3_A@200 (30 calls × 3 seeds) | V4_A@350 (30 calls × 3 seeds) |
|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.356 ± 0.055 | 0.539 ± 0.016 | 0.706 ± 0.008 |
| Judge: COMPLETE calls [n: calls] | 0.144 ± 0.087 | 0.300 ± 0.027 | 0.478 ± 0.031 |
| Judge: FAILED calls [n: calls] | 0.433 ± 0.054 | 0.222 ± 0.057 | 0.067 ± 0.027 |
| **Judge: facts CONSISTENT** [n: calls] | 0.500 ± 0.047 | 0.511 ± 0.069 | 0.567 ± 0.027 |
| Judge: facts CONTRADICTS [n: calls] | 0.433 ± 0.072 | 0.422 ± 0.069 | 0.344 ± 0.016 |
| Judge: wrong facts per call [n: calls] | 0.744 ± 0.096 | 0.633 ± 0.109 | 0.589 ± 0.103 |
| Judge: write confirmed with the right value [n: writes] | 0.240 ± 0.041 (pooled 0.275, n=102) | 0.495 ± 0.066 (pooled 0.461, n=102) | 0.689 ± 0.028 (pooled 0.637, n=102) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.053 ± 0.043 (pooled 0.059, n=102) | 0.247 ± 0.070 (pooled 0.235, n=102) | 0.472 ± 0.053 (pooled 0.451, n=102) |
| Check-line before write (score.py m3) [n: writes] | 0.028 ± 0.025 (pooled 0.039, n=102) | 0.515 ± 0.081 (pooled 0.520, n=102) | 0.654 ± 0.025 (pooled 0.676, n=102) |
| **NATURAL lines** [n: lines] | 0.759 ± 0.019 (pooled 0.758, n=1983) | 0.566 ± 0.021 (pooled 0.568, n=1244) | 0.570 ± 0.018 (pooled 0.570, n=1355) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=1983) | 0.375 (n=1244) | 0.441 (n=1355) |
| - NATURAL and pure English (en=true) [n: lines] | 0.758 (n=1983) | 0.192 (n=1244) | 0.128 (n=1355) |
| STIFF lines [n: lines] | 0.141 ± 0.009 (pooled 0.139, n=1983) | 0.180 ± 0.023 (pooled 0.175, n=1244) | 0.141 ± 0.005 (pooled 0.138, n=1355) |
| PEPPERED lines [n: lines] | 0.000 ± 0.000 (pooled 0.000, n=1983) | 0.022 ± 0.001 (pooled 0.023, n=1244) | 0.054 ± 0.006 (pooled 0.055, n=1355) |
| NONSENSE lines [n: lines] | 0.100 ± 0.011 (pooled 0.102, n=1983) | 0.233 ± 0.002 (pooled 0.234, n=1244) | 0.235 ± 0.011 (pooled 0.238, n=1355) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.085 (n=1983) | 0.159 (n=1244) | 0.207 (n=1355) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=1983) | 0.524 (n=1244) | 0.514 (n=1355) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.057 ± 0.001 (pooled 0.057, n=58364) | 0.082 ± 0.004 (pooled 0.082, n=52818) | 0.065 ± 0.002 (pooled 0.065, n=55141) |
| Trelis WER m1 [n: ref words] | 0.115 ± 0.003 (pooled 0.114, n=11107) | 0.167 ± 0.006 (pooled 0.166, n=9803) | 0.131 ± 0.003 (pooled 0.130, n=10234) |
| Model Hindi share (score.py m1) [n: calls] | 0.001 ± 0.001 | 0.486 ± 0.010 | 0.551 ± 0.005 |
| Read rate (m4) [n: calls] | 0.641 ± 0.029 | 0.508 ± 0.024 | 0.561 ± 0.036 |
| Value rate (m4) [n: calls] | 0.489 ± 0.049 | 0.381 ± 0.016 | 0.533 ± 0.072 |
| Invented-fact flag (m4) [n: calls] | 0.656 ± 0.110 | 0.278 ± 0.031 | 0.211 ± 0.016 |
| Greeting (m5) [n: calls] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Degenerate (m5) [n: calls] | 0.011 ± 0.016 | 0.011 ± 0.016 | 0.000 ± 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Words per call [n: calls] | 113.3 ± 3.7 | 102.3 ± 0.5 | 106.8 ± 0.8 |

### C: band standard (13 calls)

| metric | base (13 calls × 3 seeds) | V3_A@200 (13 calls × 3 seeds) | V4_A@350 (13 calls × 3 seeds) |
|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.385 ± 0.083 | 0.551 ± 0.048 | 0.744 ± 0.036 |
| Judge: COMPLETE calls [n: calls] | 0.128 ± 0.096 | 0.308 ± 0.063 | 0.513 ± 0.036 |
| Judge: FAILED calls [n: calls] | 0.359 ± 0.096 | 0.205 ± 0.072 | 0.026 ± 0.036 |
| **Judge: facts CONSISTENT** [n: calls] | 0.513 ± 0.096 | 0.436 ± 0.072 | 0.564 ± 0.036 |
| Judge: facts CONTRADICTS [n: calls] | 0.410 ± 0.096 | 0.462 ± 0.063 | 0.282 ± 0.072 |
| Judge: wrong facts per call [n: calls] | 0.641 ± 0.036 | 0.692 ± 0.109 | 0.513 ± 0.096 |
| Judge: write confirmed with the right value [n: writes] | 0.300 ± 0.082 (pooled 0.356, n=45) | 0.500 ± 0.108 (pooled 0.467, n=45) | 0.767 ± 0.047 (pooled 0.733, n=45) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.050 ± 0.000 (pooled 0.067, n=45) | 0.333 ± 0.103 (pooled 0.289, n=45) | 0.433 ± 0.062 (pooled 0.422, n=45) |
| Check-line before write (score.py m3) [n: writes] | 0.033 ± 0.047 (pooled 0.044, n=45) | 0.517 ± 0.062 (pooled 0.533, n=45) | 0.683 ± 0.062 (pooled 0.711, n=45) |
| **NATURAL lines** [n: lines] | 0.751 ± 0.015 (pooled 0.751, n=810) | 0.550 ± 0.030 (pooled 0.552, n=514) | 0.578 ± 0.012 (pooled 0.580, n=574) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=810) | 0.356 (n=514) | 0.450 (n=574) |
| - NATURAL and pure English (en=true) [n: lines] | 0.751 (n=810) | 0.197 (n=514) | 0.131 (n=574) |
| STIFF lines [n: lines] | 0.163 ± 0.021 (pooled 0.162, n=810) | 0.201 ± 0.046 (pooled 0.198, n=514) | 0.141 ± 0.016 (pooled 0.138, n=574) |
| PEPPERED lines [n: lines] | 0.000 ± 0.000 (pooled 0.000, n=810) | 0.023 ± 0.010 (pooled 0.025, n=514) | 0.057 ± 0.009 (pooled 0.058, n=574) |
| NONSENSE lines [n: lines] | 0.087 ± 0.008 (pooled 0.088, n=810) | 0.226 ± 0.006 (pooled 0.224, n=514) | 0.224 ± 0.018 (pooled 0.225, n=574) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.072 (n=810) | 0.163 (n=514) | 0.204 (n=574) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=810) | 0.525 (n=514) | 0.500 (n=574) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.060 ± 0.005 (pooled 0.060, n=24818) | 0.083 ± 0.000 (pooled 0.083, n=21934) | 0.061 ± 0.008 (pooled 0.061, n=22800) |
| Trelis WER m1 [n: ref words] | 0.120 ± 0.008 (pooled 0.118, n=4727) | 0.178 ± 0.010 (pooled 0.178, n=4054) | 0.125 ± 0.005 (pooled 0.125, n=4221) |
| Model Hindi share (score.py m1) [n: calls] | 0.001 ± 0.001 | 0.481 ± 0.016 | 0.544 ± 0.011 |
| Read rate (m4) [n: calls] | 0.652 ± 0.014 | 0.581 ± 0.067 | 0.563 ± 0.039 |
| Value rate (m4) [n: calls] | 0.394 ± 0.069 | 0.353 ± 0.037 | 0.472 ± 0.083 |
| Invented-fact flag (m4) [n: calls] | 0.692 ± 0.188 | 0.385 ± 0.126 | 0.077 ± 0.063 |
| Greeting (m5) [n: calls] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Degenerate (m5) [n: calls] | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Words per call [n: calls] | 110.4 ± 2.4 | 97.7 ± 1.0 | 101.9 ± 1.1 |

### C: band long (16 calls)

| metric | base (16 calls × 3 seeds) | V3_A@200 (16 calls × 3 seeds) | V4_A@350 (16 calls × 3 seeds) |
|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.302 ± 0.053 | 0.521 ± 0.039 | 0.667 ± 0.029 |
| Judge: COMPLETE calls [n: calls] | 0.125 ± 0.088 | 0.292 ± 0.078 | 0.438 ± 0.088 |
| Judge: FAILED calls [n: calls] | 0.521 ± 0.029 | 0.250 ± 0.051 | 0.104 ± 0.029 |
| **Judge: facts CONSISTENT** [n: calls] | 0.458 ± 0.029 | 0.583 ± 0.059 | 0.542 ± 0.078 |
| Judge: facts CONTRADICTS [n: calls] | 0.479 ± 0.078 | 0.375 ± 0.051 | 0.417 ± 0.029 |
| Judge: wrong facts per call [n: calls] | 0.875 ± 0.153 | 0.562 ± 0.102 | 0.688 ± 0.265 |
| Judge: write confirmed with the right value [n: writes] | 0.190 ± 0.033 (pooled 0.210, n=57) | 0.491 ± 0.062 (pooled 0.456, n=57) | 0.625 ± 0.020 (pooled 0.561, n=57) |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | 0.056 ± 0.079 (pooled 0.053, n=57) | 0.176 ± 0.054 (pooled 0.193, n=57) | 0.505 ± 0.046 (pooled 0.474, n=57) |
| Check-line before write (score.py m3) [n: writes] | 0.023 ± 0.017 (pooled 0.035, n=57) | 0.514 ± 0.097 (pooled 0.509, n=57) | 0.630 ± 0.062 (pooled 0.649, n=57) |
| **NATURAL lines** [n: lines] | 0.761 ± 0.024 (pooled 0.761, n=1117) | 0.575 ± 0.015 (pooled 0.575, n=690) | 0.561 ± 0.027 (pooled 0.559, n=740) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=1117) | 0.393 (n=690) | 0.432 (n=740) |
| - NATURAL and pure English (en=true) [n: lines] | 0.761 (n=1117) | 0.183 (n=690) | 0.127 (n=740) |
| STIFF lines [n: lines] | 0.125 ± 0.003 (pooled 0.123, n=1117) | 0.163 ± 0.013 (pooled 0.158, n=690) | 0.141 ± 0.013 (pooled 0.138, n=740) |
| PEPPERED lines [n: lines] | 0.000 ± 0.000 (pooled 0.000, n=1117) | 0.020 ± 0.005 (pooled 0.022, n=690) | 0.055 ± 0.012 (pooled 0.055, n=740) |
| NONSENSE lines [n: lines] | 0.114 ± 0.024 (pooled 0.116, n=1117) | 0.242 ± 0.003 (pooled 0.245, n=690) | 0.243 ± 0.007 (pooled 0.247, n=740) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.098 (n=1117) | 0.157 (n=690) | 0.209 (n=740) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=1117) | 0.526 (n=690) | 0.522 (n=740) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.056 ± 0.005 (pooled 0.056, n=31656) | 0.081 ± 0.008 (pooled 0.081, n=29206) | 0.070 ± 0.005 (pooled 0.069, n=30545) |
| Trelis WER m1 [n: ref words] | 0.113 ± 0.008 (pooled 0.112, n=6025) | 0.156 ± 0.009 (pooled 0.156, n=5448) | 0.133 ± 0.006 (pooled 0.132, n=5699) |
| Model Hindi share (score.py m1) [n: calls] | 0.000 ± 0.000 | 0.493 ± 0.007 | 0.558 ± 0.015 |
| Read rate (m4) [n: calls] | 0.621 ± 0.059 | 0.442 ± 0.023 | 0.543 ± 0.060 |
| Value rate (m4) [n: calls] | 0.553 ± 0.073 | 0.408 ± 0.009 | 0.562 ± 0.060 |
| Invented-fact flag (m4) [n: calls] | 0.646 ± 0.078 | 0.188 ± 0.135 | 0.312 ± 0.088 |
| Greeting (m5) [n: calls] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Degenerate (m5) [n: calls] | 0.021 ± 0.029 | 0.021 ± 0.029 | 0.000 ± 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Words per call [n: calls] | 115.9 ± 7.8 | 106.4 ± 1.4 | 111.3 ± 0.9 |

### C: band mixed (1 call)

| metric | base (1 call × 3 seeds) | V3_A@200 (1 call × 3 seeds) | V4_A@350 (1 call × 3 seeds) |
|---|---|---|---|
| **Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0) [n: calls] | 0.833 ± 0.236 | 0.667 ± 0.236 | 0.833 ± 0.236 |
| Judge: COMPLETE calls [n: calls] | 0.667 ± 0.471 | 0.333 ± 0.471 | 0.667 ± 0.471 |
| Judge: FAILED calls [n: calls] | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| **Judge: facts CONSISTENT** [n: calls] | 1.000 ± 0.000 | 0.333 ± 0.471 | 1.000 ± 0.000 |
| Judge: facts CONTRADICTS [n: calls] | 0.000 ± 0.000 | 0.667 ± 0.471 | 0.000 ± 0.000 |
| Judge: wrong facts per call [n: calls] | 0.000 ± 0.000 | 1.000 ± 0.817 | 0.000 ± 0.000 |
| Judge: write confirmed with the right value [n: writes] | - | - | - |
| **Echo** (validate.echo_ok on the confirm slot) [n: writes] | - | - | - |
| Check-line before write (score.py m3) [n: writes] | - | - | - |
| **NATURAL lines** [n: lines] | 0.826 ± 0.048 (pooled 0.821, n=56) | 0.623 ± 0.065 (pooled 0.625, n=40) | 0.608 ± 0.018 (pooled 0.610, n=41) |
| - NATURAL and Hinglish (en=false) [n: lines] | 0.000 (n=56) | 0.325 (n=40) | 0.488 (n=41) |
| - NATURAL and pure English (en=true) [n: lines] | 0.821 (n=56) | 0.300 (n=40) | 0.122 (n=41) |
| STIFF lines [n: lines] | 0.119 ± 0.053 (pooled 0.125, n=56) | 0.175 ± 0.030 (pooled 0.175, n=40) | 0.137 ± 0.080 (pooled 0.146, n=41) |
| PEPPERED lines [n: lines] | 0.000 ± 0.000 (pooled 0.000, n=56) | 0.028 ± 0.039 (pooled 0.025, n=40) | 0.000 ± 0.000 (pooled 0.000, n=41) |
| NONSENSE lines [n: lines] | 0.054 ± 0.006 (pooled 0.054, n=56) | 0.175 ± 0.030 (pooled 0.175, n=40) | 0.255 ± 0.093 (pooled 0.244, n=41) |
| - NONSENSE that is a cut-off fragment (trunc) [n: lines] | 0.036 (n=56) | 0.150 (n=40) | 0.220 (n=41) |
| Lines where Hindi carries content [n: lines] | 0.000 (n=56) | 0.475 (n=40) | 0.561 (n=41) |
| **Trelis CER m1** (roman vs roman, trimmed) [n: ref chars] | 0.036 ± 0.010 (pooled 0.035, n=1890) | 0.072 ± 0.015 (pooled 0.073, n=1678) | 0.052 ± 0.015 (pooled 0.052, n=1796) |
| Trelis WER m1 [n: ref words] | 0.080 ± 0.011 (pooled 0.079, n=355) | 0.173 ± 0.024 (pooled 0.173, n=301) | 0.153 ± 0.013 (pooled 0.153, n=314) |
| Model Hindi share (score.py m1) [n: calls] | 0.000 ± 0.000 | 0.439 ± 0.064 | 0.531 ± 0.022 |
| Read rate (m4) [n: calls] | 0.750 ± 0.177 | 0.425 ± 0.177 | 0.767 ± 0.166 |
| Value rate (m4) [n: calls] | 0.667 ± 0.000 | 0.333 ± 0.000 | 0.778 ± 0.314 |
| Invented-fact flag (m4) [n: calls] | 0.333 ± 0.471 | 0.333 ± 0.471 | 0.333 ± 0.471 |
| Greeting (m5) [n: calls] | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Degenerate (m5) [n: calls] | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Judge r2 (Hinglish register) [n: calls] | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Words per call [n: calls] | 109.3 ± 11.8 | 95.3 ± 3.3 | 99.3 ± 4.5 |

### C: standard / long, 3 seeds (seed means; pooled rates)

| model | task score standard / long | facts CONSISTENT standard / long | judge confirm+value standard / long | echo standard / long | NATURAL standard / long | NATURAL Hinglish standard / long | NONSENSE standard / long | Trelis CER m1 standard / long |
|---|---|---|---|---|---|---|---|---|
| base | 0.38 / 0.30 | 0.51 / 0.46 | 0.36 / 0.21 | 0.07 / 0.05 | 0.75 / 0.76 | 0.00 / 0.00 | 0.09 / 0.12 | 0.06 / 0.06 |
| V3_A@200 | 0.55 / 0.52 | 0.44 / 0.58 | 0.47 / 0.46 | 0.29 / 0.19 | 0.55 / 0.58 | 0.36 / 0.39 | 0.22 / 0.24 | 0.08 / 0.08 |
| V4_A@350 | 0.74 / 0.67 | 0.56 / 0.54 | 0.73 / 0.56 | 0.42 / 0.47 | 0.58 / 0.56 | 0.45 / 0.43 | 0.22 / 0.25 | 0.06 / 0.07 |

## D. Data (data/V4/STATS.md, md5 a2278a74)

- Scenarios: 162 (guidance/scenario_creation_v2.json; 72 old + new, 7 agent types incl. bank and telecom). Records: data/V4/records.json, regenerated with seeded pools and a 3-record reuse cap.
- Calls: **645 of 648** (4 gender pairings per scenario); missing food_18_g1, sub_12_g2, sub_12_g3 (none in test or val). First text run 570/648 accepted; a top-up regenerated the 78 drops and added 75.
- Length band (hash-assigned): standard 265, long 261, mixed 119 calls. Audio 878.2 min (14.64 h), 67.4-116.0 s per call, mean 81.7 s; standard 77.9 s, long 86.7 s, mixed 79.1 s mean.
- Splits: train 465 calls / 117 scenarios / 633.8 min (188 / 188 / 89 standard / long / mixed); val 60 calls / 15 scenarios / 81.6 min; test 30 calls / 30 scenarios / 40.8 min. No scenario overlaps between train, val and test.
- TTS: IndicF5 code-switch (f5cs), the four V3 voices; long and mixed calls use 24-word / 140-Devanagari chunks. Trelis CER flags (> 0.10, best of 3 tries kept): 142/9174 chunks = 1.55% (standard 1.64%, long 1.27%, mixed 1.88%). Number-word mismatches (reported only): 419/2528 chunks with numbers = 16.6%. MMS alignment fallbacks 0. Chunks still ending loud after the tail guard 0.40%.
- Hindi share (agent / customer): standard 0.544 / 0.603, long 0.576 / 0.630, mixed 0.567 / 0.610.
- Trainer window 140 s (longest train/val call 116.0 s, so every call is one window).

## E. Training curves (runs/V4_A, runs/V4_A2: loss.png, eval_metrics.csv)

Common config: rank 64, scaling 2.0, batch 8, duration_sec 140, keep_and_shift on (text stream shifted; new runs only), eval every 50 steps on val (60 calls) and control (English CONTROL set, 100 windows). Peak VRAM 37.2 GB, 5.6 s/step. Losses are not comparable with V3_A (keep_and_shift, different val set).

### V4_A (lr 1.5e-5, 1200 steps, 126 GPU-min)

| step | val total_pooled | val text | val cb1 | val cb2-8 | control total_pooled | control text | control cb2-8 | note |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.2379 | 0.8235 | 1.3696 | 2.0553 | 1.3114 | 0.4376 | 2.1921 |  |
| 50 | 1.5162 | 0.3413 | 1.1267 | 1.8632 | 0.9868 | 0.2046 | 2.1538 |  |
| 100 | 1.1860 | 0.1877 | 0.9478 | 1.7214 | 0.9774 | 0.1889 | 2.2117 | control min |
| 350 | 1.0152 | 0.1423 | 0.8237 | 1.5765 | 1.1347 | 0.2103 | 2.6955 | FINAL_CKPT (re-rank pick) |
| 400 | 1.0139 | 0.1431 | 0.8219 | 1.5696 | 1.1493 | 0.2151 | 2.7521 | val best (BEST_CKPT) |
| 1200 | 1.3010 | 0.2003 | 1.0675 | 1.5743 | 1.4169 | 0.2765 | 3.1133 | end of run |

Val bottoms at step 400 and rises to the end. Control total goes below step 0 early, then rises above step 0 from step 650; control cb2-8 (acoustic) rises from step 0 for the whole run, so the early control gain is text and cb1 only. The early-stop condition was met at step 800; the run was not stopped (NOTES 'D2 / V4 training queue').

### V4_A2 (lr 7.5e-6, 1500 steps, 158 GPU-min)

| step | val total_pooled | val text | val cb1 | val cb2-8 | control total_pooled | control text | control cb2-8 | note |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.2379 | 0.8235 | 1.3696 | 2.0553 | 1.3114 | 0.4376 | 2.1921 |  |
| 50 | 1.7520 | 0.4605 | 1.2456 | 1.9478 | 1.0524 | 0.2390 | 2.1547 |  |
| 100 | 1.3666 | 0.2650 | 1.0521 | 1.8077 | 0.9832 | 0.2004 | 2.1507 |  |
| 150 | 1.2121 | 0.1972 | 0.9641 | 1.7411 | 0.9775 | 0.1921 | 2.1822 | control min |
| 600 | 1.0326 | 0.1487 | 0.8346 | 1.5889 | 1.1336 | 0.2139 | 2.6561 | val best (BEST_CKPT), FINAL_CKPT (re-rank pick) |
| 1500 | 1.1557 | 0.1845 | 0.9285 | 1.5800 | 1.2792 | 0.2520 | 2.8957 | end of run |

Val bottoms at step 600; control total at the end (1.2792) stays just below step 0 (1.3114), but control cb2-8 also rises from step 0 (2.19 to 2.90). V4_A2 forgets English less than V4_A at the end of training (control total_pooled 1.2792 vs 1.4169); at the val-best steps the control cost is similar (1.1336 vs 1.1493).

## F. Re-rank (spec section 7; tests/V4_RERANK.md)

Top-3 val checkpoints per run, PersonaPlex on a 12-call subset (seed 1001), Gemma 4 31B naturalness judge; pick = highest NATURAL rate, tie -> lower val loss.

| run | step | val total_pooled | NATURAL | NONSENSE | pick |
|---|---|---|---|---|---|
| V4_A | 400 | 1.01392 | 99/182 (54.4%) | 32 (17.6%) | |
| V4_A | 350 | 1.01520 | 108/193 (56.0%) | 53 (27.5%) | **FINAL** |
| V4_A | 300 | 1.01933 | 108/197 (54.8%) | 37 (18.8%) | |
| V4_A2 | 600 | 1.03260 | 107/181 (59.1%) | 36 (19.9%) | **FINAL** |
| V4_A2 | 500 | 1.03305 | 105/199 (52.8%) | 39 (19.6%) | |
| V4_A2 | 450 | 1.03355 | 89/184 (48.4%) | 50 (27.2%) | |

Judge check against the hand review (V3_A@200, same 12 calls): judge 54.8% NATURAL / 27.1% NONSENSE vs hand 51% / 30%; 13/16 lines agree on air_03_g1.

## G. Caveats

1. **V4_A2 is single-seed** (seed 1001). Single-seed differences between V4_A and V4_A2 are within the seed-to-seed spread seen on the other tags (task score std up to ±0.06).
2. **V4_A step 350 vs 400 is open for the user.** The re-rank pick (350) is a tie within noise (1.6 pp spread vs ~3.6 pp SE) and has the most NONSENSE of the three; step 400 (val-best, fewest NONSENSE) was not run on the full test set.
3. **Mixed band = 1 test call**; its cells are not interpretable. Long vs standard is 16 vs 13 calls.
4. **Base's NATURAL rate is fluent pure English** (the rubric labels real English formulas NATURAL, en=true); compare the "NATURAL and Hinglish" row.
5. **Judges are Gemma 4 31B**, temperature 0; on the calibration set it was ~4 pp more lenient on NATURAL than the hand review. Task / fact judgements are per call, without human check.
6. **Echo is strict** (validate.echo_ok string match on the confirm slot); a garbled or translated value fails echo while the call judge may still accept it ("judge confirm+value" row).
7. **CER method**: Trelis Whisper-Hinglish, roman vs roman (romanised via the V4 data's own spellings, fixed-table fallback), silence-trimmed at -50 dBFS (0.2 s pad, gaps -> 0.3 s); the reference is the model's own text stream, so CER measures speech-vs-text agreement, not correctness.
8. **Training losses** are not comparable to V3_A; V4_A control cb2-8 got worse from step 0.
9. **Data**: 3 calls missing (none in test/val); 16.6% of number-bearing chunks have a number-word mismatch (not re-rendered; open question). data/V4/SAMPLES.md and gen_report.json describe the first 570-call text run; STATS.md describes the final 645 calls. The top-up regenerated the dropped calls with greeting/sign-off turns exempt from the per-band line caps and 'reference' exempt from the nukta check (data/V4/topup_go.sh header: user, 2026-10-04 ~20:00 IST).
10. V3_A@200 was trained on V3 data and is tested here on V4 inputs (new scenarios, new records, longer calls).
