# V1 comparison: pre-registered winner rule (2026-10-03, v1-compare worker)

Source data: tests/out/{base_V1,V1_A,V1_B,V1_C}/runs.csv, written by the queue's score.py. The queue ended with QUEUE DONE at 14:23:42 UTC and score exited 0, so nothing was re-run.

## Rule (NOTES, "Orchestrator decisions after Gate 1", pre-registered; applied unchanged)
- **Disqualification (metric 5):** greeting present in <50% of runs, OR degenerate 3-gram repetition (max 3-gram count >4) in >20% of runs.
- **Score:** S = mean(m3, m4, m1) - 0.05 x (invented number/date/price flags per run), where
  - m3 = m3_check_sil_rate: a check-line before the write, including >= 0.6 s of silence;
  - m4 = m4_fact_rate: prompt-fact relay;
  - m1 = m1_hindi_recall: recall of the scripted Hindi vocabulary;
  - the penalty term is m4_invented, the mean count per run.
- **Tie (|dS| < 0.02):** fewer training steps wins. Base is reported but cannot win.
- **Common subset** (NOTES, "reduced V1 test set"): 12 V1 calls at seed 1001. The calls are air_03_g1, air_16_g4, cab_07_g3, cab_11_g4, cab_16_g2, ecom_07_g4, ecom_11_g1, food_07_g1, food_11_g2, food_12_g3, sub_07_g2 and sub_11_g3. All 4 tags have all 12 runs. Gate-0 clips are excluded because they are not V1 test calls.
- **Aggregation:** mean over runs. m3 is averaged over the 9 calls that have scripted writes; the other 3 calls are None.

Caveat: V1_A/scores.csv also contains 1 leftover run (air_16_g4, seed 1002) from the killed full pass, which inflates its m3 to 0.833. That run is excluded here. base_V1/scores.csv is the full 25 x 3 set and is not the subset.

## Values on the common subset (12 calls x seed 1001)
| metric | base | A (step 250) | B (step 150) | C (step 350) |
|---|---|---|---|---|
| m5_greeting (share of runs) | 1.00 | 1.00 | 1.00 | 1.00 |
| m5_degenerate (share of runs) | 0.00 | 0.00 | 0.00 | 0.00 |
| m3_check_sil_rate (9 calls) | 0.0000 | 0.6667 | 0.3889 | 0.3889 |
| m4_fact_rate | 0.6768 | 0.7647 | 0.8551 | 0.7182 |
| m1_hindi_recall | 0.0186 | 0.6989 | 0.6806 | 0.6694 |
| m4_invented (count/run) | 0.500 | 0.0833 | 0.6667 | 0.2500 |
| m4_flag (share of runs) | 0.333 | 0.0833 | 0.3333 | 0.1667 |
| (info) m3_check_rate | 0.0 | 0.6667 | 0.3889 | 0.3889 |
| (info) m4_read_rate | 0.6921 | 0.7597 | 0.8636 | 0.7036 |
| (info) m4_value_rate | 0.5139 | 0.7743 | 0.9167 | 0.6875 |
| (info) m1_model_hindi_share | 0.0032 | 0.3376 | 0.3295 | 0.3061 |
| (info) m2_judge_vf | 0.0 | 4.00 | 3.92 | 3.42 |
| (info) m5_max_3gram | 1.67 | 2.42 | 2.33 | 2.50 |
| (info) m6_cer_fold (proxy) | 0.083 | 0.218 | 0.195 | 0.189 |

## Disqualifications
None. Every model has a greeting in 100% of runs and degenerate repetition in 0% of runs.

## S per adapter
- A: (0.6667 + 0.7647 + 0.6989)/3 - 0.05 x 0.0833 = 0.7101 - 0.0042 = **0.7059**
- B: (0.3889 + 0.8551 + 0.6806)/3 - 0.05 x 0.6667 = 0.6415 - 0.0333 = **0.6082**
- C: (0.3889 + 0.7182 + 0.6694)/3 - 0.05 x 0.2500 = 0.5922 - 0.0125 = **0.5797**
- base (not eligible): (0.0 + 0.6768 + 0.0186)/3 - 0.05 x 0.5 = **0.2068**

## Tie rule
dS(A - B) = 0.098 > 0.02, so there is no tie and the tie rule is not invoked.

## Robustness
The ranking does not change under the other readings of the rule:
| reading | A | B | C |
|---|---|---|---|
| penalty = m4_flag | 0.7059 | 0.6249 | 0.5838 |
| m4 = m4_read_rate | 0.7043 | 0.6110 | 0.5748 |
| m3 pooled over writes (14 writes) | 0.571 | 0.357 | 0.357 |

## Winner
**A, checkpoint step 250.** Path: /workspace/runs/V1_A/checkpoints/checkpoint_000250/consolidated. lr 1.5e-5, no ft_embed. The step is the best held-out-loss checkpoint chosen by pick_best.

- A leads on m3 (check-line + pause before writes) and on m1, and it invents the fewest values.
- B leads on fact relay (m4_fact 0.855, m4_value 0.917) but invents the most values (0.67 per run).

Caveat: n = 12 runs from a single seed. Per-call m3 is 0, 0.5 or 1 on only 9 calls, so the A-vs-B gap in m3 comes down to a few writes.

## Full-set numbers (reported separately; not used for the decision)
base_V1, 25 calls x 3 seeds (from scores.csv):
- m3_check_sil 0.0154
- m4_fact 0.6751
- m1_recall 0.0089
- m4_invented 0.493
- greeting 1.0
- degenerate 0.0

A, B and C have only the subset (plus A's 1 stray run).
