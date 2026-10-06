# V4 re-rank (D2 spec section 7)

Top-3 checkpoints by val total_pooled per run, re-ranked by the Gemma 4 31B naturalness judge (tests/judge.py --naturalness, v3_hinglish_review rubric: NATURAL / STIFF / PEPPERED / NONSENSE per utterance). Final pick per run = highest NATURAL rate; tie -> lower val loss.

- Test subset (12 of the 30 V4 test calls, seed 1001, no gate0): air_11_g2, air_26_g3, bank_07_g1, bank_20_g4, cab_03_g1, cab_26_g4, ecom_11_g3, ecom_23_g1, food_03_g2, food_07_g3, sub_11_g4, tel_03_g2
- Balance: 7 agent types (air, bank, cab, ecom, food 2 each; sub, tel 1 each); g1-g4 3 each (6 female / 6 male agents); length band 6 long / 5 standard / 1 mixed (air_26_g3 is the only mixed test call); writes per call 0:2, 1:4, 2:5, 3:1.
- Segmentation: tcommon.segments() (new utterance after >= 15 PAD/EPAD frames or after . ? !), deterministic; the same segmenter score.py uses.
- Judge: Gemma 4 31B, temperature 0, one prompt per call (agent and customer gender, brand), JSON schema with one label per utterance; outputs tests/out/rr4_<run>_s<step>/V4/<call>_s1001.nat.json.

## Results

| Run | Step | Val total_pooled | Control total_pooled | Calls | Utterances | NATURAL | STIFF | PEPPERED | NONSENSE (trunc) | Hindi carries content | Pick |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V4_A | 400 | 1.01392 | 1.14930 | 12 | 182 | 99/182 (54.4%) | 27 | 24 | 32 (27) | 97/182 (53.3%) |  |
| V4_A | 350 | 1.01520 | 1.13466 | 12 | 193 | 108/193 (56.0%) | 20 | 12 | 53 (43) | 94/193 (48.7%) | **FINAL** |
| V4_A | 300 | 1.01933 | 1.10650 | 12 | 197 | 108/197 (54.8%) | 40 | 12 | 37 (32) | 121/197 (61.4%) |  |
| V4_A2 | 600 | 1.03260 | 1.13362 | 12 | 181 | 107/181 (59.1%) | 28 | 10 | 36 (28) | 107/181 (59.1%) | **FINAL** |
| V4_A2 | 500 | 1.03305 | 1.10073 | 12 | 199 | 105/199 (52.8%) | 36 | 19 | 39 (30) | 114/199 (57.3%) |  |
| V4_A2 | 450 | 1.03355 | 1.08458 | 12 | 184 | 89/184 (48.4%) | 32 | 13 | 50 (38) | 99/184 (53.8%) |  |

## NATURAL / NONSENSE by length band

| Run | Step | standard NATURAL | standard NONSENSE | long NATURAL | long NONSENSE | mixed NATURAL | mixed NONSENSE |
|---|---|---|---|---|---|---|---|
| V4_A | 400 | 37/73 (50.7%) | 17/73 (23.3%) | 52/94 (55.3%) | 12/94 (12.8%) | 10/15 (66.7%) | 3/15 (20.0%) |
| V4_A | 350 | 49/78 (62.8%) | 18/78 (23.1%) | 51/102 (50.0%) | 31/102 (30.4%) | 8/13 (61.5%) | 4/13 (30.8%) |
| V4_A | 300 | 45/82 (54.9%) | 11/82 (13.4%) | 55/101 (54.5%) | 23/101 (22.8%) | 8/14 (57.1%) | 3/14 (21.4%) |
| V4_A2 | 600 | 45/77 (58.4%) | 15/77 (19.5%) | 54/92 (58.7%) | 19/92 (20.7%) | 8/12 (66.7%) | 2/12 (16.7%) |
| V4_A2 | 500 | 44/81 (54.3%) | 15/81 (18.5%) | 53/103 (51.5%) | 22/103 (21.4%) | 8/15 (53.3%) | 2/15 (13.3%) |
| V4_A2 | 450 | 44/80 (55.0%) | 21/80 (26.2%) | 40/90 (44.4%) | 25/90 (27.8%) | 5/14 (35.7%) | 4/14 (28.6%) |

## NATURAL per call (natural / utterances)

| Call | band | rr4_V4_A_s400 | rr4_V4_A_s350 | rr4_V4_A_s300 | rr4_V4_A2_s600 | rr4_V4_A2_s500 | rr4_V4_A2_s450 |
|---|---|---|---|---|---|---|---|
| air_11_g2 | long | 10/19 | 8/18 | 9/18 | 9/16 | 7/19 | 6/16 |
| air_26_g3 | mixed | 10/15 | 8/13 | 8/14 | 8/12 | 8/15 | 5/14 |
| bank_07_g1 | standard | 9/14 | 13/16 | 11/20 | 10/15 | 12/19 | 7/15 |
| bank_20_g4 | long | 4/13 | 7/15 | 5/16 | 6/13 | 7/15 | 7/15 |
| cab_03_g1 | long | 11/16 | 12/18 | 10/16 | 12/16 | 11/20 | 5/14 |
| cab_26_g4 | standard | 5/18 | 13/18 | 11/17 | 8/15 | 10/18 | 8/18 |
| ecom_11_g3 | long | 10/14 | 7/17 | 10/19 | 9/14 | 8/13 | 9/16 |
| ecom_23_g1 | long | 12/18 | 9/19 | 15/19 | 12/20 | 11/20 | 8/17 |
| food_03_g2 | standard | 6/14 | 5/14 | 6/14 | 6/13 | 6/14 | 7/12 |
| food_07_g3 | standard | 10/13 | 10/15 | 9/16 | 12/18 | 9/16 | 13/20 |
| sub_11_g4 | standard | 7/14 | 8/15 | 8/15 | 9/16 | 7/14 | 9/15 |
| tel_03_g2 | long | 5/14 | 8/15 | 6/13 | 6/13 | 9/16 | 5/12 |

## Reading the result

- **Final picks (rule applied as written):** V4_A -> step 350 (NATURAL 108/193 = 56.0%); V4_A2 -> step 600 (107/181 = 59.1%, which is also its val-best checkpoint). runs/V4_A/FINAL_CKPT and runs/V4_A2/FINAL_CKPT.
- **The differences are within noise.** With ~180-200 lines per checkpoint, one binomial SE is about 3.6 percentage points. The spread inside V4_A is 54.4-56.0% (1.6 pp), so the V4_A pick is effectively a tie decided by noise. The V4_A2 spread (48.4-59.1%) is larger, but it is still a single seed on 12 calls.
- **The V4_A pick is not the cleanest checkpoint.** Step 350 has the most NONSENSE (53/193 = 27.5%, 43 of them cut-off fragments). Step 400 has 32/182 = 17.6% and step 300 has 37/197 = 18.8%. The rubric's PEPPERED count is higher at step 400 (24, against 12 for 350 and 300). If the user prefers "fewest broken lines" over "most natural lines", V4_A would pick step 400 (its val-best) instead. This is not decided here.
- Across all six checkpoints, V4_A2@600 has the highest NATURAL rate and the lowest NONSENSE count except V4_A@400.
- **Length band:** no consistent long-vs-standard gap. The section 8 full test (30 calls × 3 seeds) is the place to answer that question.

## Judge calibration (V3_A@200, same 12 calls the hand review used)

- The same judge on tests/out/V3_A/V3/*_s1001.json (outputs in tests/out/rr4_calib/V3_A/V3/; out/V3_A was not touched): 166 segments, NATURAL 91 (54.8%), STIFF 26, PEPPERED 4, NONSENSE 45 (27.1%; 39 trunc), Hindi carries content 81 (48.8%).
- The hand review (V3_HINGLISH_REVIEW.md) found 142 utterances: NATURAL 73 (51%), STIFF 23, PEPPERED 4, NONSENSE 42 (30%), Hindi content 50%. The segment counts differ (166 vs 142) because the hand reviewer merged some splits, for example the greeting plus "Main aapki kaise help kar sakti hoon?".
- Per line on air_03_g1, after aligning the merged greeting, the judge agrees with the hand labels on 13 of 16 lines. The judge is about 3-4 pp more lenient on NATURAL.
- Caveats: the judge reads text streams, not audio. It is an LLM judge with temperature 0, and this is one seed.
