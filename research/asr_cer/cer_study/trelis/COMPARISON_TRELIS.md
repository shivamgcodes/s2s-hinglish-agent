# CER study, re-run with Trelis Whisper-Hinglish: comparison with large-v3

## 1. Model used

- **Repo id:** `Trelis/whisper-hinglish-preview`. It is a Whisper-large-v3 fine-tune of `ARTPARK-IISc/whisper-large-v3-vaani-hindi`, Apache-2.0.
- **Settings** (from `TRANSCRIBE.md` and `trelis_tx.py`):
  - bf16 on the pod GPU.
  - Decoder prompt `<|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>`, greedy decoding, `max_new_tokens` 440.
  - Audio: the same 24 kHz model wavs that large-v3 transcribed, FFT-resampled to 16 kHz.
  - Each wav is cut into chunks of 28 s or less, at the lowest-energy 20 ms frame inside the 18-28 s window. That gives 4-5 chunks per file, and the chunk texts are joined with a space.
- **Output script:** mixed. Hindi is written in Devanagari and English in Latin.
  - Numbers are written as English words ("twelve", "four hundred fifty"). The Trelis transcripts contain **0 digits**.
  - The large-v3 transcripts and the references both use digits.
- **Script mix, as the share of letters that are Devanagari** (Devanagari / (Devanagari + Latin)), mean of 10 pairs: **Trelis 0.208**, large-v3 0.690.
  - V1 pairs: Trelis 0.21-0.35, large-v3 0.63-1.00.
  - Base pairs: Trelis 0.02 and 0.01, large-v3 0.16 and 0.01.
- **Transcript length:** for all 10 pairs the Trelis transcript is longer than the large-v3 one (characters, raw text).
  - The largest gap is ecom_07_g4, at 750 vs 445 characters.

Nothing else changed from the large-v3 study: the 10 pairs, the references (byte-identical, checked) and the three scoring methods.

## 2. Results

Source: `comparison_trelis.csv`. Each cell is **Trelis / large-v3 (delta = Trelis - large-v3)**.

| pair | model | raw CER | fold CER | m1 CER | m1 WER | m2 CER | m2 WER |
|---|---|---|---|---|---|---|---|
| base_V1__cab_11_g4 | base | 0.148 / 0.116 (+0.033) | 0.151 / 0.059 (+0.092) | 0.133 / 0.063 (+0.070) | 0.218 / 0.172 (+0.046) | 0.031 / 0.062 (-0.031) | 0.080 / 0.138 (-0.057) |
| base_V1__sub_07_g2 | base | 0.116 / 0.040 (+0.076) | 0.140 / 0.047 (+0.093) | 0.101 / 0.034 (+0.067) | 0.144 / 0.068 (+0.076) | 0.016 / 0.029 (-0.013) | 0.051 / 0.103 (-0.053) |
| V1_A__ecom_07_g4 | V1_A | 0.213 / 0.517 (-0.304) | 0.132 / 0.412 (-0.280) | 0.118 / 0.387 (-0.269) | 0.157 / 0.465 (-0.307) | 0.028 / 0.399 (-0.371) | 0.095 / 0.444 (-0.349) |
| V1_A__food_11_g2 | V1_A | 0.534 / 0.631 (-0.098) | 0.350 / 0.363 (-0.013) | 0.388 / 0.367 (+0.021) | 0.468 / 0.495 (-0.028) | 0.125 / 0.336 (-0.211) | 0.220 / 0.431 (-0.211) |
| V1_A__food_12_g3 | V1_A | 0.211 / 0.443 (-0.232) | 0.121 / 0.192 (-0.071) | 0.130 / 0.177 (-0.047) | 0.241 / 0.357 (-0.116) | 0.077 / 0.104 (-0.027) | 0.188 / 0.295 (-0.107) |
| V1_A__sub_11_g3 | V1_A | 0.221 / 0.489 (-0.267) | 0.121 / 0.226 (-0.105) | 0.137 / 0.159 (-0.022) | 0.210 / 0.303 (-0.092) | 0.097 / 0.089 (+0.008) | 0.151 / 0.252 (-0.101) |
| V1_B__air_16_g4 | V1_B | 0.358 / 0.490 (-0.133) | 0.272 / 0.313 (-0.041) | 0.269 / 0.244 (+0.025) | 0.357 / 0.481 (-0.124) | 0.107 / 0.263 (-0.156) | 0.189 / 0.457 (-0.268) |
| V1_B__ecom_11_g1 | V1_B | 0.490 / 0.784 (-0.294) | 0.326 / 0.388 (-0.062) | 0.373 / 0.456 (-0.083) | 0.546 / 0.750 (-0.204) | 0.126 / 0.154 (-0.028) | 0.327 / 0.626 (-0.299) |
| V1_C__air_03_g1 | V1_C | 0.414 / 0.579 (-0.164) | 0.305 / 0.282 (+0.023) | 0.289 / 0.302 (-0.013) | 0.315 / 0.482 (-0.167) | 0.125 / 0.207 (-0.082) | 0.198 / 0.398 (-0.200) |
| V1_C__cab_07_g3 | V1_C | 0.351 / 0.443 (-0.092) | 0.255 / 0.245 (+0.011) | 0.253 / 0.246 (+0.007) | 0.389 / 0.522 (-0.133) | 0.179 / 0.187 (-0.009) | 0.256 / 0.456 (-0.200) |

**Per-model means** (Trelis / large-v3, delta)

| model | n | raw CER | fold CER | m1 CER | m1 WER | m2 CER | m2 WER |
|---|---|---|---|---|---|---|---|
| base | 2 | 0.132 / 0.078 (+0.054) | 0.146 / 0.053 (+0.093) | 0.117 / 0.048 (+0.069) | 0.181 / 0.120 (+0.061) | 0.023 / 0.045 (-0.022) | 0.066 / 0.121 (-0.055) |
| V1_A | 4 | 0.295 / 0.520 (-0.225) | 0.181 / 0.298 (-0.117) | 0.194 / 0.273 (-0.079) | 0.269 / 0.405 (-0.136) | 0.081 / 0.232 (-0.150) | 0.164 / 0.356 (-0.192) |
| V1_B | 2 | 0.424 / 0.637 (-0.213) | 0.299 / 0.350 (-0.051) | 0.321 / 0.350 (-0.029) | 0.452 / 0.615 (-0.164) | 0.116 / 0.209 (-0.092) | 0.258 / 0.541 (-0.283) |
| V1_C | 2 | 0.383 / 0.511 (-0.128) | 0.280 / 0.263 (+0.017) | 0.271 / 0.274 (-0.003) | 0.352 / 0.502 (-0.150) | 0.152 / 0.197 (-0.045) | 0.227 / 0.427 (-0.200) |
| all 10 | 10 | 0.306 / 0.453 (-0.147) | 0.217 / 0.253 (-0.035) | 0.219 / 0.244 (-0.024) | 0.305 / 0.409 (-0.105) | 0.091 / 0.183 (-0.092) | 0.176 / 0.360 (-0.185) |

**Verification flags and m2 label audit** (Trelis / large-v3)

| pair | m1 recount matches | m1 romanisation faithful | m2 maps valid | m2 recount matches | m2 judgement errors |
|---|---|---|---|---|---|
| base_V1__cab_11_g4 | yes | yes / yes | yes / yes | yes | 0/50 = 0.00 / 0.04 |
| base_V1__sub_07_g2 | yes | yes / yes | yes / yes | yes | 4/50 = 0.08 / 0.12 |
| V1_A__ecom_07_g4 | yes | yes / **no** | yes / yes | yes | 1/50 = 0.02 / 0.08 |
| V1_A__food_11_g2 | yes | yes / yes | yes / yes | yes | 2/58 = 0.034 / 0.16 |
| V1_A__food_12_g3 | yes | yes / **no** | yes / yes | yes | 3/50 = 0.06 / 0.20 |
| V1_A__sub_11_g3 | yes | yes / yes | yes / yes | yes | 1/50 = 0.02 / 0.059 |
| V1_B__air_16_g4 | yes | yes / **no** | yes / yes | yes | 0/50 = 0.00 / 0.32* |
| V1_B__ecom_11_g1 | yes | yes / yes | yes / yes | yes | 6/50 = 0.12 / 0.12 |
| V1_C__air_03_g1 | yes | yes / yes | yes / yes | yes | 2/50 = 0.04 / 0.10 |
| V1_C__cab_07_g3 | yes | yes / yes | yes / yes | yes | 12/50 = 0.24 / 0.20 |
| **total** | 10/10 | 10/10 / 7/10 | 10/10 / 10/10 | 10/10 | pooled 31/508 = **0.061** / 70/501 = 0.140 |

\* In the large-v3 run, the air_16_g4 sample was aimed at suspicious steps and was not random. The Trelis air_16 verify file lists 30 char and 20 word steps but does not say how they were chosen.

How the columns are derived:
- **m1 recount matches:** the verifier's independent CER and WER equal metrics.json to 4 decimal places.
- **m2 recount matches:** the verifier's recount of the map ops equals char_metrics.json and word_metrics.json.

## 3. What changed vs large-v3, per method

**Level (all-10 means)**
- Every method gives a lower mean for Trelis.
- The size of the drop differs a lot by method:
  - m2 WER −0.185 (about half its large-v3 value)
  - raw CER −0.147
  - m1 WER −0.105
  - m2 CER −0.092 (about half)
  - fold CER −0.035
  - m1 CER −0.024

**Base pairs (both) get worse under raw, fold and m1, and better under m2.**
- raw: +0.033 and +0.076.
- fold: +0.092 and +0.093.
- m1 CER: +0.070 and +0.067.
- m1 WER: +0.046 and +0.076.
- m2 CER: −0.031 and −0.013.
- m2 WER: −0.057 and −0.053.
- Both base transcripts are almost entirely Latin script under both ASR models (Devanagari share ≤0.16). The main textual difference is digits vs number words.
  - Example: cab_11 "RD5068 / 12 / 4821 / 3" in large-v3 vs "RD five zero six eight / twelve / forty two one / three" in Trelis.
  - The m2 cab_11 verify file lists exactly these number spans among the non-identity steps.
  - The m2 maps label spoken-form numbers as matches. m1 and score.py compare the letters, so they are charged.

**V1_A / ecom_07_g4 moves most under every method.**
- Changes: raw −0.304, fold −0.280, m1 CER −0.269, m1 WER −0.307, m2 CER −0.371, m2 WER −0.349.
- In large-v3, m1 had 263 character deletions on this pair. The earlier verify files call these real Whisper omissions (the address-update segment and the repeated instruction lines).
- In Trelis, m1 has 5 character deletions on this pair, and the Trelis transcript is 750 characters against 445.
- Under every Trelis method, ecom_07_g4 drops from worst (or near worst) of the 8 V1 pairs to the best or near best:
  - m1 CER: rank 7 → 1
  - m2 CER: rank 8 → 1
  - raw: rank 5 → 2
  - fold: rank 8 → 3

**Other large per-pair moves (|Δ| ≥ 0.15)**

| method | pairs and change |
|---|---|
| raw | ecom_11 −0.294, sub_11 −0.267, food_12 −0.232, air_03 −0.164 |
| m1 WER | ecom_11 −0.204, air_03 −0.167 |
| m2 CER | food_11 −0.211, air_16 −0.156 |
| m2 WER | ecom_11 −0.299, air_16 −0.268, food_11 −0.211, air_03 −0.200, cab_07 −0.200 |

**Small moves**
- m1 CER changes by less than ±0.05 on 6 of 10 pairs. It goes up on 5 pairs: cab_11, sub_07, food_11, air_16 and cab_07.
- fold goes up on 4 pairs (cab_11, sub_07, air_03, cab_07) and drops by 0.1 or more on only 2 (ecom_07, sub_11).

**Edit composition** (summed over 10 pairs; Trelis vs large-v3)

| | Trelis S / D / I (share of edits) | large-v3 S / D / I |
|---|---|---|
| m1 char | 283 / 90 / 877 (0.23 / 0.07 / **0.70**) | 437 / 570 / 432 (0.30 / 0.40 / 0.30) |
| m2 char | 121 / 103 / 287 (0.24 / 0.20 / 0.56) | 210 / 586 / 302 (0.19 / 0.53 / 0.28) |
| m1 word | 182 / 7 / 144 (0.55 / 0.02 / 0.43) | 282 / 85 / 87 (0.62 / 0.19 / 0.19) |
| m2 word | 122 / 10 / 59 (0.64 / 0.05 / 0.31) | 202 / 96 / 101 (0.51 / 0.24 / 0.25) |

- Deletions fall sharply under both methods: m1 char drops from 570 to 90, and m2 char from 586 to 103.
- m1 char insertions double, from 432 to 877. m2 char insertions stay about level, from 302 to 287.

**Rank agreement between the two ASR models, per method** (Spearman ρ, Trelis vs large-v3 values of the same metric over pairs)

| method | all 10 pairs | 8 V1 pairs |
|---|---|---|
| raw CER | +0.903 | +0.810 |
| fold CER | +0.491 | +0.595 |
| m1 CER | +0.588 | +0.405 |
| m1 WER | +0.891 | +0.881 |
| m2 CER | +0.347 | **−0.096** |
| m2 WER | +0.782 | +0.571 |

**Model order by mean, per method** (Trelis | large-v3)

| method | Trelis | large-v3 |
|---|---|---|
| raw | base < V1_A < V1_C < V1_B | base < V1_C < V1_A < V1_B |
| fold | base < V1_A < V1_C < V1_B | base < V1_C < V1_A < V1_B |
| m1 CER | base < V1_A < V1_C < V1_B | base < V1_A < V1_C < V1_B (V1_A ≈ V1_C, 0.273 vs 0.274) |
| m1 WER | base < V1_A < V1_C < V1_B | base < V1_A < V1_C < V1_B |
| m2 CER | base < V1_A < V1_B < V1_C | base < V1_C < V1_B < V1_A |
| m2 WER | base < V1_A < V1_C < V1_B | base < V1_A < V1_C < V1_B |

- With Trelis, 5 of the 6 metrics give the same model order: base < V1_A < V1_C < V1_B. m2 CER is the exception, with V1_C last.
- With large-v3, the orders disagreed. V1_A was 3rd under raw and fold and 4th under m2 CER; with Trelis it is 2nd under every metric.
- The only large-v3 metrics whose order is unchanged with Trelis are m1 CER, m1 WER and m2 WER.

**Agreement between methods on Trelis transcripts** (Spearman ρ, all 10 / 8 V1; large-v3 values from COMPARISON.md in brackets)

| comparison | ρ |
|---|---|
| m1 vs m2 CER | +0.845 / +0.719 [+0.782 / +0.571] |
| m1 vs m2 WER | +0.939 / +0.952 [+0.867 / +0.738] |
| raw vs m1 CER | +0.952 / +0.976 [+0.903 / +0.810] |
| raw vs m2 CER | +0.821 / +0.695 [+0.709 / +0.429] |
| fold vs m1 CER | +0.855 / +0.929 [+0.939 / +0.881] |
| fold vs m2 CER | +0.663 / +0.647 [+0.867 / +0.738] |
| raw vs fold | +0.806 / +0.976 [+0.891 / —] |

- Every between-method correlation rises with Trelis, except fold vs m2 (both subsets), and fold vs m1 and raw vs fold over all 10 pairs.

## 4. How transcript script affected each method

Facts about the Trelis transcripts:
- About 21% of their letters are Devanagari, against 69% for large-v3.
- They contain no digits. The references contain 3-27 digits per pair, and large-v3 kept digits (1-8 digit runs per pair).

| method | how it handles Devanagari | how it handles numbers | observed with Trelis |
|---|---|---|---|
| score.py raw | `deva_to_latin` table, no spaces | letters compared literally: "12" vs "twelve" is charged | Biggest CER drop of the char metrics on V1 pairs (−0.09 to −0.30), where Devanagari share fell from 0.63-1.00 to 0.21-0.35. Both base pairs, with almost no Devanagari under either model, rise by +0.03 and +0.08. |
| score.py fold | same table, then consonant skeleton | charged ("12" → "", "twelve" → "tlv") | Small mean change (−0.035). Rises on both base pairs and on air_03 and cab_07. |
| m1 (hand romanisation) | romaniser converts the Devanagari tokens (fewer with Trelis) | charged | Char insertions double (432 → 877) and m1 CER moves less than ±0.05 on 6/10 pairs. WER drops more than CER (−0.105 vs −0.024). |
| m2 (cross-script sound maps) | aligned by sound, script-independent | spoken-form numbers labelled match | Largest drops (CER −0.092, WER −0.185) and the only method where base pairs improve. |

- Across the 8 V1 pairs, the change in Devanagari share does not track the change in any metric. Spearman of Δ(Devanagari share) against Δmetric:

| metric | ρ |
|---|---|
| raw | +0.26 |
| fold | −0.14 |
| m1 CER | +0.14 |
| m1 WER | 0.00 |
| m2 CER | −0.41 |
| m2 WER | −0.28 |

- n = 8. The ecom_07 deletion change sits on top of the script change.

## 5. Reliability notes from the verify files

**Method 1**
- All 10 CER/WER values were reproduced to 4 decimal places by an independent Levenshtein.
- All 10 romanisations were judged faithful: 1:1 token alignment, Latin tokens unchanged, nothing corrected. In the large-v3 run, 3 of 10 were not faithful.
- In 5 pairs (food_11, food_12, air_16, ecom_11, cab_07), the char-level S/D/I split differs from metrics.json because equal-cost alignments break ties differently. The totals are identical, and so are the word S/D/I splits.
- Flagged romanisation choices:

| pair | flagged choice | effect |
|---|---|---|
| ecom_07 | शापकार → "shopkar" (literal "shaapkar") | would raise CER slightly |
| sub_07 | वारुण → "varun" (literal "vaarun") | CER 0.1027, WER 0.1525 if literal |
| air_03 | पी → "p" | does not change the word count |

- The base references contain truncated contractions (It', I', He', You', That'), and these inflate both base pairs. This is unchanged from the large-v3 run.

**Method 2**
- Maps are valid and recounts match on all 10 pairs.
- Pooled judgement-error rate is 31/508 = 0.061, against 0.140 for large-v3. Errors by pair:

| pair | judgement errors | what the verify file says | corrected values it gives |
|---|---|---|---|
| cab_07_g3 | 12/50 = 0.24 (the highest) | English silent letters and digraphs are charged (gh in right, igh in eight, ch/ck/wh, oo of hoon), Hindi digraphs are absorbed, and delhi → दिल्ली is counted as a sub | ~0.142 CER (82 → ~65 char errors), ~0.222 WER |
| ecom_11_g1 | 6/50 | rockerz/rockers labelled sub 3× | CER 0.1204, WER 0.2991 |
| sub_07_g2 | 4/50 | — | CER 0.0146, WER 0.0593 |
| food_12_g3 | 3/50 | — | CER ≈0.0751 |
| sub_11_g3 | 1/50 | — | CER 0.0950 |
| air_03_g1 | 2/50 | — | strict WER 0.2075; borderline CER 0.1266 |
| food_11_g2 | 2/58 | metric-neutral | — |
| ecom_07_g4 | 1/50 | metric-neutral | — |
| cab_11_g4 | 0/50 | — | — |
| air_16_g4 | 0/50 | — | — |

- **WER convention.** Multi-word hypothesis spans for one reference word are counted as one match or sub with no insertions. Examples:

| pair | spans |
|---|---|
| food_11 | fd2130 → 6 words, rs 720 → 4 |
| air_16 | indus → "in this", vs8910 → 5 words |
| air_03 | linat → "लेना था" |
| sub_07 | streambox → "string box" |

  - air_16 verify estimates a strict 1:1 WER of ~0.236, against the 0.189 reported.
  - This convention also absorbs Trelis's spelled-out numbers. It is the main reason m2 WER falls the most of all methods.
- **air_16 normalisation mismatch.** The word normaliser deletes apostrophes while the char normaliser turns them into spaces. Under the char-side normalisation, WER would be +1/127.
- **m2 ref_chars differ slightly from the large-v3 run on 2 pairs.** sub_07 is 617 vs 615 and ecom_11 is 565 vs 566. m2 ref_words differ on sub_07 (118 vs 116) and ecom_11 (107 vs 108).

**score.py**
- The functions were copied verbatim from `pod_code/hinglish/tests/score.py`, lines 227-301. They could not be imported because score.py imports `tcommon`/`hindi_share` at module level.
- The copy reproduces all 20 large-v3 raw/fold values in `comparison.csv` exactly (printed by the script).

**Table audit**
- 3 random pairs (food_11_g2, air_16_g4, cab_07_g3) were recomputed from the source files:
  - m1 CER/WER: Levenshtein on the m1-normalised reference.txt / whisper_roman.txt.
  - m2 CER/WER: op counts in char_map.jsonl / word_map.jsonl.
  - large-v3 values and deltas: from `comparison.csv`.
- All 54 checks matched.

## 6. Files

- `research/asr_cer/cer_study/trelis/compare_trelis.py`: builds the CSV and prints the self-check, Spearman values, model orders, per-pair ranks and edit composition. Stdlib only.
- `research/asr_cer/cer_study/trelis/comparison_trelis.csv`: 10 pair rows, 4 per-model mean rows and MEAN_ALL.
  - Trelis columns are unprefixed, large-v3 columns are `lv3_*`, and deltas are `d_*`.
  - Also included: `deva_share`, `digits`, and m2 `judgement_errors`/`sample_size`.
- `research/asr_cer/cer_study/trelis/TRANSCRIBE.md`, `trelis_tx.py`, `trelis_raw.json`, `inputs/`: transcription.
- `research/asr_cer/cer_study/trelis/m1/<pair>/`, `m2/<pair>/`: method outputs and verify files.
- `research/asr_cer/cer_study/comparison.csv`, `COMPARISON.md`: the large-v3 study.
