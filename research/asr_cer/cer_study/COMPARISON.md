# CER study: comparison of three ways to score Hinglish Whisper transcripts

## 1. What was compared

**Pairs.** There are 10 (reference, hypothesis) pairs. The reference is the model's own text stream, in romanised Hinglish. The hypothesis is Whisper's transcript of the model's audio, which mixes Devanagari and Latin script. The pairs cover four models:
- base_V1 (shown as `base`): cab_11_g4, sub_07_g2
- V1_A: ecom_07_g4, food_11_g2, food_12_g3, sub_11_g3
- V1_B: air_16_g4, ecom_11_g1
- V1_C: air_03_g1, cab_07_g3

base, V1_B and V1_C have only 2 pairs each.

**score.py raw / fold** (`hinglish/tests/score.py (monorepo: research/eval_harness/score.py)`)
- `raw_norm`: converts Devanagari to Latin with the fixed `deva_to_latin` table, lowercases, and keeps only ASCII alphanumerics. All spaces and punctuation are removed. CER is Levenshtein distance divided by the length of the normalised reference.
- `fold`: applies `raw_norm`, then merges digraphs (ch/chh→c, ph→f, sh→s, kh/gh/th/dh/bh/jh→plain consonant, c/q→k, w→v, z→j, x→ks), deletes every vowel plus h and y, and collapses repeated letters. The result is effectively a consonant skeleton. CER is computed the same way.

**Method 1: romanise, then score** (`w1_romanise/<pair>/score_w1.py`)
- The Whisper text is romanised by hand into `whisper_roman.txt`, with choices recorded in `romanisation_notes.txt`.
- Both sides are lowercased. Every character that is not alphanumeric or `'` becomes a space, and runs of whitespace collapse to one space. Spaces count as characters.
- CER and WER are plain Levenshtein with a backtrace that gives S/D/I counts.

**Method 2: cross-script character and word maps** (`w2_charmap/<pair>/`)
- Normalisation: the reference is lowercased with punctuation dropped and whitespace collapsed. The hypothesis has punctuation dropped and whitespace collapsed, and keeps its original script.
- Alignment is by sound. An automatic DP pass is followed by manual judgement, giving one labelled step per unit in `char_map.jsonl` and `word_map.jsonl`.
- Insertions are counted per hypothesis akshara or space unit, not per codepoint. CER = (sub+del+ins) / ref_chars, and WER is computed the same way over words.

## 2. Results

Source: `comparison.csv`. ref_chars and ref_words are Method 1's values.

| pair | model | ref ch | ref wd | raw CER | fold CER | m1 CER | m1 WER | m1 S/D/I (char) | m1 verified | m1 faithful | m2 CER | m2 WER | m2 char S/D/I | m2 word S/D/I | m2 maps valid | m2 judgement err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_V1__cab_11_g4 | base | 429 | 87 | 0.1157 | 0.0592 | 0.0629 | 0.1724 | 2/14/11 | yes | yes | 0.0615 | 0.1379 | 0/14/12 | 6/3/3 | yes | 0.04 (2/50) |
| base_V1__sub_07_g2 | base | 623 | 118 | 0.0400 | 0.0466 | 0.0337 | 0.0678 | 3/0/18 | yes | yes | 0.0293 | 0.1034 | 3/0/15 | 8/0/4 | yes | 0.12 (6/50) |
| V1_A__ecom_07_g4 | V1_A | 728 | 127 | 0.5166 | 0.4122 | 0.3874 | 0.4646 | 17/263/2 | yes | **no** | 0.3989 | 0.4444 | 12/271/7 | 10/45/1 | yes | 0.08 (4/50) |
| V1_A__food_11_g2 | V1_A | 569 | 109 | 0.6312 | 0.3632 | 0.3673 | 0.4954 | 62/69/78 | yes | yes | 0.3357 | 0.4312 | 30/82/79 | 13/16/18 | yes | 0.16 (8/50) |
| V1_A__food_12_g3 | V1_A | 599 | 112 | 0.4426 | 0.1917 | 0.1770 | 0.3571 | 68/32/6 | yes | **no** | 0.1035 | 0.2946 | 34/26/2 | 30/1/2 | yes | 0.20 (10/50) |
| V1_A__sub_11_g3 | V1_A | 642 | 119 | 0.4885 | 0.2259 | 0.1589 | 0.3025 | 35/19/48 | yes | yes | 0.0888 | 0.2521 | 16/14/27 | 18/2/10 | yes | 0.0588 (3/51) |
| V1_B__air_16_g4 | V1_B | 648 | 129 | 0.4904 | 0.3130 | 0.2438 | 0.4806 | 73/77/8 | yes | **no** | 0.2632 | 0.4567 | 32/108/30 | 30/19/9 | yes | 0.32* (16/50) |
| V1_B__ecom_11_g1 | V1_B | 566 | 108 | 0.7843 | 0.3877 | 0.4558 | 0.7500 | 97/27/134 | yes | yes | 0.1540 | 0.6262 | 31/8/48 | 41/0/26 | yes | 0.12 (6/50) |
| V1_C__air_03_g1 | V1_C | 539 | 108 | 0.5787 | 0.2818 | 0.3024 | 0.4815 | 35/53/75 | yes | yes | 0.2067 | 0.3981 | 23/50/38 | 22/9/12 | yes | 0.10 (5/50) |
| V1_C__cab_07_g3 | V1_C | 459 | 90 | 0.4432 | 0.2446 | 0.2462 | 0.5222 | 45/16/52 | yes | yes | 0.1874 | 0.4556 | 29/13/44 | 24/1/16 | yes | 0.20 (10/50) |

\* The air_16_g4 sample was aimed at suspicious steps and was not drawn at random, so its rate is not an estimate.

**Per-model means**

| model | n | raw CER | fold CER | m1 CER | m1 WER | m2 CER | m2 WER | m1 faithful | m2 judgement err (mean of pair rates) |
|---|---|---|---|---|---|---|---|---|---|
| base | 2 | 0.0779 | 0.0529 | 0.0483 | 0.1201 | 0.0454 | 0.1207 | 2/2 | 0.08 |
| V1_A | 4 | 0.5197 | 0.2983 | 0.2727 | 0.4049 | 0.2317 | 0.3556 | 2/4 | 0.1247 |
| V1_B | 2 | 0.6373 | 0.3503 | 0.3498 | 0.6153 | 0.2086 | 0.5414 | 1/2 | 0.22 (includes the targeted air_16 sample) |
| V1_C | 2 | 0.5110 | 0.2632 | 0.2743 | 0.5019 | 0.1971 | 0.4269 | 2/2 | 0.15 |
| all 10 | 10 | 0.453 | 0.253 | 0.244 | 0.409 | 0.183 | 0.360 | 7/10 | pooled 70/501 = 0.140 |

**Model order by mean CER**
- m1: base 0.048 < V1_A 0.273 ≈ V1_C 0.274 < V1_B 0.350
- m2: base 0.045 < V1_C 0.197 < V1_B 0.209 < V1_A 0.232
- raw: base 0.078 < V1_C 0.511 < V1_A 0.520 < V1_B 0.637
- fold: base 0.053 < V1_C 0.263 < V1_A 0.298 < V1_B 0.350

## 3. What the numbers show

**Agreement in CER rank** (Spearman, with tied values given their average rank)

| comparison | all 10 pairs | 8 V1 pairs |
|---|---|---|
| m1 vs m2 CER | +0.782 | +0.571 |
| m1 vs m2 WER | +0.867 | +0.738 |
| raw vs m1 | +0.903 | +0.810 |
| raw vs m2 | +0.709 | +0.429 |
| fold vs m1 | +0.939 | +0.881 |
| fold vs m2 | +0.867 | +0.738 |
| raw vs fold | +0.891 | — |

- Every metric (raw, fold, m1 and m2 CER, m1 and m2 WER) ranks the two base pairs lowest. This pushes up the all-pairs correlations, so the 8 V1 pairs are the more informative test.
- Among the V1 pairs, score.py (raw and fold) agrees with m1 more than with m2.
- Two V1 pairs change rank by more than 1. V1_B__ecom_11_g1 is rank 8 of 8 under m1 but rank 3 under m2. V1_B__air_16_g4 is rank 3 under m1 but rank 6 under m2. Every other pair moves by 1 or less.

**Level**
- m1 CER is above m2 CER in 8 of 10 pairs. The exceptions are air_16_g4 (−0.019) and ecom_07_g4 (−0.012).
- m1 WER is above m2 WER in 9 of 10 pairs. The exception is sub_07_g2 (−0.036).

**Biggest disagreement: V1_B__ecom_11_g1** (m1 CER 0.456 vs m2 0.154, a gap of +0.302)
- Whisper wrote numbers as words ("nine ek zero zero …", "four haido fifty", "fifteen"), and the reference has digits. m1 compares these letter by letter. m2 counts each number word mapped to a digit as one match (9→नाइन).
- The audit mapped the spoken numbers in m1's hypothesis back to digits. That lowered m1 to CER 0.270 and WER 0.528. Also removing the hallucinated trailing tail lowered it to CER 0.196 and WER 0.444.
- So about 0.19 of the 0.30 gap comes from how numbers are written. The pair's drop from rank 8 to rank 3 comes mostly from this convention, not from judgements about sounds.
- m2 also counts insertions per akshara: 48 here, against 61 counted per codepoint. Its own verify file relabels the pair to CER 0.170. About 25 of m2's 67 word errors come from number tokenisation and the hallucinated tail.

**Other large CER gaps (m1 minus m2)**

| pair | gap |
|---|---|
| V1_C__air_03_g1 | +0.096 |
| V1_A__food_12_g3 | +0.074 |
| V1_A__sub_11_g3 | +0.070 |

**Mix of edit types**

| | m1 | m2 |
|---|---|---|
| deletion share of character edits | 570/1439 = 0.396 | 586/1098 = 0.534 |
| deletion share of word edits | 85/454 = 0.187 | 96/399 = 0.241 |
| substitution share of word edits | 0.621 | 0.506 |
| character substitutions | 437 | 210 |
| character insertions | 432 | 302 |

- m2's higher deletion share comes mainly from its smaller denominator. It has 210 substitutions against m1's 437 and 302 insertions against 432, while raw deletions are close (586 vs 570). By pair, m2 has more deletions in 3 pairs, fewer in 5, and the same number in 2.
- One pair dominates the deletion totals. ecom_07_g4 alone has 263 of m1's 570 character deletions and 271 of m2's 586. Both verify files say these are real Whisper omissions: the address-update segment and the repeated instruction lines were dropped.
- Without ecom_07, the character deletion share is m1 0.265 and m2 0.390, and the word deletion share is m1 0.099 and m2 0.149.
- Per-pair medians of the deletion share are m1 0.314 and m2 0.424 for characters, and 0.100 and 0.138 for words.

**Pairs that score the same under both methods**
- ecom_07_g4 has the highest CER under both methods (0.387 and 0.399), and the cause is real Whisper deletions.
- sub_11_g3 and food_12_g3 are the two best V1 pairs under both methods.

## 4. Reliability and biases of each method

**score.py raw**
- This is the strictest metric, and its crude transliteration table inflates it. For example, थैंक comes out as "thaink".
- Running `raw_norm` on m1's hand-romanised hypothesis instead lowers raw CER sharply:

| pair | raw on original | raw on m1 roman |
|---|---|---|
| sub_11 | 0.489 | 0.172 |
| food_12 | 0.443 | 0.209 |
| ecom_11 | 0.784 | 0.503 |
| air_03 | 0.579 | 0.322 |

- Raw also drops spaces from both the numerator and the denominator.
- Do not use its absolute level (0.453 overall) or its ordering of the V1 models.
- Every raw and fold value in `args.json` reproduces exactly when re-run.

**score.py fold**
- It is lenient in what it counts. It deletes every vowel, h and y, merges aspirates and drops spaces, which removes exactly the places where m1's romanisation choices and m2's vowel judgements differ.
- Its denominator is a consonant skeleton, about 40–45% of m1's ref_chars (ecom_07: 296 vs 728). So it counts far fewer errors than m1 (ecom_11: about 88 vs 258).
- Its rate still lands near m1's (mean 0.253 vs 0.244; above m1 in 5 pairs and below in 5), because the smaller denominator offsets the smaller count. It is above m2 in 9 of 10 pairs; the exception is cab_11.
- It depends little on the transliteration table: on m1's roman hypothesis it gives sub_11 0.159 vs 0.226 and ecom_11 0.374 vs 0.388.
- Part of its high agreement with m1 comes from both being plain Levenshtein.

**Method 1 (romanise, then score)**
- Verification: `cer_ok` and `wer_ok` are true for all 10 pairs.
- Romanisation fidelity: `romanisation_faithful` is false for 3 pairs.

| pair | unfaithful choice | measured effect |
|---|---|---|
| ecom_07_g4 | "headpoons" instead of "headphones" | CER 0.3874→0.3846, WER 0.4646→0.4567 |
| food_12_g3 | "kan" instead of "can" | 1 word and 1 character |
| air_16_g4 | "ditels" | about 2 character edits per occurrence, and two word matches become substitutions |

The fidelity failures therefore change the score only slightly (about 0.003 CER).
- **Bias that lowers the score.** For Devanagari tokens that could be English, the romaniser chose English spellings: यूर/योर→"your", द→"the", मूवी→"movie".
- **Bias that raises the score.**
  - Literal or Hindi-convention spellings that differ from the reference: "bataye" vs "bataiye", "ho" vs "hoon", "da" vs "the" (2–3 word errors per pair in food_11 and food_12), and "tu" kept rather than "two".
  - Number words compared with digits in the reference. This is the only large effect, about 0.19 CER on ecom_11.
  - Every letter of a misspelt word is scored, so a mishearing that sounds close (aadar/order, kaleen/calling) costs several characters. This is the main reason m1 CER is above m2 CER in 8 of 10 pairs.
- **Tie-breaking.** For air_03_g1, metrics.json gives S/D/I 35/53/75 and verify gives 33/54/76. Both total 163, so CER is unchanged, but the split between edit types depends on how ties are broken. The CSV uses the metrics.json values.

**Method 2 (cross-script maps)**
- **Verification.** `char_map_valid` and `word_map_valid` are true for all 10 pairs, and recounting the maps reproduces the reported CER and WER.
- **Judgement error rate.** 70 of 501 sampled labels were wrong (0.140). The per-pair range is 0.04 (cab_11) to 0.32 (air_16).
  - The pooled figure is not a clean estimate. The air_16 sample was targeted, and cab_11's character sample clusters at steps 352–409, the Devanagari tail.
  - sub_07's 6 errors come from a normalisation bug, not from judging sounds.
  - Without air_16, the pooled rate is 54/451 = 0.120.
  - The rate measures disagreement with m2's own rules, not with a neutral standard.
  - food_12 sampled 30 characters with 10 wrong and 20 words with 0 wrong.
- **Character-level mislabels go both ways.** Of the sampled wrong labels, about 21 were too lenient and 24 too strict. The verify files' own corrected CERs also move in both directions:

| pair | reported | corrected |
|---|---|---|
| food_12 | 0.1035 | 0.117 |
| ecom_11 | 0.154 | 0.170 |
| cab_07 | 0.1874 | ~0.170 |
| air_16 | 0.263 | ~0.24 |
| sub_07 | 0.029 | 0.026 |
| air_03 | | ±0.004 |

- **Leniency built into the rules.**
  - "match" steps with an empty hypothesis absorb digraphs, silent e and the inherent schwa at zero cost. They appear in 9 of 10 character maps: air_03 147, sub_11 100, food_12 92, ecom_11 82, air_16 63, food_11 49, ecom_07 41, cab_07 30, cab_11 15, sub_07 0. air_03 has the most and also the second-largest m1−m2 gap.
  - Insertions are counted per akshara, and a digit matched to a number word costs nothing.
  - food_12 verify says the unit tables are "lenient by design" (ि accepts e, स accepts z, द accepts th, the inherent vowel accepts o).
- **Strictness built into the rules.**
  - A reference vowel realised as schwa is labelled a deletion, which inflates CER. air_16 says so, and cab_11 verify says the map "biases toward del".
  - Loanwords such as new~न्यू and update~अप are labelled substitutions.
  - Latin is compared with Latin by spelling, so the silent gh in "right" counts as 2 insertions.
- **WER is biased low.** In 6 files, the word map labels a word a match while the character map records a substitution or deletion inside it. About 18 of 25 wrong word labels are too lenient. air_03 has the same kind of mismatch ("october"), and ecom_11 and food_12 have the opposite one (character match, word substitution). The verify files give stricter WERs:

| pair | reported WER | stricter WER |
|---|---|---|
| cab_11 | 0.1379 | 0.1609 |
| ecom_07 | 0.4444 | ~0.460 |
| food_11 | 0.4312 | ~0.459 |
| sub_11 | 0.2521 | 0.2773 |
| air_16 | 0.4567 | ~0.48 |
| cab_07 | 0.4556 | 0.50 |

- **Normalisation bug.** In sub_07_g2, "12,2024" becomes "122024", which adds 4 spurious word errors and 2 character insertions. The corrected values are WER 0.0690 and CER 0.0260. This is why sub_07 is the only pair where m2 WER is above m1 WER.
- **With the verify files' corrections applied**, the m2 model means become roughly V1_C 0.188 < V1_B 0.205 < V1_A 0.235. The order is the same, but every gap is under 0.05, which is within the per-pair judgement noise.

**Issues that affect both methods**
- The two methods use different denominators, because m1 turns punctuation into a space and m2 drops it ("12 2024" vs "122024"). m1's ref_chars are 0–8 higher than m2's per pair (sub_07: 623 vs 615), and ref_words differ by 0–2.
- Both base references contain truncated contractions (You', I', He', It', That'), and these count as errors under both methods.
- Absolute CER levels cannot be compared across methods, because raw, fold, m1 and m2 each use a different denominator. Only differences within one method mean anything.

**Results to rely on, and results to treat with caution**
- Reliable:
  - base is far below every V1 model under every metric.
  - ecom_07 is the worst pair, because of real Whisper deletions.
  - sub_11 and food_12 are the best V1 pairs.
- Treat with caution:
  - The order of the V1 models. The means differ by less than 0.08 (less than 0.035 under m2), there are only 2 pairs for V1_B and V1_C, and the order changes with the method.
  - Where V1_B ranks, since it depends mostly on how ecom_11's numbers and tail are counted.
  - Deletion shares.
  - m2 WER.

## 5. Files

All paths are under `research/asr_cer/cer_study/`.

- `pairs.json`, `args.json`: pair list, reference text, and score.py raw/fold values
- `inputs/<pair>.reference.txt`, `inputs/<pair>.whisper.txt`: input texts
- `compare.py`: builds `comparison.csv` and prints the Spearman values, edit shares and per-pair gaps. It uses the Python standard library only, and re-running it reproduces the CSV byte for byte.
- `comparison.csv`: 10 pair rows and 4 model-mean rows. It also includes m2_ref_chars/words, m1 word S/D/I, and the m2 judgement error and sample-size counts.
- `w1_romanise/<pair>/`
  - `whisper_roman.txt`: the romanised hypothesis
  - `romanisation_notes.txt`: romanisation choices
  - `score_w1.py`: scorer
  - `metrics.json`: CER/WER, S/D/I and normalised strings
  - `verify.json`: cer_ok, wer_ok and romanisation_faithful
- `w2_charmap/<pair>/`
  - `char_map.jsonl` / `char_map.txt`: character map
  - `word_map.jsonl` / `word_map.txt`: word map
  - `char_map_auto.jsonl`, `align.py`: automatic DP pass
  - `finalize.py`, `word_align.py`: manual judgements
  - `char_metrics.json`, `word_metrics.json`: metrics
  - `verify.json`, `verify_check.py`: map validity, recount and sampled judgement errors
- `hinglish/tests/score.py (monorepo: research/eval_harness/score.py)`: `deva_to_latin`, `raw_norm`, `fold` and `cer`
