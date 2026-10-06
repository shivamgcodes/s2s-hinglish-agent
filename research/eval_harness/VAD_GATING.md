# VAD-gated customer vs fixed timing (V4 PersonaPlex tests, 2026-10-05)

User request (2026-10-05): instead of playing the customer at fixed scripted times, gate it with VAD so that the
customer starts speaking about 0.3 s after the model stops. Run 5 calls, choosing calls where the model did badly
before. Everything below is seed 1001, V4_A2@600 and V4_A@350. The fixed-timing numbers are the existing
section-8 runs (tests/out/V4_A2, tests/out/V4_A), unchanged.

**Bottom line:** gating works mechanically. The customer comes in 0.32 s after the model stops, and the model
gets room to confirm writes once the "hold" rule is on. But on 5 calls with 1 seed, the task-score changes are
**within seed noise**. Fixed V4_A on seeds 1002 and 1003 of the same 5 calls scores 0.70 and 0.60. That is above
both fixed s1001 (0.50) and VAD (v1 0.30, v2 0.40). The calls were picked as the worst on s1001, so regression to
the mean alone predicts improvement on any re-run. The one clear, mechanism-level finding is about
**write-value fidelity**. Once the model gets to confirm, it confirms with **wrong values** ("contact number
update ho gaya hai: 9650010001850" for 96500 18553). Under fixed timing that was partly hidden, because the
customer often cut in before the confirmation.

## What was built

- `tests/vad_gate.py` (new, pure numpy) has two parts:
  - `cut_customer_turns()` cuts each scripted customer turn out of `tests/inputs/V4/<call>.wav`, using the
    meta.json start/end with ±50 ms padding, clamped at the midpoint to the neighbouring customer turn.
  - `VadFeeder` plays those turns in script order, reacting to the model inside the 80 ms frame loop.
- Model activity in a frame: the decoded output frame's RMS is above -45 dBFS, OR the frame's text token is a
  word piece or EPAD (text cross-check). The PersonaPlex silence floor is about -74 dBFS and speech is
  -40..-15 dBFS, a cleanly bimodal split.
- Model "stopped": 4 quiet frames (0.32 s) if the last word piece ended a sentence (. ? !), otherwise 8 frames
  (0.64 s). The next customer turn starts on that frame, so the realized gap is 0.32 s, or 0.64 s after a
  mid-sentence stop. Measured: 23 of 26 starts released by `model_end` in V4_A2_vad were at 0.32 s and 3 at 0.64 s;
  in V4_A_vad all 26 were at 0.32 s.
- Release rules:
  - A customer turn that directly follows another customer turn in the script starts after 0.4 s if the model is
    silent; if the model is speaking, it waits for the model to stop.
  - First turn: after the greeting ends, or after 3 s if the model has not spoken.
  - Otherwise, if the model stays silent for 4 s, the customer starts anyway (`timeout_silent`).
  - If the model talks for 12 s from its first onset without a detected stop, the customer starts over it
    (`barge_in`).
  - Tail: the run stops after 3 s of model silence, or at `tail_max`.
  - Hard cap: 200 s of input (the context is about 221 s); unplayed turns are logged.
- **v2 options** (off by default, so v1 reproduces):
  - `hold`: if the utterance the model just finished matches the check-line / hold regex (score.py `CHECK_RE`:
    "ek minute", "rukiye", "let me check", "update kar deti hoon"...), the customer waits for the model to resume
    and stop again, or for 5 s of silence (`hold_timeout`).
  - `tail_match_fixed`: `tail_max` = the fixed input's time after its last customer turn, so post-call time
    matches the fixed run.
- `tests/driver.py`: `--customer-mode fixed|vad` (default fixed) and `--vad-params JSON`. In vad mode each run also
  writes `<run>.vad.json` (timeline: customer turns actually played with release reason, model speech intervals,
  holds, barge-ins, timeouts, overlap, stop reason) and `<run>.tmeta.json` (the inputs meta with the **actual**
  turn times). In the tmeta, an agent turn spans from the end of the previous played customer turn to the start
  of the next one. The stereo wav is L = model, R = the customer as actually played.
- `tests/run_tests.py` passes `--customer-mode` / `--vad-params` through.
- `tcommon.test_meta_path()` returns `<run>.tmeta.json` if it exists, else `tests/inputs/<V>/<call>.meta.json`.
  It is used by score.py (m3 check-line window, m5 greeting window, m4), v4_eval.py (echo window) and judge.py
  `--calljudge` (customer timeline). The judge also skips turns marked `unplayed`. Fixed tags have no tmeta, so
  their scoring is unchanged.
- Unit test: `tests/test_vad_gate.py`, CPU, fake model, 22 checks, all pass. It covers:
  - greeting → 0.32 s gap;
  - a 0.48 s mid-sentence pause does not release, a 0.8 s one does;
  - the 3 s opening timeout and the 4 s silence timeout;
  - the tail stop;
  - barge-in at 12 s and the overlap count;
  - the back-to-back 0.4 s gap;
  - the hard cap with an unplayed turn;
  - effective turn times;
  - hold: check-line, then 1.5 s pause, then confirmation → customer 0.32 s after the confirmation; hold timeout
    at 5 s.
- **Fixed mode is byte-identical.** Modified driver, fixed mode, V4_A2 food_03_g2 s1001 (scratch tag
  `vadchk_fixed_V4_A2`): `.json`, `.wav`, `_stereo.wav` and `.frames.txt` are cmp-identical to `out/V4_A2`.
  **v1 reproduces with the v2 code** (scratch tag `vadchk_v1_V4_A2`, cab_07_g2): `.json` and `.wav` are identical;
  `.vad.json` differs only by the new keys (`hold`, `hold_timeout`, `n_hold`, `n_hold_timeout`).

## Calibration (CPU, replay over the 60 fixed seed-1001 runs of V4_A2 + V4_A)

Pauses after which the model resumed by itself, with no customer audio in between:

- **After a sentence end:** 521 pauses. 12% are ≤ 0.32 s, 47% ≤ 0.64 s, 72% ≤ 1.2 s, 90% ≤ 1.6 s, 99% ≤ 2.0 s.
- **Mid-sentence:** 834 pauses. 74% are ≤ 0.32 s (so **26% exceed 0.32 s**), 92% ≤ 0.48 s, 97% ≤ 0.64 s. That is
  why `q_mid` is 0.64 s.

**The model never stays silent for more than about 2 s:** it fills silence itself. Two consequences:
1. No silence threshold separates "end of turn" from "pause between sentences". At 0.32 s the customer will
   sometimes come in between two of the model's sentences. This is accepted, because it is what the user asked
   for.
2. The "3 s of silence" tail rule never fires. Every v1 run ended at the 15 s tail cap, with about 10 s of
   post-goodbye model speech.

## Calls (picked from the seed-1001 fixed results; all have writes; 5 agent types)

| call | type / band / writes | why (fixed s1001: V4_A2 / V4_A) |
|---|---|---|
| cab_07_g2 | cab, long, 2 | FAILED + CONTRADICTS / PARTIAL + CONTRADICTS; echo 0/2 for both |
| food_03_g2 | food, standard, 1 | FAILED / PARTIAL; highest NONSENSE rate in the set (6/13, 7/14) |
| sub_11_g4 | sub, standard, 1 | PARTIAL / FAILED; no check-line (m3 0) for either |
| air_11_g2 | air, long, 2 | FAILED + CONTRADICTS / COMPLETE but 6/18 cut-off fragments (A2 5/16) |
| ecom_23_g1 | ecom, long, 3 | PARTIAL + CONTRADICTS for both; A2 echo 0/3, check-line 0.33 |

air_07_g1 was the lowest-scoring call overall, but it has no writes, so it was not picked.

## v1: the literal spec (no hold rule, 15 s tail cap) — tags V4_A2_vad, V4_A_vad

**The v1 echo / check-line numbers are a gating artifact.** In 13 of the 18 VAD confirm slots (V4_A2 8/9, V4_A
5/9), the model's only line was its check-line ("Ek minute rukiye, main check karke batata hoon"). The customer
then came back 0.32 s into the hold pause, so the model never got to confirm. That gives echo 0/9, and check-line goes "up" (0.37 → 0.80)
because the last line before each interruption is the check-line. The 15 s tail also adds about 10 s of
post-goodbye speech, which the judge sees:
- V4_A2_vad food_03_g2 "confirmed" the refund only in the tail ("Ji sir, main bilkul refund process start kar
  sakta hoon").
- Trelis transcribes tail audio that has no text tokens. This produces the CER outliers V4_A2_vad cab_07_g2 0.42
  ("help help our hundred possible questions...") and V4_A_vad ecom_23_g1 0.17.

The "tail-matched" column counts only lines that start before (last customer end + the fixed run's tail).

## V4_A2 (fixed) vs V4_A2_vad (VAD), seed 1001

| call | mode | task | facts (#wrong) | check-line | echo | judge confirm+value | NATURAL | NONSENSE | cut-off | NAT / NONS / cut-off, tail-matched | CER | overlap s | barge-in | silent timeout | duration s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | fixed | FAILED | CONTRADICTS (2) | 1.0 | 0/2 | 0/2 | 8/14 | 2/14 | 1/14 | 8/2/1 of 14 | 0.0409 | 2.72 | 0 | 0 | 91.4 |
| cab_07_g2 | vad | PARTIAL | CONSISTENT (0) | 1.0 | 0/2 | 1/2 | 6/11 | 3/11 | 3/11 | 6/2/2 of 10 | 0.4232 | 2.16 | 0 | 0 | 73.5 |
| food_03_g2 | fixed | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 6/13 | 6/13 | 1/13 | 6/6/1 of 13 | 0.1029 | 1.2 | 0 | 0 | 78.4 |
| food_03_g2 | vad | PARTIAL | CONTRADICTS (1) | 0.0 | 0/1 | 1/1 | 5/11 | 5/11 | 2/11 | 4/5/2 of 10 | 0.0458 | 0.4 | 0 | 0 | 74.6 |
| sub_11_g4 | fixed | PARTIAL | CONSISTENT (0) | 0.0 | 0/1 | 1/1 | 9/16 | 2/16 | 2/16 | 9/2/2 of 16 | 0.0556 | 1.44 | 0 | 0 | 74.1 |
| sub_11_g4 | vad | PARTIAL | CONSISTENT (0) | 1.0 | 0/1 | 1/1 | 8/12 | 1/12 | 0/12 | 7/1/0 of 11 | 0.0561 | 0.88 | 0 | 0 | 63.8 |
| air_11_g2 | fixed | FAILED | CONTRADICTS (2) | 0.5 | 0/2 | 0/2 | 9/16 | 5/16 | 5/16 | 9/5/5 of 16 | 0.0419 | 2.08 | 0 | 0 | 93.8 |
| air_11_g2 | vad | FAILED | CONSISTENT (0) | 1.0 | 0/2 | 0/2 | 6/11 | 2/11 | 1/11 | 6/2/1 of 10 | 0.021 | 0.64 | 0 | 0 | 75.9 |
| ecom_23_g1 | fixed | PARTIAL | CONTRADICTS (1) | 0.3333 | 0/3 | 2/3 | 12/20 | 4/20 | 4/20 | 12/4/4 of 20 | 0.1326 | 2.96 | 0 | 0 | 104.0 |
| ecom_23_g1 | vad | COMPLETE | CONSISTENT (0) | 1.0 | 0/3 | 3/3 | 11/12 | 0/12 | 0/12 | 9/0/0 of 10 | 0.0558 | 0.72 | 0 | 0 | 95.0 |
| **total** | fixed | score 0.20; C/P/F 0/2/3 | CONTRADICTS 3/5 (5) | 0.37 | 0/9 | 3/9 | 0.56 (79) | 0.24 | 0.16 | 0.56 / 0.24 / 0.16 (79) | - | 10.4 | 0 | 0 | 442 |
| **total** | vad | score 0.50; C/P/F 1/3/1 | CONTRADICTS 1/5 (1) | 0.80 | 0/9 | 6/9 | 0.63 (57) | 0.19 | 0.11 | 0.63 / 0.20 / 0.10 (51) | - | 4.8 | 0 | 0 | 383 |

Fixed-timing V4_A2 on other seeds (same calls; baseline for regression to the mean):

| call | seed | task | facts | check-line | echo | NATURAL | NONSENSE | cut-off |
|---|---|---|---|---|---|---|---|---|
| ecom_23_g1 | 1002 | PARTIAL | CONTRADICTS | 0.6667 | 1/3 | 10/17 | 1/17 | 1/17 |
| ecom_23_g1 | 1003 | PARTIAL | CONSISTENT | 0.6667 | 2/3 | 12/18 | 4/18 | 3/18 |

## V4_A (fixed) vs V4_A_vad (VAD), seed 1001

| call | mode | task | facts (#wrong) | check-line | echo | judge confirm+value | NATURAL | NONSENSE | cut-off | NAT / NONS / cut-off, tail-matched | CER | overlap s | barge-in | silent timeout | duration s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | fixed | PARTIAL | CONTRADICTS (2) | 0.5 | 0/2 | 0/2 | 11/16 | 3/16 | 2/16 | 11/3/2 of 16 | 0.0451 | 2.08 | 0 | 0 | 91.4 |
| cab_07_g2 | vad | PARTIAL | CONTRADICTS (1) | 0.5 | 0/2 | 1/2 | 5/12 | 2/12 | 2/12 | 5/1/1 of 10 | 0.0619 | 1.04 | 0 | 0 | 79.5 |
| food_03_g2 | fixed | PARTIAL | CONSISTENT (0) | 0.0 | 1/1 | 1/1 | 5/14 | 7/14 | 4/14 | 5/7/4 of 14 | 0.0794 | 1.68 | 0 | 0 | 78.4 |
| food_03_g2 | vad | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 8/15 | 4/15 | 3/15 | 6/3/3 of 12 | 0.0111 | 1.36 | 0 | 0 | 72.9 |
| sub_11_g4 | fixed | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 8/15 | 4/15 | 3/15 | 8/4/3 of 15 | 0.0702 | 4.4 | 0 | 0 | 74.1 |
| sub_11_g4 | vad | FAILED | CONSISTENT (0) | 1.0 | 0/1 | 0/1 | 10/12 | 2/12 | 1/12 | 9/2/1 of 11 | 0.0631 | 0.4 | 0 | 0 | 63.8 |
| air_11_g2 | fixed | COMPLETE | CONSISTENT (0) | 1.0 | 1/2 | 2/2 | 8/18 | 7/18 | 6/18 | 8/7/6 of 18 | 0.0402 | 1.84 | 0 | 0 | 93.8 |
| air_11_g2 | vad | PARTIAL | CONTRADICTS (1) | 1.0 | 0/2 | 0/2 | 7/14 | 4/14 | 4/14 | 7/3/3 of 13 | 0.0495 | 1.36 | 0 | 0 | 71.8 |
| ecom_23_g1 | fixed | PARTIAL | CONTRADICTS (2) | 1.0 | 2/3 | 1/3 | 9/19 | 6/19 | 5/19 | 9/6/5 of 19 | 0.046 | 4.08 | 0 | 0 | 104.0 |
| ecom_23_g1 | vad | PARTIAL | CONSISTENT (0) | 0.6667 | 0/3 | 2/3 | 9/12 | 3/12 | 2/12 | 8/3/2 of 11 | 0.1675 | 1.2 | 0 | 0 | 82.8 |
| **total** | fixed | score 0.50; C/P/F 1/3/1 | CONTRADICTS 2/5 (4) | 0.50 | 4/9 | 4/9 | 0.50 (82) | 0.33 | 0.24 | 0.50 / 0.33 / 0.24 (82) | - | 14.1 | 0 | 0 | 442 |
| **total** | vad | score 0.30; C/P/F 0/3/2 | CONTRADICTS 2/5 (2) | 0.63 | 0/9 | 3/9 | 0.60 (65) | 0.23 | 0.18 | 0.61 / 0.21 / 0.18 (57) | - | 5.4 | 0 | 0 | 371 |

Fixed-timing V4_A on other seeds (same calls; baseline for regression to the mean):

| call | seed | task | facts | check-line | echo | NATURAL | NONSENSE | cut-off |
|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | 1002 | PARTIAL | CONTRADICTS | 1.0 | 0/2 | 7/17 | 6/17 | 4/17 |
| cab_07_g2 | 1003 | PARTIAL | CONTRADICTS | 1.0 | 0/2 | 14/19 | 2/19 | 2/19 |
| food_03_g2 | 1002 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 6/15 | 4/15 | 3/15 |
| food_03_g2 | 1003 | COMPLETE | CONSISTENT | 0.0 | 1/1 | 4/15 | 5/15 | 5/15 |
| sub_11_g4 | 1002 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 6/15 | 6/15 | 5/15 |
| sub_11_g4 | 1003 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 8/14 | 4/14 | 4/14 |
| air_11_g2 | 1002 | PARTIAL | CONTRADICTS | 0.5 | 1/2 | 10/15 | 4/15 | 4/15 |
| air_11_g2 | 1003 | FAILED | CONTRADICTS | 0.5 | 0/2 | 7/17 | 5/17 | 5/17 |
| ecom_23_g1 | 1002 | PARTIAL | CONTRADICTS | 0.6667 | 1/3 | 12/16 | 3/16 | 3/16 |
| ecom_23_g1 | 1003 | PARTIAL | CONTRADICTS | 1.0 | 2/3 | 13/18 | 3/18 | 3/18 |
| **total** | 1002 | score 0.70; C/P/F 2/3/0 | CONTRADICTS 3 | - | 2/9 | 0.53 | 0.29 | 0.24 |
| **total** | 1003 | score 0.60; C/P/F 2/2/1 | CONTRADICTS 3 | - | 3/9 | 0.55 | 0.23 | 0.23 |


## v2 (headline): hold rule + tail matched to the fixed run — tags V4_A2_vad2, V4_A_vad2

Holds fired 1–3 times per call (V4_A2: 9 in total, V4_A: 8), with 0 hold timeouts. There were 2 barge-ins
(V4_A2 air_11_g2 and sub_11_g4: the model talked for 12 s without a detected stop) and 0 silent timeouts. All
runs end at the matched tail cap.

**Overlap** (customer audio playing while the model frame is active, same energy/text rule for both modes;
for fixed runs, "playing" = the scripted customer intervals) is lower with gating: 10.4 → 8.6 s (V4_A2) and
14.1 → 5.8 s (V4_A) over 5 calls. Calls are also shorter (442 s fixed → 410 / 400 s).

## V4_A2 (fixed) vs V4_A2_vad2 (VAD), seed 1001

| call | mode | task | facts (#wrong) | check-line | echo | judge confirm+value | NATURAL | NONSENSE | cut-off | NAT / NONS / cut-off, tail-matched | CER | overlap s | barge-in | silent timeout | duration s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | fixed | FAILED | CONTRADICTS (2) | 1.0 | 0/2 | 0/2 | 8/14 | 2/14 | 1/14 | 8/2/1 of 14 | 0.0409 | 2.72 | 0 | 0 | 91.4 |
| cab_07_g2 | vad | PARTIAL | CONTRADICTS (2) | 1.0 | 0/2 | 0/2 | 4/13 | 4/13 | 3/13 | 4/4/3 of 13 | 0.0849 | 2.96 | 0 | 0 | 82.8 |
| food_03_g2 | fixed | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 6/13 | 6/13 | 1/13 | 6/6/1 of 13 | 0.1029 | 1.2 | 0 | 0 | 78.4 |
| food_03_g2 | vad | COMPLETE | CONSISTENT (0) | 0.0 | 0/1 | 1/1 | 2/10 | 7/10 | 3/10 | 2/7/3 of 10 | 0.0951 | 1.2 | 0 | 0 | 72.3 |
| sub_11_g4 | fixed | PARTIAL | CONSISTENT (0) | 0.0 | 0/1 | 1/1 | 9/16 | 2/16 | 2/16 | 9/2/2 of 16 | 0.0556 | 1.44 | 0 | 0 | 74.1 |
| sub_11_g4 | vad | COMPLETE | CONTRADICTS (1) | 1.0 | 0/1 | 1/1 | 5/11 | 2/11 | 2/11 | 5/2/2 of 11 | 0.2717 | 1.12 | 1 | 0 | 68.5 |
| air_11_g2 | fixed | FAILED | CONTRADICTS (2) | 0.5 | 0/2 | 0/2 | 9/16 | 5/16 | 5/16 | 9/5/5 of 16 | 0.0419 | 2.08 | 0 | 0 | 93.8 |
| air_11_g2 | vad | PARTIAL | CONTRADICTS (2) | 1.0 | 0/2 | 0/2 | 10/13 | 2/13 | 2/13 | 10/2/2 of 13 | 0.0322 | 2.24 | 1 | 0 | 84.2 |
| ecom_23_g1 | fixed | PARTIAL | CONTRADICTS (1) | 0.3333 | 0/3 | 2/3 | 12/20 | 4/20 | 4/20 | 12/4/4 of 20 | 0.1326 | 2.96 | 0 | 0 | 104.0 |
| ecom_23_g1 | vad | PARTIAL | CONTRADICTS (1) | 0.6667 | 0/3 | 1/3 | 8/14 | 4/14 | 2/14 | 8/4/2 of 14 | 0.0662 | 1.12 | 0 | 0 | 102.2 |
| **total** | fixed | score 0.20; C/P/F 0/2/3 | CONTRADICTS 3/5 (5) | 0.37 | 0/9 | 3/9 | 0.56 (79) | 0.24 | 0.16 | 0.56 / 0.24 / 0.16 (79) | - | 10.4 | 0 | 0 | 442 |
| **total** | vad | score 0.70; C/P/F 2/3/0 | CONTRADICTS 4/5 (6) | 0.73 | 0/9 | 3/9 | 0.48 (61) | 0.31 | 0.20 | 0.48 / 0.31 / 0.20 (61) | - | 8.6 | 2 | 0 | 410 |

Fixed-timing V4_A2 on other seeds (same calls; baseline for regression to the mean):

| call | seed | task | facts | check-line | echo | NATURAL | NONSENSE | cut-off |
|---|---|---|---|---|---|---|---|---|
| ecom_23_g1 | 1002 | PARTIAL | CONTRADICTS | 0.6667 | 1/3 | 10/17 | 1/17 | 1/17 |
| ecom_23_g1 | 1003 | PARTIAL | CONSISTENT | 0.6667 | 2/3 | 12/18 | 4/18 | 3/18 |

## V4_A (fixed) vs V4_A_vad2 (VAD), seed 1001

| call | mode | task | facts (#wrong) | check-line | echo | judge confirm+value | NATURAL | NONSENSE | cut-off | NAT / NONS / cut-off, tail-matched | CER | overlap s | barge-in | silent timeout | duration s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | fixed | PARTIAL | CONTRADICTS (2) | 0.5 | 0/2 | 0/2 | 11/16 | 3/16 | 2/16 | 11/3/2 of 16 | 0.0451 | 2.08 | 0 | 0 | 91.4 |
| cab_07_g2 | vad | PARTIAL | CONTRADICTS (1) | 1.0 | 0/2 | 1/2 | 6/13 | 2/13 | 2/13 | 6/2/2 of 13 | 0.0653 | 2.0 | 0 | 0 | 89.6 |
| food_03_g2 | fixed | PARTIAL | CONSISTENT (0) | 0.0 | 1/1 | 1/1 | 5/14 | 7/14 | 4/14 | 5/7/4 of 14 | 0.0794 | 1.68 | 0 | 0 | 78.4 |
| food_03_g2 | vad | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 5/12 | 4/12 | 4/12 | 5/4/4 of 12 | 0.0085 | 1.36 | 0 | 0 | 67.0 |
| sub_11_g4 | fixed | FAILED | CONSISTENT (0) | 0.0 | 0/1 | 0/1 | 8/15 | 4/15 | 3/15 | 8/4/3 of 15 | 0.0702 | 4.4 | 0 | 0 | 74.1 |
| sub_11_g4 | vad | COMPLETE | MINOR (1) | 0.0 | 0/1 | 1/1 | 9/12 | 1/12 | 1/12 | 9/1/1 of 12 | 0.068 | 0.4 | 0 | 0 | 72.2 |
| air_11_g2 | fixed | COMPLETE | CONSISTENT (0) | 1.0 | 1/2 | 2/2 | 8/18 | 7/18 | 6/18 | 8/7/6 of 18 | 0.0402 | 1.84 | 0 | 0 | 93.8 |
| air_11_g2 | vad | FAILED | CONTRADICTS (2) | 1.0 | 0/2 | 0/2 | 8/16 | 5/16 | 5/16 | 8/5/5 of 16 | 0.1377 | 1.36 | 0 | 0 | 79.1 |
| ecom_23_g1 | fixed | PARTIAL | CONTRADICTS (2) | 1.0 | 2/3 | 1/3 | 9/19 | 6/19 | 5/19 | 9/6/5 of 19 | 0.046 | 4.08 | 0 | 0 | 104.0 |
| ecom_23_g1 | vad | PARTIAL | CONTRADICTS (1) | 0.6667 | 1/3 | 1/3 | 7/13 | 3/13 | 3/13 | 7/3/3 of 13 | 0.0924 | 0.64 | 0 | 0 | 92.0 |
| **total** | fixed | score 0.50; C/P/F 1/3/1 | CONTRADICTS 2/5 (4) | 0.50 | 4/9 | 4/9 | 0.50 (82) | 0.33 | 0.24 | 0.50 / 0.33 / 0.24 (82) | - | 14.1 | 0 | 0 | 442 |
| **total** | vad | score 0.40; C/P/F 1/2/2 | CONTRADICTS 3/5 (5) | 0.53 | 1/9 | 3/9 | 0.53 (66) | 0.23 | 0.23 | 0.53 / 0.23 / 0.23 (66) | - | 5.8 | 0 | 0 | 400 |

Fixed-timing V4_A on other seeds (same calls; baseline for regression to the mean):

| call | seed | task | facts | check-line | echo | NATURAL | NONSENSE | cut-off |
|---|---|---|---|---|---|---|---|---|
| cab_07_g2 | 1002 | PARTIAL | CONTRADICTS | 1.0 | 0/2 | 7/17 | 6/17 | 4/17 |
| cab_07_g2 | 1003 | PARTIAL | CONTRADICTS | 1.0 | 0/2 | 14/19 | 2/19 | 2/19 |
| food_03_g2 | 1002 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 6/15 | 4/15 | 3/15 |
| food_03_g2 | 1003 | COMPLETE | CONSISTENT | 0.0 | 1/1 | 4/15 | 5/15 | 5/15 |
| sub_11_g4 | 1002 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 6/15 | 6/15 | 5/15 |
| sub_11_g4 | 1003 | COMPLETE | CONSISTENT | 0.0 | 0/1 | 8/14 | 4/14 | 4/14 |
| air_11_g2 | 1002 | PARTIAL | CONTRADICTS | 0.5 | 1/2 | 10/15 | 4/15 | 4/15 |
| air_11_g2 | 1003 | FAILED | CONTRADICTS | 0.5 | 0/2 | 7/17 | 5/17 | 5/17 |
| ecom_23_g1 | 1002 | PARTIAL | CONTRADICTS | 0.6667 | 1/3 | 12/16 | 3/16 | 3/16 |
| ecom_23_g1 | 1003 | PARTIAL | CONTRADICTS | 1.0 | 2/3 | 13/18 | 3/18 | 3/18 |
| **total** | 1002 | score 0.70; C/P/F 2/3/0 | CONTRADICTS 3 | - | 2/9 | 0.53 | 0.29 | 0.24 |
| **total** | 1003 | score 0.60; C/P/F 2/2/1 | CONTRADICTS 3 | - | 3/9 | 0.55 | 0.23 | 0.23 |


## Summary (5 calls, seed 1001)

| | V4_A2 fixed | V4_A2 VAD v1 | V4_A2 VAD v2 | V4_A fixed | V4_A VAD v1 | V4_A VAD v2 | V4_A fixed s1002 / s1003 |
|---|---|---|---|---|---|---|---|
| task score (C/P/F) | 0.20 (0/2/3) | 0.50 (1/3/1) | 0.70 (2/3/0) | 0.50 (1/3/1) | 0.30 (0/3/2) | 0.40 (1/2/2) | 0.70 (2/3/0) / 0.60 (2/2/1) |
| facts CONTRADICTS calls | 3/5 | 1/5 | 4/5 | 2/5 | 2/5 | 3/5 | 3/5 / 3/5 |
| check-line (mean) | 0.37 | 0.80* | 0.73 | 0.50 | 0.63* | 0.53 | - |
| echo (writes) | 0/9 | 0/9* | 0/9 | 4/9 | 0/9* | 1/9 | 2/9 / 3/9 |
| judge confirm+value | 3/9 | 6/9 | 3/9 | 4/9 | 3/9 | 3/9 | - |
| NATURAL / NONSENSE / cut-off (lines) | .56/.24/.16 (79) | .63/.19/.11 (57) | .48/.31/.20 (61) | .50/.33/.24 (82) | .60/.23/.18 (65) | .53/.23/.23 (66) | .53/.29/.24 / .55/.23/.23 |
| overlap s (sum) | 10.4 | 4.8 | 8.6 | 14.1 | 5.4 | 5.8 | - |
| barge-in / silent timeout / hold | - | 0/0/- | 2/0/9 | - | 0/0/- | 0/0/8 | - |
| duration s (sum) | 442 | 383 | 410 | 442 | 371 | 400 | 442 |

\* v1 artifact: the customer cut in during the hold pause after the check-line (see above).

## Key observations

1. **Gating mechanics work.** Customer starts land exactly 0.32 s after the model's last active frame, or 0.64 s
   after a mid-sentence stop. Overlap roughly halves for V4_A. Calls are 7–10% shorter.
2. **A pure silence gate needs a hold rule.** The model has learned "check-line, pause, confirm". A 0.3 s gate
   puts the customer into that pause (v1: in 13 of 18 write slots the check-line was the model's only line). With the hold rule (v2), the model
   resumes by itself within the 5 s hold window every time (0 hold timeouts) and says "Done ji, ... updated".
3. **Write values are the real weakness.** Once given room to confirm, both adapters confirm with wrong or
   garbled values: "9650010001850" for 96500 18553, "Ajniyeh Muteen railway station" for Hazrat Nizamuddin,
   "7042086 3775" for 70420 86375, an old email read back for the new one. Echo stays 0/9 (V4_A2) and 1/9
   (V4_A), and v2 CONTRADICTS rises (V4_A2 1→4 vs v1) because those wrong values are now spoken. Fixed timing
   partly hid this, because the scripted customer often came in before the confirmation.
4. **No task-level gating effect is visible at this sample size.** V4_A2 0.20 → 0.70 looks large. But V4_A
   moves the other way (0.50 → 0.40), and fixed V4_A on other seeds of the same calls already scores 0.60–0.70.
   The calls were selected as worst-on-s1001, so expect regression to the mean. V4_A2 has an other-seed fixed
   baseline only for ecom_23_g1 (PARTIAL on s1002 and s1003).
5. **Naturalness does not improve.** NONSENSE / cut-off rates under v2 are within the fixed s1001–s1003 spread
   for V4_A (.23–.33 NONSENSE) and higher for V4_A2 (.31 vs .24). The v1 "improvement" is mostly fewer lines
   (the model was cut off at check-lines).

## Transcript excerpts (judge timeline; customer times = actually played)

air_11_g2, V4_A2, phone-number write. In the fixed run the customer comes in on schedule. In v1 the customer
cuts in after the check-line. In v2 the model confirms, but with a wrong number:

```
FIXED   [39.3] CUSTOMER: Theek hai, mera naya phone number 96500 18553 hai. Ispe saare updates bhej dena please.
        [48.4] AGENT: Ji, abhi update kar deta hoon, ek minute.
        [52.6] AGENT: Done ji, phone number update ho gaya hai: 98730 61042.        <- old number
        [60.2] AGENT: Ji sir,   [60.2] CUSTOMER: Haan, mera naya email address banerjee42@yahoo.co.in hai. ...
VAD v1  [34.4] CUSTOMER: Theek hai, mera naya phone number 96500 18553 hai. ...
        [43.4] AGENT: Ek minute rukiye, main check karke batata hoon.
        [46.5] CUSTOMER: Haan, mera naya email address banerjee42@yahoo.co.in hai. ...   <- 0.32 s into the hold
VAD v2  [43.4] AGENT: Ek minute rukiye, main check karke batata hoon.
        [48.6] AGENT: Done ji, contact number update ho gaya hai: 9650010001850.     <- confirms, wrong value
        [55.4] CUSTOMER: Haan, mera naya email address banerjee42@yahoo.co.in hai. ...
```

sub_11_g4, V4_A, refund request. Fixed: FAILED. VAD v2: judged COMPLETE, but the lines are not better:

```
FIXED   [28.7] CUSTOMER: Mujhe nahi chahiye yeh, please mere paise refund karwa dijiye.
        [33.2] AGENT: Ji ma'am, main samajh sakta hoon, please aap apna reference ID bataiye.
        [37.3] CUSTOMER: Reason simple hai, main no longer using the service, bas itna hi.
        [42.3] AGENT: Sirf wahan bahut thoda faftiya bilkul nahi hoga, ma'am.
VAD v2  [27.6] CUSTOMER: Mujhe nahi chahiye yeh, please mere paise refund karwa dijiye.
        [32.2] AGENT: Ji ma'am, yeh account ka final status active hai.
        [35.7] CUSTOMER: Reason simple hai, main no longer using the service, bas itna hi.
        [40.8] AGENT: Ji ma'am, kya main aapki koi aur help kar sakta hoon?
```

## Problems / caveats

- **Sample size:** 5 calls × 1 seed, selected as the worst on the comparison seed. Task-score deltas of ±0.3 are
  within the other-seed spread. To measure a gating effect, run all 30 calls (or these 5) with seeds 1002 and
  1003 in both modes (about 5 min of GPU per tag per seed for 5 calls).
- **0.3 s is aggressive for this model:** 47% of its own between-sentence pauses are ≥ 0.64 s, so the customer
  sometimes comes in mid-turn. Making `q_sent` a larger parameter (e.g. 0.6 s) would be a different experiment.
- **Tail:** the model never goes quiet for 3 s, so the 3 s tail rule never fires. v1 used the 15 s cap, which let
  post-goodbye speech into the judge and the Trelis CER. v2 matches the fixed run's tail instead. This deviates
  from the "e.g. 15 s" in the request, for comparability.
- **Hold detection** uses the model's own text (regex). A hold phrase the regex misses would still be cut in on.
  None was seen in these runs.
- The judge and metric windows for VAD runs use the actual times (`<run>.tmeta.json`). For agent turns, the
  "scripted" window becomes the gap between the played customer turns.

## Files

- Code (pod, /workspace/hinglish/tests):
  - new: vad_gate.py, test_vad_gate.py.
  - changed: driver.py, run_tests.py, tcommon.py, score.py, v4_eval.py, judge.py. Backups are `*.bak_vad`.
  - scratch: vad_scratch/ (rank/calibration/compare scripts, vad_go*.sh, vad_post.sh, compare*.md/json, logs/).
- Runs:
  - out/V4_A2_vad, out/V4_A_vad (v1);
  - out/V4_A2_vad2, out/V4_A_vad2 (v2; `--vad-params '{"hold": true, "tail_match_fixed": true}'`);
  - identity-check scratch tags out/vadchk_fixed_V4_A2 and out/vadchk_v1_V4_A2.
- GPU (gpu_minutes.csv rows vad_*): tests about 5 min per tag (rtf 0.40) plus 2 min per check; Trelis 1–2 min;
  Gemma suite 7 min per batch. About 40 min in total.
- Laptop copy: <laptop>/hinglish/vad_gating/ (this file; wavs/vad2/<tag>_<call>_stereo.wav,
  wavs/vad1/, wavs/fixed/<tag>_<call>_stereo.wav). Stereo L = model, R = customer.
