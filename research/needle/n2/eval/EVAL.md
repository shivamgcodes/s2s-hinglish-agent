# N2 eval: Needle v2 (tuned_full = R_e15) on the v2 test sets, against N1 tuned_full

2026-10-05. Everything ran on runpod2, the RTX 4090 pod (d7e199d0669a, root@81.27.69.179 -p 28547). The engine decodes on the CPU. Nothing ran on the L4.
This is the first time the test files have been touched. Nothing was tuned or re-selected on them.

## What was run

| model | weights | rows | path per row (the router's path) |
|---|---|---|---|
| **v2** | `/root/n2/finetune/tuned_full.cact` (md5 5361a65b…, R_e15) | `data/rows/test_{clean,opus,exact}` (136 each, 7 agent types) | `needle.Needle(tools_v2, system_text(record))` → `complete(query, 512)` → `router_v2.resolve_calls` (resolver_v2 + phone/email checks) → `score_map.v2_to_server` → `score_map.score` against `meta.gold_canonical`. The query is already `router_v2.prepare_transcript(raw Trelis)` (numconv, then romanise), as the data stage built it. Runner: `finetune/eval_engine.py`, unchanged. |
| **N1** | `/root/n2/needle_n1/tuned_full.cact` (md5 92ddd0d7…) | `data/rows/n1_test_{clean,opus,exact}` (104 each, the 5 old types; N1 tools, N1 system text, romanise only) and the `_nc` versions (the same rows, but the query goes through v2 numconv) | `Needle(N1 tools, N1 system)` → `complete` → N1 `router.resolve_calls` (N1 resolver on `order_ref`) → `score_map.n1_to_server` → the same `score_map.score` against the same gold. Runner: `eval/eval_n1.py`. |
| v2 A_e10 (informational only) | `finetune/sweep/full_A_e10.cact` | same as v2 | same as v2. Gives test evidence for the open R_e15-vs-A_e10 question. **Not a re-selection.** |

Variants:
- **clean**: Trelis transcript of the clean 16 kHz window.
- **opus**: Trelis transcript of the Opus-decoded window. This is the router's real input.
- **exact**: the gold customer text, number-converted. 56/136 exact rows include a customer turn that is only partly inside the 30 s window, so their gold text has a few words the audio lacks.

**Scoring (score_map, N1 `evaluate.compare_arg` rules)**

- REF args (order_id, ride_id, reference_id, plan, card_last4, transaction_id, service, pack, sim_number) must match the resolved value exactly.
- phone must be the same 10 digits.
- address and instruction pass at token F1 ≥ 0.6.
- location, reason and email must be equal after normalisation (strict). Token F1 ≥ 0.6 for reason and location is reported alongside as "soft".
- **correct** = the same tool list, every gold arg matching, no extra arg, and no ASK.
- **ask** = the router would not execute the call; it asks the caller instead.
- **silent wrong** = a call that ships without an ASK and is not correct, i.e. a wrong write reaches the server.

## 1. Headline: v2 on all 136 test windows

| variant | right tool | **correct** | correct (soft reason/location) | ask | silent wrong | …not wording-only | silent wrong phone | silent wrong email | missed (no call) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 122 | **68** (50%) | 78 | 12 | 53 | 43 | 1 | 2 | 3 |
| opus | 122 | **69** (51%) | 80 | 10 | 54 | 43 | 2 | 2 | 3 |
| exact | 123 | **101** (74%) | 108 | 7 | 27 | 20 | 0 | 0 | 1 |

- **Clean vs opus:** the Opus codec costs nothing (68 vs 69).
- **ASR gap:** about 32 correct calls separate Trelis text from gold text.
  - Restricted to the 118 opus rows where every NEW value is literally in the transcript: 68/118 correct and 0 silent wrong phone/email.
  - All 4 silent wrong phone/email writes come from transcripts that already carry the wrong value.
- **Exact rows, by partial first turn:**
  - partial: 41/56 correct
  - no partial: 60/80 correct (75%)
- **Opus rows, by the same split:** partial 22/56, no partial 47/80 (59%).

The informational A_e10 run on the same rows:

| | right tool | correct | soft | ask | silent wrong (not wording-only) | missed |
|---|---|---|---|---|---|---|
| clean | 116 | 61 | 70 | 17 | 52 (43) | 6 |
| opus | 116 | 61 | 72 | 16 | 52 (41) | 7 |
| exact | 113 | 79 | 89 | 9 | 47 (37) | 1 |

On test, R_e15 is ahead of A_e10 by 7–8 correct on ASR text and by 22 on exact text. On ASR text the two ship about the same number of silent wrong writes (43 vs 41–43 non-wording). The val worry that "R_e15 trades asks for wrong writes" (54 vs 33) does **not** reproduce on test.

## 2. Per argument (v2, all 7 types)

Counting rule, as in N1's per_argument: **ok / n** counts rows where the tool was right and the gold has that arg. **all** is the number of rows whose gold has the arg, whatever tool was predicted. Wrong-tool rows count as misses in `ok/all`.

| arg | class | clean ok / n (all) | opus ok / n (all) | exact ok / n (all) | note |
|---|---|---|---|---|---|
| order_id | REF | 33/37 (40) | 33/37 (40) | 32/36 (40) | misses: food_22 copies the second order FD2179 instead of FD9761 (4 windows) |
| ride_id | REF | 17/17 (24) | 17/17 (24) | 20/20 (24) | all misses are wrong-tool rows |
| reference_id | REF | 16/16 (16) | 16/16 (16) | 16/16 (16) | |
| plan | REF | 8/8 (8) | 8/8 (8) | 8/8 (8) | |
| card_last4 | REF | 8/8 (8) | 8/8 (8) | 7/7 (8) | |
| transaction_id | REF | 4/4 (4) | 4/4 (4) | 4/4 (4) | |
| service | REF | 3/5 (8) | 3/5 (8) | 4/5 (8) | 2 unresolvable refs ("Rs 49/month", "Meeting Start Hone Wali") → ASK |
| pack | REF | 3/3 (4) | 3/3 (4) | 3/3 (4) | |
| sim_number | REF | – | – | – | **untested:** the test split has no block_sim or update_alternate_number windows |
| phone | NEW | 16/20 (20) | 16/20 (20) | 20/20 (20) | all 4 misses: the transcript has the wrong or 9-digit number (2 ASK, 2 silent) |
| email | NEW | 0/4 (4) | 0/4 (4) | 4/4 (4) | none of the 4 test emails is in the transcript ("banarji 42 at yahoo dot co dot in"); 2 ASK, 2 silent |
| address | NEW | 12/18 (20) | 14/18 (20) | 17/17 (20) | |
| instruction | NEW | 12/21 (24) | 12/21 (24) | 13/20 (24) | |
| location | NEW | 1/11 strict, 7/11 soft (16) | 0/11 strict, 7/11 soft (16) | 8/14 strict, 14/14 soft (16) | strict equality is too strict for spoken places |
| reason | NEW | 4/20 strict, 8/20 soft (20) | 4/20 strict, 8/20 soft (20) | 14/20 strict, 15/20 soft (20) | |

**REF args**

- The REF schema works. On every ASR variant, every REF arg except order_id and service is at 100% of tool-matched rows.
- order_id: 4 misses, one consistent model error (it copies the other order ID).
- service: the 2 misses become ASKs, not wrong writes.

**NEW args**

- NEW free-text args (reason, location, instruction) are the main loss.
- phone and email are limited by the transcript, not by Needle. On exact text they are 20/20 and 4/4.

**Resolver** (rows with the right tool and a REF gold arg; v2 rules recomputed with `router_v2.resolve_calls` on the stored calls):

| | rows | all REF args = gold | resolver rules used |
|---|---|---|---|
| v2 clean | 98 | 92 | exact 94, unmatched 2 (→ASK), unknown_id 1, n1:name 1 |
| v2 opus | 98 | 92 | exact 95, unmatched 2 (→ASK), n1:name 1 |
| v2 exact | 99 | 94 | exact 96, unknown_id 2, unmatched 1 |
| N1 opus (native) | 61 | 59 | default_active 36, default_active_unmatched 19, id 13, name 7 |
| N1 opus (+numconv) | 59 | 56 | id 29, default_active 23, default_active_unmatched 16, name 6 |

- **v2:** the model copies the record value, and 95/98 refs resolve by the `exact` rule.
- **N1:** N1 mostly lands on the right ID by **defaulting** to the single active entity. Its order_ref is usually junk ("kar", "Select", "EC").

## 3. Per agent type (v2)

| agent type | n | clean right tool / correct / soft | opus right tool / correct / soft | exact right tool / correct / soft | opus ask / silent |
|---|---|---|---|---|---|
| airport_ticket_counter | 8 | 8 / 4 / 4 | 8 / 4 / 4 | 8 / 8 / 8 | 2 / 2 |
| bank_card_support | 16 | 16 / 11 / 13 | 16 / 12 / 14 | 15 / 15 / 15 | 0 / 4 |
| cab_ride_support | 24 | 17 / 5 / 11 | 17 / 4 / 11 | 20 / 10 / 16 | 0 / 18 |
| ecommerce_support | 32 | 31 / 16 / 16 | 31 / 17 / 17 | 29 / 27 / 27 | 2 / 13 |
| food_delivery_support | 24 | 22 / 13 / 14 | 22 / 13 / 14 | 23 / 17 / 17 | 0 / 10 |
| subscription_account_support | 16 | 16 / 9 / 10 | 16 / 9 / 10 | 16 / 13 / 14 | 0 / 7 |
| telecom_prepaid_support | 16 | 12 / 10 / 10 | 12 / 10 / 10 | 12 / 11 / 11 | 6 / 0 |

- **Bank** (new type) is strong: 12/16 on opus.
- **Telecom** (new type) has 0 silent wrong writes. Its 4 wrong-tool rows (activate_pack / deactivate_service mix-ups) are all caught by the resolver as ASK (`pack_ref:no_candidates`, `service_ref:unmatched`).
- **Cab is the weak type.** Strict location fails, and the model fires request_refund or cancel_ride on hospital pickup-change calls.

## 4. Side by side on the same 104 old-type windows: N1 vs v2

Rows are joined on example_id; the join is 104 = 104, asserted.
- **N1 native** = N1 as deployed (romanised Trelis text).
- **N1 + numconv** = the same N1 model on the v2 number-converted text. This isolates the converter's effect.

| variant | model | right tool | **correct** | soft | ask | silent wrong | silent wrong phone | silent wrong email | missed |
|---|---|---|---|---|---|---|---|---|---|
| clean | N1 native | 76 | **23** | 25 | 2 | 69 | 12 | 3 | 10 |
| clean | N1 + numconv | 75 | **37** | 40 | 3 | 58 | 1 | 4 | 6 |
| clean | v2 R_e15 | 94 | **47** | 55 | 6 | 48 | 0 | 2 | 3 |
| clean | v2 A_e10 (info) | 89 | 43 | 50 | 9 | 46 | 0 | 2 | 6 |
| opus | N1 native | 75 | **24** | 25 | 2 | 68 | 11 | 3 | 10 |
| opus | N1 + numconv | 74 | **38** | 38 | 3 | 56 | 2 | 4 | 7 |
| opus | v2 R_e15 | 94 | **47** | 56 | 4 | 50 | 1 | 2 | 3 |
| opus | v2 A_e10 (info) | 89 | 43 | 52 | 8 | 46 | 0 | 1 | 7 |
| exact | N1 native | 81 | **64** | 68 | 2 | 37 | 1 | 1 | 1 |
| exact | N1 + numconv | 82 | **66** | 70 | 2 | 35 | 0 | 1 | 1 |
| exact | v2 R_e15 | 96 | **75** | 82 | 2 | 26 | 0 | 0 | 1 |
| exact | v2 A_e10 (info) | 89 | 57 | 66 | 3 | 43 | 0 | 0 | 1 |

Paired correct on the same rows, N1 native vs v2:

| variant | both | only N1 | only v2 | neither |
|---|---|---|---|---|
| clean | 20 | 3 | 27 | 54 |
| opus | 22 | 2 | 25 | 55 |
| exact | 59 | 5 | 16 | 24 |

**Reading it (opus, the deployed input):** N1 native 24 → N1 + numconv 38 → v2 47 correct of 104.

- The **number converter alone accounts for +14.** It removes N1's phone failures: silent wrong phone goes from 11 to 2, because N1 can't add up "nine six five zero…".
- The **v2 schema + fine-tune adds +9** on top of that.
- **Right tool rises from 75 to 94.** N1 confused pickup and drop and missed cab calls.

Per arg on the 104 opus rows (ok / tool-matched n / all rows):

| arg | N1 native | N1 + numconv | v2 |
|---|---|---|---|
| order_id | 31/31/40 | 29/29/40 | 33/37/40 |
| ride_id | 7/7/24 | 7/7/24 | 17/17/24 |
| reference_id | 15/15/16 | 15/15/16 | 16/16/16 |
| plan | 6/8/8 | 5/8/8 | 8/8/8 |
| phone | **0**/11/12 | 9/11/12 | 9/12/12 |
| email | 0/3/4 | 0/4/4 | 0/4/4 |
| address | 3/15/16 | 8/14/16 | 11/14/16 |
| instruction | 6/14/24 | 7/13/24 | 12/21/24 |
| location (strict / soft) | 0 / 0 of 2 | 0 / 0 of 2 | 0 / 7 of 11 |
| reason (strict / soft) | 2 / 3 of 15 | 2 / 2 of 15 | 2 / 4 of 16 |

**N1's phones on opus:** 0/11 are correct on the rows where N1 picked the right tool. Some values are not 10 digits ('96502053', '9650205373853', '74202086375'). Others are 10 digits and still wrong ('7420203875', '4220386375', '3004020543'), so a 10-digit check alone would not have caught all of them. N1's router has no phone check at all, so every one ships. Running numconv before Needle is what fixes this: N1 + numconv gets 9/11. This is the "ASR writes phones as words" failure from N1's REPORT, now confirmed on V4.

Per agent type, opus, correct (N1 native / N1 + numconv / v2):

| agent type | correct |
|---|---|
| airport | 0 / 4 / 4 |
| cab | 3 / 3 / 4 |
| ecommerce | 9 / 15 / 17 |
| food | 5 / 10 / 13 |
| subscription | 7 / 6 / 9 |

The full per-variant tables are in eval.json.

## 5. Latency (CPU, single process, runpod2)

- **Rows:** the same 60 opus old-type windows for both models (`eval/lat/ids.json`), plus 3 warm-up rows that were run and then dropped.
- **What was timed:** `complete()` only. Agent build is excluded.
- **Repeats:** two passes each, run one after the other with nothing else in the container.

| model | pass 1 median / p95 / max (ms) | pass 2 median / p95 / max (ms) |
|---|---|---|
| N1 tuned_full | 533 / 692 / 2177 | 571 / 758 / 2571 |
| v2 tuned_full | 432 / 575 / 2505 | 454 / 639 / 2513 |

- v2 is not slower than N1, even though v2 decodes a reasoning line. The engine decodes a think block for both.
- **Max outliers are degenerate repetition** that runs to the token cap:
  - v2 on food_23_g4: `'under office isliyeeeeeee…'`, which ships no call.
  - N1 on food_22_g2: `L'O'O'O'O…`.
- **CPU caveat:** runpod2 is an AMD EPYC 7542 container with a CFS quota of 10.2 CPUs (64 visible). The host is shared, with a load average of about 20–25 from other tenants. The laptop CPU is different, and so are its absolute numbers (N1 REPORT: 0.2–0.4 s on the laptop).

## 6. Top failure patterns (v2, opus), with verbatim examples

Failure groups on opus (68 not correct of 136):

| group | count |
|---|---|
| wrong argument: reason | 16 |
| wrong argument: location | 11 |
| wrong tool | 11 |
| wrong argument: instruction | 8 |
| email | 4 |
| address | 4 |
| phone | 4 |
| missed call | 3 |
| order_id | 3 |
| service | 2 |
| instruction + order_id | 1 |

In 17 of these 68 the gold value is not in the transcript (meta `grounded` false). That covers all 4 email, all 4 phone and 3 location rows.

**F1. Reason copied from the wrong span, or with an ASR word attached (16).**

Strict matching fails even on near-verbatim copies, and some reasons come from the wrong span altogether:
- `bank_20_g1` — Q `…ka dispute raise kar do region duplicate charge for same fuel purchase hai`
  - gold reason `Duplicate charge for same fuel purchase`
  - got `Region duplicate charge for same fuel purchase`. The ASR word "region" (spoken "reason") is kept.
- `ecom_22_g1` — Q `…order EC1543 ke liye refund chahiye kyunki ordered wrong item by mistake`
  - got reason `ke liye refund chahiye`. The model picked the wrong span.
- `ecom_22_g2` — Q `…reason ye hai ki ordered wrong item by mistake forest essentials facial uptan 50 G`
  - got `forest essentials facial uptan 50 G`. The model picked the wrong span.

**F2. Location: partial, padded or wrong span (11; 0/11 strict, 7/11 soft).**
- `cab_03_g1` — Q `…ab main IGI airport terminal 3 arrival gate 6 par hoon`
  - gold `IGI Airport Terminal 3, Arrival Gate 6`
  - got `Main Thoda, Confused, Airport Terminal 3`
- `cab_03_g2` — got `airport terminal 3`. Truncated.
- `cab_03_g4` — got `pickup IGI airport terminal 3 arrival gate 6`. Right span plus one leading word; strict fails, soft passes.

**F3. Wrong tool (11: cab 5, all in scenario cab_26; telecom 4, all caught as ASK; ecom 1; food 1).**
- `cab_26_g1` — Q `…relative different wing mein hai naya address hai kokilaben hospital andheri west gate 4 discharge lounge`
  - gold change_pickup_location
  - got `request_refund {reason: 'Patient reported an injury during a discharge'}`. Ships silently.
- `cab_26_g2__cl1` — Q `…ek aur request hai please help passenger with a wheelchair`
  - gold add_driver_instruction
  - got `request_refund {reason: 'Wheelchair access'}`
- `cab_26_g3` — gold change_pickup_location; got `cancel_ride {ride_ref: 'RD5350'}`. **A destructive wrong write.**
- Telecom: `tel_20_g1/g3/g4` want deactivate_service 'Cine-Buzz Premium' and get `activate_pack {pack_ref: 'Premium'}`. Caught by the resolver as an ASK (`pack_ref:no_candidates`). `tel_03_g1__cl1` is the reverse (deactivate_service 'DataBoost 25' for activate_pack), caught as `service_ref:unmatched`.
- Others, which ship: `cab_26_g4__cl1` (instruction → change_pickup_location), `ecom_23_g2__cl2` (instruction → update_contact_number), `food_07_g4` (address → add_delivery_instruction).

**F4. Instruction rephrased or hallucinated (8).**
- `ecom_07_g3` — Q `…please likh lijiye hand package to brother at the door`
  - got `Packed aiye, please leave the parcel and call on arrival`
  - Reasoning `'kiye' -> instruction`: the model invents a training-like instruction.
- `ecom_07_g2` — got `Package to brother door woh, wheelchair accessible`. An invented tail.
- `cab_26_g3__cl1` — got `Passenger wheelchair`, gold `Please help passenger with a wheelchair`. F1 is below 0.6.

**F5. Email not in the transcript (4/4 test email rows, transcript-limited).**
- Trelis + romanise gives `banarji 42 at yahoo dot co dot in` for banerjee42@yahoo.co.in.
- Outputs:
  - `banarji42.co` → ASK (`email_malformed`)
  - `banati2@yahoo.co` → **ships**
  - `banarji.42@yahoo.co` → **ships**
- The model also drops the final `.in`, which is a model error on top of the ASR one.

**F6. Phone wrong in the transcript (4/4 transcript-limited).**
- `bank_11_g2`: the transcript says `9014096697` (gold 9015096697) and it ships. Needle copied it correctly.
- `ecom_07_g4`: `6391071564` vs 6397071564. Ships.
- `ecom_07_g1` / `g3`: 9 digits → ASK `phone_not_10_digits:9`. The safety net works.

**F7. Engine suppressed the call (missed, 3).**
- In 2 of 3 the model did emit a call, but the engine moved it to `suppressed_calls`, which the router ignores:
  - `cab_07_g1__cl1` suppressed `change_drop_location {location: "Hai used bhi Chang's Wagon, Saket, Hai Zagangi"}`
  - `cab_07_g3` suppressed `change_pickup_location {location: 'Mavera Pic, Hajrat Nijyamuddin, Rajiv Gandhi'}`
- The third (`food_23_g4`) is the degenerate `isliyeeee…` loop.

**F8. Wrong record ID copied (order_id, 4 windows of food_22).**
- Instruction-after-address-change calls copy the other order `FD2179` instead of `FD9761`. Ships silently.

**F9. Reasoning names an arg that the call omits (1, but dangerous).**
- `bank_07_g3__cl1`: the reasoning says `'flat 502 hirandani gardens powai mumbai' -> address`, but the call is `request_card_replacement {card_ref: '4651'}` with no address.
- The router then fills the **registered address on file**, so the card ships to the old address without any ASK.

**N1 (opus, native) top groups, for contrast:**

| group | count |
|---|---|
| wrong tool | 19 |
| reason | 13 |
| address | 12 |
| phone | 11 |
| missed | 10 |

Examples:
- phone `nine six five zero zero one eight five five three` → `96502053` (ships)
- pickup → change_drop_location
- order_ref `kar` / `Select` / `EC`
- reason `I am in a fault, no refund`
- address `H seventy three mayoor`

With numconv, N1's phone and address errors mostly go away. Its wrong tool rises to 23, mostly change_drop_location for pickup, sometimes as 2 calls.

## 7. Notes and hiccups

- **Longest N1 row.** The N1 exact row of about 1028 tokens (cab_07_g2__cl0) decoded without error in 662 ms and produced a call (a wrong tool). There was no crash. I did not check whether the engine truncated the input.
- **N1 resolver input.** The N1 resolver ran on the V4 record unchanged; it has primary_id, secondary_id, distractors and facts. No fields were patched.
- **Load.** The host load average of about 20–25 during the runs comes from other tenants. Our container was idle apart from the one engine. The runs were sequential, with one engine at a time.
- **Tooling hiccup.** My first local wait loop had a logic bug and never exited. I stopped it and replaced it. The runs were not affected.

## Files

All under `/root/n2/eval`:
- `EVAL.md`
- `eval.json`: every table above, per variant, including the side-by-side per arg and per type, and A_e10.
- `runs/*.json`: per row, with the query, gold, function_calls, suppressed calls, reasoning, server args, per-arg detail and latency.
- `lat/latency.json` and `lat/*.json`
- `eval_n1.py`, `analyse.py`, `run_all.sh`, `lat/run_lat.sh`
