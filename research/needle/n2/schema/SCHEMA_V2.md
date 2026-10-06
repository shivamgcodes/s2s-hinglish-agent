# Needle v2 tool schema (N2, 2026-10-05)

Code: `tools_v2.py` (schemas), `resolver_v2.py` (REF → record value / ASK), `targets.py` (gold writes → training targets), `router_v2.py` (inference glue), `score_map.py` (N1-comparable scoring), `test_resolver_v2.py` (17 tests, all pass on V4). Tool JSON per agent type is in `tools_json/`.

## The idea in one paragraph

Needle runs only after the agent's check-line, so each call should produce one write. Every argument is one of two kinds.

- **REF**: the value is already in the record, which is in Needle's system text. The model copies the exact value from the record: the order/ride/booking/account/transaction/charge ID, the card's last 4 digits, or the plan, service or pack name. The resolver maps that copy to the record value. It also accepts spoken phrases ("Coursera wala", "purana order", "25 wala pack") and returns **ASK** when it can't pick exactly one value.
  - For a number already on file (`block_sim`), the model writes only `registered` or `alternate`, never the digits.
- **NEW**: the caller says a value that isn't in the record: a new phone number, new email, new address, pickup/drop place, instruction or reason. The model copies it from the transcript after numbers have been converted.
  - Only these arguments carry free-form values, so this is the only place the model writes a phone number.

## Where each scripted value comes from (V4 `calls.jsonl`, 645 calls / 647 writes)

I checked every `writes[].args` value against `records.json`. Two places to look:
- **information**: the record text the model sees.
- **caller_values**: what the caller says.

| agent type | tool | writes[] arg | source in V4 | class | model arg | n |
|---|---|---|---|---|---|---|
| food / ecom | cancel_order | order_id | info, = primary_id 24/24 | REF order | `order_ref` | 24 |
| food / ecom | change_delivery_address | order_id | info, primary 48/48 (caller said it 27/48) | REF order | `order_ref` | 48 |
| | | address | caller_values 48/48 | NEW | `address` (req) | |
| food / ecom | add_delivery_instruction | order_id | info, primary 51/51 | REF order | `order_ref` | 51 |
| | | instruction | caller_values (paraphrased in 3) | NEW | `instruction` (req) | |
| food / ecom | update_contact_number | phone | caller_values 36/36 (new, family, friend) | NEW | `phone` (req) | 36 |
| food / ecom | request_refund | reference_id | info, = order ID 24/24 | REF order | `order_ref` | 24 |
| | | reason | caller_values | NEW | `reason` (req) | |
| cab | cancel_ride / change_pickup_location / change_drop_location / add_driver_instruction | ride_id | info, primary 88/88 | REF ride | `ride_ref` | 88 |
| | | location | caller_values (new_pickup / new_drop) 52/52 | NEW | `location` (req) | |
| | | instruction | caller_values | NEW | `instruction` (req) | |
| cab | update_contact_number | phone | caller_values 20/20 | NEW | `phone` | 20 |
| cab | request_refund | reference_id | info, = ride ID 4/4 | REF ride | `ride_ref` | 4 |
| sub | cancel_subscription / pause_subscription | plan | info, current plan 34/34 (caller said it only 8/34) | REF plan | `plan_ref` | 34 |
| sub | update_contact_number / update_email_address | phone / email | caller_values 46/46 | NEW | `phone` / `email` | 46 |
| sub | request_refund | reference_id | info, = account ID 8/8 | REF account | `account_ref` | 8 |
| air | request_ticket_cancellation | pnr | info, PNR 20/20 | REF booking | `booking_ref` | 20 |
| air | request_refund | reference_id | info: PNR 16, **charge ref** (duplicate charge / baggage fee, = secondary_id) 8 | REF booking | `booking_ref` | 24 |
| air | update_contact_number / update_email_address | phone / email | caller_values 28/28 | NEW | `phone` / `email` | 28 |
| bank | block_card | card_last4 | info, the record's only card 28/28 (caller said it 18/28) | REF card | `card_ref` | 28 |
| bank | raise_transaction_dispute | transaction_id | info, = primary_id 20/20; caller said the ID 20/20 | REF transaction | `transaction_ref` | 20 |
| | | reason | caller_values | NEW | `reason` (req) | |
| bank | request_card_replacement | card_last4 | info 20/20 (caller said it 6/20) | REF card | `card_ref` | 20 |
| | | address | caller_values 20/20, **never** the registered address | NEW, may be omitted | `address` (optional) | |
| bank | update_contact_number / update_email_address | phone / email | caller_values 28/28 | NEW | `phone` / `email` | 28 |
| tel | deactivate_service | service | info (active-services list) 20/20, caller said it 20/20 | REF service | `service_ref` | 20 |
| tel | activate_pack | pack | info (available-packs list) 20/20 | REF pack | `pack_ref` | 20 |
| tel | block_sim | sim_number | info: = the record's registered `phone` 16/16 | REF phone_on_file | `number_ref` (`registered`/`alternate`) | 16 |
| tel | update_alternate_number | phone | caller_values 20/20 | NEW | `phone` (req) | 20 |
| tel | request_refund | reference_id | info, = primary_id (TL…) 20/20 | REF charge | `charge_ref` | 20 |
| | | reason | caller_values | NEW | `reason` (req) | |

Checks behind the table, all in `test_resolver_v2.py`:
- Every gold REF value appears verbatim in `build_data.system_text(record)` (469/469).
- Every gold REF value is among the resolver's candidates.
- The round trip gold → target → resolver → server args gives back the gold for all 647 check-lines.

## Schema rules

- **No raw ID arguments.** Each reference has its own `*_ref` name: `order_ref`, `ride_ref`, `booking_ref`, `account_ref`, `plan_ref`, `card_ref`, `transaction_ref`, `service_ref`, `pack_ref`, `charge_ref`, `number_ref`.
- **REF arguments are optional.** In N1, Needle withheld calls whose required argument had no span. If the model leaves a REF out, the resolver uses the record's only candidate, or the primary one.
- **NEW arguments are required.** The exception is `request_card_replacement.address`; see below.
- **`update_contact_number` no longer has `order_ref`.** No V4 script carries an entity for it.
- **Descriptions contain no example values**: no phone number, email, address, ID or place. They state the format only. N1 copied `9800011085` from its old phone description, and its old address example "Flat 204, Sector 15" collides with bank_01's real "Flat 204, Prestige…". A test enforces this: no 4-digit runs, no `x@y.z`, no ID-shaped strings.
- **Phone target** is exactly 10 digits with no spaces. **Email target** is lower-case.

## Resolver v2 (`resolver_v2.resolve(kind, ref, record)`)

Each kind gets its candidates from the record:

| kind | candidates |
|---|---|
| ID kinds | `[A-Z]{2}dddd` tokens in `information`, plus `primary_id`, each with the words around it, e.g. "Older order FD3076 from Sagar Ratna", "TX6742 Coursera … Rs 4,127 on 26 November", "duplicate charge WS5041". `transaction` is limited to the TX prefix. |
| plan | Current plan from the facts, plus the previous plan from the distractors. |
| card | "ending dddd", "Card dddd", or `facts.card_last4`. Every V4 bank record has exactly one card. |
| service / pack | Parsed from the active-services and available-packs lists. Prices and the "(…)" part are dropped. |
| phone_on_file | The registered phone and the alternate number on record. |

The rules are tried in this order; the first one that leaves exactly one candidate wins:
1. **exact**: the trained target.
2. **ID / digit block**: also covers card last 4 and phone suffix.
3. **N1 phrase rules**: old types only.
4. **containment**: the name inside the ref, e.g. "cricket alerts band karo".
5. **words**: content words unique to one candidate's context: merchant, restaurant, amount. A number in the ref must be on that candidate, so "DataBoost" alone gives ASK when both DataBoost 10 and DataBoost 25 exist.
6. **ordinal**: naya/latest picks the primary; purana/previous picks the non-primary one.
7. **default**: an empty ref picks the only candidate, or the primary for order, ride, booking, account, plan, card and phone_on_file.

ASK cases:
- An empty ref when there are 2+ transactions, services, packs or charges.
- An ID-shaped ref that isn't in the record (`unknown_id`), so a misheard or invented ID never silently becomes the primary. Since the review fix (REVIEW-schema-1), this also covers spoken or digit-only forms ("7138", "F D seven one three eight", Devanagari digits): for ID kinds, a ref with an ID shape or a run of 3+ digits that matched nothing gives ASK. Before the fix these went to the primary.
- A card or phone number whose digits don't match the record.
- Tied matches. Example: bank_20 has two Indian Oil charges for the same amount, so "Indian Oil" gives ASK.

Before matching, the ref goes through the same `numconv` + romanise steps as the transcript, so "data boost pachchees" and Devanagari refs still match.

## Ambiguous cases and how I handled them

1. **`request_card_replacement.address`.** In V4 all 20 are a new address the caller says, and none is the registered address. I made it a NEW, **optional** argument:
   - if it's omitted, or says "same / registered / wahi address", the resolver uses the registered address on file;
   - if the record has no registered address, it returns ASK.
   - **There is no training data for the omitted path**, so it's untrained. If you'd rather make it required and always ASK when it's missing, it's a one-line change in `tools_v2._T`.
2. **Air `request_refund`** points at the PNR 16 times and at a separate charge ref (WS5041 duplicate charge, NW8638 baggage fee) 8 times. Both are `booking_ref`; the target is the exact ID. An empty `booking_ref` defaults to the PNR, which would be wrong for those 8, so the model has to copy the charge ID.
3. **`block_sim.sim_number`** is the registered phone in all 16. The target is `number_ref: "registered"`, so the model never copies the 10 digits. `alternate` maps to the alternate number on record; there are no V4 examples of it.
4. **Subscription plan.** In sub_04 the script wrote "CineMax Ultra" but the record says "StreamBox CineMax Ultra". Targets and gold are canonicalised to the record value.
5. **Gold is almost always the primary entity.** The exceptions are 8 air charge refs, plus choosing among 2 services or packs (40) and 2 transactions (20). So **V4 barely trains or tests choosing the older or other order/ride/booking.** For old types the resolver's phrase and ordinal rules come from N1 and are unit-tested, but the model has little data to learn the non-primary choice from.
6. **Phone numbers the caller gives for someone else** (`family_phone`, `friend_phone`, `phone` in caller_values) are still NEW `phone`.
7. **Paraphrased NEW values.** Some reason/instruction values in the script are not exactly what the caller said (7 reasons, 5 instructions on `text_roman`). `targets.grounded()` flags these so the data stage can filter them or take the transcript span instead.

## Training target style (`targets.target_call(..., ref_style=...)`)

- **`canonical`** (default; DECISIONS 2026-10-05): every REF argument is always filled with the exact record value, and phone_on_file gets `registered`/`alternate`. This matches "if a record is to be picked from the record, just have it pick the right one". It also removes the N1 named-vs-omitted ambiguity in targets.
- **`mentioned`**: a REF argument is present only when its value, or its ID digits, appears in the transcript; otherwise it's left out and the resolver defaults. This is close to N1. On `text_roman` windows, the ref is mentioned for order 57/147, ride 51/92, plan 8/34 and card 20/48.

The data stage can switch between them; the resolver accepts both.

## Check-line → gold

A write belongs to the **last** check-line strictly between its `caller_turn_idx` and its `confirm_turn_idx` (`targets.checkline_writes`). On V4, all 647 check-lines get exactly one write. The looser "any check-line in between" rule would give 2 writes to 9 check-lines: air_15, where one caller turn asks for cancel + refund.

## Router v2 (`router_v2.route(agent_type, record, transcript, weights)`)

The interface is the same as N1's `router.route`.

1. `prepare_transcript(transcript)` = `numconv.convert` (the sibling module, imported) and then N1's `romanise` on whatever is still in Devanagari.
   - **The data stage must build training inputs with this same function.**
2. The system string is N1's `build_data.system_text(record)`; it contains the record. It works on all 162 V4 records, including bank and telecom.
3. Needle runs with `tools_v2.tools_for(agent_type)`.
4. REF arguments go through `resolver_v2`.
5. NEW arguments are checked:
   - phone must be 10 digits, otherwise ASK;
   - email is normalised ("at the rate", "dot") and must look like an email, otherwise ASK;
   - text must not be empty;
   - a missing required argument gives ASK.

Each call returns `server_args` (named as in `writes[]`), `refs` with the rule used, `ask`/`ask_reasons`, and the N1 keys `order_ref`/`resolved_id`/`resolver_rule`. The default weights path is `$NEEDLE_V2_WEIGHTS` or `/root/n2/models/needle_v2.cact`, which doesn't exist until training.

## N1-comparable scoring (`score_map.py`)

Both systems are mapped into the server's `writes[]` shape and scored against the same canonical gold for each check-line.

- **N1:** `resolved_id` (from the N1 resolver) fills `order_id`/`ride_id`/`pnr`/`reference_id`.
  - For cancel/pause subscription, N1's `order_ref` is resolved with `resolver_v2`'s plan kind. The gold is a plan name, which N1's resolver can't output.
  - N1's `order_ref` on `update_contact_number` is ignored.
- **v2:** `server_args`.

How arguments are compared:

| argument | rule |
|---|---|
| ID / card / plan / service / pack / sim | exact |
| phone | 10-digit equal |
| address, instruction | token F1 ≥ 0.6 |
| location, reason, email | N1's normalised-equality rules, imported from N1 `evaluate.compare_arg` (token F1 is also reported) |

A call that ends in ASK counts as not executed, i.e. not correct. The comparison covers the 5 old types only; bank and telecom are v2-only.

For a fair comparison, N1 `tuned_full` has to be re-run on the same V4 30-second inputs. Those inputs are converted with `prepare_transcript`, which also covers N1's romanised-input convention.
