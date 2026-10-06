"""Assembles REPORT.md: hand-written sections (below) + results/report_tables.md (make_report.py) +
tool schemas as sent (tools_json/) + data stats + resolver tests. Run make_report.py first."""
import json, os
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was this script's own dir
P = lambda *a: os.path.join(HERE, *a)
N1_PKG = os.environ.get("N1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                 "packages", "needle_router", "n1"))  # tools_json/ lives in the package

HEAD = r'''# N1: Needle 3 as the Hinglish action router — report

2026-10-04. Spec: `handoffs/N1_needle_router.md` §5-6. Decision log: `DECISIONS.md` (sections A-J).
Every number below comes from `results/<model>_<test>.json` (scored by `evaluate.py`), and
`make_report.py` regenerates all the tables. Every failure is listed verbatim, with the model's reasoning string, in
`REPORT_APPENDIX_failures.md`.

## 1. Results in brief

- **Tool choice: the fine-tune fixed it.** Tool match on held-out positives: base 30/216 (a) and 3/216 (b); tuned_full 195/216 (a) and 173/216 (b). On test_n0, Hinglish went from 1/24 to 19/24.
- **Resolver (on tool-matched rows):** tuned_full gets the gold ID on 194/195 (a) and 172/173 (b).
  - These held-out records have one active entity, and 151/195 (a) of those calls omit order_ref, so the default rule decides most of them. They do not test N0's two-order case.
  - On test_n0, the only set with two active orders, tuned_full's resolver is right on 31/39 tool-matched rows (Hinglish 11/19; base 11/14).
  - 4 test_n0 gold refs are unresolvable by the resolver itself (E8).
- **Negatives: the tuned models call a tool on chit-chat. This blocks using router.py on free-running turns.** Empty-list accuracy: base 278/324 (a) and 301/324 (b); tuned_full 8/324 (a) and 7/324 (b); tuned_l8 32/324 (a).
  - Base's empty lists are almost all suppressed calls (274/278 on a, 300/301 on b). If those had shipped, base would score 4/324 (a) and 1/324 (b), against tuned_full 7/324 and 6/324 (E9 table, §5).
  - Base's suppressor also withholds most positives (185/216 on a). So raw decoding did not get worse.
  - What changed is that tuned models have no confidence head and so no confidence gate (the engine still withholds a few ungrounded calls: tuned_full rows n0 1, a 2, b 16, §4.4), and the fine-tune taught `[]` only on a path the engine does not use.
  - Cause, confirmed by diagnostic §4.1: train.jsonl has no `reasoning` lines, so every training target starts with `<tool_call>`. The engine always decodes a `<think>` block first.
  - With the training render (no think), the e10 weights return `[]` on 30/30 held-out negatives. With `<think>` forced, they return `[]` on 1/30.
  - Per tool and per negative type (§4.4, `results/neg_analysis.json`): tuned_full ships a call on 316/324 (a) and 317/324 (b) negatives, one call per row, spread over all 13 tools of the five tool sets (most: request_refund a 77 / b 86, cancel_order 66 / 63, request_ticket_cancellation 52 / 42, cancel_ride 30 / 32). The shipped tool is the call's own write tool on only 43/316 (a) and 49/317 (b). Every negative type is hit: greeting_or_first 57/57 (a), in-write-window 104/104 (a), after_write 80/83 (a).
- **Base reproduction:** base re-run on pod2 gives 0 differing `function_calls` rows out of 1,140 against the laptop base run (§4.3). Laptop and pod2 also agree row for row on all three models on test_n0.
- **tuned_l8 (the 8-layer rung) is not usable.**
  - test_n0: tool match 10/48, with `cancel_order` on almost every row.
  - It produces looping reasoning: rows over 1,000 chars are n0 2, a 2, b 15.
  - The base 8-layer rung with no LoRA scores tool match 14/48 and empty list 3/12, so the rung itself is weak too (§4.2).
  - The LoRA was trained at full depth only. `needle finetune` has no layers option.
- **Laptop latency (test_n0, 60 rows, one process; pass 2 figures, §3):**

  | model | median | p95 |
  |---|---|---|
  | base | 364 ms | 1,059 ms |
  | tuned_full | 349 ms | 618 ms |
  | tuned_l8 | 103 ms | 208 ms |
- **Chosen model:** `tuned_full.cact` (e10 adapter, 20 layers). router.py uses it.
  - It should only be called on turns the client already flags with a check-line. On free chit-chat it calls a tool (`"Theek hai, thank you so much for the help."` → `cancel_ride`, §7).
- **Correct = tool + real args + resolver.** Correct on positives:
  - tuned_full: a 130/216, b 79/216, n0 27/48.
  - base: a 28/216, b 2/216, n0 11/48.
- **Argument errors are now the main loss on positives**, mostly phone (a 7/40, b 3/33) and email/location/reason on (b).
  - 17 (a) and 31 (b) tuned_full phone predictions are `9800011085`, which no test gold contains.
  - That number is one of the 5 train phone values (each on 24 lines). The model never predicts the other 4.
  - It is also the `e.g.` value in the tool schema's phone description (tools.py `_PHONE`), so the schema example is the likely source.

## 2. Setup

| item | value |
|---|---|
| models | base = package `needle3.cact` (3.0.2 archive, md5 71c31b0b…); tuned_l8 = `tuned_l8.cact` (e10 LoRA, `--layers 8`, md5 53649254…); tuned_full = `tuned_full.cact` (e10 LoRA, 20 layers, md5 92ddd0d7…) |
| chosen adapter | `finetune/sweep/adapter_e10.safetensors` (md5 d6205037…), val token_mean 0.1855 (FINETUNE.md) |
| engine | cactus-needle 3.0.6, `libneedle.so` md5 679dc570… (identical on laptop and pod2) |
| test files | test_n0 `10f91e24…`, test_heldout_a `21ad08b0…`, test_heldout_b `c6c2bdb9…` (unchanged since G5/H1) |
| system | each row's `system` = `build_data.system_text(record)`, starting `date: 2026-10-04 Sun 10:00; user: …`; `Needle(..., auto_date=False)`; `agent._system_text == row.system` asserted per group; `run_meta.auto_date` false in every raw file |
| call | `agent.reset(); agent.complete(query, 512)`, turn 1 only, one untimed warm-up per (tools, system) group |
| accuracy runs | tuned_l8, tuned_full and base_pod2: pod2 CPU, one process at a time with `taskset -c 0-29` (`results/run_all_seq.sh`). base: the earlier laptop run (`results/base_*`, E15), reproduced exactly on pod2 |
| latency runs | laptop CPU (12 cores, 22 GB), test_n0, one process at a time, engine threads not capped (the engine has no thread setting; N0 did not cap either, E2), MemAvailable watchdog at 6 GB (never hit: 10-11 GB available) |
| scoring | evaluate.py docstring and DECISIONS E4-E12. Phone digits-only; address/instruction token F1 ≥ 0.6; other args normalised equality; order_ref normalised equality, or omission when gold omits (A1); resolver on the predicted order_ref against the record |

Slices: the test file fixes the rendering (test_n0 and heldout_a = (a) text as written; heldout_b = (b) Trelis ASR).
English vs Hinglish is the hindi_share < 0.10 label. Held-out English is only 17 rows (3 positives), so the
English-vs-Hinglish comparison rests on test_n0 (30/30).

## 3. Latency (laptop CPU, test_n0, 60 rows, ms, `complete()` only)

| model | pass | 1-min load at start | median | p95 | max | median / p95 excl. first row | rows with runaway reasoning | decode tok/s (median) | peak engine RAM |
|---|---|---|---|---|---|---|---|---|---|
| base | E15 run (05:13) | 0.17 | 341.9 | 908.7 | 2433 | 345.9 / 918.9 | 0 | 248 | 102 MB |
| base | pass 1 | 0.35 | 416.0 | 1040.8 | 2600 | 417.5 / 1052.8 | 0 | 222 | 102 MB |
| tuned_l8 | pass 1 | 5.55 | 116.3 | 899.3 | 1971 | 116.6 / 916.3 | 2 | 582 | 76 MB |
| tuned_full | pass 1 | 6.09 | 354.6 | 760.5 | 1141 | 355.4 / 766.0 | 0 | 220 | 156 MB |
| tuned_full | pass 2 | 0.70 | 348.8 | 618.2 | 784 | 350.2 / 620.6 | 0 | 232 | 156 MB |
| tuned_l8 | pass 2 | 0.96 | 102.8 | 207.5 | 864 | 102.4 / 207.6 | 2 | 618 | 76 MB |
| base | pass 2 | 0.99 | 364.1 | 1059.3 | 2827 | 364.7 / 1063.6 | 0 | 234 | 101 MB |

- Pass 1 (`results/laptop_latency.sh`, tags `laptop_*`) ran the models back to back. Pass 2 (`laptop_latency_r2.sh`, tags `laptop_r2_*`) ran them in reverse order and waited for the 1-min load to fall below 1.0 before each model.
- Browsers were open during both passes. Pass 2 is the headline.
- Outputs were identical in both passes (same `function_calls`, same correct counts).
- Latency is mostly decode length: the base p95 comes from long reasoning strings. tuned_l8 is fast per token, but on 2 rows the reasoning loops.
- Pod2 latencies in the result files are not comparable with the laptop and are not used.
  - Four parallel engine processes under pod2's ~31-CPU cgroup quota ran at 6-14 s per row. The runs were restarted one at a time (J2).

## 4. Diagnostics

### 4.1 Why the tuned models fire on negatives (`finetune/diag_think.py` → `finetune/sweep/diag_think.json`)
Setup: greedy JAX decode on pod2 GPU, e10 LoRA merged, CQ W4 numerics (as in `finetune._score_quantised`). The rows are the first 30 negatives and the first 30 positives of test_heldout_a, each decoded with two prompts:

| prompt | negatives: `[]` | positives: exact calls |
|---|---|---|
| plain: training render, the target starts at `<tool_call>` | 30/30 | 20/30 |
| think: the same prompt + `<think>\n` | 1/30 | 13/30 |

- Agreement with the engine's `function_calls` on these 60 rows:
  - Tool names: think path 57/60, plain path 30/60.
  - Exact calls: think path 38/60, plain path 19/60.
- Every engine response, base and tuned, carries a non-empty `reasoning` string. So the engine opens a think block before the call.
- train.jsonl has no `reasoning` field (GUIDE_NOTES §1: the think block is rendered only if reasoning is non-empty). The fine-tune therefore taught `[]` only on the no-think path. The val loss was also measured on that path, so it did not show the problem.

### 4.2 8-layer rung (test_n0)
| model | tool match (pos) | empty list (neg) | correct (pos) |
|---|---|---|---|
| base (20 layers) | 14/48 | 10/12 | 11/48 |
| base_l8 (rung 8, no LoRA; `finetune/sweep/base_l8.cact`, md5 3ecae504…) | 14/48 | 3/12 | 5/48 |
| tuned_l8 | 10/48 | 0/12 | 4/48 |
| tuned_full | 39/48 | 2/12 | 27/48 |

### 4.3 Reproduction and cross-machine checks (`results/report_checks.json`)
| comparison | rows | rows whose `function_calls` differ | correct (rows) |
|---|---|---|---|
| base (laptop, E15) vs base_pod2 / test_n0 | 60 | 0 | 21 vs 21 |
| base vs base_pod2 / test_heldout_a | 540 | 0 | 306 vs 306 |
| base vs base_pod2 / test_heldout_b | 540 | 0 | 303 vs 303 |
| laptop_tuned_full vs tuned_full (pod2) / test_n0 | 60 | 0 | 29 vs 29 |
| laptop_tuned_l8 vs tuned_l8 (pod2) / test_n0 | 60 | 0 | 4 vs 4 |
| laptop_base vs base / test_n0 | 60 | 0 | 21 vs 21 |

## 5. Tables per test set × rendering × model (§5 metrics)

Column definitions:
- **correct (pos):** tool match, real-argument match and the resolver returns the gold ID.
- **real args:** every argument except order_ref.
- **order_ref = gold (named):** calls where the gold names the entity.
- **order_ref omitted:** calls where gold omits order_ref (A1) and the prediction omits it too.
- **resolver:** `resolver.resolve(predicted order_ref)` equals the gold entity ID.
- **empty list:** negatives whose `function_calls == []`.

The E9 table also shows each model with base's suppressed-only negatives counted as calls, because base's
empty lists are mostly suppressions (274/278 on a), and tuned models have no confidence head.

'''

TAIL = r'''
## 6. Failures

The Failure groups table above has the counts per test × model. `REPORT_APPENDIX_failures.md` lists every non-correct row of every model and test verbatim: query, gold, function_calls, suppressed_calls and the model's `reasoning` string. The base rows come from the laptop run and the tuned rows from pod2. The same data, with the raw engine response, is in `results/<model>_<test>.json` → `failures`.

## 7. Router (`router.py`)

`route(agent_type, record, transcript)` returns `[{name, arguments, order_ref, resolved_id, resolver_rule}]`.
- It uses the same pipeline as the eval: `tools.tools_for(agent_type)`, the `build_data.system_text(record)` pinned system with `auto_date=False`, `tuned_full.cact`, then `resolver.resolve`.
- The gate is "a call shipped": with no confidence head, every `function_calls` entry is returned.
- `resolved_id None` (rule `ambiguous`) means the server must ask which entity.
- Check (`python router.py`, `results/router_selftest.txt`, run on the laptop): routed 3 held-out positives. Their function_calls are identical to the eval run's for the same rows, 3/3.
  - (a) V1:cab_15_g2:w2: correct, RD5520 via `default_active`.
  - (a) V1:air_11_g1:w0: phone `980012105` against gold `9800012105` (one 0 dropped).
  - (b) V1:air_11_g1:w0: phone is the leaked `9800011085`.
  - All three resolve to the gold entity.

```python
import json, router                                   # needle venv, CPU; tuned_full.cact next to router.py
rec = json.load(open("src/records.json"))["cab_15"]   # the frontloaded record for this call
calls = router.route("cab_ride_support", rec, "Achha, ek instruction add kar do, passenger wearing a red scarf.")
# [{'name': 'add_driver_instruction', 'arguments': {'instruction': 'Passenger wearing a red scarf'},
#   'order_ref': None, 'resolved_id': 'RD5520', 'resolver_rule': 'default_active'}]
router.route("cab_ride_support", rec, "Theek hai, thank you so much for the help.")
# [{'name': 'cancel_ride', ...}]  <- known defect (§4.1): only route turns flagged by a check-line
for c in calls:
    print(c["name"], c["resolved_id"] or "ASK WHICH ONE", c["arguments"])
router.close()
```
'''

def main():
    i5 = HEAD.index("## 5. Tables")
    neg = open(P("results", "neg_analysis.md"), encoding="utf-8").read()   # neg_analysis.py (J12)
    L = [HEAD[:i5] + neg + "\n" + HEAD[i5:], open(P("results", "report_tables.md"), encoding="utf-8").read(), TAIL]
    L.append("\n## 8. Tool schemas as sent (`tools_json/<agent_type>.json` = `tools.tools_for(agent_type)`, "
             "identical to every test row's `tools`: 1,080/1,080 held-out rows checked)\n")
    for f in sorted(os.listdir(os.path.join(N1_PKG, "tools_json"))):
        L.append(f"### {f[:-5]}\n\n```json\n" + json.dumps(json.load(open(os.path.join(N1_PKG, "tools_json", f))), ensure_ascii=False, indent=1) + "\n```\n")
    ds = open(P("DATA_STATS.md"), encoding="utf-8").read()
    i = ds.index("## Final Needle datasets")
    L.append("\n## 9. Data stats (from DATA_STATS.md; full file next to this report)\n")
    L.append(ds[i:].replace("## Final Needle datasets", "### Final Needle datasets").replace("\n### ", "\n#### ").replace("#### Final", "### Final"))
    L.append("\n## 10. Resolver tests (`test_resolver.py` → `test_resolver_output.txt`)\n\n```\n"
             + open(P("test_resolver_output.txt"), encoding="utf-8").read().rstrip() + "\n```\n")
    L.append("Gold order_ref resolver sanity on the test files: test_n0 52/56 (4 noisy spellings → ambiguous, E8), "
             "heldout a 216/216, heldout b 216/216.\n")
    if os.path.exists(P("results", "audit.md")):   # audit.py (K-section)
        L.append("\n" + open(P("results", "audit.md"), encoding="utf-8").read())
    open(P("REPORT.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote REPORT.md", sum(len(x) for x in L))

main()
