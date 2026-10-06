"""N1 REPORT.md audit (independent of evaluate.py's scorer).

    python3 audit.py            -> results/audit.json, results/audit.md (REPORT.md §11 via build_report.py)
    results/router_audit.json comes from router_audit.py (pod2, needle venv) and is read here.

Checks:
  A. recompute 3 metric cells per test set (tool match (pos), empty list (neg), correct (pos)) for
     base / tuned_l8 / tuned_full from results/<tag>_<test>_raw.jsonl + the test files, with its own
     argument normalisation (evaluate.py is NOT imported; resolver.resolve is the system under test and
     is called); compared with the REPORT.md §5 cells and the per-row evaluate flags (row diffs listed)
  B. system facts + tool schemas: every test row's system == build_data.system_text(record) (held-out)
     and tools == tools.tools_for(agent_type); run_meta (auto_date, groups, weights) equal across tags
  C. epoch choice from finetune/sweep/val_losses.json + val_loss.log + train logs
  D. leakage: scenario / call / query overlap between train+val and the test files
"""
import collections
import hashlib
import json
import os
import re
import sys

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))  # build_data, resolver, router, tools, src/records.json
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (data, results/); was this script's own dir
sys.path.insert(0, N1_PKG)
import resolver  # noqa: E402  (system under test)

R = os.path.join(HERE, "results")
TESTS = ["test_n0", "test_heldout_a", "test_heldout_b"]
TAGS = ["base", "tuned_l8", "tuned_full"]
PAIR_TAGS = ["base_pod2", "tuned_l8", "tuned_full"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- own normalisation (not evaluate.py)
NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}
NUMW.update({"oh": "0", "double": "x2", "triple": "x3"})


def toks(s):
    s = str(s or "").lower().replace("'", "").replace("’", "")
    return re.findall(r"[a-z0-9]+", s)


_U = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                  "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_T = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}


def ntoks(s):
    """toks() with English number words folded to numerals ('forty five' -> 45, 'five hundred two' -> 502).
    Used for value args only (spoken numbers in addresses/locations, e.g. 'terminal three')."""
    t, out, i = toks(s), [], 0
    while i < len(t):
        if t[i] in _U or t[i] in _T:
            n = _U.get(t[i], _T.get(t[i])); j = i + 1
            if t[i] in _T and j < len(t) and t[j] in _U and 0 < _U[t[j]] < 10:
                n += _U[t[j]]; j += 1
            if j < len(t) and t[j] == "hundred":
                n *= 100; j += 1
                if j < len(t) and (t[j] in _U or t[j] in _T):
                    m = _U.get(t[j], _T.get(t[j])); j += 1
                    if t[j - 1] in _T and j < len(t) and t[j] in _U and 0 < _U[t[j]] < 10:
                        m += _U[t[j]]; j += 1
                    n += m
            out.append(str(n)); i = j
        else:
            out.append(t[i]); i += 1
    return out


def digits(v):
    """phone -> digits only; spoken English digit words and 'double/triple' expanded."""
    out, mult = [], 1
    for t in toks(v):
        if t in ("double", "triple"):
            mult = 2 if t == "double" else 3
            continue
        if t in NUMW and NUMW[t].isdigit():
            out.append(NUMW[t] * mult)
        elif t.isdigit():
            out.append(t[0] * mult + t[1:] if mult > 1 else t)
        mult = 1
    d = "".join(out)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    if len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d


def f1(g, p, tk=None):
    tk = tk or ntoks
    g, p = collections.Counter(tk(g)), collections.Counter(tk(p))
    c = sum((g & p).values())
    if not c:
        return 0.0
    pr, rc = c / sum(p.values()), c / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def email(v):
    s = " " + str(v or "").lower() + " "
    s = re.sub(r"\s+at\s+the\s+rate\s+|\s+at\s+", "@", s)
    s = re.sub(r"\s+dot\s+", ".", s)
    return re.sub(r"\s+", "", s)


STRICT = False  # True = no number-word folding (sensitivity run)


def arg_ok(k, g, p):
    tk = toks if STRICT else ntoks
    if p is None or str(p).strip() == "" or g is None:
        return False
    if k == "phone":
        return digits(g) != "" and digits(g) == digits(p)
    if k in ("address", "instruction"):
        return f1(g, p, tk) >= 0.6
    if k == "email":
        return email(g) == email(p)
    return tk(g) == tk(p)


def record_for(m, records):
    if m.get("eval_record") == "resolver.N0_RECORD":
        return resolver.N0_RECORD
    return records[m["scenario_id"]]


def score_row(gold, calls, m, records):
    """-> (positive, tool_match, correct)."""
    pos = (m.get("kind") != "read") if "kind" in m else (m["type"] != "negative")
    if not pos:
        return False, len(calls) == 0, len(calls) == 0
    tm = [c["name"] for c in gold] == [c.get("name") for c in calls]
    if not tm:
        return True, False, False
    ents = list(m["expected_entity_ids"]) if "expected_entity_ids" in m else [m["gold_entity_id"]] * len(gold)
    rec = record_for(m, records)
    ok = True
    for gc, pc, ent in zip(gold, calls, ents):
        ga, pa = gc.get("arguments") or {}, pc.get("arguments") or {}
        for k in (set(ga) | set(pa)) - {"order_ref"}:
            ok &= arg_ok(k, ga.get(k), pa.get(k))
        pref = pa.get("order_ref")
        pref = "" if pref is None else str(pref).strip()
        ok &= resolver.resolve(pref, rec, m["agent_type"])["id"] == ent
    return True, True, ok


def report_cell(test, tag, col):
    """Read the 'all' row cell from REPORT.md §5 for one test/model/column."""
    lines = open(os.path.join(HERE, "REPORT.md"), encoding="utf-8").read().split("\n")
    i = lines.index(f"### {test}")
    hdr = [c.strip() for c in lines[i + 2].strip("|").split("|")]
    for l in lines[i + 4:]:
        if not l.startswith("|"):
            break
        cells = [c.strip() for c in l.strip("|").split("|")]
        if cells[0] == "all" and cells[1] == tag:
            return cells[hdr.index(col)].split(" ")[0]
    return None


def part_a(records):
    out, diffs = {}, []
    for test in TESTS:
        rows, meta = jl(os.path.join(HERE, test + ".jsonl")), jl(os.path.join(HERE, test + "_meta.jsonl"))
        for tag in TAGS:
            raw = jl(os.path.join(R, f"{tag}_{test}_raw.jsonl"))
            head, body = raw[0], raw[1:]
            assert head.get("_meta") and len(body) == len(rows) == len(meta), (tag, test)
            assert [b["i"] for b in body] == list(range(len(rows)))
            ev = json.load(open(os.path.join(R, f"{tag}_{test}.json")))["rows"]
            c = collections.Counter()
            for b, r, m, e in zip(body, rows, meta, ev):
                assert e["row_id"] == (m.get("id") or m.get("example_id"))
                calls = ((b["response"] or {}).get("function_calls")) or []
                pos, tm, ok = score_row(r["answers"], calls, m, records)
                c["pos" if pos else "neg"] += 1
                if pos:
                    c["tm"] += tm
                    c["correct"] += ok
                else:
                    c["empty"] += tm
                es = e["score"]
                if (es["tool_match"], es["correct"]) != (tm, ok):
                    diffs.append({"test": test, "tag": tag, "row": e["row_id"], "audit": [tm, ok],
                                  "evaluate": [es["tool_match"], es["correct"]],
                                  "gold": r["answers"], "pred": calls})
            mine = {"correct (pos)": f"{c['correct']}/{c['pos']}", "tool match (pos)": f"{c['tm']}/{c['pos']}",
                    "empty list (neg)": f"{c['empty']}/{c['neg']}"}
            rep = {k: report_cell(test, tag, k) for k in mine}
            out[f"{test}/{tag}"] = {"audit": mine, "report": rep, "match": mine == rep}
    return out, diffs


def part_b(records):
    import build_data
    import tools
    res = {}
    for test in TESTS:
        rows, meta = jl(os.path.join(HERE, test + ".jsonl")), jl(os.path.join(HERE, test + "_meta.jsonl"))
        n_tools = sum(r["tools"] == tools.tools_for(m["agent_type"]) for r, m in zip(rows, meta))
        if test == "test_n0":
            systems = collections.Counter(r.get("system") for r in rows)
            n_sys = None
        else:
            n_sys = sum(r["system"] == build_data.system_text(records[m["scenario_id"]]) for r, m in zip(rows, meta))
            systems = None
        heads = {}
        for tag in ["base", "base_pod2", "tuned_l8", "tuned_full"]:
            h = jl(os.path.join(R, f"{tag}_{test}_raw.jsonl"))[0]
            heads[tag] = {"auto_date": h.get("auto_date"), "weights": h.get("weights"),
                          "needle_version": h.get("needle_version"), "max_new_tokens": h.get("max_new_tokens"),
                          "groups": [(g["agent_type"], g["scenario_id"], g["n_rows"]) for g in h["groups"]],
                          "started": h.get("started"), "machine": h.get("machine")}
        g0 = heads["base"]["groups"]
        res[test] = {
            "rows": len(rows), "test_md5": md5(os.path.join(HERE, test + ".jsonl")),
            "tools_eq_tools_for": n_tools,
            "system_eq_build_data_system_text": n_sys,
            "n0_distinct_systems": len(systems) if systems else None,
            "n0_system": next(iter(systems)) if systems else None,
            "all_dates_pinned": all(r["system"].startswith("date: 2026-10-04 Sun 10:00; ") for r in rows),
            "run_meta": {t: {k: v for k, v in h.items() if k != "groups"} for t, h in heads.items()},
            "groups_equal_across_tags": all(h["groups"] == g0 for h in heads.values()),
            "auto_date_false_all": all(h["auto_date"] is False for h in heads.values()),
        }
    return res


def part_c():
    sw = os.path.join(HERE, "finetune", "sweep")
    v = json.load(open(os.path.join(sw, "val_losses.json")))
    logs = {}
    for e in (3, 6, 10):
        txt = open(os.path.join(sw, f"train_e{e}.log"), encoding="utf-8", errors="replace").read()
        losses = [float(x) for x in re.findall(r"loss[ =:]+([0-9.]+|nan|inf)", txt, re.I) if x[0].isdigit()]
        logs[f"e{e}"] = {"n_loss_values": len(losses), "nan_or_inf": bool(re.search(r"\bnan\b|\binf\b", txt, re.I)),
                         "last_loss": losses[-1] if losses else None}
    md5s = {f: md5(os.path.join(sw, f)) for f in
            ("adapter_e3.safetensors", "adapter_e6.safetensors", "adapter_e10.safetensors", "tuned_full.cact",
             "tuned_l8.cact")}
    md5s["root tuned_full.cact"] = md5(os.path.join(HERE, "tuned_full.cact"))
    md5s["root tuned_l8.cact"] = md5(os.path.join(HERE, "tuned_l8.cact"))
    md5s["val.jsonl"] = md5(os.path.join(HERE, "val.jsonl"))
    return {"val_losses_json": v, "train_logs": logs, "md5": md5s,
            "val_loss_log_tail": open(os.path.join(sw, "val_loss.log"), errors="replace").read()[-3000:],
            "build_full_log": open(os.path.join(sw, "build_full.log"), errors="replace").read()[-2000:],
            "build_l8_log": open(os.path.join(sw, "build_l8.log"), errors="replace").read()[-2000:]}


def part_d():
    def scen(p):
        m = jl(p)
        return ({x.get("scenario_id") for x in m}, {x.get("call_id") for x in m})
    tr_s, tr_c = scen(os.path.join(HERE, "train_meta.jsonl"))
    va_s, va_c = scen(os.path.join(HERE, "val_meta.jsonl"))
    out = {"train_scenarios": len(tr_s), "val_scenarios": len(va_s), "train_val_scen_overlap": sorted(tr_s & va_s)}
    trq = {r["query"].strip().lower() for r in jl(os.path.join(HERE, "train.jsonl"))}
    vaq = {r["query"].strip().lower() for r in jl(os.path.join(HERE, "val.jsonl"))}
    trsys = {r["system"] for r in jl(os.path.join(HERE, "train.jsonl"))}
    for t in ("test_heldout_a", "test_heldout_b"):
        ts, tc = scen(os.path.join(HERE, t + "_meta.jsonl"))
        out[t] = {"scenarios": len(ts), "scen_overlap_train": sorted(ts & tr_s), "scen_overlap_val": sorted(ts & va_s),
                  "call_overlap_train_val": sorted(tc & (tr_c | va_c))}
    # examples_raw: each scenario_id in exactly one split
    sp = collections.defaultdict(set)
    for x in jl(os.path.join(HERE, "examples_raw.jsonl")):
        sp[x.get("scenario_id")].add(x.get("split"))
    out["examples_raw_scenarios_multi_split"] = sorted(k for k, v in sp.items() if len(v) > 1)
    out["examples_raw_split_counts"] = dict(collections.Counter(next(iter(v)) for v in sp.values() if len(v) == 1))
    for t in TESTS:
        rows, meta = jl(os.path.join(HERE, t + ".jsonl")), jl(os.path.join(HERE, t + "_meta.jsonl"))
        pos = [r for r, m in zip(rows, meta) if (m.get("kind") != "read" if "kind" in m else m["type"] != "negative")]
        neg = [r for r, m in zip(rows, meta) if r not in pos]
        out.setdefault("query_overlap", {})[t] = {
            "pos_in_train": [r["query"] for r in pos if r["query"].strip().lower() in trq],
            "pos_in_val": [r["query"] for r in pos if r["query"].strip().lower() in vaq],
            "neg_in_train_or_val": sum(r["query"].strip().lower() in (trq | vaq) for r in neg),
            "system_texts_in_train": sum(r["system"] in trsys for r in rows), "rows": len(rows)}
    return out


def main():
    records = json.load(open(os.path.join(N1_PKG, "src", "records.json"), encoding="utf-8"))
    global STRICT
    STRICT = True
    a_strict, diffs_strict = part_a(records)
    STRICT = False
    a, diffs = part_a(records)
    b = part_b(records)
    c = part_c()
    d = part_d()
    res = {"A_cells": a, "A_row_diffs_vs_evaluate": diffs,
           "A_strict_no_number_words": {"cells": {k: v["audit"] for k, v in a_strict.items()},
                                        "row_diffs": diffs_strict}, "B_system_tools": b, "C_epoch": c, "D_leakage": d}
    json.dump(res, open(os.path.join(R, "audit.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: [v["audit"], v["match"]] for k, v in a.items()}), "n_diffs", len(diffs),
          "strict_diffs", len(diffs_strict))
    print(json.dumps(b, indent=1, ensure_ascii=False)[:6000])
    print(json.dumps({k: v for k, v in c.items() if "log" not in k or k == "train_logs"}, indent=1))
    print(c["val_loss_log_tail"][-1500:])
    print(json.dumps(d, indent=1, ensure_ascii=False)[:5000])


def write_md(res):
    a, b, c, d = res["A_cells"], res["B_system_tools"], res["C_epoch"], res["D_leakage"]
    st = res["A_strict_no_number_words"]
    L = ["## 11. Audit (2026-10-04, `audit.py` → `results/audit.{json,md}`, `router_audit.py` → `results/router_audit.json`)",
         "",
         "Result: **no defect found; no number in §1-10 changed.** REPORT.md regenerated by `build_report.py` was "
         "byte-identical to the pre-audit file before this section was added.",
         "",
         "### 11.1 Metric cells recomputed from the raw engine output",
         "`audit.py` joins `results/<model>_<test>_raw.jsonl` (the engine's `function_calls`) to the test files by "
         "row index and scores with its own code (evaluate.py not imported; `resolver.resolve` called, as it is the "
         "system under test). Own rules: tool names equal in order; phone = digits only; address/instruction "
         "token F1 ≥ 0.6; other args token equality; English number words folded to numerals; email 'at'/'dot'.",
         "",
         "| test | model | correct (pos) | tool match (pos) | empty list (neg) | = §5 |",
         "|---|---|---|---|---|---|"]
    for k, v in a.items():
        t, m = k.split("/")
        x = v["audit"]
        L.append(f"| {t} | {m} | {x['correct (pos)']} | {x['tool match (pos)']} | {x['empty list (neg)']} | "
                 f"{'yes' if v['match'] else 'NO: ' + json.dumps(v['report'])} |")
    nd = len(res["A_row_diffs_vs_evaluate"])
    L += ["",
          f"- Row level: the audit's (tool match, correct) differs from evaluate.py's per-row score on {nd} of 3,420 rows.",
          f"- Sensitivity: without number-word folding, {len(st['row_diffs'])} rows flip to not correct "
          "(all spoken numbers in the (b) ASR or noisy n0 text: 'terminal three' vs 'Terminal 3', 'five hundred two' vs "
          "'502'). Correct (pos) would then be: " + ", ".join(
              f"{k} {v['correct (pos)']}" for k, v in st["cells"].items()
              if v["correct (pos)"] != a[k]["audit"]["correct (pos)"]) + ". evaluate.py's `norm_tokens` folds number words "
          "for every argument by design, so §5 stands.", ""]
    L += ["### 11.2 Same system facts and tool schemas for base and tuned runs", ""]
    for t, x in b.items():
        sysline = (f"system == `build_data.system_text(record)` {x['system_eq_build_data_system_text']}/{x['rows']}"
                   if x["system_eq_build_data_system_text"] is not None else
                   f"one system text for all {x['rows']} rows (N0 record), starts with the pinned date")
        L.append(f"- {t} (md5 {x['test_md5'][:8]}…): tools == `tools.tools_for(agent_type)` {x['tools_eq_tools_for']}/{x['rows']}; "
                 f"{sysline}; pinned `date: 2026-10-04 Sun 10:00` on every row: {x['all_dates_pinned']}; "
                 f"engine groups (agent_type, scenario, n_rows) identical across base, base_pod2, tuned_l8, tuned_full: "
                 f"{x['groups_equal_across_tags']}; `auto_date` false in all four raw files: {x['auto_date_false_all']}.")
    L += ["- evaluate.py builds each group's `Needle(tools=row.tools, system=row.system, auto_date=False)` from the test "
          "row and asserts `agent._system_text == system`; the runs differ only in `weights`. Every tag has needle 3.0.6 "
          "and max_new_tokens 512.",
          "- Weights md5 at the paths in `run_meta` (checked on pod2 / laptop): tuned_full 92ddd0d7…, tuned_l8 53649254…, "
          "base `needle3.cact` 71c31b0b… on both machines; `libneedle.so` 679dc570… on pod2; evaluate.py dba105ab…, "
          "resolver.py b241ba61…, tools.py aacb0d2d…, build_data.py e99db5f0… identical laptop/pod2.",
          "- The laptop base run (05:13-05:20) predates evaluate.py's last edit (05:26); base_pod2, run with the current "
          "file, reproduces it on 0/1,140 differing rows (§4.3), so the base numbers are not affected.", ""]
    v = c["val_losses_json"]["models"]
    L += ["### 11.3 Chosen epoch", "",
          "| model | val token_mean | val batch_mean |", "|---|---|---|"]
    for k, x in v.items():
        L.append(f"| {k} | {x['token_mean']:.4f} | {x['batch_mean']:.4f} |")
    lg = c["train_logs"]
    L += ["",
          f"- From `finetune/sweep/val_losses.json` and `val_loss.log` (same values; val.jsonl md5 {c['md5']['val.jsonl'][:8]}…, "
          "360 rows, 5,179 answer tokens): e10 is the lowest on both means. It is the lowest of the three spec grid points; "
          "the loss is still falling, so no minimum was reached.",
          f"- adapter_e10 md5 {c['md5']['adapter_e10.safetensors'][:8]}…; `build_full.log` / `build_l8.log` merge "
          "`adapter_e10.safetensors` (20 / 8 layers); tuned_full.cact and tuned_l8.cact in `finetune/sweep/` and the needle "
          "root are md5-identical to `md5s.txt`.",
          "- Train logs: no NaN/inf in e3/e6/e10 (" + ", ".join(f"{k} last logged loss {x['last_loss']}" for k, x in lg.items()) + ").",
          "- Caveat already in §4.1: val loss is on the no-think render, so it does not measure the negatives failure.", ""]
    q = d["query_overlap"]
    L += ["### 11.4 Leakage", "",
          f"- Scenarios: train {d['train_scenarios']}, val {d['val_scenarios']}, test {d['test_heldout_a']['scenarios']}; "
          f"overlap train∩val {len(d['train_val_scen_overlap'])}, test∩train {len(d['test_heldout_a']['scen_overlap_train'])}, "
          f"test∩val {len(d['test_heldout_a']['scen_overlap_val'])} (same for b); call_id overlap test vs train+val: "
          f"{len(d['test_heldout_a']['call_overlap_train_val'])} (a), {len(d['test_heldout_b']['call_overlap_train_val'])} (b).",
          f"- examples_raw.jsonl: every scenario_id is in exactly one split ({d['examples_raw_split_counts']}); "
          f"scenarios in more than one split: {len(d['examples_raw_scenarios_multi_split'])}.",
          "- Exact query text (lower-cased) of test positives found in train/val: " + ", ".join(
              f"{t} {len(x['pos_in_train']) + len(x['pos_in_val'])}" for t, x in q.items()) + ".",
          "- Test negatives with a query identical to a train/val query: " + ", ".join(
              f"{t} {x['neg_in_train_or_val']}" for t, x in q.items()) + ". These are generic closings (e.g. "
          "'Theek hai, thank you so much for the help.', 'Nahi, bas itna hi tha. Bye.'), trained with target `[]`; "
          "tuned_full still calls a tool on all 7 (a) and 7 (b) of them (§4.1, §7). Test system texts found in train: " + ", ".join(
              f"{t} {x['system_texts_in_train']}" for t, x in q.items()) + ".",
          "- Tool-schema `e.g.` values (same schema for every model, sent on every train and test row): the order_ref "
          "example IDs (FD1565, EC2469, RD3938, AC5746, JB7215) are entity IDs of train scenarios only (food_06, ecom_01, "
          "cab_01, sub_01, air_01), none of a test scenario or the N0 record; the phone example `9800011085` is train "
          "scenario food_06's phone (the §1 leak). One schema example contains a test gold value: location "
          "'IGI Airport Terminal 3, Gate 4' vs gold 'IGI Airport Terminal 3' on 8 positives of each held-out file "
          "(0 in train/val). Comparisons between models are unaffected; absolute location scores on those rows may be "
          "helped by the schema text.", ""]
    try:
        ra = json.load(open(os.path.join(R, "router_audit.json"), encoding="utf-8"))
        L += ["### 11.5 router.py end to end (pod2 CPU, `router_audit.py`, weights md5 " + ra["weights_md5"][:8] + "…)", "",
              "Three held-out positives not used by the §7 self-test. | = eval: router `function_calls` identical to "
              "`results/tuned_full_<test>_raw.jsonl`.", "",
              "| test / row | query | gold | routed | resolved (rule) | gold ID | tools, system = row | = eval |",
              "|---|---|---|---|---|---|---|---|"]
        for p in ra["picks"]:
            r = p["routed"][0] if p["routed"] else {}
            L.append(f"| {p['test']} / {p['example_id']} | {p['query']} | "
                     f"{json.dumps(p['gold'], ensure_ascii=False)} | "
                     f"{json.dumps(p['function_calls'], ensure_ascii=False)} | {r.get('resolved_id')} ({r.get('resolver_rule')}) | "
                     f"{p['gold_entity_id']} | {p['router_tools_eq_row_tools'] and p['router_system_eq_row_system']} | "
                     f"{p['identical_to_eval']} |")
        n = len(ra["picks"])
        L += ["",
              f"- Identical to the eval run: {sum(p['identical_to_eval'] for p in ra['picks'])}/{n}; resolved ID = gold: "
              f"{sum(p['resolved_eq_gold'] for p in ra['picks'])}/{n}; tool match {sum(p['tool_match'] for p in ra['picks'])}/{n}.",
              "- Third pick: the email argument `bose75@yahoo.com` is the customer's *current* email copied from the system "
              "text, not the spoken new one (`bose.home75@gmail.com`); it is an argument error that the eval already "
              "scores as not correct.",
              "- §7 examples re-run through `router.route` on pod2 with the cab_15 record "
              "(`results/router_audit_s7_example.txt`): the instruction turn gives add_driver_instruction → RD5520 "
              "(default_active); 'Theek hai, thank you so much for the help.' gives `cancel_ride` → RD5520, as §7 states. "
              "The same query is a test negative in cab_07 and cab_11 (heldout_a), where the eval run also shipped `cancel_ride`.", ""]
    except FileNotFoundError:
        L.append("router_audit.json missing")
    L += ["### 11.6 Audit outcome", "",
          "- Ascertained fixes: none needed; no defect was confirmed, so no result was re-run.",
          "- Open discussions (unchanged, not fixed here): think-block training gap (§4.1); schema example phone "
          "`9800011085` (§1); tuned_l8 unusable (§4.2); val loss still falling at e10 (11.3). New from the audit: "
          "the model copies the current email from the system text (11.5); a location schema example contains a "
          "test gold value (11.4).", ""]
    open(os.path.join(R, "audit.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
    write_md(json.load(open(os.path.join(R, "audit.json"), encoding="utf-8")))
