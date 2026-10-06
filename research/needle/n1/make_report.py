"""Builds REPORT.md tables + REPORT_APPENDIX_failures.md from results/*.json (N1 spec §5-6).

Writes results/report_tables.md (pasted into REPORT.md by hand-written prose around it) and
REPORT_APPENDIX_failures.md (every failure verbatim, with the model's reasoning string).
"""
import json
import os
from collections import Counter

HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was this script's own dir
R = os.path.join(HERE, "results")
TESTS = ["test_n0", "test_heldout_a", "test_heldout_b"]
MODELS = [("base", "base"), ("tuned_l8", "tuned_l8"), ("tuned_full", "tuned_full")]
RUNAWAY_CHARS = 1000  # reasoning longer than this = the decode looped (tuned_l8 pattern)


def load(tag, test):
    p = os.path.join(R, f"{tag}_{test}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def c(f):
    return "-" if not f or not f.get("n") else f"{f['k']}/{f['n']} ({f['pct']:.1f}%)"


COLS = [("correct_positives", "correct (pos)"), ("tool_exact_match_positives", "tool match (pos)"),
        ("real_args_match_of_tool_matched", "real args (of tool ok)"),
        ("order_ref_match_gold_named_calls", "order_ref = gold (named)"),
        ("order_ref_omitted_agree_calls", "order_ref omitted (gold omits)"),
        ("args_match_strict_of_tool_matched", "args incl. order_ref (of tool ok)"),
        ("resolver_ok_of_tool_matched", "resolver (of tool ok)"),
        ("resolver_ok_of_positives", "resolver (all pos)"),
        ("empty_list_accuracy_negatives", "empty list (neg)")]


def runaway(d):
    return sum(1 for r in d["rows"] if len(r.get("reasoning") or "") > RUNAWAY_CHARS)


def main():
    L = []
    data = {(m, t): load(tag, t) for m, tag in MODELS for t in TESTS}
    # ---- main tables
    for t in TESTS:
        L += [f"### {t}", "", "| slice | model | rows | " + " | ".join(x[1] for x in COLS) + " |",
              "|---|---|---|" + "---|" * len(COLS)]
        d0 = data[("base", t)]
        slices = ["all"] + list(d0["metrics"]["by_rendering_language"].keys())
        for s in slices:
            for m, _ in MODELS:
                d = data[(m, t)]
                if d is None:
                    continue
                a = d["metrics"]["overall"] if s == "all" else d["metrics"]["by_rendering_language"].get(s)
                if a is None:
                    continue
                L.append(f"| {s} | {m} | {a['rows']} | " + " | ".join(c(a[x[0]]) for x in COLS) + " |")
        L.append("")
    # ---- shipped / suppressed view
    L += ["### Shipped vs suppressed (E9)", "",
          "| test | model | pos shipped | pos withheld only | pos no call | correct (pos) if withheld had shipped | "
          "neg: empty | neg: suppressed-only (counted as empty) | neg: empty if suppressed had shipped | rows with runaway reasoning (>1000 chars) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in TESTS:
        for m, _ in MODELS:
            d = data[(m, t)]
            if d is None:
                continue
            a = d["metrics"]["overall"]
            e, so = a["empty_list_accuracy_negatives"], a["negatives_with_suppressed_calls_only"]
            L.append(f"| {t} | {m} | {c(a['positives_shipped'])} | {c(a['positives_withheld_only'])} | "
                     f"{c(a['positives_no_call_at_all'])} | {c(a['correct_positives_if_suppressed_shipped'])} | "
                     f"{e['k']}/{e['n']} | {so} | {e['k'] - so}/{e['n']} | {runaway(d)} |")
    L.append("")
    # ---- multi-call
    L += ["### Multi-call order (test_n0, 8 gold multi-call rows)", "", "| model | classes |", "|---|---|"]
    for m, _ in MODELS:
        d = data[(m, "test_n0")]
        if d:
            L.append(f"| {m} | " + ", ".join(f"{k}: {v}" for k, v in sorted(d["multi_call_order"].items())) + " |")
    L.append("")
    # ---- per gold tool, held-out a/b
    L += ["### Tool match per gold tool (positives)", "", "| test | gold tool | n | base | tuned_l8 | tuned_full |",
          "|---|---|---|---|---|---|"]
    for t in TESTS:
        d0 = data[("base", t)]
        for tool, a in d0["metrics"]["by_gold_tool"].items():
            cells = []
            for m, _ in MODELS:
                d = data[(m, t)]
                b = d["metrics"]["by_gold_tool"].get(tool) if d else None
                cells.append(c(b["tool_exact_match_positives"]) if b else "-")
            L.append(f"| {t} | {tool} | {a['positives']} | " + " | ".join(cells) + " |")
    L.append("")
    # ---- per-argument
    L += ["### Per-argument (aligned calls of tool-matched rows; ok_rule / n)", "",
          "| test | argument | base | tuned_l8 | tuned_full |", "|---|---|---|---|---|"]
    for t in TESTS:
        keys = []
        for m, _ in MODELS:
            d = data[(m, t)]
            for k in (d or {}).get("per_argument", {}):
                if k not in keys:
                    keys.append(k)
        for k in keys:
            cells = []
            for m, _ in MODELS:
                v = (data[(m, t)] or {}).get("per_argument", {}).get(k)
                cells.append(f"{v.get('ok_rule', 0)}/{v.get('n', 0)}" if v else "-")
            L.append(f"| {t} | {k} | " + " | ".join(cells) + " |")
    L.append("")
    # ---- failure groups
    L += ["### Failure groups", "", "| test | model | failures | groups |", "|---|---|---|---|"]
    for t in TESTS:
        for m, _ in MODELS:
            d = data[(m, t)]
            if d:
                L.append(f"| {t} | {m} | {len(d['failures'])} | " + ", ".join(
                    f"{k} {v}" for k, v in sorted(d["failure_groups"].items(), key=lambda x: -x[1])) + " |")
    L.append("")
    # ---- confusion matrices
    L += ["### Confusion matrices (row level; rows = gold, columns = predicted function_calls; 'none' = [])", ""]
    for t in TESTS:
        for m, _ in MODELS:
            d = data[(m, t)]
            if not d:
                continue
            cm = d["confusion_matrix"]
            preds = sorted({p for x in cm.values() for p in x})
            L += [f"#### {t} / {m}", "", "| gold \\ pred | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
            for g, x in cm.items():
                L.append(f"| {g} | " + " | ".join(str(x.get(p, "")) for p in preds) + " |")
            L.append("")
    open(os.path.join(R, "report_tables.md"), "w", encoding="utf-8").write("\n".join(L))

    # ---- appendix: every failure verbatim
    A = ["# N1 appendix: every failure, verbatim", "",
         "Generated by make_report.py from `results/<model>_<test>.json` `failures`. One entry per non-correct row: "
         "query as sent, gold answers, the model's function_calls and suppressed_calls, and its reasoning string. "
         "base = laptop run (results/base_*), tuned_* = pod2 runs.", ""]
    for m, tag in MODELS:
        for t in TESTS:
            d = data[(m, t)]
            if not d:
                continue
            A += [f"## {m} / {t}: {len(d['failures'])} failures", ""]
            for f in d["failures"]:
                A.append(f"- **{f['row_id']}** [{f['language']}, {f['rendering']}] ({f['group']}): query `{f['query']}`; "
                         f"gold `{json.dumps(f['gold'], ensure_ascii=False)}`; function_calls "
                         f"`{json.dumps(f['function_calls'], ensure_ascii=False)}`; suppressed "
                         f"`{json.dumps(f['suppressed_calls'], ensure_ascii=False)}`; reasoning `{f['reasoning']}`")
            A.append("")
    open(os.path.join(HERE, "REPORT_APPENDIX_failures.md"), "w", encoding="utf-8").write("\n".join(A))

    # ---- reproduction + cross-machine checks
    out = {}
    for a_tag, b_tag, tests in [("base", "base_pod2", TESTS), ("laptop_tuned_full", "tuned_full", ["test_n0"]),
                                ("laptop_tuned_l8", "tuned_l8", ["test_n0"]), ("laptop_base", "base", ["test_n0"])]:
        for t in tests:
            x, y = load(a_tag, t), load(b_tag, t)
            if not (x and y):
                continue
            diff = [r1["row_id"] for r1, r2 in zip(x["rows"], y["rows"]) if r1["function_calls"] != r2["function_calls"]]
            out[f"{a_tag} vs {b_tag} / {t}"] = {"rows": len(x["rows"]), "differing_function_calls": len(diff),
                                                "ids": diff[:20],
                                                "correct": [x["metrics"]["overall"]["correct_all_rows"]["k"],
                                                            y["metrics"]["overall"]["correct_all_rows"]["k"]]}
    # latency on laptop
    lat = {}
    for tag in ("laptop_base", "laptop_tuned_l8", "laptop_tuned_full", "base"):
        d = load(tag, "test_n0")
        if d:
            a, e = d["metrics"]["overall"], d["latency_extra"]["excluding_first_row_of_each_group"]
            rows = d["rows"]
            nr = [r["latency_ms"] for r in rows if len(r.get("reasoning") or "") <= RUNAWAY_CHARS]
            nr.sort()
            lat[tag] = {"n": a["latency_n"], "median": a["latency_ms_median"], "p95": a["latency_ms_p95"],
                        "max": a["latency_ms_max"], "excl_first": e, "runaway_rows": runaway(d),
                        "median_excl_runaway": round(nr[len(nr) // 2], 1) if nr else None,
                        "loadavg_start": d["run_meta"]["loadavg_start"], "started": d["run_meta"]["started"],
                        "machine": d["run_meta"]["machine"], "cpu_count": d["run_meta"]["cpu_count"],
                        "decode_tps_median": sorted((r.get("raw_response") or {}).get("decode_tps") or 0 for r in rows)[len(rows) // 2]
                        if rows and "raw_response" in rows[0] else None}
    out["laptop_latency"] = lat
    json.dump(out, open(os.path.join(R, "report_checks.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
