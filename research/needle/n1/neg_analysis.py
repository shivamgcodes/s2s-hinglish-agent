"""Negatives analysis (J12): which calls each model emits on negative rows (gold = []), by tool and by negative type.
Reads results/<model>_<test>.json + examples_raw.jsonl (neg_kind, in_write_window, call's own write tool).
Writes results/neg_analysis.json and results/neg_analysis.md (included in REPORT.md by build_report.py)."""
import json, os, collections
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was this script's own dir
R = os.path.join(HERE, "results")
MODELS = ["base", "tuned_l8", "tuned_full"]
TESTS = ["test_n0", "test_heldout_a", "test_heldout_b"]

ex, call_tools = {}, collections.defaultdict(set)
for line in open(os.path.join(HERE, "examples_raw.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["type"] == "negative":
        ex[r["example_id"]] = r
    else:
        for g in r.get("answers") or []:
            call_tools[r["call_id"]].add(g["name"])


def names(calls):
    return [c["name"] for c in calls or []]


def analyse(model, test):
    d = json.load(open(os.path.join(R, f"{model}_{test}.json"), encoding="utf-8"))
    neg = [r for r in d["rows"] if not r["positive"]]
    out = {"n": len(neg), "rows_shipped": 0, "rows_suppressed_only": 0, "rows_empty_no_suppressed": 0,
           "calls_shipped": 0, "calls_suppressed": 0, "multi_call_rows_shipped": 0,
           "tool_shipped": collections.Counter(), "tool_suppressed": collections.Counter(),
           "shipped_tool_is_calls_own_write_tool": 0, "by_type": {}}
    groups = collections.defaultdict(lambda: [0, 0, 0])   # key -> [n, rows_shipped, rows_suppressed_only]
    for r in neg:
        s, u = names(r["function_calls"]), names(r["suppressed_calls"])
        out["calls_shipped"] += len(s); out["calls_suppressed"] += len(u)
        out["tool_shipped"].update(s); out["tool_suppressed"].update(u)
        if s:
            out["rows_shipped"] += 1
            out["multi_call_rows_shipped"] += len(s) > 1
        elif u:
            out["rows_suppressed_only"] += 1
        else:
            out["rows_empty_no_suppressed"] += 1
        if test == "test_n0":
            keys = [("language", r["language"])]
        else:
            m = ex[r["row_id"]]
            if s and set(s) & call_tools.get(m["call_id"], set()):
                out["shipped_tool_is_calls_own_write_tool"] += 1
            keys = [("neg_kind", m["neg_kind"]), ("in_write_window", str(m["in_write_window"])),
                    ("neg_kind x window", f"{m['neg_kind']} / {'in' if m['in_write_window'] else 'out'}"),
                    ("language", r["language"]), ("agent_type", r["agent_type"]),
                    ("variant", m["variant"]), ("is_test_call", str(m["is_test_call"]))]
        for k in keys:
            g = groups[k]; g[0] += 1; g[1] += bool(s); g[2] += (not s and bool(u))
    for (dim, val), (n, sh, so) in sorted(groups.items()):
        out["by_type"].setdefault(dim, {})[val] = {"n": n, "rows_shipped": sh, "rows_suppressed_only": so}
    out["tool_shipped"] = dict(out["tool_shipped"].most_common())
    out["tool_suppressed"] = dict(out["tool_suppressed"].most_common())
    return out


def main():
    res = {t: {m: analyse(m, t) for m in MODELS} for t in TESTS}
    json.dump(res, open(os.path.join(R, "neg_analysis.json"), "w"), indent=1, ensure_ascii=False)
    L = ["### 4.4 Negatives analysis: calls emitted on negative rows (`neg_analysis.py` → `results/neg_analysis.json`)", "",
         "Rows whose gold is `[]`. 'shipped' = `function_calls` non-empty; 'suppressed only' = `function_calls` empty but "
         "`suppressed_calls` non-empty (the engine withheld the call); 'empty' = neither. base = laptop run (E15), "
         "tuned_l8 / tuned_full = pod2 runs. Negative type for held-out rows is `neg_kind` from `examples_raw.jsonl` "
         "(greeting_or_first = customer turn ≤ 1; after_write = after the call's confirm turn; other) and "
         "`in_write_window`; test_n0's 12 negatives are N0 `kind=read` rows, split by language only.", "",
         "| test | model | negatives | rows shipped | rows suppressed only | rows empty | calls shipped | calls suppressed | multi-call rows (shipped) | shipped tool = the call's own write tool |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t in TESTS:
        for m in MODELS:
            a = res[t][m]
            own = "n/a" if t == "test_n0" else f"{a['shipped_tool_is_calls_own_write_tool']}/{a['rows_shipped']}"
            L.append(f"| {t} | {m} | {a['n']} | {a['rows_shipped']} | {a['rows_suppressed_only']} | "
                     f"{a['rows_empty_no_suppressed']} | {a['calls_shipped']} | {a['calls_suppressed']} | "
                     f"{a['multi_call_rows_shipped']} | {own} |")
    tf = {t: res[t]["tuned_full"] for t in TESTS}
    L += ["", "Tuned rows can carry suppressed calls although confidence is None: the engine still withholds a call it "
          "cannot ground (GUIDE_NOTES: a required field with no span goes to `suppressed_calls`). tuned_full negatives "
          "with suppressed-only calls: " + ", ".join(f"{t} {tf[t]['rows_suppressed_only']}" for t in TESTS) +
          ". That is why tuned_full's empty list (" + ", ".join(f"{tf[t]['n'] - tf[t]['rows_shipped']}/{tf[t]['n']}" for t in TESTS) +
          ") is one row higher than its E9 'if suppressed had shipped' figure where that count is 1.", ""]
    for t in TESTS:
        L += [f"#### {t}: calls on negatives per tool (shipped; suppressed in brackets)", ""]
        tools = sorted(set().union(*[set(res[t][m]["tool_shipped"]) | set(res[t][m]["tool_suppressed"]) for m in MODELS]),
                       key=lambda x: -sum(res[t][m]["tool_shipped"].get(x, 0) for m in MODELS))
        L += ["| tool | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for tool in tools:
            L.append(f"| {tool} | " + " | ".join(
                f"{res[t][m]['tool_shipped'].get(tool, 0)} ({res[t][m]['tool_suppressed'].get(tool, 0)})" for m in MODELS) + " |")
        L += ["", f"#### {t}: negatives with a shipped call, by negative type (rows shipped / n; suppressed-only rows in brackets)", "",
              "| dimension | value | n | " + " | ".join(MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
        for dim, vals in res[t]["base"]["by_type"].items():
            for val, b in vals.items():
                L.append(f"| {dim} | {val} | {b['n']} | " + " | ".join(
                    f"{res[t][m]['by_type'][dim][val]['rows_shipped']} ({res[t][m]['by_type'][dim][val]['rows_suppressed_only']})"
                    for m in MODELS) + " |")
        L.append("")
    open(os.path.join(R, "neg_analysis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
