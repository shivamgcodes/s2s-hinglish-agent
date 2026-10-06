"""Audit: router.py end to end on 3 held-out positives NOT used by router._selftest (rows 15, 85 of a; 15 of b,
results/router_selftest.txt). Picks: first heldout_a positive with a named order_ref, first heldout_b positive with a named
order_ref, first heldout_b positive whose gold tool is not in the earlier two picks.
Per pick: router's tools/system == the test row's; router function_calls == evaluate.py's recorded ones;
routed resolved_id == gold entity.   -> results/router_audit.json

    python router_audit.py [--weights PATH]      (needle venv, CPU)
"""
import argparse
import hashlib
import json
import os
import sys

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))  # build_data, resolver, router, tools, src/records.json
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (data, results/); was this script's own dir
sys.path.insert(0, N1_PKG)
import router  # noqa: E402
import tools  # noqa: E402

SELFTEST_ROWS = {("test_heldout_a", 15), ("test_heldout_a", 85), ("test_heldout_b", 15)}  # results/router_selftest.txt


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=router.DEFAULT_WEIGHTS)
    a = ap.parse_args()
    recs = json.load(open(os.path.join(N1_PKG, "src", "records.json"), encoding="utf-8"))
    data = {t: (jl(os.path.join(HERE, t + ".jsonl")), jl(os.path.join(HERE, t + "_meta.jsonl")),
                jl(os.path.join(HERE, "results", f"tuned_full_{t}_raw.jsonl"))[1:])
            for t in ("test_heldout_a", "test_heldout_b")}
    picks, used_tools = [], set()

    def first(t, cond):
        rows, meta, _ = data[t]
        for i, (r, m) in enumerate(zip(rows, meta)):
            if m["type"] != "negative" and (t, i) not in SELFTEST_ROWS and (t, i) not in picks and cond(r, m):
                return (t, i)

    named = lambda r, m: any((c["arguments"] or {}).get("order_ref") for c in r["answers"])  # noqa: E731
    for t in ("test_heldout_a", "test_heldout_b"):
        p = first(t, named)
        picks.append(p)
        used_tools.add(data[t][0][p[1]]["answers"][0]["name"])
    picks.append(first("test_heldout_b", lambda r, m: r["answers"][0]["name"] not in used_tools))

    out = {"weights": a.weights, "weights_md5": hashlib.md5(open(a.weights, "rb").read()).hexdigest(), "picks": []}
    for t, i in picks:
        rows, meta, raw = data[t]
        r, m = rows[i], meta[i]
        rec = recs[m["scenario_id"]]
        system = router.system_for(rec)
        resp = router.route_raw(m["agent_type"], rec, r["query"], weights=a.weights)
        routed = router.resolve_calls(resp.get("function_calls"), rec, m["agent_type"])
        ev = raw[i]["response"]["function_calls"]
        out["picks"].append({
            "test": t, "row": i, "example_id": m["example_id"], "query": r["query"], "gold": r["answers"],
            "gold_entity_id": m["gold_entity_id"],
            "router_tools_eq_row_tools": tools.tools_for(m["agent_type"]) == r["tools"],
            "router_system_eq_row_system": system == r["system"],
            "function_calls": resp.get("function_calls"), "eval_function_calls": ev,
            "identical_to_eval": resp.get("function_calls") == ev,
            "routed": routed, "reasoning": resp.get("reasoning"),
            "tool_match": [c["name"] for c in r["answers"]] == [c["name"] for c in routed],
            "resolved_eq_gold": all(c["resolved_id"] == m["gold_entity_id"] for c in routed) and bool(routed),
        })
    router.close()
    json.dump(out, open(os.path.join(HERE, "results", "router_audit.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
