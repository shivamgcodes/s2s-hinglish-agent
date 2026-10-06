"""N2 eval: N1 tuned_full (N1 schema, N1 router) engine-decoded on the N1-format test rows (5 old agent types).

Per row (same path as needle_n1/router.py route()):
  needle.Needle(tools=row.tools (N1 tools.tools_for), system=row.system (N1 build_data.system_text), weights=N1
  tuned_full.cact, auto_date=False) -> reset() -> complete(query, 512) -> N1 router.resolve_calls (N1 resolver on
  order_ref) -> score_map.n1_to_server -> score_map.score(gold = meta.gold_canonical).
The record passed to the N1 resolver is the calls.jsonl record unchanged (V4 records carry primary_id /
secondary_id / distractors / facts, which N1 resolver.entities_from_record reads).
Usage: python eval_n1.py --weights /root/n2/needle_n1/tuned_full.cact --rows /root/n2/data/rows/n1_test_clean --out X.json
"""
import argparse, json, os, sys, time

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[4]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
sys.path.insert(1, str(REPO / "research" / "needle" / "n2" / "schema"))  # score_map.py
import score_map  # noqa: E402  (puts N1 dir on sys.path via n1path)
import n1path  # noqa: E402
sys.path.insert(0, n1path.N1_DIR)
import router as n1router  # noqa: E402  (N1 router.py)

CALLS = {}
for l in open(f"{N2_ROOT}/V4/calls.jsonl"):
    c = json.loads(l)
    CALLS[c["call_id"]] = c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="file with example_ids to run (in that order)")
    a = ap.parse_args()
    import needle
    rows = [json.loads(l) for l in open(a.rows + ".jsonl")]
    metas = [json.loads(l) for l in open(a.rows + "_meta.jsonl")]
    idx = list(range(len(rows)))
    if a.ids:
        pos = {m["example_id"]: i for i, m in enumerate(metas)}
        idx = [pos[x.strip()] for x in open(a.ids) if x.strip()]
    else:
        idx.sort(key=lambda i: (rows[i]["system"], metas[i]["agent_type"]))
    if a.limit:
        idx = idx[:a.limit]
    out, agents, t0 = [], {}, time.time()
    for i in idx:
        row, meta = rows[i], metas[i]
        call = CALLS[meta["call_id"]]
        rec, at = call["record"], meta["agent_type"]
        assert row["system"] == n1router.system_for(rec), "system mismatch vs N1 router"
        key = (row["system"], json.dumps(row["tools"]))
        if key not in agents:
            for ag in agents.values():
                ag.close()
            agents = {key: needle.Needle(tools=row["tools"], system=row["system"], weights=a.weights,
                                         auto_date=False)}
        ag = agents[key]
        ag.reset()
        t = time.perf_counter()
        resp = ag.complete(row["query"], n1router.MAX_NEW_TOKENS) or {}
        ms = 1000 * (time.perf_counter() - t)
        fc = resp.get("function_calls") or []
        resolved = n1router.resolve_calls(fc, rec, at)
        pred = score_map.n1_to_server(resolved, rec, at)
        gold = [meta["gold_canonical"]]
        s = score_map.score(gold, pred, at)
        silent = bool(pred) and not s["correct"] and not any(p["ask"] for p in pred)
        bad_args = sorted({k for r in s["per_arg"] for k, v in r.items() if not v["ok"]})
        out.append({"i": i, "example_id": meta["example_id"], "variant": meta.get("variant"), "agent_type": at,
                    "tool": meta["tool"], "query": row["query"], "gold": gold, "function_calls": fc,
                    "suppressed_calls": resp.get("suppressed_calls"), "reasoning": resp.get("reasoning"),
                    "resolved": resolved, "server": pred,
                    "tool_match": s["tool_match"], "args_match": s["args_match"], "correct": s["correct"],
                    "ask": any(p["ask"] for p in pred), "silent_wrong": silent,
                    "silent_wrong_phone": silent and "phone" in bad_args,
                    "silent_wrong_email": silent and "email" in bad_args,
                    "bad_args": bad_args, "group": s.get("group"), "per_arg": s["per_arg"], "latency_ms": round(ms)})
        print(json.dumps({k: out[-1][k] for k in ("example_id", "tool_match", "correct", "group", "latency_ms")}),
              flush=True)
    for ag in agents.values():
        ag.close()
    out.sort(key=lambda r: r["i"])
    json.dump({"summary": {"weights": a.weights, "rows": a.rows, "n": len(out), "wall_s": round(time.time() - t0, 1),
                           "correct": sum(r["correct"] for r in out), "tool_match": sum(r["tool_match"] for r in out)},
               "rows": out}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
