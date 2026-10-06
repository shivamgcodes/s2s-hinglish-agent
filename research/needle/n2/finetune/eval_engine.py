"""N2: engine-decoded scoring of Needle .cact weights on v2 rows (val selection; CPU, parallel shard processes).

Per row: needle.Needle(tools=row.tools, system=row.system, weights, auto_date=False) -> reset() -> complete(query, 512)
(router_v2.MAX_NEW_TOKENS) -> router_v2.resolve_calls(function_calls, record, agent_type) -> score_map.v2_to_server
-> score_map.score(gold = meta.gold_canonical). Same path as router_v2.route_raw with convert=False (rows already carry
prepare_transcript(raw)).
Usage: python eval_engine.py --weights X.cact --rows /root/n2/data/rows/val --out X_val.json [--procs 24]
Reports: tool_match, correct, ask, silent_wrong (a call shipped without ASK that is not correct), silent wrong phone /
email, missed (no call), suppressed (engine moved calls to suppressed), think stats; split by variant (clean/opus).
"""
import argparse, json, os, subprocess, sys, time

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[4]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
sys.path.insert(1, str(REPO / "research" / "needle" / "n2" / "schema"))  # score_map.py
import router_v2, score_map, targets  # noqa: E402

CALLS = None


def _calls():
    global CALLS
    if CALLS is None:
        CALLS = {}
        for l in open(f"{N2_ROOT}/V4/calls.jsonl"):
            c = json.loads(l)
            CALLS[c["call_id"]] = c
    return CALLS


def work(args):
    weights, items, part_path = args
    import needle
    out = []
    pf = open(part_path, "w")
    agents = {}
    for i, row, meta in items:
        call = _calls()[meta["call_id"]]
        rec, at = call["record"], meta["agent_type"]
        assert row["system"] == router_v2.system_for(rec), "system mismatch vs router_v2"
        key = (row["system"], json.dumps(row["tools"]))
        if key not in agents:
            for a in agents.values():
                a.close()
            agents = {key: needle.Needle(tools=row["tools"], system=row["system"], weights=weights, auto_date=False)}
        ag = agents[key]
        ag.reset()
        t = time.perf_counter()
        resp = ag.complete(row["query"], router_v2.MAX_NEW_TOKENS) or {}
        ms = 1000 * (time.perf_counter() - t)
        fc = resp.get("function_calls") or []
        resolved = router_v2.resolve_calls(fc, rec, at)
        pred = score_map.v2_to_server(resolved)
        gold = [meta["gold_canonical"]]
        s = score_map.score(gold, pred, at)
        silent = bool(pred) and not s["correct"] and not any(p["ask"] for p in pred)
        bad_args = sorted({k for r in s["per_arg"] for k, v in r.items() if not v["ok"]})
        out.append({"i": i, "example_id": meta["example_id"], "variant": meta.get("variant"), "agent_type": at,
                    "tool": meta["tool"], "query": row["query"], "gold": gold, "function_calls": fc,
                    "suppressed_calls": resp.get("suppressed_calls"), "reasoning": resp.get("reasoning"),
                    "server": pred, "ask_reasons": [r["ask_reasons"] for r in resolved],
                    "tool_match": s["tool_match"], "args_match": s["args_match"], "correct": s["correct"],
                    "ask": any(p["ask"] for p in pred), "silent_wrong": silent,
                    "silent_wrong_phone": silent and "phone" in bad_args,
                    "silent_wrong_email": silent and "email" in bad_args,
                    "bad_args": bad_args, "group": s.get("group"), "per_arg": s["per_arg"], "latency_ms": round(ms)})
        pf.write(json.dumps(out[-1], ensure_ascii=False) + "\n"); pf.flush()
    for a in agents.values():
        a.close()
    return out


def summarise(rs):
    n = len(rs)
    f = lambda k: sum(1 for r in rs if r[k])
    d = {"n": n, "tool_match": f("tool_match"), "correct": f("correct"), "ask": f("ask"),
         "silent_wrong": f("silent_wrong"), "silent_wrong_phone": f("silent_wrong_phone"),
         "silent_wrong_email": f("silent_wrong_email"),
         "missed": sum(1 for r in rs if not r["function_calls"]),
         "suppressed": sum(1 for r in rs if r["suppressed_calls"]),
         "multi_call": sum(1 for r in rs if len(r["function_calls"]) > 1)}
    groups = {}
    for r in rs:
        if not r["correct"]:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
    d["fail_groups"] = dict(sorted(groups.items(), key=lambda x: -x[1]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", required=True, help="path prefix: <rows>.jsonl and <rows>_meta.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cpu0", type=int, default=8, help="first CPU for shard 0 (CPUs below are left for training)")
    ap.add_argument("--cpus", type=int, default=0, help="CPUs per shard")
    ap.add_argument("--shard", type=int, default=None, help="internal: run one shard in this process")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows + ".jsonl")]
    metas = [json.loads(l) for l in open(a.rows + "_meta.jsonl")]
    assert len(rows) == len(metas)
    # group rows by (system, tools) so each worker builds few agents
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["system"], metas[i]["agent_type"]))
    chunks = [[] for _ in range(a.procs)]
    k = -1; last = None
    for i in order:
        key = rows[i]["system"]
        if key != last:
            k = (k + 1) % a.procs; last = key
        chunks[k].append((i, rows[i], metas[i]))
    parts = [a.out + f".part{k}.jsonl" for k in range(a.procs)]
    if a.shard is not None:  # child: one shard, rows streamed to its part file
        work((a.weights, chunks[a.shard], parts[a.shard]))
        return
    t0 = time.time()
    # independent processes (multiprocessing.Pool hung once with the engine's worker subprocesses)
    # Pod CPU note (runpod2): the container has a CFS quota of 10.2 CPUs (cpu.cfs_quota_us 1020000) although 64 are
    # visible, and the engine starts 12 threads pinned by index within the allowed CPU set. Many parallel engines (or an
    # engine taskset to few CPUs) thrash: 25-90 s/row instead of ~0.7 s. Default: ONE shard, unpinned. --cpus > 0
    # pins shard k to its own CPU block (only useful on a host without a quota).
    pin = (lambda k: ["taskset", "-c", f"{a.cpu0 + a.cpus * k}-{a.cpu0 + a.cpus * k + a.cpus - 1}"]) if a.cpus > 0 \
        else (lambda k: [])
    procs = [subprocess.Popen(pin(k) + [
                               sys.executable, os.path.abspath(__file__), "--weights", a.weights, "--rows", a.rows,
                               "--out", a.out, "--procs", str(a.procs), "--shard", str(k)],
                              stderr=open(parts[k] + ".err", "w"))
             for k in range(a.procs) if chunks[k]]
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), f"shard failures: {rcs}"
    res = [json.loads(l) for k in range(a.procs) if chunks[k] for l in open(parts[k])]
    assert len(res) == len(rows), (len(res), len(rows))
    for k in range(a.procs):
        for f in (parts[k], parts[k] + ".err"):
            if os.path.exists(f):
                os.remove(f)
    res.sort(key=lambda r: r["i"])
    summ = {"weights": a.weights, "rows": a.rows, "wall_s": round(time.time() - t0, 1), "all": summarise(res)}
    for v in sorted({r["variant"] for r in res}):
        summ[v] = summarise([r for r in res if r["variant"] == v])
    by_tool = {}
    for r in res:
        b = by_tool.setdefault(r["tool"], [0, 0, 0])
        b[0] += 1; b[1] += r["tool_match"]; b[2] += r["correct"]
    summ["by_tool(n,tool_match,correct)"] = by_tool
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": res}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
