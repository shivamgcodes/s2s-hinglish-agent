"""N2 eval: aggregate the per-row engine runs in runs/ into eval.json (+ tables consumed by EVAL.md).
No model calls. v2 REF resolver rules are recomputed with router_v2.resolve_calls on the stored function_calls
(deterministic, same code as the run)."""
import json, os, sys
from collections import Counter, defaultdict

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[4]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
sys.path.insert(1, str(REPO / "research" / "needle" / "n2" / "schema"))  # score_map.py
import router_v2, score_map, tools_v2  # noqa: E402

RUNS = f"{N2_ROOT}/eval/runs"
ROWS = f"{N2_ROOT}/data/rows"
REF = score_map.REF_GOLD_ARGS
SOFT = {"reason", "location"}  # strict (normalised equality) in the N1 rule; soft = token F1 >= 0.6
CALLS = {}
for l in open(f"{N2_ROOT}/V4/calls.jsonl"):
    c = json.loads(l); CALLS[c["call_id"]] = c


def load(name):
    p = f"{RUNS}/{name}.json"
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def metas(f):
    return {m["example_id"]: m for m in (json.loads(l) for l in open(f"{ROWS}/{f}_meta.jsonl"))}


def soft_ok(r):
    if not r["tool_match"] or r["ask"]:
        return False
    return all(v["ok"] or (k in SOFT and (v.get("token_f1") or 0) >= 0.6) for pa in r["per_arg"] for k, v in pa.items())


def headline(rs):
    n = len(rs)
    c = lambda k: sum(1 for r in rs if r[k])
    return {"n": n, "tool_match": c("tool_match"), "correct": c("correct"),
            "correct_soft_reason_location": sum(soft_ok(r) for r in rs), "ask": c("ask"),
            "silent_wrong": c("silent_wrong"), "silent_wrong_phone": c("silent_wrong_phone"),
            "silent_wrong_email": c("silent_wrong_email"), "missed": sum(1 for r in rs if not r["function_calls"]),
            "multi_call": sum(1 for r in rs if len(r["function_calls"]) > 1),
            "ask_but_args_right": sum(1 for r in rs if r["ask"] and r["tool_match"] and r["args_match"])}


def per_arg(rs):
    t = defaultdict(lambda: Counter())
    for r in rs:
        gargs = r["gold"][0]["args"]
        for k in gargs:
            t[k]["n_all_rows"] += 1
        if not r["tool_match"]:
            continue
        for k, v in r["per_arg"][0].items():
            d = t[k]
            if v["gold"] is not None:
                d["n_tool_matched"] += 1
                d["ok"] += v["ok"]
                d["missing"] += v["pred"] in (None, "")
                if k in SOFT or k in ("address", "instruction", "email"):
                    d["f1_ge_0.6"] += (v.get("token_f1") or 0) >= 0.6
            else:
                d["extra"] += 1
    out = {}
    for k, d in sorted(t.items(), key=lambda x: (x[0] not in REF, x[0])):
        out[k] = {"class": "REF" if k in REF else "NEW", **dict(d),
                  "ok_of_all_rows": d["ok"]}
    return out


def resolver_v2(rs):
    rules, okc, n = Counter(), 0, 0
    for r in rs:
        if not r["tool_match"]:
            continue
        m_rec = CALLS[r["example_id"].split("__")[0]]["record"]
        res = router_v2.resolve_calls(r["function_calls"], m_rec, r["agent_type"])
        for c in res:
            for marg, ref in c["refs"].items():
                if ref["kind"] in tools_v2.REF_KINDS:
                    rules[ref["rule"]] += 1
        refk = [k for k in r["gold"][0]["args"] if k in REF]
        if refk:
            n += 1
            okc += all(r["per_arg"][0][k]["ok"] for k in refk)
    return {"rows_tool_matched_with_ref_gold": n, "all_ref_args_resolved_to_gold": okc,
            "rules_on_tool_matched_calls": dict(rules.most_common())}


def resolver_n1(rs):
    rules, okc, n = Counter(), 0, 0
    for r in rs:
        if not r["tool_match"]:
            continue
        for c in r["resolved"]:
            rules[c["resolver_rule"]] += 1
        refk = [k for k in r["gold"][0]["args"] if k in REF]
        if refk:
            n += 1
            okc += all(r["per_arg"][0][k]["ok"] for k in refk)
    return {"rows_tool_matched_with_ref_gold": n, "all_ref_args_resolved_to_gold": okc,
            "rules_on_tool_matched_calls": dict(rules.most_common())}


def by_type(rs):
    d = defaultdict(list)
    for r in rs:
        d[r["agent_type"]].append(r)
    return {k: {"n": len(v), "tool_match": sum(r["tool_match"] for r in v), "correct": sum(r["correct"] for r in v),
                "correct_soft": sum(soft_ok(r) for r in v), "ask": sum(r["ask"] for r in v),
                "silent_wrong": sum(r["silent_wrong"] for r in v)} for k, v in sorted(d.items())}


def by_tool(rs):
    d = defaultdict(list)
    for r in rs:
        d[r["tool"]].append(r)
    return {k: {"n": len(v), "tool_match": sum(r["tool_match"] for r in v), "correct": sum(r["correct"] for r in v)}
            for k, v in sorted(d.items())}


def groups(rs, meta):
    g, lim = Counter(), Counter()
    for r in rs:
        if r["correct"]:
            continue
        g[r["group"]] += 1
        m = meta[r["example_id"]]
        gr = m.get("grounded") or {}
        if gr and not all(gr.values()):
            lim[r["group"]] += 1
    return {"groups": dict(g.most_common()), "of_which_value_not_in_transcript": dict(lim)}


def block(rs, meta, kind):
    b = {"headline": headline(rs), "per_arg": per_arg(rs), "by_agent_type": by_type(rs), "by_tool": by_tool(rs),
         "fail_groups": groups(rs, meta)}
    b["resolver"] = resolver_v2(rs) if kind == "v2" else resolver_n1(rs)
    if any(meta[r["example_id"]].get("exact_has_partial_turn") is not None for r in rs):
        for flag in (True, False):
            sub = [r for r in rs if meta[r["example_id"]].get("exact_has_partial_turn") is flag]
            b[f"partial_turn_{flag}"] = headline(sub)
    gsub = [r for r in rs if all((meta[r["example_id"]].get("grounded") or {}).values())]
    b["value_in_transcript_rows"] = headline(gsub)
    return b


def main():
    out = {"v2": {}, "Ae10_informational": {}, "n1": {}, "side_by_side": {}}
    for v in ("clean", "opus", "exact"):
        meta = metas(f"test_{v}")
        rs = load(f"v2_test_{v}")
        if rs:
            out["v2"][v] = block(rs, meta, "v2")
        ra = load(f"Ae10_test_{v}")
        if ra:
            out["Ae10_informational"][v] = block(ra, meta, "v2")
        for suf in ("", "_nc"):
            n1m = metas(f"n1_test_{v}{suf}")
            rn = load(f"n1_n1_test_{v}{suf}")
            if rn:
                out["n1"][v + suf] = block(rn, n1m, "n1")
        # side by side on the same 104 old-type windows
        rn, rnc = load(f"n1_n1_test_{v}"), load(f"n1_n1_test_{v}_nc")
        if rs and rn and rnc:
            ids = {r["example_id"] for r in rn}
            assert ids == {r["example_id"] for r in rnc}
            v2old = [r for r in rs if r["example_id"] in ids]
            assert len(v2old) == len(ids) == 104, (len(v2old), len(ids))
            sb = {"N1 native": headline(rn), "N1 + numconv": headline(rnc), "v2 R_e15": headline(v2old)}
            if ra:
                sb["v2 A_e10 (info)"] = headline([r for r in ra if r["example_id"] in ids])
            sb["per_arg"] = {"N1 native": per_arg(rn), "N1 + numconv": per_arg(rnc), "v2 R_e15": per_arg(v2old)}
            sb["by_agent_type"] = {"N1 native": by_type(rn), "N1 + numconv": by_type(rnc), "v2 R_e15": by_type(v2old)}
            # paired: rows each side gets right
            c1 = {r["example_id"] for r in rn if r["correct"]}
            c2 = {r["example_id"] for r in v2old if r["correct"]}
            sb["paired_correct"] = {"both": len(c1 & c2), "only_N1": len(c1 - c2), "only_v2": len(c2 - c1),
                                    "neither": 104 - len(c1 | c2)}
            out["side_by_side"][v] = sb
    json.dump(out, open(f"{N2_ROOT}/eval/eval.json", "w"), ensure_ascii=False, indent=1)
    for k in ("v2", "n1", "Ae10_informational"):
        for v, b in out[k].items():
            print(k, v, json.dumps(b["headline"]))
    for v, sb in out["side_by_side"].items():
        print("SBS", v, {k: (x["correct"], x["tool_match"]) for k, x in sb.items() if "n" in x}, sb["paired_correct"])


if __name__ == "__main__":
    main()
