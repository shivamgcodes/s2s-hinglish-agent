"""Re-score round-1 candidates of trial runs with the frozen counter + current deterministic checks (no judge)."""
import json, sys, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
import common, generate, validate as V
scen = common.load_scenarios(); recs = json.load(open(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/data/records.json"))
interrupt, twopart_new = common.assignments(scen)
for d in sys.argv[1:]:
    v = os.path.basename(d.rstrip("/")).split("_")[-1]
    # judge verdicts from gen_log
    judged = {}
    for l in open(f"{d}/gen_log.jsonl"):
        r = json.loads(l)
        if r["attempt"] == 1:
            judged[(r["call_id"], r["sample"])] = any(f.startswith("llm_judge") for f in r["failures"])
    per = collections.Counter(); calls = collections.defaultdict(list); n = 0
    for l in open(f"{d}/raw_outputs.jsonl"):
        r = json.loads(l)
        if r["attempt"] != 1: continue
        cid = r["call_id"]; sid = cid.rsplit("_", 1)[0]; n += 1
        try:
            raw = common.extract_json(r["text"])
            call = generate.postprocess(cid, v, scen, recs[sid], raw, cid in interrupt)
            # two-part flag as used at generation time (t7 used the final assignment)
            F = V.check_call(call, scen[sid], cid in interrupt, cid in twopart_new)
        except Exception as e:
            F = [("parse", str(e))]
        if judged.get((cid, r["sample"])): F.append(("llm_judge", ""))
        for k in {f[0] for f in F}: per[k] += 1
        calls[cid].append(not F)
    sp = sum(sum(x) for x in calls.values()); b3 = sum(any(x) for x in calls.values())
    print(f"{d}: per-sample pass {sp}/{n}, best-of-3 round-1 pass {b3}/{len(calls)}; per-sample fails {dict(per)}")
