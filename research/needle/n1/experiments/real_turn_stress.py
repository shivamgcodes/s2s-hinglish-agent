"""Out-of-sample stress check of resolver.py: feed the WHOLE request turn (confirm_turn_idx - 2) of
every V1/V3 write as order_ref (worst case for junk spans), as text_roman and as asr_roman.
Gold = the ID in writes[].args if it is an entity ID, else the record's primary_id.
Reports misroutes (non-gold ID or None) by rule and split; not accuracy (default->primary is free)."""
import collections, json, os, sys
from pathlib import Path  # monorepo shim: N1 modules from packages/needle_router/n1 ($N1_DIR)
REPO = Path(__file__).resolve().parents[4]
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was the script's parent dir
sys.path.insert(0, N1_PKG)
from resolver import resolve
src = os.path.join(HERE, "src")
rec = json.load(open(os.path.join(N1_PKG, "src", "records.json")))
hold = set(json.load(open(os.path.join(src, "holdout.json")))["test_scenarios"])
asr = {}
for l in open(os.path.join(HERE, "asr_turns.jsonl")):
    a = json.loads(l); asr[(a["variant"], a["call_id"], a["turn"])] = a["asr_roman"]
rows = []
for V in ("V1", "V3"):
    for l in open(os.path.join(src, f"{V}_calls.jsonl")):
        c = json.loads(l); r = rec[c["scenario_id"]]
        ids = {r["primary_id"]} | {e for e in [r.get("secondary_id")] if e}
        for w in c["writes"]:
            i = w["confirm_turn_idx"] - 2
            gold = next((v for v in w["args"].values() if v in ids), r["primary_id"])
            for rend, text in (("a_text_roman", c["turns"][i]["text_roman"]), ("b_asr_roman", asr.get((V, c["call_id"], i)))):
                if text is None:
                    continue
                g = resolve(text, r, r["agent_type"])
                rows.append({"variant": V, "call_id": c["call_id"], "split": "heldout" if c["scenario_id"] in hold else "train",
                             "rendering": rend, "turn": i, "text": text, "gold": gold, "got": g["id"], "rule": g["rule"],
                             "ok": g["id"] == gold})
c = collections.Counter(); n = collections.Counter(); byrule = collections.Counter()
for x in rows:
    k = (x["split"], x["rendering"]); n[k] += 1
    if not x["ok"]:
        c[k] += 1; byrule[(x["split"], x["rendering"], x["rule"])] += 1
print("misroutes (whole request turn used as order_ref):")
for k in sorted(n): print(f"  {k[0]:<8} {k[1]:<13} {c[k]}/{n[k]}")
print("misroutes by rule:")
for k, v in sorted(byrule.items()): print(" ", k, v)
print("rules fired (all rows):", collections.Counter(x["rule"] for x in rows).most_common())
print("misroute rows:")
for x in rows:
    if not x["ok"]:
        print(f"  [{x['split']}/{x['rendering']}] {x['call_id']} t{x['turn']} gold {x['gold']} got {x['got']} ({x['rule']}): {x['text']}")
json.dump(rows, open(os.path.join(HERE, "experiments", "real_turn_stress.json"), "w"), indent=1, ensure_ascii=False)
