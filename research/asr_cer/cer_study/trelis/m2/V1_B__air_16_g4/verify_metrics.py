#!/usr/bin/env python3
"""Verify char_map.jsonl reconstructs normalised ref/hyp exactly; write char_metrics.json and char_map.txt."""
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "char_map.jsonl"), encoding="utf-8")]
ref = open(os.path.join(HERE, "ref_norm.txt"), encoding="utf-8").read()
hyp = open(os.path.join(HERE, "hyp_norm.txt"), encoding="utf-8").read()

ok_ref = "".join(r["ref"] for r in rows) == ref
ok_hyp = "".join(r["hyp"] for r in rows) == hyp
ok_idx = all(r["i"] == k for k, r in enumerate(rows))
bad = [r for r in rows if (r["op"] == "del" and (r["hyp"] or len(r["ref"]) != 1))
       or (r["op"] == "ins" and (r["ref"] or not r["hyp"]))
       or (r["op"] == "sub" and (len(r["ref"]) != 1 or not r["hyp"]))
       or (r["op"] == "match" and (not r["ref"] or not r["hyp"]))]
print("ref reconstructs:", ok_ref, "| hyp reconstructs:", ok_hyp, "| idx ok:", ok_idx, "| malformed:", len(bad))
if not (ok_ref and ok_hyp and ok_idx) or bad:
    sys.exit(1)

c = Counter(r["op"] for r in rows)
matched_chars = sum(len(r["ref"]) for r in rows if r["op"] == "match")
n = len(ref)
assert matched_chars + c["sub"] + c["del"] == n
m = {
    "pair": "V1_B__air_16_g4",
    "ref_chars": n,
    "hyp_chars": len(hyp),
    "match_steps": c["match"], "matched_ref_chars": matched_chars,
    "sub": c["sub"], "del": c["del"], "ins": c["ins"],
    "cer": round((c["sub"] + c["del"] + c["ins"]) / n, 4),
}
json.dump(m, open(os.path.join(HERE, "char_metrics.json"), "w"), indent=2, ensure_ascii=False)

# char_map.txt : chunks of steps as aligned ref / hyp / op lines
OPC = {"match": "=", "sub": "S", "del": "D", "ins": "I"}
lines = []
for s in range(0, len(rows), 25):
    chunk = rows[s:s + 25]
    cells = []
    for r in chunk:
        rr = r["ref"].replace(" ", "␣") or "·"
        hh = r["hyp"].replace(" ", "␣") or "·"
        w = max(len(rr), len(hh), 1)
        cells.append((rr.ljust(w), hh.ljust(w), OPC[r["op"]].ljust(w)))
    lines.append(f"# steps {s}-{s + len(chunk) - 1}")
    lines.append("ref: " + "|".join(c[0] for c in cells))
    lines.append("hyp: " + "|".join(c[1] for c in cells))
    lines.append("op : " + "|".join(c[2] for c in cells))
    lines.append("")
lines.append(json.dumps(m, ensure_ascii=False))
open(os.path.join(HERE, "char_map.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(m)
