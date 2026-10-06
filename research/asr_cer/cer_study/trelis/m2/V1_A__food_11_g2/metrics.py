#!/usr/bin/env python3
"""Verify char_map.jsonl against normalised texts; write char_metrics.json and char_map.txt."""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ref = open(os.path.join(HERE, "ref_norm.txt"), encoding="utf-8").read()
hyp = open(os.path.join(HERE, "hyp_norm.txt"), encoding="utf-8").read()
steps = [json.loads(l) for l in open(os.path.join(HERE, "char_map.jsonl"), encoding="utf-8")]

assert "".join(s["ref"] for s in steps) == ref, "ref concat mismatch"
assert "".join(s["hyp"] for s in steps) == hyp, "hyp concat mismatch"
assert [s["i"] for s in steps] == list(range(len(steps)))
for s in steps:
    assert s["op"] in ("match", "sub", "del", "ins")
    assert len(s["ref"]) <= 1
    if s["op"] == "ins": assert s["ref"] == "" and s["hyp"] != ""
    if s["op"] == "del": assert s["hyp"] == "" and s["ref"] != ""
    if s["op"] in ("match", "sub"): assert s["ref"] != ""
    # sub may carry empty hyp when it is a later letter of a multi-letter akshara group

c = Counter(s["op"] for s in steps)
n = len(ref)
assert c["match"] + c["sub"] + c["del"] == n
cer = (c["sub"] + c["del"] + c["ins"]) / n
metrics = {"pair": "V1_A__food_11_g2", "ref_chars": n, "hyp_chars": len(hyp), "steps": len(steps),
           "match": c["match"], "sub": c["sub"], "del": c["del"], "ins": c["ins"], "cer": round(cer, 4)}
json.dump(metrics, open(os.path.join(HERE, "char_metrics.json"), "w"), indent=2, ensure_ascii=False)

ab = {"match": "=", "sub": "S", "del": "D", "ins": "I"}
with open(os.path.join(HERE, "char_map.txt"), "w", encoding="utf-8") as f:
    W = 40
    for k in range(0, len(steps), W):
        blk = steps[k:k + W]
        cols = [max(1, len(s["ref"]), len(s["hyp"])) for s in blk]
        f.write("ref: " + "|".join((s["ref"] or "·").replace(" ", "_").ljust(w) for s, w in zip(blk, cols)) + "\n")
        f.write("hyp: " + "|".join((s["hyp"] or "·").replace(" ", "_").ljust(w) for s, w in zip(blk, cols)) + "\n")
        f.write("op:  " + "|".join(ab[s["op"]].ljust(w) for s, w in zip(blk, cols)) + "\n\n")
print(json.dumps(metrics))
