import json
from collections import Counter
from normalize import norm
REF = norm(open("reference.txt", encoding="utf-8").read())
HYP = norm(open("whisper.txt", encoding="utf-8").read())
rows = [json.loads(l) for l in open("char_map.jsonl", encoding="utf-8")]
r = "".join(x["ref"] for x in rows); h = "".join(x["hyp"] for x in rows)
assert r == REF, ("REF mismatch", r, REF)
assert h == HYP, ("HYP mismatch", h, HYP)
for x in rows:
    assert x["op"] in ("match", "sub", "del", "ins")
    if x["op"] == "del": assert x["ref"] and not x["hyp"]
    elif x["op"] == "ins": assert x["hyp"] and not x["ref"]
    else: assert len(x["ref"]) == 1 and x["hyp"]
c = Counter(x["op"] for x in rows)
n = len(REF)
assert c["match"] + c["sub"] + c["del"] == n
m = {"pair": "base_V1__sub_07_g2", "ref_chars": n, "hyp_chars": len(HYP),
     "match": c["match"], "sub": c["sub"], "del": c["del"], "ins": c["ins"],
     "cer": round((c["sub"] + c["del"] + c["ins"]) / n, 6),
     "normalisation": "NFC, lowercase, apostrophes deleted, other punctuation->space, whitespace collapsed; units = ref characters, hyp chunks (Devanagari codepoints incl. matras)"}
json.dump(m, open("char_metrics.json", "w"), indent=2, ensure_ascii=False)
with open("char_map.txt", "w", encoding="utf-8") as f:
    for x in rows:
        f.write(f"{x['i']:4d}  ref={x['ref']!r:6} hyp={x['hyp']!r:12} {x['op']:5}  {x['why']}\n")
print(json.dumps(m, indent=2, ensure_ascii=False))
for x in rows:
    if x["op"] != "match" or x["why"] != "identical": print(x)
