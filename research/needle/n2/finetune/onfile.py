"""How many silent wrong email / phone writes equal a value already on the record (system text)? A router guard
(new value == on-file value -> ASK) would stop those."""
import json, os, re, sys
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir
rows = [json.loads(l) for l in open(f"{N2_ROOT}/data/rows/val.jsonl")]
for t in sys.argv[1:]:
    d = json.load(open(f"{N2_ROOT}/finetune/evals/engine_val_{t}.json"))
    n = same = 0
    for r in d["rows"]:
        if not r["silent_wrong"] or not r["server"]:
            continue
        sysl = rows[r["i"]]["system"].lower()
        for k in ("email", "phone"):
            v = r["server"][0]["args"].get(k)
            if k in r["bad_args"] and v:
                n += 1
                same += (v.lower() in sysl) if k == "email" else (v in re.sub(r"\D", "", sysl))
    print(t, "silent wrong email/phone writes", n, "equal to an on-file value", same)
