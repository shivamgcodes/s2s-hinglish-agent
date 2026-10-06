"""Silent wrong writes split: 'soft' = only reason/location differ with token F1 >= 0.6 (wording, not a wrong value);
'harmful' = any other wrong arg / wrong tool shipped without ASK. Per tag, by cause and by tool."""
import json, os, sys, collections
F = os.environ.get("N2_ROOT", "/root/n2") + "/finetune/evals"
for t in sys.argv[1:]:
    d = json.load(open(f"{F}/engine_val_{t}.json"))
    harm, soft, cause, tools = 0, 0, collections.Counter(), collections.Counter()
    for r in d["rows"]:
        if not r["silent_wrong"]:
            continue
        hard = [k for pa in r["per_arg"] for k, v in pa.items() if not v["ok"]
                and not (k in ("reason", "location") and (v.get("token_f1") or 0) >= 0.6)]
        if r["tool_match"] and not hard:
            soft += 1
            continue
        harm += 1
        c = "wrong tool/extra" if not r["tool_match"] else ",".join(sorted(set(hard)))
        cause[c] += 1; tools[r["tool"]] += 1
    print(t, "silent soft", soft, "harmful", harm, dict(cause))
