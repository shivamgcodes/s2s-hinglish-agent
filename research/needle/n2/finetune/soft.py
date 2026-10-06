"""Secondary view (not the selection metric): correct when reason/location also use the token-F1 >= 0.6 rule that
score_map already applies to address/instruction; plus correct on non-email rows and per-arg failure counts."""
import json, os, sys, collections
F = os.environ.get("N2_ROOT", "/root/n2") + "/finetune/evals"
for t in sys.argv[1:]:
    d = json.load(open(f"{F}/engine_val_{t}.json"))
    soft = noemail = noemail_n = 0
    bad = collections.Counter()
    for r in d["rows"]:
        ok = r["tool_match"] and not r["ask"] and bool(r["per_arg"])
        if ok:
            for pa in r["per_arg"]:
                for k, v in pa.items():
                    if v["ok"]:
                        continue
                    if k in ("reason", "location") and (v.get("token_f1") or 0) >= 0.6:
                        continue
                    ok = False
        soft += ok
        if r["tool"] != "update_email_address":
            noemail_n += 1; noemail += r["correct"]
        for k in r["bad_args"]:
            bad[k] += 1
    print(t, "correct_soft", soft, "| correct on non-email rows", f"{noemail}/{noemail_n}", "| bad args", dict(bad))
