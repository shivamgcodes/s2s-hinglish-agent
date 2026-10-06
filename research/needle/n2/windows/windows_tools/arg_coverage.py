"""Per-arg coverage for N2 windows: is each gold write arg recoverable from the window's (oracle) customer text,
from the record, both, or neither? Adds meta.writes[].arg_coverage and summary.json["arg_coverage"].
Rule: ID/phone/email/number-like values -> normalised substring (lowercase, alnum only; digits kept); free text
(reason/location/instruction etc.) -> token F1 >= 0.6 against the best contiguous span of the same length +-50%.
Oracle text = script text_roman of customer turns overlapping the window (partial turns included in full - upper
bound). Usage: python arg_coverage.py --win DIR --v4 V4DIR"""
import argparse, json, re
from collections import Counter, defaultdict

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def toks(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

def is_idlike(v):
    v = str(v)
    return bool(re.search(r"\d", v)) and len(toks(v)) <= 3 or "@" in v

def best_f1(gold, text):
    g = toks(gold); t = toks(text)
    if not g or not t:
        return 0.0
    best = 0.0
    G = Counter(g)
    for L in range(max(1, int(len(g) * 0.5)), int(len(g) * 1.5) + 2):
        for i in range(0, max(1, len(t) - L + 1)):
            S = Counter(t[i:i + L]); ov = sum((S & G).values())
            if ov:
                p, r = ov / sum(S.values()), ov / len(g); best = max(best, 2 * p * r / (p + r))
    return best

def found(v, text):
    if is_idlike(v):
        return norm(v) != "" and norm(v) in norm(text)
    return best_f1(v, text) >= 0.6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win"); ap.add_argument("--v4"); a = ap.parse_args()
    recs = {c["call_id"]: c["record"] for c in map(json.loads, open(f"{a.v4}/calls.jsonl"))}
    rows = [json.loads(l) for l in open(f"{a.win}/meta.jsonl")]
    agg = defaultdict(Counter); tot = Counter(); nullrec = 0
    R4 = json.load(open(f"{a.v4}/records.json"))
    for r in rows:
        r["record_id"] = R4[r["scenario_id"]]["record_id"]
        nullrec += r["record_id"] is None
        wt = " ".join(t["text_roman"] for t in r["customer_turns_in_window"])
        rec = recs[r["call_id"]]
        rt = " ".join([str(rec.get(k, "")) for k in ("primary_id", "secondary_id", "phone", "information", "customer_name")]
                      + [str(v) for v in rec.get("facts", {}).values()] + [str(v) for v in rec.get("distractors", [])])
        for w in r["writes"]:
            cov = {}
            for k, v in w["args"].items():
                iw, ir = found(v, wt), found(v, rt)
                b = "both" if iw and ir else "window_only" if iw else "record_only" if ir else "neither"
                cov[k] = b; agg[f"{w['tool']}.{k}"][b] += 1; tot[b] += 1
            w["arg_coverage"] = cov
    with open(f"{a.win}/meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = json.load(open(f"{a.win}/summary.json"))
    S["arg_coverage_rule"] = __doc__.split("Usage")[0].strip()
    S["arg_coverage_total"] = dict(tot)
    S["arg_coverage_per_tool_arg"] = {k: dict(v) for k, v in sorted(agg.items())}
    S["n_rows_null_record_id"] = nullrec
    json.dump(S, open(f"{a.win}/summary.json", "w"), indent=1, ensure_ascii=False)
    print("null record_id", nullrec); print(dict(tot))
    for k, v in sorted(agg.items()):
        print(k, dict(v))

main()
