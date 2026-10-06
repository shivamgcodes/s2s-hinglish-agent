#!/usr/bin/env python3
"""Build comparison.csv for the CER/WER study (stdlib only).

Reads args.json (score.py CERs), w1_romanise/<pair>/{metrics,verify}.json (Method 1)
and w2_charmap/<pair>/{char_metrics,word_metrics,verify}.json (Method 2).
Writes comparison.csv (one row per pair + one mean row per model) and prints
Spearman rank correlations and deletion shares to stdout.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORDER = ["base_V1", "V1_A", "V1_B", "V1_C"]
MODEL_LABEL = {"base_V1": "base"}


def load(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return json.load(f)


def find_key(d, key):
    """Return d[key] at top level, else first match inside nested dicts."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def build_row(a):
    p = a["pair_id"]
    m1 = load("w1_romanise", p, "metrics.json")
    v1 = load("w1_romanise", p, "verify.json")
    c2 = load("w2_charmap", p, "char_metrics.json")
    w2 = load("w2_charmap", p, "word_metrics.json")
    v2 = load("w2_charmap", p, "verify.json")
    err = find_key(v2, "judgement_errors_in_sample")
    size = find_key(v2, "sample_size")
    return {
        "pair_id": p,
        "model": MODEL_LABEL.get(a["model"], a["model"]),
        "call_id": a["call_id"],
        "ref_chars": m1["ref_chars"],
        "ref_words": m1["ref_words"],
        "m2_ref_chars": c2["ref_chars"],
        "m2_ref_words": w2["ref_words"],
        "scorepy_cer_raw": a["scorepy_cer_raw"],
        "scorepy_cer_fold": a["scorepy_cer_fold"],
        "m1_cer": m1["cer"],
        "m1_wer": m1["wer"],
        "m1_cer_S": m1["cer_S"],
        "m1_cer_D": m1["cer_D"],
        "m1_cer_I": m1["cer_I"],
        "m1_wer_S": m1["wer_S"],
        "m1_wer_D": m1["wer_D"],
        "m1_wer_I": m1["wer_I"],
        "m1_verified": bool(v1.get("cer_ok")) and bool(v1.get("wer_ok")),
        "m1_romanisation_faithful": bool(v1.get("romanisation_faithful")),
        "m2_cer": c2["cer"],
        "m2_wer": w2["wer"],
        "m2_char_sub": c2["sub"],
        "m2_char_del": c2["del"],
        "m2_char_ins": c2["ins"],
        "m2_word_sub": w2.get("sub", 0),
        "m2_word_del": w2.get("del", 0),
        "m2_word_ins": w2.get("ins", 0),
        "m2_maps_valid": bool(v2.get("char_map_valid")) and bool(v2.get("word_map_valid")),
        "m2_judgement_errors": err,
        "m2_sample_size": size,
        "m2_judgement_error_rate": round(err / size, 4) if err is not None and size else None,
    }


def main():
    args = load("args.json")
    rows = [build_row(a) for a in args]
    rows.sort(key=lambda r: (MODEL_ORDER.index(next(k for k in MODEL_ORDER if MODEL_LABEL.get(k, k) == r["model"])), r["pair_id"]))
    cols = list(rows[0].keys())

    means = []
    for mk in MODEL_ORDER:
        label = MODEL_LABEL.get(mk, mk)
        grp = [r for r in rows if r["model"] == label]
        m = {"pair_id": f"MEAN_{label}", "model": label, "call_id": f"n={len(grp)}"}
        for c in cols[3:]:
            vals = [r[c] for r in grp if r[c] is not None]
            if vals and isinstance(vals[0], bool):
                m[c] = f"{sum(vals)}/{len(vals)}"
            elif vals:
                m[c] = round(sum(vals) / len(vals), 4)
            else:
                m[c] = None
        means.append(m)

    out = os.path.join(HERE, "comparison.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        w.writerows(means)
    print("wrote", out)

    # Spearman rank agreement over the pairs
    metrics = ["scorepy_cer_raw", "scorepy_cer_fold", "m1_cer", "m2_cer", "m1_wer", "m2_wer"]
    print("\nSpearman rho over", len(rows), "pairs")
    for i, x in enumerate(metrics):
        for y in metrics[i + 1:]:
            print(f"  {x:17s} vs {y:17s}: {spearman([r[x] for r in rows], [r[y] for r in rows]):+.3f}")

    # deletion share of total edits
    def share(s, d, i):
        S = sum(r[s] for r in rows); D = sum(r[d] for r in rows); I = sum(r[i] for r in rows)
        T = S + D + I
        return f"S={S} D={D} I={I} total={T} del_share={D / T:.3f} sub_share={S / T:.3f} ins_share={I / T:.3f}"
    print("\nEdit composition (summed over pairs)")
    print("  m1 char:", share("m1_cer_S", "m1_cer_D", "m1_cer_I"))
    print("  m2 char:", share("m2_char_sub", "m2_char_del", "m2_char_ins"))
    print("  m1 word:", share("m1_wer_S", "m1_wer_D", "m1_wer_I"))
    print("  m2 word:", share("m2_word_sub", "m2_word_del", "m2_word_ins"))

    print("\nPer-pair differences (m1 - m2)")
    for r in sorted(rows, key=lambda r: -abs(r["m1_cer"] - r["m2_cer"])):
        print(f"  {r['pair_id']:22s} dCER={r['m1_cer'] - r['m2_cer']:+.4f} dWER={r['m1_wer'] - r['m2_wer']:+.4f}"
              f"  m1_ins={r['m1_cer_I']} m2_ins={r['m2_char_ins']} m1_sub={r['m1_cer_S']} m2_sub={r['m2_char_sub']}")
    n = len(rows)
    print("\nOverall means: " + ", ".join(f"{c}={sum(r[c] for r in rows) / n:.4f}" for c in metrics))
    te = sum(r["m2_judgement_errors"] for r in rows); ts = sum(r["m2_sample_size"] for r in rows)
    print(f"m2 judgement errors pooled: {te}/{ts} = {te / ts:.3f}")


if __name__ == "__main__":
    main()
