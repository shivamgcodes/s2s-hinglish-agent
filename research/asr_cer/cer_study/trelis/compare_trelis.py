#!/usr/bin/env python3
"""Build comparison_trelis.csv: Trelis/whisper-hinglish-preview re-run of the CER/WER study vs large-v3.

Reads (stdlib only):
  inputs/<pair>.{reference,whisper}.txt            Trelis transcripts -> score.py raw/fold CER (computed here)
  m1/<pair>/{metrics,verify}.json                  Method 1 (romanise, then Levenshtein)
  m2/<pair>/{char_metrics,word_metrics,verify}.json  Method 2 (cross-script sound maps)
  ../comparison.csv                                large-v3 values (earlier study)
  ../inputs/<pair>.whisper.txt                     large-v3 transcripts (script statistics only)
Writes comparison_trelis.csv (10 pair rows + 4 per-model mean rows + 1 ALL mean row) and prints
self-checks, Spearman rank agreement and script statistics to stdout.

score.py's raw_norm/fold/cer cannot be imported (score.py imports tcommon/hindi_share at module level,
which only exist on the pod), so the block between the COPIED markers is lines 227-301 of
hinglish/tests/score.py (monorepo: research/eval_harness/score.py), copied verbatim.
"""
import csv
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
MODEL_ORDER = ["base", "V1_A", "V1_B", "V1_C"]

# ---- COPIED VERBATIM from pod_code/hinglish/tests/score.py lines 227-301 (begin) ----
_C = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
      "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
      "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
      "ष": "sh", "स": "s", "ह": "h", "ळ": "l"}
_NUK = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_V = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
      "औ": "au", "ऑ": "o", "ऍ": "e"}
_M = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
      "ॉ": "o", "ॅ": "e"}
_DD = str.maketrans("०१२३४५६७८९", "0123456789")


def deva_to_latin(s):
    s = unicodedata.normalize("NFD", s).translate(_DD)
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch in _C:
            nuk = i + 1 < len(s) and s[i + 1] == "़"
            out.append(_NUK.get(ch, _C[ch]) if nuk else _C[ch])
            i += 2 if nuk else 1
            nxt = s[i] if i < len(s) else ""
            if nxt in _M:
                out.append(_M[nxt]); i += 1
            elif nxt == "्":
                i += 1
            elif "ऀ" <= nxt <= "ॿ" and nxt not in "ंँः":
                out.append("a")
            elif nxt in "ंँः":
                out.append("a")
            # word-final: schwa deleted
            continue
        if ch in _V:
            out.append(_V[ch])
        elif ch in "ंँ":
            out.append("n")
        elif ch == "ः":
            out.append("h")
        elif ch in "़्":
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def raw_norm(s):
    return "".join(c for c in deva_to_latin(s).lower() if c.isalnum() and c.isascii())


def fold(s):
    s = raw_norm(s)
    for a, b in (("chh", "C"), ("ch", "C"), ("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("bh", "b"), ("jh", "j"), ("c", "k"), ("C", "c"), ("w", "v"), ("z", "j"), ("q", "k"),
                 ("x", "ks")):
        s = s.replace(a, b)
    s = re.sub(r"[aeiouhy]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def cer(ref, hyp):
    return (lev(ref, hyp) / len(ref)) if ref else (0.0 if not hyp else 1.0)

# ---- COPIED VERBATIM (end) ----


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def find_key(d, key):
    """d[key] at top level, else first match in nested dicts (depth-first)."""
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def scorepy(ref_txt, hyp_txt):
    return (round(cer(raw_norm(ref_txt), raw_norm(hyp_txt)), 4),
            round(cer(fold(ref_txt), fold(hyp_txt)), 4))


DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def script_stats(txt):
    d, l = len(DEVA.findall(txt)), len(LATIN.findall(txt))
    return round(d / (d + l), 4) if d + l else 0.0, len(re.findall(r"\d", txt))


def m1_fields(p):
    m = load(HERE, "m1", p, "metrics.json")
    v = load(HERE, "m1", p, "verify.json")
    # verifier's independent recount sits at top level, under 'independent', or under 'computed'
    vc = first(v.get("cer"), find_key(v.get("independent"), "cer"), find_key(v.get("computed"), "cer"))
    vw = first(v.get("wer"), find_key(v.get("independent"), "wer"), find_key(v.get("computed"), "wer"))
    verified = vc is not None and vw is not None and abs(vc - m["cer"]) < 5e-5 and abs(vw - m["wer"]) < 5e-5
    return {
        "m1_cer": round(m["cer"], 4), "m1_wer": round(m["wer"], 4),
        "m1_cer_S": m["cer_S"], "m1_cer_D": m["cer_D"], "m1_cer_I": m["cer_I"],
        "m1_wer_S": m["wer_S"], "m1_wer_D": m["wer_D"], "m1_wer_I": m["wer_I"],
        "m1_ref_chars": m["ref_chars"], "m1_ref_words": m["ref_words"],
        "m1_verified": verified,
        "m1_romanisation_faithful": bool(v.get("romanisation_faithful")),
    }


def m2_fields(p):
    c = load(HERE, "m2", p, "char_metrics.json")
    w = load(HERE, "m2", p, "word_metrics.json")
    v = load(HERE, "m2", p, "verify.json")
    rc = first(find_key(v, "cer_recount"), find_key(v.get("char_recount"), "cer"))
    rw = first(find_key(v, "wer_recount"), find_key(v.get("word_recount"), "wer"))
    recount_ok = rc is not None and rw is not None and abs(rc - c["cer"]) < 5e-4 and abs(rw - w["wer"]) < 5e-4
    audit = v.get("audit", {})
    err = first(find_key(v, "judgement_errors_in_sample"), find_key(v, "judgement_errors"),
                audit.get("wrong_labels") if isinstance(audit.get("wrong_labels"), int) else None)
    size = find_key(v, "sample_size")
    if size is None and isinstance(audit.get("char_steps"), list):  # air_16_g4 lists steps instead of a size
        size = len(audit["char_steps"]) + len(audit.get("word_steps", []))
    if size is None and isinstance(audit.get("char_steps_sampled"), int):  # sub_07_g2 gives counts only
        size = audit["char_steps_sampled"] + audit.get("word_steps_sampled", 0)
    return {
        "m2_cer": round(c["cer"], 4), "m2_wer": round(w["wer"], 4),
        "m2_char_sub": c["sub"], "m2_char_del": c["del"], "m2_char_ins": c["ins"],
        "m2_word_sub": w["sub"], "m2_word_del": w["del"], "m2_word_ins": w["ins"],
        "m2_ref_chars": c["ref_chars"], "m2_ref_words": w["ref_words"],
        "m2_maps_valid": bool(v.get("maps_valid")),
        "m2_recount_matches": recount_ok,
        "m2_judgement_errors": err, "m2_sample_size": size,
        "m2_judgement_error_rate": round(err / size, 4) if err is not None and size else None,
    }


LV3_COLS = ["scorepy_cer_raw", "scorepy_cer_fold", "m1_cer", "m1_wer", "m2_cer", "m2_wer",
            "m1_verified", "m1_romanisation_faithful", "m2_maps_valid", "m2_judgement_error_rate"]
DELTA_COLS = ["scorepy_cer_raw", "scorepy_cer_fold", "m1_cer", "m1_wer", "m2_cer", "m2_wer", "m2_judgement_error_rate"]


def lv3_value(r, c):
    x = r[c]
    if x in ("True", "False"):
        return x == "True"
    return float(x) if x not in ("", None) else None


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
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


def mean_row(grp, cols, pair_id, model):
    m = {"pair_id": pair_id, "model": model, "call_id": f"n={len(grp)}"}
    for c in cols[3:]:
        vals = [r[c] for r in grp if r[c] is not None]
        if vals and isinstance(vals[0], bool):
            m[c] = f"{sum(vals)}/{len(vals)}"
        elif vals:
            m[c] = round(sum(vals) / len(vals), 4)
        else:
            m[c] = None
    return m


def main():
    with open(os.path.join(STUDY, "comparison.csv"), encoding="utf-8") as f:
        lv3 = {r["pair_id"]: r for r in csv.DictReader(f) if not r["pair_id"].startswith("MEAN_")}

    rows, checks = [], []
    for p, L in lv3.items():
        ref = read(os.path.join(HERE, "inputs", f"{p}.reference.txt"))
        hyp_t = read(os.path.join(HERE, "inputs", f"{p}.whisper.txt"))
        hyp_l = read(os.path.join(STUDY, "inputs", f"{p}.whisper.txt"))
        ref_l = read(os.path.join(STUDY, "inputs", f"{p}.reference.txt"))
        raw_t, fold_t = scorepy(ref, hyp_t)
        raw_l, fold_l = scorepy(ref_l, hyp_l)  # self-check: copied functions reproduce large-v3 values
        checks.append((p, ref == ref_l, raw_l, float(L["scorepy_cer_raw"]), fold_l, float(L["scorepy_cer_fold"])))
        deva_t, dig_t = script_stats(hyp_t)
        deva_l, dig_l = script_stats(hyp_l)
        r = {"pair_id": p, "model": L["model"], "call_id": L["call_id"],
             "scorepy_cer_raw": raw_t, "scorepy_cer_fold": fold_t}
        r.update(m1_fields(p))
        r.update(m2_fields(p))
        r.update({"deva_share": deva_t, "digits": dig_t})
        for c in LV3_COLS:
            r[f"lv3_{c}"] = lv3_value(L, c)
        r.update({"lv3_deva_share": deva_l, "lv3_digits": dig_l})
        for c in DELTA_COLS:
            a, b = r[c], r[f"lv3_{c}"]
            r[f"d_{c}"] = round(a - b, 4) if a is not None and b is not None else None
        rows.append(r)

    rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), r["pair_id"]))
    cols = list(rows[0].keys())
    means = [mean_row([r for r in rows if r["model"] == mk], cols, f"MEAN_{mk}", mk) for mk in MODEL_ORDER]
    means.append(mean_row(rows, cols, "MEAN_ALL", "all"))
    # deltas on mean rows = difference of the means (equal to mean of deltas up to rounding)
    out = os.path.join(HERE, "comparison_trelis.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        w.writerows(means)
    print("wrote", out)

    print("\nSelf-check: copied score.py functions vs large-v3 comparison.csv (ref_same, raw calc/csv, fold calc/csv)")
    for p, same, a, b, c, d in checks:
        print(f"  {p:22s} ref_same={same} raw {a:.4f}/{b:.4f} {'OK' if abs(a - b) < 1e-4 else 'DIFF'}"
              f"  fold {c:.4f}/{d:.4f} {'OK' if abs(c - d) < 1e-4 else 'DIFF'}")

    metrics = ["scorepy_cer_raw", "scorepy_cer_fold", "m1_cer", "m1_wer", "m2_cer", "m2_wer"]
    v1 = [r for r in rows if r["model"] != "base"]
    print("\nSpearman rho, Trelis vs large-v3, same metric over pairs (all 10 / 8 V1)")
    for c in metrics:
        print(f"  {c:17s} {spearman([r[c] for r in rows], [r['lv3_' + c] for r in rows]):+.3f}"
              f" / {spearman([r[c] for r in v1], [r['lv3_' + c] for r in v1]):+.3f}")
    print("\nSpearman rho between methods, Trelis transcripts (all 10 / 8 V1)")
    for i, x in enumerate(metrics):
        for y in metrics[i + 1:]:
            print(f"  {x:17s} vs {y:17s} {spearman([r[x] for r in rows], [r[y] for r in rows]):+.3f}"
                  f" / {spearman([r[x] for r in v1], [r[y] for r in v1]):+.3f}")
    print("\nModel order by mean (Trelis | large-v3)")
    for c in metrics:
        t = sorted(means[:4], key=lambda m: m[c])
        l = sorted(means[:4], key=lambda m: m["lv3_" + c])
        print(f"  {c:17s} " + " < ".join(f"{m['model']} {m[c]:.3f}" for m in t)
              + "  |  " + " < ".join(f"{m['model']} {m['lv3_' + c]:.3f}" for m in l))
    print("\nPer-pair rank under each metric among 8 V1 pairs (Trelis rank / large-v3 rank)")
    for c in metrics:
        rt, rl = ranks([r[c] for r in v1]), ranks([r["lv3_" + c] for r in v1])
        print(f"  {c:17s} " + ", ".join(f"{r['call_id']} {a:g}/{b:g}" for r, a, b in zip(v1, rt, rl)))
    def comp(rs, s, d, i):
        S = sum(r[s] for r in rs); D = sum(r[d] for r in rs); I = sum(r[i] for r in rs); T = S + D + I
        return f"S={S} D={D} I={I} total={T} sub={S / T:.3f} del={D / T:.3f} ins={I / T:.3f}"
    print("\nEdit composition, Trelis (summed over 10 pairs)")
    print("  m1 char:", comp(rows, "m1_cer_S", "m1_cer_D", "m1_cer_I"))
    print("  m2 char:", comp(rows, "m2_char_sub", "m2_char_del", "m2_char_ins"))
    print("  m1 word:", comp(rows, "m1_wer_S", "m1_wer_D", "m1_wer_I"))
    print("  m2 word:", comp(rows, "m2_word_sub", "m2_word_del", "m2_word_ins"))
    te = sum(r["m2_judgement_errors"] for r in rows); ts = sum(r["m2_sample_size"] for r in rows)
    print(f"\nm2 judgement errors pooled (Trelis): {te}/{ts} = {te / ts:.3f}")


if __name__ == "__main__":
    main()
