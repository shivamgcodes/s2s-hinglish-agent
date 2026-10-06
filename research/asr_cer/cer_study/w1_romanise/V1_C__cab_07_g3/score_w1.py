#!/usr/bin/env python3
import json, os, re, sys

D = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w']|_", " ", s)  # keep letters, digits, apostrophe
    return " ".join(s.split())


def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    # dp of (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                sub = prev[j - 1]
            else:
                p = prev[j - 1]
                sub = (p[0] + 1, p[1] + 1, p[2], p[3])
            d = prev[j]; d = (d[0] + 1, d[1], d[2] + 1, d[3])
            ins = cur[j - 1]; ins = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
            cur.append(min(sub, d, ins, key=lambda t: t[0]))
        prev = cur
    return prev[m]


ref_raw = open(os.path.join(D, "reference.txt"), encoding="utf-8").read()
hyp_raw = open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read()
r, h = norm(ref_raw), norm(hyp_raw)
c = lev(list(r), list(h))
rw, hw = r.split(), h.split()
w = lev(rw, hw)
out = {
    "pair": "V1_C__cab_07_g3",
    "cer": round(c[0] / len(r), 4), "cer_edits": c[0], "cer_S": c[1], "cer_D": c[2], "cer_I": c[3],
    "ref_chars": len(r), "hyp_chars": len(h),
    "wer": round(w[0] / len(rw), 4), "wer_edits": w[0], "wer_S": w[1], "wer_D": w[2], "wer_I": w[3],
    "ref_words": len(rw), "hyp_words": len(hw),
    "scorepy_cer_raw": 0.4432, "scorepy_cer_fold": 0.2446,
    "normalised_reference": r, "normalised_hypothesis": h,
}
json.dump(out, open(os.path.join(D, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in out.items() if not k.startswith("normalised")}, indent=2))
