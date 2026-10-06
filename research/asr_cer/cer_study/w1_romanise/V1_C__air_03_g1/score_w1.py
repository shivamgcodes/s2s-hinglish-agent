#!/usr/bin/env python3
"""W1 scorer: CER/WER between reference (model text stream) and romanised Whisper hypothesis."""
import json, os, re, sys

D = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower()
    s = "".join(c if (c.isalpha() or c.isdigit() or c == "'") else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def lev(ref, hyp):
    """Levenshtein with backtrace; returns (dist, S, D, I)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    assert S + Dl + I == dp[n][m]
    return dp[n][m], S, Dl, I


ref = norm(open(os.path.join(D, "reference.txt"), encoding="utf-8").read())
hyp = norm(open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read())

cd, cS, cD, cI = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
wd, wS, wD, wI = lev(rw, hw)

out = {
    "pair": "V1_C__air_03_g1",
    "cer": round(cd / len(ref), 4), "cer_dist": cd, "cer_S": cS, "cer_D": cD, "cer_I": cI,
    "ref_chars": len(ref), "hyp_chars": len(hyp),
    "wer": round(wd / len(rw), 4), "wer_dist": wd, "wer_S": wS, "wer_D": wD, "wer_I": wI,
    "ref_words": len(rw), "hyp_words": len(hw),
    "ref_norm": ref, "hyp_norm": hyp,
    "scorepy_cer_raw": 0.5787, "scorepy_cer_fold": 0.2818,
}
json.dump(out, open(os.path.join(D, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in out.items() if k not in ("ref_norm", "hyp_norm")}, indent=2))
