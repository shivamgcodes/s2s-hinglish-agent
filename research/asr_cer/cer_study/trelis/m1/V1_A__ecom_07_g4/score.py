import json, os, re

D = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower()
    s = "".join(c if (c.isalnum() or c == "'") else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def lev(r, h):
    # returns (S, D, I) with min total edits; backtrace
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            if r[i - 1] != h[j - 1]:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            Dl += 1
            i -= 1
        else:
            I += 1
            j -= 1
    assert S + Dl + I == dp[n][m]
    return S, Dl, I


ref = norm(open(os.path.join(D, "reference.txt"), encoding="utf-8").read())
hyp = norm(open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read())

cS, cD, cI = lev(ref, hyp)
rw, hw = ref.split(" "), hyp.split(" ")
wS, wD, wI = lev(rw, hw)
out = {
    "cer": (cS + cD + cI) / len(ref), "cer_S": cS, "cer_D": cD, "cer_I": cI, "ref_chars": len(ref),
    "wer": (wS + wD + wI) / len(rw), "wer_S": wS, "wer_D": wD, "wer_I": wI, "ref_words": len(rw),
    "ref_norm": ref, "hyp_norm": hyp,
}
json.dump(out, open(os.path.join(D, "metrics.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_norm")}, indent=2))
