import json, re, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w']|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def lev(a, b):
    # returns (S, D, I) via full DP with backtrace
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j-1]+c, dp[i-1][j]+1, dp[i][j-1]+1)
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1):
            if a[i-1] != b[j-1]: S += 1
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    assert S + Dl + I == dp[n][m]
    return S, Dl, I
ref = norm(open(os.path.join(D, "reference.txt"), encoding="utf-8").read())
hyp = norm(open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read())
cS, cD, cI = lev(ref, hyp)
rw, hw = ref.split(" "), hyp.split(" ")
wS, wD, wI = lev(rw, hw)
out = dict(cer=(cS+cD+cI)/len(ref), wer=(wS+wD+wI)/len(rw), cer_S=cS, cer_D=cD, cer_I=cI,
           ref_chars=len(ref), wer_S=wS, wer_D=wD, wer_I=wI, ref_words=len(rw))
json.dump(out, open(os.path.join(D, "metrics.json"), "w"), indent=2)
print(ref); print(hyp); print(json.dumps(out, indent=2))
