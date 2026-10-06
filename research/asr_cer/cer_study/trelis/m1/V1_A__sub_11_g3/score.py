import json, re, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalnum() or c == "'") else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()
def lev(a, b):
    # returns (dist, S, D, I) with backtrace
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j-1] + (a[i-1] != b[j-1]), dp[i-1][j] + 1, dp[i][j-1] + 1)
    i, j, S, Dd, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            S += a[i-1] != b[j-1]; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            Dd += 1; i -= 1
        else:
            I += 1; j -= 1
    return dp[n][m], S, Dd, I
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
cd, cS, cD, cI = lev(ref, hyp)
rw, hw = ref.split(), hyp.split()
wd, wS, wD, wI = lev(rw, hw)
out = dict(cer=cd/len(ref), cer_S=cS, cer_D=cD, cer_I=cI, ref_chars=len(ref),
           wer=wd/len(rw), wer_S=wS, wer_D=wD, wer_I=wI, ref_words=len(rw))
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w'), indent=2)
print(json.dumps(out, indent=2))
