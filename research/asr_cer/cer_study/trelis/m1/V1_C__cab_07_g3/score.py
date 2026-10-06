import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalpha() or c.isdigit() or c == "'") else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()
def lev(r, h):
    n, m = len(r), len(h)
    dp = [[None]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = (i, 0, i, 0)
    for j in range(m+1): dp[0][j] = (j, 0, 0, j)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]: c = dp[i-1][j-1]
            else:
                a = dp[i-1][j-1]; s = (a[0]+1, a[1]+1, a[2], a[3])
                b = dp[i-1][j];   d = (b[0]+1, b[1], b[2]+1, b[3])
                e = dp[i][j-1];   ins = (e[0]+1, e[1], e[2], e[3]+1)
                c = min(s, d, ins)
            dp[i][j] = c
    return dp[n][m]
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
ce, cs, cd, ci = lev(ref, hyp)
rw, hw = ref.split(), hyp.split()
we, ws, wd, wi = lev(rw, hw)
out = dict(cer=ce/len(ref), cer_S=cs, cer_D=cd, cer_I=ci, ref_chars=len(ref),
           wer=we/len(rw), wer_S=ws, wer_D=wd, wer_I=wi, ref_words=len(rw),
           ref_norm=ref, hyp_norm=hyp)
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w'), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
