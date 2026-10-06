import json, re, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w']|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def lev(r, h):
    n, m = len(r), len(h)
    d = [[None]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = (i, 0, i, 0)
    for j in range(m+1): d[0][j] = (j, 0, 0, j)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]; continue
            a = d[i-1][j-1]; b = d[i-1][j]; c = d[i][j-1]
            d[i][j] = min((a[0]+1, a[1]+1, a[2], a[3]),
                          (b[0]+1, b[1], b[2]+1, b[3]),
                          (c[0]+1, c[1], c[2], c[3]+1))
    return d[n][m]
ref = norm(open(os.path.join(D, "reference.txt"), encoding="utf-8").read())
hyp = norm(open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read())
ce, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
we, ws, wd, wi = lev(rw, hw)
out = dict(cer=ce/len(ref), cer_S=cs, cer_D=cd, cer_I=ci, ref_chars=len(ref),
           wer=we/len(rw), wer_S=ws, wer_D=wd, wer_I=wi, ref_words=len(rw),
           norm_reference=ref, norm_hypothesis=hyp)
json.dump(out, open(os.path.join(D, "metrics.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
