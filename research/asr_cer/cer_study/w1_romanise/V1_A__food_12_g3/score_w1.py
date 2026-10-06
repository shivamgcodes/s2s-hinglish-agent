import json, re, sys, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalnum() or c == "'") else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()
def lev(r, h):
    n, m = len(r), len(h)
    d = [[None]*(m+1) for _ in range(n+1)]
    d[0][0] = (0,0,0,0)
    for i in range(1,n+1): d[i][0] = (i,0,i,0)
    for j in range(1,m+1): d[0][j] = (j,0,0,j)
    for i in range(1,n+1):
        for j in range(1,m+1):
            if r[i-1] == h[j-1]:
                diag = d[i-1][j-1]
            else:
                c = d[i-1][j-1]; diag = (c[0]+1,c[1]+1,c[2],c[3])
            c = d[i-1][j]; dl = (c[0]+1,c[1],c[2]+1,c[3])
            c = d[i][j-1]; ins = (c[0]+1,c[1],c[2],c[3]+1)
            d[i][j] = min(diag, dl, ins, key=lambda t: t[0])
    return d[n][m]
ref = norm(open(os.path.join(D,'reference.txt'),encoding='utf-8').read())
hyp = norm(open(os.path.join(D,'whisper_roman.txt'),encoding='utf-8').read())
ce, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
we, ws, wd, wi = lev(rw, hw)
out = dict(pair='V1_A__food_12_g3', cer=round(ce/len(ref),4), cer_edits=ce, cer_S=cs, cer_D=cd, cer_I=ci,
           ref_chars=len(ref), hyp_chars=len(hyp),
           wer=round(we/len(rw),4), wer_edits=we, wer_S=ws, wer_D=wd, wer_I=wi,
           ref_words=len(rw), hyp_words=len(hw),
           ref_normalised=ref, hyp_normalised=hyp,
           scorepy_cer_raw=0.4426, scorepy_cer_fold=0.1917)
assert cs+cd+ci == ce and ws+wd+wi == we
json.dump(out, open(os.path.join(D,'metrics.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps({k:v for k,v in out.items() if 'normalised' not in k}, indent=1))
