import json, re, sys, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalnum() or c == "'") else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()
def lev(r, h):
    n, m = len(r), len(h)
    # dp of (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if r[i-1] == h[j-1]:
                cands = [prev[j-1]]
            else:
                c, s, d, ins = prev[j-1]; cands = [(c+1, s+1, d, ins)]
            c, s, d, ins = prev[j]; cands.append((c+1, s, d+1, ins))
            c, s, d, ins = cur[j-1]; cands.append((c+1, s, d, ins+1))
            cur.append(min(cands))
        prev = cur
    return prev[m]
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
cc, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
wc, ws, wd, wi = lev(rw, hw)
out = dict(pair='V1_A__ecom_07_g4', cer=round(cc/len(ref), 4), cer_edits=cc, cer_S=cs, cer_D=cd, cer_I=ci,
           ref_chars=len(ref), hyp_chars=len(hyp),
           wer=round(wc/len(rw), 4), wer_edits=wc, wer_S=ws, wer_D=wd, wer_I=wi,
           ref_words=len(rw), hyp_words=len(hw),
           ref_normalised=ref, hyp_normalised=hyp,
           scorepy_cer_raw=0.5166, scorepy_cer_fold=0.4122)
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w'), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in out.items() if 'normalised' not in k}, indent=1))
