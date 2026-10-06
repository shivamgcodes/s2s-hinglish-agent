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
                best = prev[j-1]
            else:
                c, s, d, ins = prev[j-1]; best = (c+1, s+1, d, ins)
            c, s, d, ins = prev[j]
            if c + 1 < best[0]: best = (c+1, s, d+1, ins)
            c, s, d, ins = cur[j-1]
            if c + 1 < best[0]: best = (c+1, s, d, ins+1)
            cur.append(best)
        prev = cur
    return prev[m]
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
cc, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
wc, ws, wd, wi = lev(rw, hw)
out = dict(pair='V1_B__air_16_g4', cer=round(cc/len(ref), 4), cer_S=cs, cer_D=cd, cer_I=ci, cer_dist=cc, ref_chars=len(ref), hyp_chars=len(hyp),
           wer=round(wc/len(rw), 4), wer_S=ws, wer_D=wd, wer_I=wi, wer_dist=wc, ref_words=len(rw), hyp_words=len(hw),
           ref_norm=ref, hyp_norm=hyp, scorepy_cer_raw=0.4904, scorepy_cer_fold=0.313)
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if not k.endswith('_norm')}, indent=1))
