import json, re, sys, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalpha() or c.isdigit() or c == "'") else ' ' for c in s)
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
                c = prev[j-1]; best = (c[0]+1, c[1]+1, c[2], c[3])
            d = prev[j]; cand = (d[0]+1, d[1], d[2]+1, d[3])
            if cand[0] < best[0]: best = cand
            ins = cur[j-1]; cand = (ins[0]+1, ins[1], ins[2], ins[3]+1)
            if cand[0] < best[0]: best = cand
            cur.append(best)
        prev = cur
    return prev[m]
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
cc, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
wc, ws, wd, wi = lev(rw, hw)
out = dict(pair='base_V1__cab_11_g4', cer=round(cc/len(ref), 4), cer_edits=cc, cer_S=cs, cer_D=cd, cer_I=ci,
           ref_chars=len(ref), wer=round(wc/len(rw), 4), wer_edits=wc, wer_S=ws, wer_D=wd, wer_I=wi, ref_words=len(rw),
           ref_norm=ref, hyp_norm=hyp, scorepy_cer_raw=0.1157, scorepy_cer_fold=0.0592)
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
