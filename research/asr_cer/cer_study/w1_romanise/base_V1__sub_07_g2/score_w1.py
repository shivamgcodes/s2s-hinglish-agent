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
            sub = r[i-1] != h[j-1]
            a = prev[j-1]; c1 = (a[0] + sub, a[1] + sub, a[2], a[3])
            b = prev[j];   c2 = (b[0] + 1, b[1], b[2] + 1, b[3])
            c = cur[j-1];  c3 = (c[0] + 1, c[1], c[2], c[3] + 1)
            cur.append(min(c1, c2, c3))
        prev = cur
    return prev[m]
ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
cc = lev(list(ref), list(hyp)); rw, hw = ref.split(), hyp.split(); wc = lev(rw, hw)
out = dict(pair='base_V1__sub_07_g2',
    cer=round(cc[0] / len(ref), 6), cer_edits=cc[0], cer_S=cc[1], cer_D=cc[2], cer_I=cc[3], ref_chars=len(ref), hyp_chars=len(hyp),
    wer=round(wc[0] / len(rw), 6), wer_edits=wc[0], wer_S=wc[1], wer_D=wc[2], wer_I=wc[3], ref_words=len(rw), hyp_words=len(hw),
    normalised_reference=ref, normalised_hypothesis=hyp,
    scorepy_cer_raw=0.04, scorepy_cer_fold=0.0466)
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w'), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if not k.startswith('normalised')}, indent=2))
