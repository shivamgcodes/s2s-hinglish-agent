import json, os, re
D = os.path.dirname(os.path.abspath(__file__))

def norm(s):
    s = s.lower()
    s = ''.join(c if (c.isalpha() or c.isdigit() or c == "'") else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()

def lev(r, h):
    n, m = len(r), len(h)
    # dp cell: (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if r[i-1] == h[j-1]:
                best = prev[j-1]
            else:
                c = prev[j-1]; s = (c[0]+1, c[1]+1, c[2], c[3])
                c = prev[j];   d = (c[0]+1, c[1], c[2]+1, c[3])
                c = cur[j-1];  ins = (c[0]+1, c[1], c[2], c[3]+1)
                best = min(s, d, ins)
            cur.append(best)
        prev = cur
    return prev[m]

ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper_roman.txt'), encoding='utf-8').read())
c = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
w = lev(rw, hw)
out = {
    'cer': c[0] / len(ref), 'cer_S': c[1], 'cer_D': c[2], 'cer_I': c[3], 'ref_chars': len(ref),
    'wer': w[0] / len(rw), 'wer_S': w[1], 'wer_D': w[2], 'wer_I': w[3], 'ref_words': len(rw),
    'ref_norm': ref, 'hyp_norm': hyp,
}
json.dump(out, open(os.path.join(D, 'metrics.json'), 'w'), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
