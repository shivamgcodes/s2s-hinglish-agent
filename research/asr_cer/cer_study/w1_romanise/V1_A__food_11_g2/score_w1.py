import json, re, sys, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w']|_", " ", s)  # keep letters/digits/apostrophe
    return re.sub(r"\s+", " ", s).strip()
def lev(r, h):
    n, m = len(r), len(h)
    # dp cell = (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if r[i-1] == h[j-1]:
                c = prev[j-1]
            else:
                s = prev[j-1]; s = (s[0]+1, s[1]+1, s[2], s[3])
                d = prev[j];   d = (d[0]+1, d[1], d[2]+1, d[3])
                ins = cur[j-1]; ins = (ins[0]+1, ins[1], ins[2], ins[3]+1)
                c = min(s, d, ins, key=lambda t: t[0])
            cur.append(c)
        prev = cur
    return prev[m]
ref = norm(open(os.path.join(D, "reference.txt"), encoding="utf-8").read())
hyp = norm(open(os.path.join(D, "whisper_roman.txt"), encoding="utf-8").read())
ce, cs, cd, ci = lev(list(ref), list(hyp))
rw, hw = ref.split(), hyp.split()
we, ws, wd, wi = lev(rw, hw)
out = {"pair": "V1_A__food_11_g2",
       "cer": round(ce/len(ref), 4), "cer_edits": ce, "cer_S": cs, "cer_D": cd, "cer_I": ci, "ref_chars": len(ref), "hyp_chars": len(hyp),
       "wer": round(we/len(rw), 4), "wer_edits": we, "wer_S": ws, "wer_D": wd, "wer_I": wi, "ref_words": len(rw), "hyp_words": len(hw),
       "ref_normalised": ref, "hyp_normalised": hyp,
       "scorepy_cer_raw": 0.6312, "scorepy_cer_fold": 0.3632}
json.dump(out, open(os.path.join(D, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in out.items() if "normalised" not in k}, indent=1))
