#!/usr/bin/env python3
"""Word-level cross-script alignment map + WER for base_V1__cab_11_g4.
Alignment decisions (same sound?) are hand-judged and encoded in OVERRIDES;
everything else is an exact (case-insensitive) match walked in order.
Rule used: Devanagari rendering that differs only by a vowel shade (हव~have,
दे~day, राजश~rajesh, यो~you) = match; a dropped/added consonant or a
different word (aarti~aati, its~इस, i~im, he~hell) = sub."""
import json, unicodedata
def norm(s, low):
    s = s.lower() if low else s
    return ''.join(c for c in s if not unicodedata.category(c).startswith('P')).split()
R = norm(open('reference.txt', encoding='utf-8').read(), True)
H = norm(open('whisper.txt', encoding='utf-8').read(), False)
M=lambda w="same word, same script":("match",w)
steps = []  # (ref, hyp, op, why)
ri = hi = 0
def take(op, why, r=True, h=True):
    global ri, hi
    rw = R[ri] if r else ''; hw = H[hi] if h else ''
    if r: ri += 1
    if h: hi += 1
    steps.append((rw, hw, op, why))
# hand-judged special steps keyed by (ref index, hyp index)
SPECIAL = {
 (5,5):  [("sub","Whisper split 'citycab' into two words; 'city' vs 'citycab' differ as tokens",1,1),
          ("ins","second half of the split 'City Cab'",0,1)],
 (13,14):[("sub","'Aati' drops the r of 'Aarti' (consonant missing) -> different word",1,1)],
 (21,22):[("match","'it' (ref stream truncated It'll) ~ 'It'",1,1),
          ("ins","Whisper expanded It'll -> 'It will'; ref has no 'will' token",0,1)],
 (29,31):[("match","He' (He's) ~ 'He'",1,1),
          ("ins","Whisper wrote 'He is'; ref token was the truncated He'",0,1)],
 (36,39):[("del","'Okay' not transcribed by Whisper",1,0)],
 (39,41):[("sub","ref 'I'' -> 'i' vs hyp \"I'm\" -> 'im': different tokens",1,1)],
 (65,67):[("sub","ref He' -> 'he' vs hyp \"he'll\" -> 'hell': different tokens",1,1)],
 (70,72):[("sub","it's vs इस ('is'): t sound missing -> different word",1,1)],
 (71,73):[("match","rajesh ~ राजश (raajash; vowel shade, same name)",1,1)],
 (72,74):[("match","no ~ नो",1,1)],
 (73,75):[("match","problem ~ प्रॉब्लम",1,1)],
 (74,76):[("match","have ~ हव (vowel shade of English 'have')",1,1)],
 (75,77):[("del","'a' not present in Whisper's 'हव ग्रेट दे'",1,0)],
 (76,77):[("match","great ~ ग्रेट",1,1)],
 (77,78):[("match","day ~ दे (Devanagari rendering of 'day')",1,1)],
 (78,79):[("sub","aarti vs आती (aati): r sound missing -> different word",1,1)],
 (79,80):[("match","thank ~ थैंक",1,1)],
 (80,81):[("match","you ~ यू",1,1)],
 (81,82):[("match","you ~ यो (yo; vowel shade, same word)",1,1)],
 (82,83):[("match","welcome ~ वेलकम",1,1)],
 (83,84):[("match","have ~ हव",1,1)],
 (84,85):[("del","'a' not present in Whisper's 'हव ग्रेट दे'",1,0)],
 (85,85):[("match","great ~ ग्रेट",1,1)],
 (86,86):[("match","day ~ दे",1,1)],
}
while ri < len(R) or hi < len(H):
    if (ri, hi) in SPECIAL:
        for op, why, r, h in SPECIAL.pop((ri, hi)): take(op, why, r, h)
    elif ri < len(R) and hi < len(H) and R[ri] == H[hi].lower():
        take("match", "same word, same script")
    else:
        raise SystemExit(f"unhandled at ref {ri} {R[ri:ri+3]} hyp {hi} {H[hi:hi+3]}")
assert not SPECIAL, SPECIAL
assert ' '.join(s[0] for s in steps if s[0]) == ' '.join(R)
assert ' '.join(s[1] for s in steps if s[1]) == ' '.join(H)
with open('word_map.jsonl','w',encoding='utf-8') as f:
    for i,(r,h,op,why) in enumerate(steps):
        f.write(json.dumps({"i":i,"ref":r,"hyp":h,"op":op,"why":why},ensure_ascii=False)+"\n")
c = {k: sum(1 for s in steps if s[2]==k) for k in ("match","sub","del","ins")}
N = len(R); wer = (c["sub"]+c["del"]+c["ins"])/N
json.dump({**c,"ref_words":N,"hyp_words":len(H),"wer":round(wer,4),
  "note":"cross-script sound alignment; vowel-shade renderings = match, consonant drop/different token = sub"},
  open('word_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
with open('word_map.txt','w',encoding='utf-8') as f:
    f.write(f"{'i':>3}  {'ref':<12} | {'hyp':<12} | op\n")
    for i,(r,h,op,why) in enumerate(steps): f.write(f"{i:>3}  {r or '-':<12} | {h or '-':<12} | {op}\n")
    f.write(f"\n{c}  ref_words={N}  WER={wer:.4f}\n")
print(c, N, len(H), round(wer,4))
