import json, unicodedata
PAIR = "V1_B__air_16_g4"
def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return s.split()
ref = norm(open('reference.txt').read()); hyp = norm(open('whisper.txt').read())
# cross-script / spoken-form equivalents judged same sound (per char_map.jsonl 'same sound' steps)
SAME = {('abhishek','अभिषेक'),('main','मैं'),('aapki','आपकी'),('kaise','कैसे'),('kar','कर'),('sakta','सकता'),
        ('hoon','हूँ'),('ji','जी'),('aapka','आपका'),('aur','और'),('hai','है'),('jyoti','ज्योति'),('woh','वह'),
        ('ki','कि'),('haan','हाँ'),('ek','एक'),('rukiye','रुकिए'),('koi','कोई'),('baat','बात')}
# ordered special alignments: (ref words, hyp words, op, why)
SPECIAL = [
 (['indus'], ['in','this'], 'sub', "indus heard as 'in this' (d->h, u->i, extra t/space per char map)"),
 (['airways'], ['airway'], 'sub', "final s dropped: airway"),
 (['bataiye'], ['बताएं'], 'sub', "bataiye vs बताएं: i dropped, extra nasal ं"),
 (['vs8910'], ['vs','eight','nine','one','z'], 'sub', "891 spoken as digits but final 0 heard as z"),
 (['402'], ['four','hundred','two'], 'match', "402 spoken as four hundred two"),
 (['is'], ['its'], 'sub', "is heard as it's (extra t)"),
 (['mumbai'], ['mubi'], 'sub', "m and a dropped: mubi"),
 (['22'], ['twenty','two'], 'match', "22 spoken as twenty two"),
 (['is'], ['it'], 'sub', "is heard as it (s->t)"),
 (['cancelled'], ['cancel'], 'sub', "-led dropped: cancel"),
 (['715'], ['seven','fifteen'], 'match', "7:15 spoken as seven fifteen"),
 (['mittal'], ['मितालसी'], 'sub', "mittal+sahi merged into मितालसी; one t dropped, sahi's 'ah' lost -> sound differs"),
 (['sahi'], [], 'del', "sahi absorbed into मितालसी (only सी heard), counted as deletion"),
 (['402'], ['four','hundred','two'], 'match', "402 spoken as four hundred two"),
 (['is'], ['it'], 'sub', "is heard as it (s->t)"),
 (['cancelled'], ['cancel'], 'sub', "-led dropped: cancel"),
 (['kijiye'], ['कै','जी'], 'sub', "kijiye heard as कै जी (i->ै, ye dropped, split)"),
 (['is'], ['age'], 'sub', "is heard as age"),
 (['402'], ['four','hundred','two'], 'match', "402 spoken as four hundred two"),
 (['3000'], ['three','thousand'], 'match', "3,000 spoken as three thousand"),
 (['a'], [], 'del', "a merged into इससे (a->इ), counted as deletion"),
 (['safe'], ['इससे'], 'sub', "a safe heard as इससे (f->स): sound differs"),
 (['nahi'], ['नहीं'], 'sub', "extra nasal ं in नहीं (char map ins)"),
]
rows = []; ri = hi = 0; si = 0
def add(r, h, op, why):
    global ri, hi
    rows.append({"i": len(rows), "ref": ' '.join(r), "hyp": ' '.join(h), "op": op, "why": why})
    ri += len(r); hi += len(h)
while ri < len(ref):
    if si < len(SPECIAL):
        R, H, op, why = SPECIAL[si]
        if ref[ri:ri+len(R)] == R and hyp[hi:hi+len(H)] == H and not (R == H):
            add(R, H, op, why); si += 1; continue
    r, h = ref[ri], hyp[hi] if hi < len(hyp) else None
    if r == h: add([r], [h], 'match', 'same word'); continue
    if (r, h) in SAME: add([r], [h], 'match', f'{r} = {h}, same sound (cross-script)'); continue
    raise SystemExit(f'unhandled ref[{ri}]={r!r} hyp[{hi}]={h!r} next special={SPECIAL[si] if si<len(SPECIAL) else None}')
assert si == len(SPECIAL), si
while hi < len(hyp):
    add([], [hyp[hi]], 'ins', 'trailing hallucinated Devanagari text, not in reference')
R = ' '.join(x['ref'] for x in rows if x['ref']).split(); H = ' '.join(x['hyp'] for x in rows if x['hyp']).split()
assert R == ref, 'ref mismatch'; assert H == hyp, 'hyp mismatch'
c = {k: sum(x['op'] == k for x in rows) for k in ('match','sub','del','ins')}
m = {"pair": PAIR, "ref_words": len(ref), "hyp_words": len(hyp), **c,
     "wer": round((c['sub']+c['del']+c['ins'])/len(ref), 6), "ref_norm": ' '.join(ref), "hyp_norm": ' '.join(hyp)}
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False)+'\n')
json.dump(m, open('word_metrics.json','w'), ensure_ascii=False, indent=2)
with open('word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'op':<5}  {'ref':<12}  {'hyp':<24}  why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['op']:<5}  {x['ref']:<12}  {x['hyp']:<24}  {x['why']}\n")
    f.write(f"\nref_words={len(ref)} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} WER={m['wer']}\n")
print({k:v for k,v in m.items() if 'norm' not in k})
