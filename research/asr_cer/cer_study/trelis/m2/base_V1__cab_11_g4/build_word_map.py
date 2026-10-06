import json, re, unicodedata
def norm(s):
    s = s.lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return s.split()
ref = norm(open('reference.txt').read()); hyp = norm(open('whisper.txt').read())
cm = json.load(open('char_metrics.json'))
assert ref == cm['ref_norm'].split() and hyp == cm['hyp_norm'].split()
# special alignments: (ref_word_index, ref, hyp, op, why); everything else 1:1 identical match
rows = []
ri = hi = 0
def add(r, h, op, why):
    global ri, hi
    rows.append({"i": len(rows), "ref": r, "hyp": h, "op": op, "why": why})
    ri += len(r.split()); hi += len(h.split())
special = {
  ('citycab',): ("city cab", "match", "compound written as two words, same sound"),
  ('aarti', 'आती'): ("आती", "sub", "r of aarti not heard (आती): sound differs"),
  ('rd5068',): ("rd five zero six eight", "match", "digits spoken individually, same sound"),
  ('take',): ("takes", "sub", "extra s: takes vs take"),
  ('12',): ("twelve", "match", "12 spoken as twelve"),
  ('rajesh', 'राजेश'): ("राजेश", "match", "rajesh = राजेश, same sound"),
  ('i', 'im'): ("im", "sub", "i' heard as i'm (extra m)"),
  ('4821',): ("forty two one", "sub", "4821 heard as forty two one (eight dropped, twenty->two)"),
  ('3',): ("three", "match", "3 spoken as three"),
  ('he', 'hell'): ("hell", "sub", "he' heard as he'll (extra ll)"),
  ('you', 'youre'): ("youre", "sub", "you heard as you're (extra re)"),
}
while ri < len(ref) or hi < len(hyp):
    r = ref[ri] if ri < len(ref) else None; h = hyp[hi] if hi < len(hyp) else None
    if (r,) in special:
        s = special[(r,)]; add(r, s[0], s[1], s[2]); continue
    if (r, h) in special:
        s = special[(r, h)]; add(r, s[0], s[1], s[2])
        continue
    if r == 'he' and h == 'he' and hyp[hi+1] == 'is' and ref[ri+1] == 'driving':
        add('he', 'he', 'match', 'same word'); add('', 'is', 'ins', "extra 'is' in hypothesis (ref has truncated he')"); continue
    if r == h:
        add(r, h, 'match', 'same word'); continue
    raise SystemExit(f'unhandled {ri} {r!r} {hi} {h!r}')

R = ' '.join(x['ref'] for x in rows if x['ref']).split(); H = ' '.join(x['hyp'] for x in rows if x['hyp']).split()
assert R == ref, 'ref mismatch'; assert H == hyp, 'hyp mismatch'
c = {k: sum(x['op'] == k for x in rows) for k in ('match','sub','del','ins')}
m = {"pair": "base_V1__cab_11_g4", "ref_words": len(ref), "hyp_words": len(hyp), **c,
     "wer": round((c['sub']+c['del']+c['ins'])/len(ref), 6), "ref_norm": ' '.join(ref), "hyp_norm": ' '.join(hyp)}
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False)+'\n')
json.dump(m, open('word_metrics.json','w'), ensure_ascii=False, indent=2)
with open('word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'op':<5}  {'ref':<12}  {'hyp':<24}  why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['op']:<5}  {x['ref']:<12}  {x['hyp']:<24}  {x['why']}\n")
    f.write(f"\nref_words={len(ref)} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} WER={m['wer']}\n")
print({k:v for k,v in m.items() if 'norm' not in k})
