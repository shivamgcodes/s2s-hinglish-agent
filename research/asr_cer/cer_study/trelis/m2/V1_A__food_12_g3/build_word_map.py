#!/usr/bin/env python3
"""Word-level sound alignment for V1_A__food_12_g3 (ref Roman Hinglish vs Trelis Whisper).
A ref word is 'match' only if every char of it is 'match' in char_map.jsonl; cross-checked below."""
import json, os, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return s.split()
ref = norm(open(os.path.join(HERE, 'reference.txt')).read())
hyp = norm(open(os.path.join(HERE, 'whisper.txt')).read())
cm = json.load(open(os.path.join(HERE, 'char_metrics.json')))
assert ref == cm['ref_normalised'].split() and hyp == cm['hyp_normalised'].split()

S = 'sub'; Mt = 'match'
X = [  # (ref, hyp, op, why)
 ('hello','hello',Mt,'same word'),('thank','thank',Mt,'same word'),('you','you',Mt,'same word'),('for','for',Mt,'same word'),('calling','calling',Mt,'same word'),
 ('biteway','डिपे',S,'brand name: b->ड, t->प, w absent; sound differs'),
 ('this','this',Mt,'same word'),('is','is',Mt,'same word'),
 ('kavya','कावा',S,'y of -vya not heard; sound differs'),
 ('main','मैं',Mt,'main = मैं, same sound'),('aapki','आपकी',Mt,'same sound across scripts'),('kaise','कैसे',Mt,'same sound across scripts'),
 ('help','help',Mt,'same word'),('kar','कर',Mt,'same sound across scripts'),('sakti','सकती',Mt,'same sound across scripts'),
 ('hoon','हूँ',Mt,'same sound across scripts'),('ji','जी',Mt,'same sound across scripts'),('your','your',Mt,'same word'),('order','order',Mt,'same word'),
 ('fd2243','if the two two four three',S,"letter F 'ef'~'if' and digits fine, but D heard as 'the'"),
 ('from','फॉर्म',S,"fro heard as for (metathesis): फॉर्म"),
 ('behrouz','बेहरूष',S,'z heard as sh (ष)'),
 ('is','is',Mt,'same word'),('out','out',Mt,'same word'),('for','for',Mt,'same word'),('delivery','delivery',Mt,'same word'),('and','and',Mt,'same word'),
 ('aapka','आपका',Mt,'same sound across scripts'),('eta','eta',Mt,'same word'),('abhi','अभी',Mt,'same sound across scripts'),
 ('12','trail',S,"12 ('twelve') heard as 'trail'"),
 ('minutes','में',S,'only m+n heard, rest of minutes absent'),
 ('ji','जी',Mt,'same sound across scripts'),('the','the',Mt,'same word'),('instruction','instruction',Mt,'same word'),('is','is',Mt,'same word'),('leave','leave',Mt,'same word'),
 ('at','out',S,'at heard as out'),
 ('main','main',Mt,'same word'),('gate','gate',Mt,'same word'),('which','which',Mt,'same word'),('is','is',Mt,'same word'),('currently','currently',Mt,'same word'),('set','set',Mt,'same word'),
 ('ji','जी',Mt,'same sound across scripts'),('bilkul','बिलकुल',Mt,'same sound across scripts'),('please','please',Mt,'same word'),
 ('check','take',S,'check heard as take (ch->t)'),
 ('and','and',Mt,'same word'),('confirm','confirm',Mt,'same word'),('the','the',Mt,'same word'),('correct','correct',Mt,'same word'),('instruction','instruction',Mt,'same word'),
 ('then','then',Mt,'same word'),('we','we',Mt,'same word'),('can','can',Mt,'same word'),('update','update',Mt,'same word'),('it','it',Mt,'same word'),
 ('let','let',Mt,'same word'),('me','me',Mt,'same word'),('check','check',Mt,'same word'),('just','just',Mt,'same word'),('now','now',Mt,'same word'),
 ('haan','हाँ',Mt,'same sound across scripts'),('ji','जी',Mt,'same sound across scripts'),('abhi','अभी',Mt,'same sound across scripts'),('update','update',Mt,'same word'),
 ('kar','कर',Mt,'same sound across scripts'),('deti','देती',Mt,'same sound across scripts'),('hoon','हूँ',Mt,'same sound across scripts'),
 ('done','धन',S,'d heard as aspirated dh (धन)'),
 ('ji','जी',Mt,'same sound across scripts'),('the','the',Mt,'same word'),('instruction','instruction',Mt,'same word'),('is','is',Mt,'same word'),('updated','updated',Mt,'same word'),
 ('ring','bring',S,'extra b: bring vs ring'),('bell','bill',S,'e heard as i'),('and','in',S,'and heard as in'),
 ('hand','hand',Mt,'same word'),('over','over',Mt,'same word'),
 ('personalna','persona',S,'-lna not heard'),
 ('theek','ठीक',Mt,'same sound across scripts'),('hai','है',Mt,'same sound across scripts'),('ya','या',Mt,'same sound across scripts'),
 ('muskurein','मुस्कुरेन',Mt,'same sound across scripts (ein = एन)'),('aapka','आपका',Mt,'same sound across scripts'),('delivery','delivery',Mt,'same word'),
 ('bilkul','बिल्कुल',Mt,'same sound across scripts'),('safe','safe',Mt,'same word'),('ho','हो',Mt,'same sound across scripts'),
 ('theek','दहिक',S,'th (ठ) heard as द + separate ह'),
 ('hai','है',Mt,'same sound across scripts'),('ji','जी',Mt,'same sound across scripts'),('the','the',Mt,'same word'),('request','request',Mt,'same word'),('is','is',Mt,'same word'),
 ('submitted','submit',S,'-ted ending not heard'),
 ('ji','जी',Mt,'same sound across scripts'),('thank','thank',Mt,'same word'),('you','you',Mt,'same word'),('for','for',Mt,'same word'),('calling','calling',Mt,'same word'),
 ('biteway','bitpay',S,'w heard as p, e absent'),
 ('have','have',Mt,'same word'),('a','a',Mt,'same word'),('nice','nice',Mt,'same word'),
 ('din','day',S,'din heard as day'),
 ('end','एन',S,'final d absent'),('song','song',Mt,'same word'),
 ('ji','जीत',S,"t of following 'thank' attached to ji (जीत): sound differs"),
 ('thank','है',S,"thank heard as है (t moved to previous word, a->ai, nk absent)"),
]
rows = [{"i": i, "ref": r, "hyp": h, "op": o, "why": w} for i, (r, h, o, w) in enumerate(X)]
R = ' '.join(x['ref'] for x in rows if x['ref']).split(); H = ' '.join(x['hyp'] for x in rows if x['hyp']).split()
assert R == ref, 'ref mismatch'; assert H == hyp, 'hyp mismatch'

# cross-check against char map: per ref word, all chars match <=> word match
steps = [json.loads(l) for l in open(os.path.join(HERE, 'char_map.jsonl'))]
words, cur, ok = [], '', True
for s in steps:
    if s['ref'] == ' ' or (s['ref'] == '' and s['hyp'] == ' '):
        if s['ref'] == ' ':
            words.append((cur, ok)); cur, ok = '', True
            if s['op'] != 'match': words[-1] = (words[-1][0], False); ok = False
        else:
            ok = False
        continue
    cur += s['ref']; ok = ok and s['op'] == 'match'
words.append((cur, ok))
assert [w for w, _ in words] == ref
for x, (w, okc) in zip(rows, words):
    assert (x['op'] == 'match') == okc, (x, okc)

c = {k: sum(x['op'] == k for x in rows) for k in ('match', 'sub', 'del', 'ins')}
m = {"pair": "V1_A__food_12_g3", "ref_words": len(ref), "hyp_words": len(hyp), **c,
     "wer": round((c['sub'] + c['del'] + c['ins']) / len(ref), 6), "ref_norm": ' '.join(ref), "hyp_norm": ' '.join(hyp),
     "concat_verified": True}
with open(os.path.join(HERE, 'word_map.jsonl'), 'w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
json.dump(m, open(os.path.join(HERE, 'word_metrics.json'), 'w'), ensure_ascii=False, indent=2)
with open(os.path.join(HERE, 'word_map.txt'), 'w') as f:
    f.write(f"{'i':>3}  {'op':<5}  {'ref':<12}  {'hyp':<26}  why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['op']:<5}  {x['ref']:<12}  {x['hyp']:<26}  {x['why']}\n")
    f.write(f"\nref_words={len(ref)} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} WER={m['wer']}\n")
print({k: v for k, v in m.items() if 'norm' not in k})
