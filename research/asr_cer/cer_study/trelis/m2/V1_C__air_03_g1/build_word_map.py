import json, unicodedata
def norm(s):
    s = s.lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return s.split()
ref = norm(open('reference.txt').read()); hyp = norm(open('whisper.txt').read())
cm = json.load(open('char_metrics.json'))
assert ref == cm['normalised_reference'].split() and hyp == cm['normalised_hypothesis'].split()
rows = []
def add(r, h, op, why):
    rows.append({"i": len(rows), "ref": r, "hyp": h, "op": op, "why": why})
# explicit (ref, hyp, op, why) sequence, aligned by sound per char_map.jsonl
SAME = 'same word'
TR = 'same sound, Devanagari spelling'
seq = []
def m(*ws):
    for w in ws: seq.append((w, w, 'match', SAME))
def t(r, h): seq.append((r, h, 'match', TR))
m('hello','thank','you','for')
seq.append(('calling','coming','sub','calling heard as coming (a dropped, ll->om)'))
seq.append(('udaan','उर्दांश','sub','udaan heard with extra r and sh (urdaansh)'))
seq.append(('air','dear','sub','air heard as dear (extra d, ai->ea)'))
m('this','is'); t('aditi','अदिति'); t('main','मैं'); t('aapki','आपकी'); t('kaise','कैसे'); m('help')
t('kar','कर'); t('sakti','सकती'); t('hoon','हूँ'); t('ji','जी')
m('no','problem','please','tell','me'); t('aapka','आपका'); m('booking','id'); t('ya','या'); m('pnr','number')
t('bataiye','बताइए'); t('ji','जी'); t('aapka','आपका'); m('flight','ua')
seq.append(('402','four hundred two','match','402 spoken as four hundred two, same number'))
m('is','from','delhi','to','mumbai','on')
seq.append(('22','twenty two','match','22 spoken as twenty two'))
m('october'); t('ji','जी'); m('your','gate','is')
seq.append(('15c','fifteen c','match','15c spoken as fifteen c'))
m('and','seat')
seq.append(('12f','twilling','sub','12f (twelve eff) heard as twilling: different sound'))
t('aur','और')
seq.append(('a','eat','sub','a heard as eat (extra e, t)'))
m('meal','preference','vegetarian'); t('hai','है'); t('ji','जी'); m('boarding','starts','at')
seq.append(('630','six thirty','match','6:30 spoken as six thirty'))
seq.append(('pm','पी','sub','pm heard as pee only (em missing)'))
t('aur','और'); m('flight','departure')
seq.append(('715','seven fifteen','match','7:15 spoken as seven fifteen'))
m('pm','from','terminal')
seq.append(('3','three','match','3 spoken as three'))
t('ji','जी'); m('the','current','terminal','is','terminal')
seq.append(('3','three','match','3 spoken as three'))
m('which','is','listed','on','your')
seq.append(('record','data','sub','record heard as data: different word'))
t('achha','अच्छा'); t('ji','जी'); m('thank','you','for','the','info')
seq.append(('linat','लेना था','sub','linat heard as lena tha (i->e, t->th, extra aa, split into two words)'))
t('hoon','हूँ'); t('ek','एक'); m('minute'); t('ji','जी'); m('have','a','safe','journey','thank','you','for','calling')
seq.append(('udaan','उदांशिये','sub','udaan air run together as udaanshiye (extra shi, air -> ye)'))
seq.append(('air','','del','air not separately present; only -ye merged into previous hyp word, r missing'))
for w in 'हैंक्स मेरी पुका हैंचे inter हैं सहेजी deal कर सकती हो'.split():
    seq.append(('', w, 'ins', 'trailing hallucinated speech not in reference'))
for s in seq: add(*s)
R = ' '.join(x['ref'] for x in rows if x['ref']).split(); H = ' '.join(x['hyp'] for x in rows if x['hyp']).split()
assert R == ref, 'ref mismatch'; assert H == hyp, 'hyp mismatch'
c = {k: sum(x['op'] == k for x in rows) for k in ('match','sub','del','ins')}
assert c['match']+c['sub']+c['del'] == len(ref)
mt = {"pair": "V1_C__air_03_g1", "ref_words": len(ref), "hyp_words": len(hyp), **c,
     "errors": c['sub']+c['del']+c['ins'],
     "wer": round((c['sub']+c['del']+c['ins'])/len(ref), 6), "ref_norm": ' '.join(ref), "hyp_norm": ' '.join(hyp)}
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False)+'\n')
json.dump(mt, open('word_metrics.json','w'), ensure_ascii=False, indent=2)
with open('word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'op':<5}  {'ref':<12}  {'hyp':<24}  why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['op']:<5}  {x['ref']:<12}  {x['hyp']:<24}  {x['why']}\n")
    f.write(f"\nref_words={len(ref)} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} WER={mt['wer']}\n")
print({k:v for k,v in mt.items() if 'norm' not in k})
