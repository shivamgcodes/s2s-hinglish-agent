import json, unicodedata
def norm(s):
    s = s.lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return s.split()
ref = norm(open('reference.txt').read()); hyp = norm(open('whisper.txt').read())
cm = json.load(open('char_metrics.json'))
assert ref == cm['normalised_reference'].split() and hyp == cm['normalised_hypothesis'].split()
# Ordered alignment, hand-built from char_map.jsonl. (ref, hyp, op, why)
M = 'match'
A = [
 ('hello','haggle','sub',"hello heard as 'haggle' (char map: e/l/l/o subs + inserted e)"),
 ('thank','thank',M,'same word'),('you','you',M,'same word'),('for','for',M,'same word'),('calling','calling',M,'same word'),
 ('ridenow','right now','sub',"brand 'ridenow' heard as 'right now' (char map: d->t sub, e->space sub, extra gh)"),
 ('this','this',M,'same word'),('is','is',M,'same word'),
 ('divya','दिव्या',M,'divya = दिव्या, same sound'),
 ('main','मैं',M,'main = मैं, same sound'),
 ('aapki','आपकी',M,'aapki = आपकी, same sound'),
 ('kaise','कैसे',M,'kaise = कैसे, same sound'),
 ('help','help',M,'same word'),
 ('kar','कर',M,'kar = कर, same sound'),
 ('sakti','सकती',M,'sakti = सकती, same sound'),
 ('hoon','हो','sub','hoon (/huː~/) heard as हो (/hoː/): vowel differs, nasal missing'),
 ('ji','जी',M,'ji = जी, same sound'),
 ('sorry','sorry',M,'same word'),
 ('lekin','लेकिन',M,'lekin = लेकिन, same sound'),
 ('this','this',M,'same word'),('is','is',M,'same word'),('only','only',M,'same word'),('for','for',M,'same word'),('ride','ride',M,'same word'),
 ('rd4616','rb four six one six','sub','booking id letter D heard as B (digits spoken individually otherwise same)'),
 ('aapka','आपका',M,'aapka = आपका, same sound'),
 ('address','address',M,'same word'),('correct','correct',M,'same word'),
 ('set','सेठ','sub','set heard as सेठ: aspirated retroflex ठ vs plain t'),
 ('kiji','जी','sub',"kiji heard as just जी; 'ki' dropped"),
 ('checking','taking','sub',"checking heard as 'taking' (ch->t, e->a)"),
 ('just','just',M,'same word'),('a','a',M,'same word'),('sec','sec',M,'same word'),
 ('ek','एक',M,'ek = एक, same sound'),
 ('minute','minute',M,'same word'),('done','done',M,'same word'),
 ('ji','g',M,"ji written as letter 'g' (pronounced jee), same sound per char map"),
 ('the','the',M,'same word'),('pickup','pickup',M,'same word'),('is','is',M,'same word'),('now','now',M,'same word'),
 ('igi','igi',M,'same word'),('airport','airport',M,'same word'),('terminal','terminal',M,'same word'),
 ('3','three',M,'3 spoken as three'),
 ('delhi','दिल्ली','sub',"delhi vs दिल्ली: char map counts ref silent 'h' vs geminate ल as sub"),
 ('checking','taking','sub',"checking heard as 'taking' (ch->t, e->a)"),
 ('just','just',M,'same word'),('a','a',M,'same word'),
 ('sec','sick','sub',"sec heard as 'sick'"),
 ('ek','eight','sub',"ek heard as 'eight'"),
 ('minute','minute',M,'same word'),
 ('ji','जी',M,'ji = जी, same sound'),
 ('bilkul','बिलकुल',M,'bilkul = बिलकुल, same sound'),
 ('the','the',M,'same word'),('drop','drop',M,'same word'),('is','is',M,'same word'),
 ('updated','a pretty air','sub',"updated heard as 'a pretty air'"),
 ('aerocity','aerocity',M,'same word'),
 ('delhi','दिल्ली','sub',"delhi vs दिल्ली: char map counts ref silent 'h' vs geminate ल as sub"),
 ('building','building',M,'same word'),
 ('1','one',M,'1 spoken as one'),('2','two',M,'2 spoken as two'),
 ('ji','जी',M,'ji = जी, same sound'),
 ('your','your',M,'same word'),('driver','driver',M,'same word'),('is','is',M,'same word'),
 ('sandeep','संदीप',M,'sandeep = संदीप, same sound'),
 ('aur','our','sub',"aur heard as English 'our' (a->o)"),
 ('car','car',M,'same word'),('is','is',M,'same word'),
 ('white','bike','sub',"white heard as 'bike'"),
 ('maruti','maruti',M,'same word'),('swift','swift',M,'same word'),('dl','dl',M,'same word'),
 ('01','zero one',M,'01 spoken as zero one'),
 ('koi','कोई',M,'koi = कोई, same sound'),('baat','बात',M,'baat = बात, same sound'),('nahi','नहीं',M,'nahi = नहीं, same sound (per char map)'),
 ('ji','जी',M,'ji = जी, same sound'),
 ('thank','thank',M,'same word'),('you','you',M,'same word'),('for','for',M,'same word'),('calling','calling',M,'same word'),
 ('ridenow','right now','sub',"brand 'ridenow' heard as 'right now'"),
 ('','jake','ins',"extra word 'jake'"),
 ('have','have',M,'same word'),('a','a',M,'same word'),('nice','nice',M,'same word'),
 ('din','day','sub',"din heard as 'day'"),
] + [('',w,'ins','trailing hallucinated word') for w in ['take','you','take','मिलते','makes']]
rows = [{"i": k, "ref": r, "hyp": h, "op": op, "why": why} for k,(r,h,op,why) in enumerate(A)]
R = ' '.join(x['ref'] for x in rows if x['ref']).split(); H = ' '.join(x['hyp'] for x in rows if x['hyp']).split()
assert R == ref, 'ref mismatch'; assert H == hyp, 'hyp mismatch'
c = {k: sum(x['op'] == k for x in rows) for k in ('match','sub','del','ins')}
m = {"pair": "V1_C__cab_07_g3", "ref_words": len(ref), "hyp_words": len(hyp), **c,
     "wer": round((c['sub']+c['del']+c['ins'])/len(ref), 6), "ref_norm": ' '.join(ref), "hyp_norm": ' '.join(hyp)}
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False)+'\n')
json.dump(m, open('word_metrics.json','w'), ensure_ascii=False, indent=2)
with open('word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'op':<5}  {'ref':<12}  {'hyp':<24}  why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['op']:<5}  {x['ref']:<12}  {x['hyp']:<24}  {x['why']}\n")
    f.write(f"\nref_words={len(ref)} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} WER={m['wer']}\n")
print({k:v for k,v in m.items() if 'norm' not in k})
