"""Word-level cross-script alignment (Roman ref vs Devanagari Whisper) + WER.
Ops are hand-judged by sound; this script walks both token streams in order,
emits word_map.jsonl / word_map.txt / word_metrics.json, and verifies coverage."""
import json, unicodedata
def toks(s, lower):
    if lower: s = s.lower()
    # drop punctuation (deleted, so 'H-12' -> 'h12', consistent with char map normalisation)
    s = ''.join(ch for ch in s if not unicodedata.category(ch).startswith('P'))
    return s.split()
R = toks(open('reference.txt', encoding='utf-8').read(), True)
H = toks(open('whisper.txt', encoding='utf-8').read(), False)

# (op, why) in alignment order. M=match S=sub I=ins D=del
M, S, I = 'match', 'sub', 'ins'
FOR = (S, "'for' heard as 'par' (f->p, different vowel): different word")
THE = (M, "'the' spoken 'da/de' in Indian English: same word")
NUM_SPLIT = "spoken number split by Whisper into several words; first aligned as sub, rest ins"
plan = [
 (M,''),(M,''),(M,''),FOR,(S,"calling vs 'kaleen': -ing lost, different word"),
 (S,"bazaarly vs 'bejaarli': first vowel a->e"),(S,"'main' (I) vs 'mein' (in): different word/vowel"),
 (M,''),(S,"'main' (I) vs 'mein' (in)"),(M,''),(M,''),(M,''),(M,''),(M,''),
 (S,"hoon vs 'ho': nasal long u lost"),(M,''),THE,(M,''),(M,''),FOR,
 (S,"boAt vs 'but': vowel o->u"),(S,"rockerz vs 'rakars': o->a"),
 (S,"450 -> 'phor' ("+NUM_SPLIT+")"),(I,"'haido' (hundred?) part of 450"),(I,"'50' part of 450"),
 (S,"headphones vs 'hetphones': d->t"),(M,"dental d spelling of midnight"),(S,"black vs 'brack': l->r"),
 (M,''),(M,''),(M,''),THE,(M,''),(M,''),FOR,(S,"lokesh vs 'lokhish'"),(M,"spelling variant"),
 (S,"on vs 'un': vowel differs"),(M,''),
 (S,"98000 vs 'nine' ("+NUM_SPLIT+")"),(I,"'ek' ~ eight, part of 98000"),(I,"zero, part of 98000"),(I,"zero, part of 98000"),(I,"zero, part of 98000"),
 (S,"01299 vs 'zero' ("+NUM_SPLIT+")"),(I,"zero, part of 01299"),(I,"one, part of 01299"),(I,"two, part of 01299"),(I,"nine, part of 01299"),(I,"nine, part of 01299"),
 (M,''),THE,(M,''),(M,''),FOR,(S,"boAt vs 'bort': extra r"),(M,''),
 (S,"450 -> 'phor' ("+NUM_SPLIT+")"),(I,"'haido' part of 450"),(I,"'fifty' part of 450"),
 (S,"and vs 'en': d lost, vowel differs"),(M,''),(M,"shipped pronounced 'shipt'"),(S,"hain vs 'hai': nasal lost"),(M,''),
 THE,(M,''),(M,''),(S,"is vs 'ich': s->ch"),
 (S,"h12 vs 'ech' (h-twelve split into 2 words)"),(I,"'twelve' part of h12"),
 (S,"sector vs 'siktar': vowel e->i"),(M,"15 spoken 'fifteen': same sound, digits vs word"),(M,''),
 (S,"delhi vs 'dili': vowel differs"),(M,''),(M,''),(M,''),
 (M,''),(S,"remind vs 'mind': 're' lost"),(S,"me vs 'mein': nasal added, different word"),(S,"gaadi vs 'kaari': g->k, d->r"),(M,''),(M,''),
 (M,''),(M,"spelling variant"),THE,(M,''),(S,"is vs 'ich'"),(S,"scheduled vs 'shirul'"),FOR,(M,''),
 (S,"25 vs 'trinki' (twenty-five split)"),(I,"'five' part of 25"),(M,"Hindi spelling of October"),
 (M,''),(M,''),THE,(M,''),(S,"is vs 'ich'"),(S,"correct vs 'karek': final t lost"),(M,''),THE,(M,''),FOR,
 (S,"boAt vs 'boot': vowel o->oo"),(M,''),
 (S,"450 -> 'phor' ("+NUM_SPLIT+")"),(I,"'haido' part of 450"),(I,"'50' part of 450"),
 (M,''),(S,"friday vs 'praiday': f->p"),(M,"same digits"),(M,''),
 (M,''),(M,''),FOR,(S,"calling vs 'kuling': vowel differs"),(M,"z written ज (no nukta): same sound"),
 (M,''),(M,''),(M,"dental d spelling of good"),(M,''),(M,''),
] + [(I,"trailing Whisper hallucination, not in reference")]*9

rows=[]; ri=hi=0
for k,(op,why) in enumerate(plan):
    r = R[ri] if op in (M,S,'del') else ''
    h = H[hi] if op in (M,S,I) else ''
    if op in (M,S,'del'): ri+=1
    if op in (M,S,I): hi+=1
    rows.append({"i":k,"ref":r,"hyp":h,"op":op,"why":why})
assert ri==len(R) and hi==len(H), (ri,len(R),hi,len(H))
assert ' '.join(x['ref'] for x in rows if x['ref'])==' '.join(R)
assert ' '.join(x['hyp'] for x in rows if x['hyp'])==' '.join(H)
with open('word_map.jsonl','w',encoding='utf-8') as f:
    for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
with open('word_map.txt','w',encoding='utf-8') as f:
    f.write(f"{'i':>3}  {'ref':<14} | {'hyp':<16} | {'op':<5} | why\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['ref']:<14} | {x['hyp']:<16} | {x['op']:<5} | {x['why']}\n")
c={o:sum(x['op']==o for x in rows) for o in ('match','sub','del','ins')}
m={"pair":"V1_B__ecom_11_g1","ref_words":len(R),"hyp_words":len(H),**c,
   "errors":c['sub']+c['del']+c['ins'],"wer":round((c['sub']+c['del']+c['ins'])/len(R),4),
   "notes":"ref punctuation deleted (H-12 -> h12). Digit tokens matched 1-1 when spoken form equals one hyp word (15~फिफ्टीन, 25~25); multi-word spoken numbers scored sub+ins. 'for'->'पार' scored sub (7x). 9 trailing hallucinated words are ins."}
json.dump(m,open('word_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(json.dumps(m,ensure_ascii=False,indent=1))
