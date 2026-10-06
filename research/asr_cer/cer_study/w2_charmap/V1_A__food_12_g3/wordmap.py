# Word-level cross-script alignment (judgement-coded ops) + WER.
# Policy: 'match' = hyp is a standard/plausible Devanagari (or Latin) rendering of the same spoken word
# (incl. Indian-accent conventions such as दा for 'the', कन for unstressed 'can').
# 'sub' = a phoneme is changed so it is a different word / non-word (e.g. हेल्ब for help, b!=p).
import json
from tok import ref_words, hyp_words
R=ref_words(); H=hyp_words()
SUB={5:'Biteway heard as Deepay (brand misrecognised)',8:'Kavya -> Kava (y dropped)',12:'help -> हेल्ब (p heard as b)',
18:'order -> आदर (Hindi word "aadar", r lost)',19:'FD2243 -> एफ़टी + 2243 split; D heard as T, digits inserted separately',
20:'from -> पॉम (f/r lost)',21:'Behrouz -> बहरूस (z heard as s)',24:'for -> पर (different word)',30:'12 (twelve) -> ट्रेल',
31:'minutes -> में (different word)',37:'at -> आद (t heard as d)',38:'English "main" (gate) -> Hindi में',43:'set -> सित (vowel change, "sit")',
46:'please -> प्रीज (l heard as r)',47:'check -> ठेक (ch heard as th)',56:'update -> अपेट (truncated)',57:'it -> एक (different word)',
62:'now -> न (truncated)',66:'update -> अप्तेज (garbled)',69:'hoon -> हो (different word ho)',75:'updated -> अपेटेड (d lost)',
76:'ring -> प्रिंग (extra p)',77:'bell -> बिल (bill)',81:'personalna -> परसना (syllable lost)',91:'theek -> दहिक (garbled)',
95:'"the request" merged into दरिक्विस्ट; aligned to request',97:'submitted -> हरमट',102:'calling -> कोरिंग (l heard as r)',
103:'Biteway split into बिट + पे; aligned to बिट',108:'end -> इन (different word)'}
DEL={94:'"the" merged into following hyp word दरिक्विस्ट'}
INS_AFTER={19:'2243 split off from FD2243',103:'पे = second half of Biteway split'}
WHY_M={33:'accented "the"',50:'accented "the"',72:'accented "the"',55:'unstressed can /kən/',70:'done /dʌn/',85:'muskurein ~ मुस्कुरेन nasal spelling variant',
88:'bilkul spelling variant',28:'same Latin token'}
rows=[];h=0
for i,r in enumerate(R):
    if i in DEL: rows.append(dict(ref=r,hyp='',op='del',why=DEL[i])); continue
    op='sub' if i in SUB else 'match'
    rows.append(dict(ref=r,hyp=H[h],op=op,why=SUB.get(i,WHY_M.get(i,'same word by sound'))));h+=1
    if i in INS_AFTER: rows.append(dict(ref='',hyp=H[h],op='ins',why=INS_AFTER[i]));h+=1
assert h==len(H),(h,len(H))
assert ' '.join(x['ref'] for x in rows if x['ref'])==' '.join(R)
assert ' '.join(x['hyp'] for x in rows if x['hyp'])==' '.join(H)
with open('word_map.jsonl','w',encoding='utf-8') as f:
    for k,x in enumerate(rows): f.write(json.dumps(dict(i=k,**x),ensure_ascii=False)+'\n')
c={o:sum(x['op']==o for x in rows) for o in ['match','sub','del','ins']}
N=len(R);wer=(c['sub']+c['del']+c['ins'])/N
json.dump(dict(**c,ref_words=N,hyp_words=len(H),wer=round(wer,4)),open('word_metrics.json','w'),indent=2)
with open('word_map.txt','w',encoding='utf-8') as f:
    f.write(f"{'i':>3}  {'ref':<14} | {'hyp':<14} | op\n")
    for k,x in enumerate(rows): f.write(f"{k:>3}  {x['ref']:<14} | {x['hyp']:<14} | {x['op']}\n")
print(c,N,len(H),wer)
