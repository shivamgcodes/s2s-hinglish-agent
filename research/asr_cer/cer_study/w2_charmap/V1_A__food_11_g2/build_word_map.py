# Word-level alignment, built as an ordered list of (n_ref, n_hyp, op, why) steps
# consumed against the normalised token streams. Cross-script sound judgements are manual.
import json
from norm import ref_words, hyp_words
R=ref_words(open('reference.txt').read()); H=hyp_words(open('whisper.txt').read())
M='match'; S='sub'; D='del'; I='ins'
steps=[]
def m(n,why='same word by sound'): steps.extend([(M,why)]*n)
m(5,'identical / same sound')
steps.append((S,"TiffinGo vs Tiffinbo: g->b, Whisper misheard brand name"))
m(3); m(7,'Hindi/English words transliterated; हेल्ब ~ help (b/p voicing variant only)'); m(1,'ji ~ जी')
# region where Whisper repeated "aapki kaise help kar sakta hoon" instead of the status sentence
steps += [(S,'ji vs आपकी: Whisper repeated earlier phrase'),(S,'the vs कैसे: repeated phrase'),(S,'order vs हेल्ब: repeated phrase')]
steps += [(D,'not transcribed (Whisper repeated earlier phrase instead)')]*7
steps.append((M,'kar ~ कर'))
steps.append((S,'deta vs सकता: different word (repeated phrase)'))
steps += [(D,'not transcribed')]*4
steps.append((M,'hoon ~ हूँ'))
steps.append((D,'yaa not transcribed'))
steps.append((M,'delivery ~ दिलिवरी'))
steps.append((S,'sahi vs से: different word'))
steps += [(M,'the ~ दा'),(M,'order ~ आर्डर')]
steps.append((S,'includes vs इंक्रूट्स (inkroots): l->r and d->t, mis-heard'))
m(3,'kashmiri biryani aur ~ कश्मीरी बिर्यानी और')
steps.append((I,'extra word बर्हानी (truncated/ghost of Burhalia)'))
steps += [(M,'ji ~ जी'),(M,'the ~ दा'),(M,'order ~ आर्डर')]
steps.append((S,'fd2130 vs एफटी: Whisper split code into "FT" + "2130"; D heard as T'))
steps.append((I,'2130 second half of the split fd2130'))
steps.append((S,'includes vs इंक्रूट्स: mis-heard'))
m(2,'kashmiri biryani'); steps += [(M,'burhalia ~ बर्हालिया'),(M,'raita ~ रेहिता (spelling variant)')]
m(6,'aur coca cola ji bilkul the ~ और कोका कोला जी बिल्कुल द')
m(3,'coca cola is')
steps.append((S,'included vs इंप्रूडेड (impruded): mis-heard'))
m(5,'in the delivery price of ~ इन द डिलिवरी प्राइस आफ')
steps.append((D,'rs: Whisper put "rupees" after the number (reordered)'))
steps.append((M,'720 ~ 720'))
steps.append((I,'रुपी = spoken "Rs" moved after the number'))
m(5,'ji the total fare is ~ जी द टोटल फैर इस')
steps.append((D,'rs reordered after number'))
steps.append((M,'720'))
steps.append((I,'रुपी = reordered Rs'))
m(4,'aur the order is')
steps.append((S,'updated vs अपिडेल (apidel): mis-heard'))
m(3,'in progress ka')
steps.append((I,'extra word है'))
m(5,'ji the order(~आउडर vowel variant) total is')
steps.append((D,'rs reordered after number'))
steps.append((M,'720'))
steps.append((I,'रुपिये = reordered Rs'))
m(3,'aur status abhi')
steps.append((S,'update vs अप्टेल (aptel): mis-heard'))
m(3,'kar deta(~दीता vowel variant) hoon')
m(9,'ji thank you for calling tiffingo have a good')
steps.append((S,'day vs देजी: Whisper merged "day ji" into one word'))
steps.append((D,'ji merged into देजी'))
steps += [(I,'hallucinated/garbled tail not in reference')]*12
ri=hi=0; rows=[]
for k,(op,why) in enumerate(steps):
    r = R[ri] if op in (M,S,D) else ''
    h = H[hi] if op in (M,S,I) else ''
    if op in (M,S,D): ri+=1
    if op in (M,S,I): hi+=1
    rows.append({'i':k,'ref':r,'hyp':h,'op':op,'why':why})
assert ' '.join(x['ref'] for x in rows if x['ref'])==' '.join(R), (ri,len(R))
assert ' '.join(x['hyp'] for x in rows if x['hyp'])==' '.join(H), (hi,len(H))
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
c={o:sum(x['op']==o for x in rows) for o in (M,S,D,I)}
met={'ref_words':len(R),'hyp_words':len(H),**c,'errors':c[S]+c[D]+c[I],'wer':round((c[S]+c[D]+c[I])/len(R),4)}
json.dump(met,open('word_metrics.json','w'),indent=1)
with open('word_map.txt','w') as f:
    f.write(f"{'#':>3}  {'ref':<12} | {'hyp':<14} | op\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['ref'] or '-':<12} | {x['hyp'] or '-':<14} | {x['op']}\n")
    f.write(json.dumps(met)+'\n')
print(met)
