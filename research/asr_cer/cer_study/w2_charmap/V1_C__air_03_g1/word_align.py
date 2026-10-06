# Hand-judged cross-script word alignment (sound-based), then verified + scored programmatically.
import json
from tok import norm_ref, norm_hyp
R=norm_ref(open('reference.txt').read()); H=norm_hyp(open('whisper.txt').read())
M,S,D,I='match','sub','del','ins'
# spec: (op, why) consumed in order; match/sub consume 1 ref + 1 hyp, del 1 ref, ins 1 hyp
spec=[]
def add(op,why=''): spec.append((op,why))
for _ in range(7): add(D,'opening greeting "hello thank you for calling udaan air" absent from Whisper transcript')
for w in ['this','is','aditi']: add(M,'same word, same script')
add(M,'main ~ मैं'); add(M,'aapki ~ आपकी'); add(M,'kaise ~ कैसे'); add(M,'help ~ हेल्प'); add(M,'kar ~ कर'); add(M,'sakti ~ सकती'); add(M,'hoon ~ हूँ')
add(M,'ji ~ जी'); add(M,'no ~ नो'); add(M,'problem ~ प्रॉब्लम'); add(M,'please ~ प्लीज'); add(M,'tell ~ टेल'); add(M,'me ~ मी'); add(M,'aapka ~ आपका'); add(M,'booking ~ बुकिंग')
add(S,'id ~ आईरी: "ai-ri" vs "ai-dee", D heard as R -> different word')
add(M,'ya ~ या')
add(M,'pnr ~ पियानार: Whisper spelling of the letter sequence P-N-R (p,n,r in order); borderline, judged same spoken token')
add(M,'number ~ नमबर (spelling variant)'); add(M,'bataiye ~ बताये (spelling variant)')
add(M,'ji ~ जी'); add(M,'aapka ~ आपका')
add(S,'flight ~ प्लाइट: "plight", f heard as p')
add(S,'Whisper merged "UA 402" into one token UA402; ua aligned to it as sub')
add(D,'402 absorbed into merged hyp token UA402')
add(M,'is ~ इस'); add(S,'from ~ प्रॉम: "prom", f heard as p'); add(M,'delhi ~ देली (spelling variant)'); add(M,'to ~ तु (spelling variant)')
add(S,'mumbai ~ मुमी: "mumi", missing -bai'); add(S,'on ~ और: different word "aur"'); add(M,'22 ~ 22'); add(M,'october ~ अक्टोबर')
add(M,'ji ~ जी'); add(M,'your ~ योर'); add(M,'gate ~ गेट'); add(M,'is ~ इस')
add(S,'15c ~ 15: Whisper split 15C into "15" + "सी"; first half aligned as sub'); add(I,'सी = the "C" of 15C split off as separate token')
add(S,'and ~ एंट: "ent", d heard as t'); add(M,'seat ~ सीट'); add(S,'12f ~ थ्वेले: garbled "twelve", F missing')
add(M,'aur ~ और'); add(S,'a ~ इड: different sound'); add(M,'meal ~ मिल (vowel-length spelling variant)')
add(S,'preference ~ प्रेपरेंस: "preperence", f heard as p'); add(M,'vegetarian ~ विजिटारियन (spelling variant)'); add(M,'hai ~ है')
add(M,'ji ~ जी'); add(M,'boarding ~ बोर्डिंग'); add(M,'starts ~ स्टार्ट्स'); add(M,'at ~ एट'); add(M,'6 ~ 6'); add(M,'30 ~ 30'); add(S,'pm ~ पी: only "P", M missing')
add(M,'aur ~ और'); add(S,'flight ~ प्लाइट: "plight", f heard as p'); add(M,'departure ~ डेपार्चर'); add(M,'7 ~ 7'); add(M,'15 ~ 15')
add(S,'pm ~ पी: Whisper split P M into two tokens; first half aligned as sub'); add(I,'एम = the "M" of PM split off as separate token')
add(S,'from ~ थ्रॉंग: "throng", different word'); add(M,'terminal ~ टर्मिनल'); add(M,'3 ~ 3')
add(M,'ji ~ जी'); add(M,'the ~ दा (spelling variant of "the")'); add(M,'current ~ करंट'); add(M,'terminal ~ टर्मिनल'); add(M,'is ~ इस'); add(M,'terminal ~ टर्मिनल')
add(S,'3 ~ थी: "thee", r missing; different word ("thi")'); add(M,'which ~ विच'); add(M,'is ~ इस'); add(M,'listed ~ लिस्टेड'); add(M,'on ~ ओन'); add(M,'your ~ यूर'); add(S,'record ~ एकर: "ekar", different word')
add(M,'achha ~ अच्छा'); add(M,'ji ~ जी'); add(M,'thank ~ थैंक'); add(M,'you ~ यू'); add(M,'for ~ फॉर'); add(M,'the ~ दे (spelling variant of "the")'); add(M,'info ~ इंफो')
add(S,'linat ~ लेना: "lena", different word'); add(S,'hoon ~ थूँ: "thoon", th instead of h'); add(M,'ek ~ एक'); add(M,'minute ~ मिनट')
add(M,'ji ~ जी'); add(M,'have ~ हैव'); add(D,'"a safe" merged by Whisper into one token असेफ; article "a" deleted'); add(S,'safe ~ असेफ: merged "a safe" -> "asef"'); add(S,'journey ~ चर्णी: "charni", j heard as ch, different word')
add(M,'thank ~ थैंक'); add(M,'you ~ यू'); add(M,'for ~ फॉर'); add(M,'calling ~ कोलिंग (vowel spelling variant)'); add(M,'udaan ~ उदान'); add(S,'air ~ चिये: different sound')
for _ in range(10): add(I,'trailing hallucinated/garbled Whisper text with no reference counterpart')
ri=hi=0; rows=[]
for k,(op,why) in enumerate(spec):
    r=h=''
    if op in (M,S,D): r=R[ri]; ri+=1
    if op in (M,S,I): h=H[hi]; hi+=1
    rows.append({'i':k,'ref':r,'hyp':h,'op':op,'why':why})
assert ri==len(R) and hi==len(H), (ri,len(R),hi,len(H))
assert ' '.join(x['ref'] for x in rows if x['ref'])==' '.join(R)
assert ' '.join(x['hyp'] for x in rows if x['hyp'])==' '.join(H)
with open('word_map.jsonl','w') as f:
    for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
c={o:sum(x['op']==o for x in rows) for o in (M,S,D,I)}
wer=(c[S]+c[D]+c[I])/len(R)
met={'ref_words':len(R),'hyp_words':len(H),**c,'wer':round(wer,4),'concat_check':'pass'}
json.dump(met,open('word_metrics.json','w'),indent=1,ensure_ascii=False)
with open('word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'ref':<14}| {'hyp':<14}| op\n")
    for x in rows: f.write(f"{x['i']:>3}  {x['ref']:<14}| {x['hyp']:<14}| {x['op']}\n")
print(met)
