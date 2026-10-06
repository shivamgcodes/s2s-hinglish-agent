# Word-level cross-script alignment. Alignment decisions (same sound?) were made by
# hand-judgement and encoded below as a list of (ref_idx|None, hyp_idx|None, op, why).
import json
from norm import ref_words, hyp_words
R=ref_words(); H=hyp_words()
steps=[]
def M(r,h,why='same word by sound'): steps.append((r,h,'match',why))
def S(r,h,why): steps.append((r,h,'sub',why))
def D(r,why): steps.append((r,None,'del',why))
def I(h,why): steps.append((None,h,'ins',why))
sub1={5:'brand ShopKart heard as शापकार (shaapkaar) - final t missing',
      35:"'for' heard as पर (par)",36:"'boAt' heard as बुथ (buth)",
      42:"'for' heard as पर (par)",48:'number 25 transcribed as 20',
      56:"'minute' heard as मिना (mina)"}
for k in range(0,53):
    if k in sub1: S(k,k,sub1[k])
    elif k==39: M(k,k,'headphones ~ हेडपून्स: same word, vowel-shifted rendering')
    elif k==46: M(k,k,'date ~ डियेट: same word, spelling variant')
    else: M(k,k)
D(53,"'a sec' merged by Whisper into one token आरसेक; 'a' deleted")
S(54,53,"'sec' ~ आरसेक (aarsek): merged with 'a', extra r sound")
M(55,54,'ek ~ एक'); S(56,55,sub1[56])
for k in range(57,73): D(k,'segment "done ji ... for delivery" (address update) absent from Whisper transcript')
for k,h in zip(range(73,85),range(56,68)):
    if k==82: S(k,h,"'registered' transcribed as 'register' (missing -ed)")
    else: M(k,h)
for k in range(85,93): D(k,'segment "the instruction is added: aapka street number bataiye" absent from Whisper')
for k,h in zip(range(93,100),range(68,75)):
    if k==96: S(k,h,'number 25 transcribed as 20')
    else: M(k,h,'same word by sound (दिलीवरी/देट/अक्तुबर are spelling variants)' if k in(93,94,97) else 'same word by sound')
for k in range(100,117): D(k,'repeated instruction segment + "aur" absent from Whisper')
M(117,75,'delivery ~ दिलीवरी'); M(118,76,'date ~ देट')
I(77,"Whisper repeats 'फ्राइडे' (Friday) not present in this reference sentence")
S(119,78,'number 25 transcribed as 20')
M(120,79,'october ~ अक्तुबर'); M(121,80,'ko ~ को'); M(122,81,'hai ~ है')
for k in range(123,126): D(k,'trailing "thank you for" (truncated reference) absent from Whisper')
rows=[]
for i,(r,h,op,why) in enumerate(steps):
    rows.append({'i':i,'ref':R[r] if r is not None else '','hyp':H[h] if h is not None else '','op':op,'why':why})
# checks
assert [x['ref'] for x in rows if x['ref']]==R, 'ref mismatch'
assert [x['hyp'] for x in rows if x['hyp']]==H, 'hyp mismatch'
assert ' '.join(x['ref'] for x in rows if x['ref'])==' '.join(R)
assert ' '.join(x['hyp'] for x in rows if x['hyp'])==' '.join(H)
c={o:sum(1 for x in rows if x['op']==o) for o in ('match','sub','del','ins')}
N=len(R); wer=(c['sub']+c['del']+c['ins'])/N
with open('word_map.jsonl','w',encoding='utf-8') as f:
    for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
json.dump({'ref_words':N,'hyp_words':len(H),**c,'wer':round(wer,4),
  'formula':'(sub+del+ins)/ref_words','concat_check':'passed'},open('word_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
with open('word_map.txt','w',encoding='utf-8') as f:
    f.write(f"{'i':>4}  {'ref':<14} | {'hyp':<14} | op\n")
    for x in rows: f.write(f"{x['i']:>4}  {x['ref']:<14} | {x['hyp']:<14} | {x['op']}\n")
print(c,N,len(H),round(wer,4))
