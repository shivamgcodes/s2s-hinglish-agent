import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))
def strip_punct(s):
    return ''.join(c if not unicodedata.category(c).startswith('P') else ' ' for c in s)
ref = strip_punct(open(f'{D}/reference.txt').read().lower()).split()
hyp = strip_punct(open(f'{D}/whisper.txt').read()).split()
# Hand-judged cross-script alignment (ref, hyp, op, why)
M='match';S='sub';X='del';I='ins'
steps = [
("hello","I",S,"'I' (from 'I will') does not sound like 'hello'"),
("","will",I,"extra word, Whisper heard 'hello' as 'I will'"),
("thank","thank",M,"same"),("you","you",M,"same"),("for","for",M,"same"),("calling","calling",M,"same"),
("ridenow","right",S,"brand 'RideNow' heard as 'right now': 1 ref word split into 2; first half 'right' != 'ridenow'"),
("","now",I,"second half of the split 'right now'"),
("this","This",M,"same"),("is","is",M,"same"),("divya","Divya",M,"same"),
("main","मैं",M,"main ~ मैं"),("aapki","आपकी",M,"aapki ~ आपकी"),("kaise","कैसे",M,"kaise ~ कैसे"),
("help","हिल्प",M,"help ~ हिल्प (vowel-variant transliteration of same word)"),
("kar","कर",M,"kar ~ कर"),("sakti","सकती",M,"sakti ~ सकती"),("hoon","हूँ",M,"hoon ~ हूँ"),
("ji","जी",M,"ji ~ जी"),("sorry","सॉरी",M,"sorry ~ सॉरी"),("lekin","लेकिन",M,"lekin ~ लेकिन"),
("this","this",M,"same"),("is","is",M,"same"),("only","only",M,"same"),("for","for",M,"same"),("ride","ride",M,"same"),
("rd4616","RB4616",S,"D vs B: different ID"),
("aapka","आपका",M,"aapka ~ आपका"),("address","address",M,"same"),("correct","correct",M,"same"),("set","सेट",M,"set ~ सेट"),
("kiji","जी",S,"'kiji' heard as 'ji' (syllable 'ki' lost)"),
("checking","टेकिंग",S,"checking vs 'teking': ch->t, different word"),
("just","जस्ट",M,"just ~ जस्ट"),
("a","असेक",S,"'a sec' merged into one token 'असेक'; aligned to 'a'"),
("sec","",X,"merged into previous hyp token 'असेक'"),
("ek","एक",M,"ek ~ एक"),("minute","मिना",S,"minute vs 'mina': final syllable lost"),
("done","डन",M,"done ~ डन"),("ji","जी",M,"ji ~ जी"),("the","दे",M,"the ~ दे (Indian pronunciation of 'the')"),
("pickup","पिक",S,"'pickup' split into 'पिक अप'; first half != whole word"),
("","अप",I,"second half of split 'पिक अप'"),
("is","इस",M,"is ~ इस"),("now","नाओ",M,"now ~ नाओ"),
("igi","आई",S,"'IGI' spelled as 3 letters 'आई जी आई'; 1 ref word -> 3 hyp words"),
("","जी",I,"letter 'G' of IGI"),("","आई",I,"letter 'I' of IGI"),
("airport","एरपोर्ट",M,"airport ~ एरपोर्ट"),("terminal","टेर्मिनल",M,"terminal ~ टेर्मिनल"),
("3","थ्री",M,"3 ~ थ्री (three)"),("delhi","दिली",M,"delhi ~ दिली (vowel-variant spelling)"),
("checking","टेकिंग",S,"checking vs 'teking'"),("just","जस्ट",M,"just ~ जस्ट"),
("a","आ",M,"a ~ आ"),("sec","सेक",M,"sec ~ सेक"),("ek","एक",M,"ek ~ एक"),
("minute","मिना",S,"minute vs 'mina'"),("ji","जी",M,"ji ~ जी"),("bilkul","बिलकुल",M,"bilkul ~ बिलकुल"),
("the","दर",S,"'the drop' heard as 'dar rop'; 'dar' != 'the'"),
("drop","रॉप",S,"'rop' missing initial d"),
("is","इस",M,"is ~ इस"),("updated","अपेटेर",S,"'apeter' != 'updated'"),
("aerocity","एरो",S,"'Aerocity' split into 'एरो सिटी'"),("","सिटी",I,"second half of split 'एरो सिटी'"),
("delhi","डिली",M,"delhi ~ डिली (variant spelling)"),
("building","बिल्बिंग",S,"'bilbing': d->b consonant change"),
("1","वन",M,"1 ~ वन (one)"),("2","दू",S,"'du' vs 'two': t->d consonant change"),
("ji","जी",M,"ji ~ जी"),("your","योर",M,"your ~ योर"),("driver","ड्राइवर",M,"driver ~ ड्राइवर"),("is","इस",M,"is ~ इस"),
("sandeep","संदी",S,"'sandi': final 'p' lost, different name"),
("aur","और",M,"aur ~ और"),("car","कार",M,"car ~ कार"),("is","इस",M,"is ~ इस"),
("white","भाइक",S,"'bhaik' != 'white'"),("maruti","मारोटी",M,"maruti ~ मारोटी (vowel variant)"),
("swift","स्विप",S,"'swip': final 'ft' -> 'p'"),
("dl","डी",S,"'DL' spelled as 2 letters 'डी एल'"),("","एल",I,"letter 'L' of DL"),
("01","जी",S,"'01' (zero one) heard as 'जी रोवा'"),("","रोवा",I,"rest of 'zero one' -> 'रोवा'"),
("koi","कोई",M,"koi ~ कोई"),("baat","बात",M,"baat ~ बात"),("nahi","नहीं",M,"nahi ~ नहीं"),("ji","जी",M,"ji ~ जी"),
("thank","Thank",M,"same"),("you","you",M,"same"),("for","for",M,"same"),("calling","calling",M,"same"),
("ridenow","right",S,"'RideNow' heard as 'right now'"),("","now",I,"second half of 'right now'"),
("","Jake",I,"hallucinated / no reference counterpart"),
("have","have",M,"same"),("a","a",M,"same"),("nice","nice",M,"same"),
("din","day",S,"Hindi 'din' vs English 'day' (same meaning, different word)"),
("","Take",I,"trailing hallucination"),("","you",I,"trailing hallucination"),("","take",I,"trailing hallucination"),
("","milte",I,"trailing hallucination"),("","Makes",I,"trailing hallucination"),("","us",I,"trailing hallucination"),
]
r=[s[0] for s in steps if s[0]]; h=[s[1] for s in steps if s[1]]
assert r==ref, ("ref mismatch", r, ref)
assert h==hyp, ("hyp mismatch", h, hyp)
for s in steps:
    op=s[2]
    assert (op==X)==(s[1]=='') and (op==I)==(s[0]=='')
c={k:sum(1 for s in steps if s[2]==k) for k in (M,S,X,I)}
with open(f'{D}/word_map.jsonl','w') as f:
    for i,s in enumerate(steps):
        f.write(json.dumps({"i":i,"ref":s[0],"hyp":s[1],"op":s[2],"why":s[3]},ensure_ascii=False)+"\n")
with open(f'{D}/word_map.txt','w') as f:
    f.write(f"{'i':>3}  {'ref':<12} | {'hyp':<12} | op\n")
    for i,s in enumerate(steps):
        f.write(f"{i:>3}  {s[0]:<12} | {s[1]:<12} | {s[2]}\n")
N=len(ref); err=c[S]+c[X]+c[I]
m={"pair":"V1_C__cab_07_g3","ref_words":N,"hyp_words":len(hyp),"match":c[M],"sub":c[S],"del":c[X],"ins":c[I],
   "errors":err,"wer":round(err/N,4),"checks":{"ref_concat_exact":True,"hyp_concat_exact":True}}
json.dump(m,open(f'{D}/word_metrics.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(m))
