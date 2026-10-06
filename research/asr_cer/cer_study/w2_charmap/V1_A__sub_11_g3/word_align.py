#!/usr/bin/env python3
"""Word-level cross-script (Roman ref vs Devanagari/Latin Whisper hyp) alignment for V1_A__sub_11_g3.
Alignment decisions are manual sound-based judgements, encoded below; the script verifies
that the ref/hyp columns reproduce the normalised token sequences, then writes outputs.
Judgement rule: 'match' if hyp is a plausible spelling of the same pronunciation (vowel length,
dental/retroflex, Indian 'the'->'de', reduced final cluster nd->n / ld->l); 'sub' if a consonant
or vowel quality was misheard into a different sound (p->b, l->r, o->u, etc.) or a different word."""
import json, os, unicodedata

D = os.path.dirname(os.path.abspath(__file__))

def strip_punct(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('P'))

ref = strip_punct(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read().lower()).split()
hyp = strip_punct(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read()).split()

M, S, DEL, INS = 'match', 'sub', 'del', 'ins'
# (ref, hyp, op, why)
A = [
 ('hello','Hello',M,'same word'),('thank','thank',M,'same'),('you','you',M,'same'),('for','for',M,'same'),
 ('calling','calling',M,'same'),
 ('tuneplus','Tuna',S,'brand split by Whisper; first half misheard tune->Tuna'),
 ('','Plus',INS,'second half of split brand name'),
 ('this','This',M,'same'),('is','is',M,'same'),('simran','Simran',M,'same'),
 ('main','मैं',M,'main~मैं'),('aapki','आपकी',M,'same'),('kaise','कैसे',M,'same'),
 ('help','हेल्ब',S,'p heard as b (helb)'),('kar','कर',M,'same'),('sakti','सकती',M,'same'),('hoon','हूँ',M,'same'),
 ('ji','जी',M,'same'),('please','प्लीज',M,'please'),('aapka','आपका',M,'same'),('account','अकाउंट',M,'same'),
 ('id','आईडी',M,'I-D'),('ya','या',M,'same'),('phone','फून',S,'o heard as u (phoon)'),('number','नंबर',M,'same'),
 ('bataiye','बताये',M,'spelling variant'),('aur','और',M,'same'),('koi','की',S,'different word (ki)'),
 ('ji','जी',M,'same'),('pallavi','पलवी',M,'spelling variant of name'),('ji','जी',M,'same'),
 ('your','यूर',M,'your'),('current','करंट',M,'same'),('plan','प्लान',M,'same'),('is','इस',M,'is'),
 ('premium','प्रीमियम',M,'same'),('monthly','मंथली',M,'same'),('at','एट',M,'same'),
 ('rs','',DEL,'Rs not spoken before number; Whisper says dollar after it'),
 ('499','499',M,'same'),
 ('','डुलार',INS,'"dollar" inserted after 499 (currency read differently)'),
 ('and','इन',S,'and heard as in'),('status','स्टेटस',M,'same'),('active','एक्टिव',M,'same'),('hai','है',M,'same'),
 ('ji','जी',M,'same'),('please','प्लीज',M,'same'),('the','दे',M,'Indian-English "the"=de'),
 ('reason','वीजन',S,'r heard as v (veezan)'),('why','भाई',S,'different word (bhai)'),
 ('you','यू',M,'same'),('want','वांट',M,'same'),('to','तो',M,'to'),('cancel','कैंसल',M,'same'),
 ('the','द',M,'the'),('premium','प्रीमियम',M,'same'),('monthly','मंथली',M,'same'),('plan','प्लान',M,'same'),
 ('ji','जी',M,'same'),('i','आई',M,'I'),('understand','अंडरस्टेन',M,'reduced final -nd'),('your','यूर',M,'your'),
 ('problem','प्राबलम',M,'spelling variant'),('ji','जी',M,'same'),('bilkul','बिल्कुल',M,'same'),
 ('please','प्रीज',S,'l heard as r (preez)'),('the','दे',M,'the=de'),('reason','रीजन',M,'same'),('in','इन',M,'same'),
 ('the','दे',M,'the=de'),('reason','रीजन',M,'same'),('field','फिल',M,'reduced final -ld, short vowel'),
 ('aur','और',M,'same'),('main','मैं',M,'same'),('abhi','आभी',M,'vowel-length variant'),('theek','ठीक',M,'same'),
 ('hai','है',M,'same'),('ji','जी',M,'same'),('please','प्रीज',S,'l heard as r (preez)'),
 ('provide','प्रवाइद',M,'spelling variant'),('the','दे',M,'the=de'),('reason','रीजन',M,'same'),('in','इन',M,'same'),
 ('english','इंग्ली',S,'final -sh missing (ingli)'),('ji','जी',M,'same'),('bilkul','बिल्कुल',M,'same'),
 ('thank','थे',S,'truncated/misheard (the)'),('ji','जी',M,'same'),('bilkul','बिल्कुल',M,'same'),
 ('the','बे',S,'th heard as b (be)'),('cancellation','कैंसलेशन',M,'same'),('is','इज',M,'same'),
 ('submitted','सब्मिटेड',M,'same'),('for','फर',M,'for'),('your','योर',M,'your'),('account','अकाउंट',M,'same'),
 ('theek','ठीक',M,'same'),('hai','है',M,'same'),('ek','एक',M,'same'),('minute','मिनट',M,'same'),
 ('main','मैं',M,'same'),('check','चेक',M,'same'),('karti','करती',M,'same'),('hoon','हूँ',M,'same'),
 ('done','दन',M,'done ~ dun'),('ji','जी',M,'same'),
 ('the','दरिक्वेस्ट',S,'Whisper merged "the request" into one token; aligned to "the"'),
 ('request','',DEL,'absorbed into merged token दरिक्वेस्ट'),
 ('is','इद',S,'s heard as d (id)'),('submitted','सबमिटेड',M,'same'),('for','पर',S,'f heard as p (par)'),
 ('ji','जी',M,'same'),('thank','थैंक',M,'same'),('you','यू',M,'same'),('for','पर',S,'f heard as p (par)'),
 ('calling','पलिंग',S,'k heard as p (paling)'),
 ('tuneplus','टून',S,'brand split; "tune" part (counted sub since 1 ref word -> 2 hyp tokens)'),
 ('','अपलस',INS,'"a-plus" remainder of split brand'),
 ('have','हैव',M,'same'),('a','आ',M,'a'),('nice','नाइस',M,'same'),('din','दिन',M,'same'),
] + [('', w, INS, 'trailing Whisper hallucination: repeats "theek ek minute main check karti hoon"')
     for w in ['ठीक','एक','मिनट','मैं','चेक','करती','हूँ']]

r_rec = [a[0] for a in A if a[0]]
h_rec = [a[1] for a in A if a[1]]
assert r_rec == ref, ('ref mismatch', next((i, x, y) for i, (x, y) in enumerate(zip(r_rec, ref)) if x != y) if r_rec[:len(ref)] != ref[:len(r_rec)] else (len(r_rec), len(ref)))
assert h_rec == hyp, ('hyp mismatch', next(((i, x, y) for i, (x, y) in enumerate(zip(h_rec, hyp)) if x != y), (len(h_rec), len(hyp))))
for r, h, op, _ in A:
    assert (op == INS) == (r == '') and (op == DEL) == (h == '')

with open(os.path.join(D, 'word_map.jsonl'), 'w', encoding='utf-8') as f:
    for i, (r, h, op, why) in enumerate(A):
        f.write(json.dumps({'i': i, 'ref': r, 'hyp': h, 'op': op, 'why': why}, ensure_ascii=False) + '\n')

with open(os.path.join(D, 'word_map.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{'i':>4}  {'ref':<14} | {'hyp':<14} | op\n" + '-' * 44 + '\n')
    for i, (r, h, op, why) in enumerate(A):
        f.write(f"{i:>4}  {r or '-':<14} | {h or '-':<14} | {op}{'' if op == M else '   # ' + why}\n")

c = {k: sum(1 for a in A if a[2] == k) for k in (M, S, DEL, INS)}
n = len(ref)
wer = (c[S] + c[DEL] + c[INS]) / n
met = {'pair': 'V1_A__sub_11_g3', 'ref_words': n, 'hyp_words': len(hyp), 'steps': len(A),
       'match': c[M], 'sub': c[S], 'del': c[DEL], 'ins': c[INS], 'errors': c[S] + c[DEL] + c[INS],
       'wer': round(wer, 4),
       'wer_excluding_trailing_hallucination': round((c[S] + c[DEL] + c[INS] - 7) / n, 4),
       'notes': 'ref = lowercased, punctuation dropped; hyp = punctuation dropped; cross-script sound-based alignment, manual judgement in word_align.py. 7 of the insertions are a trailing Whisper repeat hallucination.',
       'concat_checks': {'ref_reproduced': r_rec == ref, 'hyp_reproduced': h_rec == hyp}}
json.dump(met, open(os.path.join(D, 'word_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(met, ensure_ascii=False, indent=2))
