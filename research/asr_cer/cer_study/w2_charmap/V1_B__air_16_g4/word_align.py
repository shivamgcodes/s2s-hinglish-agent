"""Word-level cross-script alignment for V1_B__air_16_g4.
Alignment steps are hand-decided (same-sound judgement); this script expands them,
verifies the ref/hyp concatenation invariants, and writes word_map.jsonl/.txt + word_metrics.json.
Step tuple: (n_ref, n_hyp, op, why) consumed sequentially from the normalised word lists."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_norm import ref_words, hyp_words
D = os.path.dirname(os.path.abspath(__file__))
R = ref_words(open(f'{D}/reference.txt').read()); H = hyp_words(open(f'{D}/whisper.txt').read())
M = lambda why='same word': (1, 1, 'match', why)
S = lambda why: (1, 1, 'sub', why)
Dl = lambda why: (1, 0, 'del', why)
I = lambda why: (0, 1, 'ins', why)
FP = 'f->p: Whisper heard a different consonant (plight/fram/paar), not a spelling variant'
IS = '"is" heard as ich/it (s->ch/t), different word'
steps = [M()]*5 + [
 S('"Indus" heard as "in" (+ inserted "this")'), I('extra word from mishearing "Indus" as "in this"'),
 S('"airways" -> "airway" (plural -s lost)'),
 M(), M(), M('abhishek, case only'),
 M('main ~ मैं'), M('aapki ~ आपकी'), M('kaise ~ कैसे'), M(), M('kar ~ कर'), M('sakta ~ सकता'), M('hoon ~ हूँ'),
 M('ji ~ जी'), M('ji ~ जी'), M(), M('aapka ~ आपका'), M(), M('id ~ ID'),
 S('bataiye vs बताएं (bataen): different verb form'),
 M('the ~ The'), M('id ~ ID'), M(), S('VS8910 vs VS891Z: last char 0->Z'),
 M('ji ~ जी'), M('your ~ योर'), S('flight ~ प्लाइट: '+FP),
 S('"in 402" merged by Whisper into IN-402 (punct dropped -> IN402); in->IN402 sub'), Dl('402 absorbed into merged IN402'),
 S(IS), S('from ~ प्रम: '+FP), M('delhi ~ दिली (gemination variant)'), M('to ~ तू'),
 S('mumbai ~ मूवी: different word'), M('on ~ ओन'), M(), M('november ~ नवंबर'),
 S('the ~ द्वा: distorted, not "the"'), M('status ~ स्टेटस'), S(IS),
 M('ji ~ जी'), M('the ~ दा (Devanagari convention for "the")'), S('flight ~ प्लाइट: '+FP), M('status ~ स्टेटस'), S(IS),
 S('cancelled ~ कैंसल: -ed suffix missing'),
 M('aur ~ और'), S('flight ~ प्लाइट: '+FP), M('time ~ टाइम'), M('7:15 ~ 7-15 (punct dropped)'),
 S('pm ~ पीम: "pee-em" collapsed to one syllable "peem"'), M('hai ~ है'),
 M('the ~ दा'), M('booking ~ बुकिंग'), S(IS), S('for ~ पार: '+FP),
 M('jyoti ~ जोती (spelling variant)'), M('mittal ~ मिताल (vowel-length variant)'), S('sahi ~ सी: different word'),
 M('ji ~ जी'), M('the ~ द'), M('reason ~ रीजन'), S('woh ~ वा: vowel o->aa, not a standard spelling of वो'), M('hai ~ है'),
 S('ki ~ जी: different word'), M('your ~ यूर'), S('flight ~ प्लाइट: '+FP),
 S('"in 402" merged into IN402; in->IN402 sub'), Dl('402 absorbed into merged IN402'),
 S('is ~ इट (it): different word'), S('cancelled ~ कैंसल: -ed suffix missing'),
 M('ji ~ जी'), M('haan ~ हाँ'), M('ji ~ जी'), M('please ~ प्लीज'), M('update ~ अपडेट'), M('the ~ द'), M('new ~ न्यू'),
 M('details ~ दिटेल्स (द/ड variant)'), M('aur ~ और'),
 S('resubmit split by Whisper into री + सब्मिट; resubmit->री sub'), I('सब्मिट: second half of split "resubmit"'),
 S('kijiye ~ के: Whisper split ki-ji-ye into के जी; kijiye->के sub'), I('जी: second half of split "kijiye"'),
 M('the ~ द'), M('booking ~ बुकिंग'), M('is ~ इज'), M('for ~ फार (old-style Devanagari for "for")'),
 S('"a man" merged into अमेन; a->अमेन sub'), Dl('man absorbed into merged अमेन'),
 M('with ~ विद'), M('ji ~ जी'), M('please ~ प्लीज'), M('update ~ अपडेट'), M('the ~ द'), M('details ~ दिटेल्स'),
 M('then ~ देन'), M('ek ~ एक'), M('minute ~ मिनट'), M('rukiye ~ रुकिये'),
 Dl('"done" not in Whisper'), M('ji ~ जी'),
] + [Dl('missing span: "the refund request ... 3,000 hai, ji" not transcribed by Whisper')]*14 + [
 M('you ~ यू'), M('are ~ आर'), M('welcome ~ वेलकम'),
 S('have ~ है (hai): different word'), Dl('"a" not in Whisper'), S('safe ~ इसे: different word'),
 M('koi ~ कोई'), M('baat ~ बात'), M('nahi ~ नहीं'),
] + [I('trailing hallucination "एक मिनट होके इसी फरमी ची" not in reference')]*6
ri = hi = 0; rows = []
for k, (nr, nh, op, why) in enumerate(steps):
    r = R[ri] if nr else ''; h = H[hi] if nh else ''
    ri += nr; hi += nh
    rows.append({'i': k, 'ref': r, 'hyp': h, 'op': op, 'why': why})
assert ' '.join(x['ref'] for x in rows if x['ref']) == ' '.join(R), 'ref mismatch'
assert ' '.join(x['hyp'] for x in rows if x['hyp']) == ' '.join(H), 'hyp mismatch'
with open(f'{D}/word_map.jsonl', 'w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
c = {o: sum(x['op'] == o for x in rows) for o in ('match', 'sub', 'del', 'ins')}
N = len(R); wer = (c['sub'] + c['del'] + c['ins']) / N
met = {'pair': 'V1_B__air_16_g4', 'ref_words': N, 'hyp_words': len(H), **c, 'errors': c['sub']+c['del']+c['ins'],
       'wer': round(wer, 4), 'concat_check': 'passed',
       'notes': 'Cross-script word match by sound. f->p, is->ich, cancelled->कैंसल, merges (IN402, अमेन) and splits (री सब्मिट, के जी) counted as errors.'}
json.dump(met, open(f'{D}/word_metrics.json', 'w'), ensure_ascii=False, indent=2)
w = max(len(x['ref']) for x in rows) + 2
with open(f'{D}/word_map.txt', 'w') as f:
    f.write(f"{'i':>4} | {'ref':<{w}} | {'hyp':<14} | op\n")
    for x in rows: f.write(f"{x['i']:>4} | {x['ref'] or '-':<{w}} | {x['hyp'] or '-':<14} | {x['op']}\n")
print(met)
