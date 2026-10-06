#!/usr/bin/env python3
"""Apply manual (judgement) corrections to char_map_auto.jsonl, split insertions into
akshara units, verify concatenation invariants, write char_map.jsonl / .txt / char_metrics.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import norm_ref, norm_hyp, D

S = [json.loads(l) for l in open(f'{D}/char_map_auto.jsonl')]
ref = norm_ref(open(f'{D}/reference.txt').read())
hyp = norm_hyp(open(f'{D}/whisper.txt').read())

M = 'match'
ACC = [('a','अ',M,''),('c','क',M,''),('c','',M,'doubled c = single /k/'),
       ('o','ा',M,"'ou' in account = /au/; a-part"),('u','उ',M,"'ou' = /au/; u-part"),('n','ं',M,''),('t','ट',M,'')]
TH = "'th' of English 'the' is conventionally written द; h is part of the digraph"
RANGES = {
    (106,112): ACC, (514,520): ACC,
    (114,117): [('i','आई',M,'acronym ID: letter name /aai/'),('d','डी',M,'acronym ID: letter name /dee/')],
    (147,149): [('k','क',M,''),('o','','del','hyp की has no o of koi'),('i','ी',M,'')],
    (205,213): [('r','','del','Rs not spoken/heard before the number'),('s','','del','Rs not heard'),
                (' ','','del','space of dropped Rs'),('4','4',M,''),('9','9',M,''),('9','9',M,''),
                (' ',' ',M,''),('','डुलार ','ins','hyp adds currency word dollar after the number')],
    (214,216): [('a','इ','sub',"and ~ in: vowel a vs i"),('n','न',M,''),('d','','del','final d absent in hyp')],
    (257,260): [('w','भ','sub','w heard as bh'),('h','',M,'part of wh digraph'),('y','ाई',M,"y in 'why' = /ai/")],
    (309,310): [('i','आई',M,"pronoun I = /aai/")],
    (378,382): [('f','फ',M,''),('i','ि',M,"'ie' = one i vowel (length ignored)"),('e','',M,"part of 'ie' digraph"),
                ('l','ल',M,''),('d','','del','final d absent in hyp')],
    (421,423): [('v','व',M,''),('i','ाइ',M,"i in provide = /ai/")],
    (475,477): [('t','ब','sub','th (/d/) heard as b'),('h','',M,'part of th digraph'),('e','े',M,'')],
    (636,670): [('h','ह',M,''),('a','ै',M,"'a' in have ~ ै"),('v','व',M,''),('e','',M,'silent final e'),(' ',' ',M,''),
                ('a','आ',M,"article 'a' ~ आ"),(' ',' ',M,''),('n','न',M,''),('i','ाइ',M,"i in nice = /ai/"),
                ('c','स',M,"c = /s/"),('e','',M,'silent final e'),(' ',' ',M,''),('d','द',M,''),('i','ि',M,''),('n','न',M,''),
                ('',' ठीक एक मिनट मैं चेक करती हूँ','ins','hallucinated repeat of an earlier phrase after the final word')],
}
OPS = {
    31: ('sub', "tune heard as 'Tuna': e vs a"),
    124: ('sub', "phone vs फून: o vs uu vowel"),
    130: (M, 'anusvara before ब is pronounced m'),
    139: ('del', "bataiye vs बताये: i not present"),
    172: (M, 'doubled r in current = single /r/'),
    191: (M, "premium ~ प्रीमियम: य glide spelling of i-u"), 290: (M, "premium ~ प्रीमियम: य glide spelling of i-u"),
    247: (M, TH), 282: (M, TH), 354: (M, TH), 368: (M, TH), 428: (M, TH), 573: (M, TH),
    321: ('del', 'final d of understand absent in hyp'),
    393: ('sub', 'abhi vs आभी: short a vs long aa'),
    447: ('del', "final 'sh' of English absent in hyp (इंग्ली)"),
    461: ('sub', "thank vs थे: a vs e"),
    484: (M, 'doubled l = single /l/'),
    487: (M, "'ti' in -tion = /sh/ (श)"), 488: (M, "part of -tion /sh/"),
    500: (M, 'doubled t = single /t/'), 592: (M, 'doubled t = single /t/'),
    506: (M, "unstressed 'for' ~ फर (schwa)"),
    511: (M, "your ~ योर: ou = one vowel"),
    550: (M, 'ck = single /k/'),
    579: (M, "'qu' = /kw/, u ~ व"),
    597: ('sub', 'f vs p (for heard as पर)'), 614: ('sub', 'f vs p (for heard as पर)'),
    620: (M, 'doubled l = single /l/'),
    630: ('sub', "silent e of tune realised as अ (heard 'toon aplas')"),
}

def aksharas(s):
    """split a hyp string into units: space / latin / digit / Devanagari akshara (incl. conjuncts)"""
    units = []
    for k, c in enumerate(s):
        attach = units and (c in 'ािीुूेैोौॉृंँः़्' or (units[-1].endswith('्') and 'ऀ' <= c <= 'ॿ'))
        if attach: units[-1] += c
        else: units.append(c)
    return units

out = []; k = 0
starts = {a: (a, b) for (a, b) in RANGES}
while k < len(S):
    if k in starts:
        a, b = starts[k]
        orig_r = ''.join(S[x]['ref'] for x in range(a, b+1)); orig_h = ''.join(S[x]['hyp'] for x in range(a, b+1))
        new = RANGES[(a, b)]
        assert ''.join(n[0] for n in new) == orig_r, (a, orig_r)
        assert ''.join(n[1] for n in new) == orig_h, (a, orig_h, ''.join(n[1] for n in new))
        for r, h, op, why in new: out.append({'ref': r, 'hyp': h, 'op': op, 'why': why})
        k = b + 1; continue
    st = dict(S[k])
    if k in OPS: st['op'], st['why'] = OPS[k]
    elif st['op'] == 'match' and st['why'].startswith('near-equivalent'):
        st['why'] = st['why'].replace('near-equivalent', 'same sound (judged)')
    if st['op'] == 'ins' and st['hyp'] == ' ' and k not in OPS:
        st['why'] = 'hyp splits TunePlus into two words (extra space)'
    out.append({'ref': st['ref'], 'hyp': st['hyp'], 'op': st['op'], 'why': st['why']}); k += 1

# split insertions into akshara units so each inserted unit counts once
final = []
for st in out:
    if st['op'] == 'ins':
        for u in aksharas(st['hyp']): final.append({'ref': '', 'hyp': u, 'op': 'ins', 'why': st['why']})
    else: final.append(st)
assert all(not (s['op'] == 'ins' and s['hyp'] == '') for s in final)
for i, s in enumerate(final): s['i'] = i
final = [{'i': s['i'], 'ref': s['ref'], 'hyp': s['hyp'], 'op': s['op'], 'why': s['why']} for s in final]

assert ''.join(s['ref'] for s in final) == ref, 'ref concat mismatch'
assert ''.join(s['hyp'] for s in final) == hyp, 'hyp concat mismatch'
assert sum(1 for s in final if s['op'] != 'ins') == len(ref)

with open(f'{D}/char_map.jsonl', 'w') as f:
    for s in final: f.write(json.dumps(s, ensure_ascii=False) + '\n')

cnt = {o: sum(s['op'] == o for s in final) for o in ['match', 'sub', 'del', 'ins']}
N = len(ref)
cer = (cnt['sub'] + cnt['del'] + cnt['ins']) / N
metrics = {'pair': 'V1_A__sub_11_g3', 'ref_chars': N, 'hyp_chars': len(hyp), 'steps': len(final), **cnt,
           'errors': cnt['sub'] + cnt['del'] + cnt['ins'], 'cer': round(cer, 4),
           'notes': 'ins counted per hyp akshara/space unit; ref = lowercased, punctuation dropped, whitespace collapsed; '
                    'hyp = punctuation dropped, whitespace collapsed; alignment by sound (auto DP + manual judgement in finalize.py)',
           'concat_checks': {'ref_reproduced': True, 'hyp_reproduced': True}}
json.dump(metrics, open(f'{D}/char_metrics.json', 'w'), ensure_ascii=False, indent=2)

# human-readable: blocks of ~80 ref chars; each step padded to common width
mk = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
lines = [f"CER {cer:.4f}  ref_chars={N}  match={cnt['match']} sub={cnt['sub']} del={cnt['del']} ins={cnt['ins']}",
         "rows: REF / HYP / OP  (' ' match, S sub, D del, I ins; '·' = empty side)", '']
blk = []; nref = 0
def flush():
    r = h = o = ''
    for s in blk:
        rr = s['ref'] if s['ref'] else '·'
        hh = s['hyp'].replace(' ', '␣') if s['hyp'] else '·'
        rr = rr.replace(' ', '␣')
        w = max(len(rr), len(hh), 1)
        r += rr.ljust(w) + '|'; h += hh.ljust(w) + '|'; o += mk[s['op']].ljust(w) + '|'
    lines.extend(['REF ' + r, 'HYP ' + h, 'OP  ' + o, ''])
for s in final:
    blk.append(s)
    if s['ref']: nref += 1
    if nref >= 80 and s['ref'] == ' ': flush(); blk = []; nref = 0
if blk: flush()
open(f'{D}/char_map.txt', 'w').write('\n'.join(lines) + '\n')
print(json.dumps(metrics, ensure_ascii=False))
