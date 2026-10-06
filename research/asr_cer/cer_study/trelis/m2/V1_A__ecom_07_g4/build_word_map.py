#!/usr/bin/env python3
"""Word-level sound-aligned map for V1_A__ecom_07_g4 (derived from char_map.jsonl)."""
import json, os, collections
from build_map import norm
D = os.path.dirname(os.path.abspath(__file__))
REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read()).split()
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read()).split()

# ref word -> (hyp span, op, why) for non-1:1 / non-match cases, keyed by ref index
SPECIAL = {
    5:  (1, 'sub', 'shopkart vs शापकार: o~aa and final t missing (char map)'),
    36: (1, 'sub', 'boat vs बुथ्रॉकर्स (first half): o vs u, t vs th, oa not realised'),
    37: (0, 'del', 'rockerz merged into previous hyp word; z vs s'),
    38: (3, 'match', '450 spoken as four hundred fifty'),
    48: (1, 'sub', '25 vs twenty: units missing'),
    61: (5, 'sub', 'EC vs AC; 3147 spoken as three one four seven'),
    66: (2, 'sub', 'H vs is; 12 spoken as twelve'),
    68: (1, 'match', '15 spoken as fifteen'),
    70: (1, 'sub', 'delhi vs दिल्ली: e vs i, h vs geminate l'),
    82: (1, 'sub', 'registered vs register: final ed missing'),
    88: (1, 'sub', 'added vs aired: dd vs ir'),
    96: (1, 'sub', '25 vs twenty: units missing'),
    111: (1, 'sub', 'added vs aired: dd vs ir'),
    119: (1, 'sub', '25 vs twenty: units missing'),
}
rows, j = [], 0
for i, w in enumerate(REF):
    span, op, why = SPECIAL.get(i, (1, 'match', ''))
    h = ' '.join(HYP[j:j+span]); j += span
    if op == 'match' and not why and h != w:
        why = 'same sound, Devanagari' if h != 'g' else 'letter g pronounced jee = ji'
    rows.append({'i': len(rows), 'ref': w, 'hyp': h, 'op': op, 'why': why})
assert j == len(HYP), (j, len(HYP))
# check by concatenation
assert ' '.join(r['ref'] for r in rows if r['ref']).split() == REF
assert ' '.join(r['hyp'] for r in rows if r['hyp']).split() == HYP
for r in rows:
    if r['op'] == 'del': assert r['hyp'] == ''
    if r['op'] == 'ins': assert r['ref'] == ''
with open(f'{D}/word_map.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
c = collections.Counter(r['op'] for r in rows)
n = len(REF)
assert c['match'] + c['sub'] + c['del'] == n
m = {'pair': 'V1_A__ecom_07_g4', 'ref_words': n, 'hyp_words': len(HYP), 'match': c['match'], 'sub': c['sub'],
     'del': c['del'], 'ins': c['ins'], 'wer': round((c['sub'] + c['del'] + c['ins']) / n, 6), 'verified_concat': True}
json.dump(m, open(f'{D}/word_metrics.json', 'w'), indent=2)
with open(f'{D}/word_map.txt', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(f"{r['i']:>3}  {r['op']:<5}  {r['ref']:<12}  {r['hyp'] or '_':<24}  {r['why']}\n")
    f.write('\n' + json.dumps(m) + '\n')
print(json.dumps(m, indent=2))
