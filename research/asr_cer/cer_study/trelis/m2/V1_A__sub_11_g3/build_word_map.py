#!/usr/bin/env python3
"""Word-level sound-aligned map for V1_A__sub_11_g3 (derived from char_map.jsonl)."""
import json, os, collections
from align import norm
D = os.path.dirname(os.path.abspath(__file__))
REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read()).split()
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read()).split()

# ref index -> (hyp span, op, why); ins rows are emitted via INS_AFTER (ref index after which hyp words are inserted)
SPECIAL = {
    5:   (2, 'sub', 'tuneplus vs tuna plus: e vs a (char map), split into two words'),
    26:  (1, 'sub', 'koi vs कि: o not heard'),
    28:  (1, 'sub', 'pallavi vs पलवी: geminate l lost'),
    37:  (0, 'del', 'rs not heard (currency spoken after number as dollar)'),
    38:  (4, 'match', '499 spoken as four hundred ninety nine'),
    39:  (1, 'sub', 'and vs in: a vs i, d not heard'),
    56:  (1, 'match', 'letter g pronounced jee = ji'),
    69:  (1, 'sub', 'field vs feel: i vs e, final d not heard'),
    84:  (0, 'del', 'thank not heard'),
    87:  (0, 'del', 'the merged into next hyp word as de'),
    88:  (1, 'sub', 'cancellation vs decancellation: extra de prefix (from the, th vs d)'),
    113: (1, 'sub', 'calling vs pulling: c vs p, a vs u'),
    114: (2, 'match', 'tuneplus spoken as tune plus, same sound split into two words'),
}
INS_AFTER = {38: (1, 'dollar has no counterpart in reference (ref has rs before the number)'),
             118: (7, 'repeated tail theek ek minute main check karti hoon not in reference')}
rows, j = [], 0
for i, w in enumerate(REF):
    span, op, why = SPECIAL.get(i, (1, 'match', ''))
    h = ' '.join(HYP[j:j+span]); j += span
    if op == 'match' and not why and h != w:
        why = 'same sound, Devanagari'
    rows.append({'i': len(rows), 'ref': w, 'hyp': h, 'op': op, 'why': why})
    if i in INS_AFTER:
        n, why = INS_AFTER[i]
        for k in range(n):
            rows.append({'i': len(rows), 'ref': '', 'hyp': HYP[j], 'op': 'ins', 'why': why}); j += 1
assert j == len(HYP), (j, len(HYP))
assert ' '.join(r['ref'] for r in rows if r['ref']).split() == REF
assert ' '.join(r['hyp'] for r in rows if r['hyp']).split() == HYP
for r in rows:
    if r['op'] == 'del': assert r['hyp'] == ''
    if r['op'] == 'ins': assert r['ref'] == ''
    if r['op'] == 'match' and r['hyp'] == r['ref']: assert not r['why']
with open(f'{D}/word_map.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
c = collections.Counter(r['op'] for r in rows)
n = len(REF)
assert c['match'] + c['sub'] + c['del'] == n
m = {'pair': 'V1_A__sub_11_g3', 'ref_words': n, 'hyp_words': len(HYP), 'match': c['match'], 'sub': c['sub'],
     'del': c['del'], 'ins': c['ins'], 'wer': round((c['sub'] + c['del'] + c['ins']) / n, 6), 'verified_concat': True}
json.dump(m, open(f'{D}/word_metrics.json', 'w'), indent=2)
with open(f'{D}/word_map.txt', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(f"{r['i']:>3}  {r['op']:<5}  {r['ref'] or '_':<12}  {r['hyp'] or '_':<24}  {r['why']}\n")
    f.write('\n' + json.dumps(m) + '\n')
print(json.dumps(m, indent=2))
