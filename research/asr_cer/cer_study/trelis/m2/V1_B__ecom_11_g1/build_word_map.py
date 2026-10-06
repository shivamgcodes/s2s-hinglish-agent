#!/usr/bin/env python3
"""Word-level sound-aligned map for V1_B__ecom_11_g1 (derived from char_map.jsonl)."""
import json, os, collections
from align import normalise as norm
D = os.path.dirname(os.path.abspath(__file__))
REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read()).split()
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read()).split()

H450 = '450 vs four hydro fifty: spurious "hydro" (misheard hundred) inserted (char map)'
# ref index -> (hyp span, op, why) for non-default cases
SPECIAL = {
    5:  (1, 'sub', 'bazaarly vs बेजारली: a vs e (char map)'),
    6:  (1, 'sub', 'main vs में: ai vs e (char map)'),
    8:  (1, 'sub', 'main vs में: ai vs e (char map)'),
    14: (1, 'sub', 'hoon vs हो: second o and final n not heard (char map)'),
    20: (1, 'sub', 'boat vs boot: a vs o (char map)'),
    21: (1, 'sub', 'rockerz vs rockers: z vs s (char map)'),
    22: (3, 'sub', H450),
    32: (1, 'sub', 'for vs पर: f vs p, o missing (char map)'),
    36: (1, 'sub', 'phone heard as four (char map)'),
    37: (5, 'sub', '98000 vs nine एक zero zero zero: 8 transcribed as एक (one)'),
    38: (5, 'match', '01299 spoken as zero one two nine nine'),
    43: (1, 'sub', 'for vs पर: f vs p, o missing (char map)'),
    45: (1, 'sub', 'rockerz vs rockers: z vs s (char map)'),
    46: (3, 'sub', H450),
    50: (1, 'sub', 'hain vs है: final n not heard (char map)'),
    55: (0, 'del', '"is" not heard (char map: i deleted, s->h merged into hh)'),
    56: (2, 'sub', 'h12 vs hh twelve: extra h from "is" (s vs h)'),
    58: (1, 'match', '15 spoken as fifteen'),
    65: (1, 'sub', 'remind vs mind: re not heard (char map)'),
    66: (1, 'sub', 'me vs make: extra ak inserted (char map)'),
    67: (1, 'sub', 'gaadi ka ek merged into आधीकारीक: g missing, d vs ध, extra र, e vs ी'),
    68: (0, 'del', 'ka merged into previous hyp word आधीकारीक'),
    69: (0, 'del', 'ek merged into previous hyp word आधीकारीक'),
    78: (2, 'match', '25 spoken as twenty five'),
    90: (1, 'sub', 'boat vs boot: a vs o (char map)'),
    91: (1, 'sub', 'rockerz vs rockers: z vs s (char map)'),
    92: (3, 'sub', '450 vs for hydro fifty: for~four homophone, spurious "hydro" inserted (char map)'),
    95: (2, 'match', '25 spoken as twenty five'),
}
rows, j = [], 0
for i, w in enumerate(REF):
    span, op, why = SPECIAL.get(i, (1, 'match', ''))
    h = ' '.join(HYP[j:j+span]); j += span
    if op == 'match' and not why and h != w:
        why = 'same sound, Devanagari'
    rows.append({'i': len(rows), 'ref': w, 'hyp': h, 'op': op, 'why': why})
for h in HYP[j:]:
    rows.append({'i': len(rows), 'ref': '', 'hyp': h, 'op': 'ins',
                 'why': 'trailing hallucinated speech after the call ends (char map)'})
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
m = {'pair': 'V1_B__ecom_11_g1', 'ref_words': n, 'hyp_words': len(HYP), 'match': c['match'], 'sub': c['sub'],
     'del': c['del'], 'ins': c['ins'], 'wer': round((c['sub'] + c['del'] + c['ins']) / n, 6), 'verified_concat': True}
json.dump(m, open(f'{D}/word_metrics.json', 'w'), indent=2)
with open(f'{D}/word_map.txt', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(f"{r['i']:>3}  {r['op']:<5}  {r['ref'] or '_':<12}  {r['hyp'] or '_':<24}  {r['why']}\n")
    f.write('\n' + json.dumps(m) + '\n')
print(json.dumps(m, indent=2))
