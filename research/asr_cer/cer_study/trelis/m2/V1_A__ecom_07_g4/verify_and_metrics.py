#!/usr/bin/env python3
import json, os, collections
from build_map import norm  # noqa
D = os.path.dirname(os.path.abspath(__file__))
REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read())
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read())
rows = [json.loads(l) for l in open(f'{D}/char_map.jsonl', encoding='utf-8')]
r = ''.join(x['ref'] for x in rows); h = ''.join(x['hyp'] for x in rows)
assert r == REF, 'ref concat mismatch'
assert h == HYP, 'hyp concat mismatch'
for x in rows:
    assert x['op'] in ('match','sub','del','ins')
    if x['op'] == 'ins': assert x['ref'] == '' and x['hyp']
    else: assert len(x['ref']) == 1
    if x['op'] == 'del': assert x['hyp'] == ''
c = collections.Counter(x['op'] for x in rows)
n = len(REF)
assert c['match'] + c['sub'] + c['del'] == n
m = {'pair': 'V1_A__ecom_07_g4', 'ref_chars': n, 'hyp_len': len(HYP), 'match': c['match'], 'sub': c['sub'],
     'del': c['del'], 'ins': c['ins'], 'cer': round((c['sub']+c['del']+c['ins'])/n, 6), 'verified_concat': True}
json.dump(m, open(f'{D}/char_metrics.json', 'w'), indent=2)
with open(f'{D}/char_map.txt', 'w', encoding='utf-8') as f:
    for x in rows:
        if x['op'] != 'match' or x['why']:
            pass
    W = 60
    for s in range(0, len(rows), W):
        blk = rows[s:s+W]
        f.write('ref: ' + '|'.join(x['ref'] or '_' for x in blk) + '\n')
        f.write('hyp: ' + '|'.join(x['hyp'] or '_' for x in blk) + '\n')
        f.write('op : ' + '|'.join({'match':'.','sub':'S','del':'D','ins':'I'}[x['op']] for x in blk) + '\n\n')
    f.write('NON-MATCH STEPS:\n')
    for x in rows:
        if x['op'] != 'match':
            f.write(f"{x['i']}\t{x['op']}\tref={x['ref']!r}\thyp={x['hyp']!r}\t{x['why']}\n")
print(json.dumps(m, indent=2))
