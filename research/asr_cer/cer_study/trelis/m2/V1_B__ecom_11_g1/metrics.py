#!/usr/bin/env python3
"""Verify char_map.jsonl reconstructs the normalised texts exactly; write char_metrics.json and char_map.txt."""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
norm = json.load(open(os.path.join(HERE, 'normalised.json'), encoding='utf-8'))
steps = [json.loads(l) for l in open(os.path.join(HERE, 'char_map.jsonl'), encoding='utf-8')]

ref_cat = ''.join(s['ref'] for s in steps)
hyp_cat = ''.join(s['hyp'] for s in steps)
assert ref_cat == norm['ref'], 'ref concat mismatch'
assert hyp_cat == norm['hyp'], 'hyp concat mismatch'
for k, s in enumerate(steps):
    assert s['i'] == k
    assert s['op'] in ('match', 'sub', 'del', 'ins')
    if s['op'] == 'del':
        assert s['ref'] and not s['hyp']
    if s['op'] == 'ins':
        assert s['hyp'] and not s['ref']
    if s['op'] in ('match', 'sub'):
        assert s['ref'] and s['hyp']

c = collections.Counter(s['op'] for s in steps)
ref_chars = len(norm['ref'])
errs = c['sub'] + c['del'] + c['ins']
m = {
    'pair': 'V1_B__ecom_11_g1',
    'ref_chars': ref_chars,
    'hyp_chars': len(norm['hyp']),
    'steps': len(steps),
    'match': c['match'], 'sub': c['sub'], 'del': c['del'], 'ins': c['ins'],
    'errors': errs,
    'cer': round(errs / ref_chars, 4),
    'verified_concat': True,
}
json.dump(m, open(os.path.join(HERE, 'char_metrics.json'), 'w'), indent=1)

with open(os.path.join(HERE, 'char_map.txt'), 'w', encoding='utf-8') as f:
    W = 100
    for a in range(0, len(steps), 25):
        blk = steps[a:a + 25]
        cells = []
        for s in blk:
            w = max(len(s['ref']), len(s['hyp']), 1)
            cells.append((s['ref'].replace(' ', '_') or '-', s['hyp'].replace(' ', '_') or '-',
                          {'match': '=', 'sub': 'S', 'del': 'D', 'ins': 'I'}[s['op']]))
        f.write('ref: ' + ' | '.join(x[0] for x in cells) + '\n')
        f.write('hyp: ' + ' | '.join(x[1] for x in cells) + '\n')
        f.write('op : ' + ' | '.join(x[2] for x in cells) + '\n\n')
print(json.dumps(m))
