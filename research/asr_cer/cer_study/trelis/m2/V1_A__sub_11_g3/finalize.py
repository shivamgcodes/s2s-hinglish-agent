#!/usr/bin/env python3
"""Verify char_map.jsonl against normalised texts, then write char_metrics.json and char_map.txt."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.dirname(os.path.abspath(__file__))
from align import norm  # noqa: E402  (re-runs alignment; idempotent)

ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read())
rows = [json.loads(l) for l in open(os.path.join(D, 'char_map.jsonl'), encoding='utf-8')]

assert ''.join(r['ref'] for r in rows) == ref, 'ref concat mismatch'
assert ''.join(r['hyp'] for r in rows) == hyp, 'hyp concat mismatch'
for k, r in enumerate(rows):
    assert r['i'] == k
    assert r['op'] in ('match', 'sub', 'del', 'ins')
    if r['op'] == 'sub': assert len(r['ref']) == 1 and r['hyp']
    if r['op'] == 'del': assert len(r['ref']) == 1 and r['hyp'] == ''
    if r['op'] == 'ins': assert r['ref'] == '' and r['hyp']
    if r['op'] == 'match': assert r['ref'] and r['hyp']

S = sum(r['op'] == 'sub' for r in rows)
De = sum(r['op'] == 'del' for r in rows)
I = sum(r['op'] == 'ins' for r in rows)
M = sum(len(r['ref']) for r in rows if r['op'] == 'match')
N = len(ref)
assert M + S + De == N
metrics = {'ref_chars': N, 'hyp_chars': len(hyp), 'match_ref_chars': M, 'sub': S, 'del': De, 'ins': I,
           'cer': round((S + De + I) / N, 4), 'steps': len(rows)}
json.dump(metrics, open(os.path.join(D, 'char_metrics.json'), 'w'), indent=2)

with open(os.path.join(D, 'char_map.txt'), 'w', encoding='utf-8') as f:
    W = 40
    for s in range(0, len(rows), W):
        chunk = rows[s:s + W]
        f.write('ref: ' + '|'.join(r['ref'] or '·' for r in chunk) + '\n')
        f.write('hyp: ' + '|'.join(r['hyp'] or '·' for r in chunk) + '\n')
        f.write('op : ' + '|'.join({'match': '=', 'sub': 'S', 'del': 'D', 'ins': 'I'}[r['op']] for r in chunk) + '\n\n')
print('VERIFIED', metrics)
