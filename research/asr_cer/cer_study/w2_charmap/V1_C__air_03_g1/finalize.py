#!/usr/bin/env python3
"""Verify char_map.jsonl reproduces normalised ref/hyp, compute CER, render char_map.txt."""
import json, os, unicodedata
D = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(f'{D}/normalised.json', encoding='utf-8'))
steps = [json.loads(l) for l in open(f'{D}/char_map.jsonl', encoding='utf-8') if l.strip()]
for k, s in enumerate(steps): s['i'] = k
r = ''.join(s['ref'] for s in steps); h = ''.join(s['hyp'] for s in steps)
assert r == N['REF'], 'REF mismatch'
assert h == N['HYP'], 'HYP mismatch'
for s in steps:
    assert s['op'] in ('match','sub','del','ins')
    if s['op'] == 'del': assert s['ref'] and not s['hyp']
    if s['op'] == 'ins': assert s['hyp'] and not s['ref']
    if s['op'] in ('match','sub'): assert s['ref']
with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for s in steps: f.write(json.dumps(s, ensure_ascii=False) + '\n')
c = {o: sum(1 for s in steps if s['op'] == o) for o in ('match','sub','del','ins')}
nref = len(N['REF'])
cer = (c['sub'] + c['del'] + c['ins']) / nref
met = {'pair': 'V1_C__air_03_g1', 'ref_chars': nref, 'hyp_chars': len(N['HYP']), 'steps': len(steps), **c,
       'cer': round(cer, 4), 'formula': '(sub+del+ins)/ref_chars', 'ins_unit': 'one hyp span (akshara / Latin char / space) per ins step'}
json.dump(met, open(f'{D}/char_metrics.json', 'w'), indent=1)
def w(t):  # display width approx: combining marks are zero-width
    return sum(0 if unicodedata.category(ch) in ('Mn','Mc') else 1 for ch in t)
lines, cur, cnt = [], [], 0
for s in steps:
    cur.append(s); cnt += len(s['ref'])
    if cnt >= 80: lines.append(cur); cur, cnt = [], 0
if cur: lines.append(cur)
mk = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
out = [f"V1_C__air_03_g1  CER={cer:.4f}  ref_chars={nref}  match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']}",
       "rows: REF / HYP / OP (' '=match S=sub D=del I=ins); '·' = empty side, '_' = space", '']
for ln in lines:
    a = b = o = ''
    for s in ln:
        rr = (s['ref'] or '·').replace(' ', '_'); hh = (s['hyp'] or '·').replace(' ', '_')
        wd = max(w(rr), w(hh), 1)
        a += rr + ' ' * (wd - w(rr)); b += hh + ' ' * (wd - w(hh)); o += mk[s['op']] + ' ' * (wd - 1)
    out += ['REF ' + a, 'HYP ' + b, 'OP  ' + o, '']
open(f'{D}/char_map.txt', 'w', encoding='utf-8').write('\n'.join(out))
print(json.dumps(met))
