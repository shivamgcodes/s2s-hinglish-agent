#!/usr/bin/env python3
"""Character-level, sound-aligned map for base_V1__cab_11_g4.
Roman stretches are aligned with Levenshtein DP; cross-script names and
number/number-word stretches are aligned by hand (manual steps)."""
import json, re, unicodedata

def norm(s):
    s = s.lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()

REF = norm(open('reference.txt', encoding='utf-8').read())
HYP = norm(open('whisper.txt', encoding='utf-8').read())

def dp(r, h):
    n, m = len(r), len(h)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j-1] + (r[i-1] != h[j-1]), D[i-1][j]+1, D[i][j-1]+1)
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + (r[i-1] != h[j-1]):
            op = 'match' if r[i-1] == h[j-1] else 'sub'
            out.append((r[i-1], h[j-1], op, 'same letter' if op == 'match' else 'different letter/sound')); i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j] + 1:
            out.append((r[i-1], '', 'del', 'not heard in hypothesis')); i -= 1
        else:
            out.append(('', h[j-1], 'ins', 'extra in hypothesis')); j -= 1
    return out[::-1]

M = lambda r, h, op, why: (r, h, op, why)
# segments: ('auto', ref_text, hyp_text) or ('man', [steps])
SEG = [
 ('auto', 'hello thank you for calling citycab how can i assist you today yes ',
          'hello thank you for calling city cab how can i assist you today yes '),
 ('man', [M('a', 'आ', 'match', 'aa = आ long vowel'), M('a', '', 'match', 'second a of aa digraph, long vowel present in आ'),
          M('r', '', 'del', 'r of aarti not heard (आती)'),
          M('t', 'त', 'match', 't = त'), M('i', 'ी', 'match', 'i = ी matra')]),
 ('auto', ' your ride rd', ' your ride rd'),
 ('man', [M('5', ' five', 'match', 'digit 5 spoken as five'), M('0', ' zero', 'match', 'digit 0 spoken as zero'),
          M('6', ' six', 'match', 'digit 6 spoken as six'), M('8', ' eight', 'match', 'digit 8 spoken as eight')]),
 ('auto', ' is on the way it take about ', ' is on the way it takes about '),
 ('man', [M('1', 'twelve', 'match', '12 spoken as twelve (same number)'), M('2', '', 'match', 'second digit of 12, covered by twelve')]),
 ('auto', ' minutes driver is ', ' minutes driver is '),
 ('man', [M('r', 'र', 'match', 'r = र'), M('a', 'ा', 'match', 'a = ा matra'), M('j', 'ज', 'match', 'j = ज'),
          M('e', 'े', 'match', 'e = े matra'), M('s', 'श', 'match', 'sh = श'), M('h', '', 'match', 'h of sh digraph, sound present in श')]),
 ('auto', ' he driving a white maruti suzuki swift okay i have i sorry but i dont have that the drivers number is ',
          ' he is driving a white maruti suzuki swift okay i have im sorry but i dont have that the drivers number is '),
 ('man', [M('4', 'forty', 'match', '48xx spoken forty-eight: forty heard'),
          M('8', '', 'del', 'eight of forty-eight not heard'),
          M('2', ' two', 'sub', 'twenty (of twenty-one) heard as two'),
          M('1', ' one', 'match', 'one of twenty-one heard')]),
 ('auto', ' you can ask him when he arrives sure thing that should be gate ',
          ' you can ask him when he arrives sure thing that should be gate '),
 ('man', [M('3', 'three', 'match', '3 spoken as three')]),
 ('auto', ' he wait for you there its rajesh no problem have a great day aarti thank you you welcome have a great day',
          ' hell wait for you there its rajesh no problem have a great day aarti thank you youre welcome have a great day'),
]

steps = []
for s in SEG:
    steps += dp(s[1], s[2]) if s[0] == 'auto' else s[1]

rows = [dict(i=k, ref=r, hyp=h, op=op, why=w) for k, (r, h, op, w) in enumerate(steps)]
cr = ''.join(x['ref'] for x in rows); ch = ''.join(x['hyp'] for x in rows)
assert cr == REF, ('REF mismatch', cr, REF)
assert ch == HYP, ('HYP mismatch', ch, HYP)
for x in rows:
    assert len(x['ref']) <= 1
    assert (x['op'] == 'ins') == (x['ref'] == '')
    assert x['op'] != 'del' or x['hyp'] == ''

with open('char_map.jsonl', 'w', encoding='utf-8') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
cnt = {o: sum(x['op'] == o for x in rows) for o in ('match', 'sub', 'del', 'ins')}
N = len(REF)
assert cnt['match'] + cnt['sub'] + cnt['del'] == N
met = dict(pair='base_V1__cab_11_g4', ref_chars=N, hyp_chars=len(HYP), **cnt,
           cer=round((cnt['sub'] + cnt['del'] + cnt['ins']) / N, 6),
           ref_norm=REF, hyp_norm=HYP)
json.dump(met, open('char_metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
with open('char_map.txt', 'w', encoding='utf-8') as f:
    W = 100
    for k in range(0, len(rows), W):
        blk = rows[k:k+W]
        wd = [max(len(x['ref']), len(x['hyp']), 1) for x in blk]
        sym = {'match': '.', 'sub': 'S', 'del': 'D', 'ins': 'I'}
        f.write('ref: ' + '|'.join((x['ref'] or '_').replace(' ', '␣').ljust(w) for x, w in zip(blk, wd)) + '\n')
        f.write('hyp: ' + '|'.join((x['hyp'] or '_').replace(' ', '␣').ljust(w) for x, w in zip(blk, wd)) + '\n')
        f.write('op:  ' + '|'.join(sym[x['op']].ljust(w) for x, w in zip(blk, wd)) + '\n\n')
    for x in rows:
        if x['op'] != 'match': f.write(f"{x['i']}\t{x['op']}\t{x['ref']!r}\t{x['hyp']!r}\t{x['why']}\n")
print(json.dumps({k: v for k, v in met.items() if 'norm' not in k}))
