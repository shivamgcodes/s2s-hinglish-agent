import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s, lower):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower() if lower else s
ref = norm(open(f'{D}/reference.txt', encoding='utf-8').read(), True)
hyp = norm(open(f'{D}/whisper.txt', encoding='utf-8').read(), False)  # case kept; compared case-insensitively
# hyp units: Latin chars individually; Devanagari grouped into aksharas (base + combining marks)
units = []
for c in hyp:
    if units and unicodedata.category(c) in ('Mn', 'Mc') and 'ऀ' <= c <= 'ॿ':
        units[-1] += c
    else:
        units.append(c)
n, m = len(ref), len(units)
eq = lambda a, b: a == b.lower()
INF = 10**9
dp = [[0]*(m+1) for _ in range(n+1)]
for i in range(n+1): dp[i][0] = i
for j in range(m+1): dp[0][j] = j
for i in range(1, n+1):
    for j in range(1, m+1):
        dp[i][j] = min(dp[i-1][j-1] + (0 if eq(ref[i-1], units[j-1]) else 1), dp[i-1][j]+1, dp[i][j-1]+1)
ops = []; i, j = n, m
while i or j:
    if i and j and dp[i][j] == dp[i-1][j-1] + (0 if eq(ref[i-1], units[j-1]) else 1):
        ops.append((ref[i-1], units[j-1], 'match' if eq(ref[i-1], units[j-1]) else 'sub')); i -= 1; j -= 1
    elif i and dp[i][j] == dp[i-1][j] + 1:
        ops.append((ref[i-1], '', 'del')); i -= 1
    else:
        ops.append(('', units[j-1], 'ins')); j -= 1
ops.reverse()
def why(r, h, op):
    if op == 'match': return 'case-insensitive match' if r != h else ''
    if op == 'sub': return f"'{r}' heard as '{h}'"
    if op == 'del': return 'reference char not in whisper (e.g. truncated contraction / punctuation-joined number)' if r else ''
    if 'ऀ' <= h[0] <= 'ॿ': return 'trailing Devanagari "करते हैं" hallucinated by whisper after end of reference'
    return 'extra char in whisper (e.g. full contraction re/m/s, space after comma in date)'
recs = [{"i": k, "ref": r, "hyp": h, "op": op, "why": why(r, h, op)} for k, (r, h, op) in enumerate(ops)]
assert ''.join(x['ref'] for x in recs) == ref
assert ''.join(x['hyp'] for x in recs) == hyp
with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for x in recs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
c = {k: sum(1 for x in recs if x['op'] == k) for k in ('match', 'sub', 'del', 'ins')}
cer = (c['sub'] + c['del'] + c['ins']) / n
json.dump({**c, "ref_chars": n, "hyp_chars": len(hyp), "hyp_units": m, "cer": round(cer, 6),
           "normalised_reference": ref, "normalised_hypothesis": hyp,
           "notes": "punctuation (Unicode P*) dropped, whitespace collapsed; ref lowercased, hyp case kept but compared case-insensitively; Devanagari ins grouped per akshara"},
          open(f'{D}/char_metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
# txt view
mk = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
lines = []; chunk = []; rc = 0
def flush():
    a = b = o = ''
    for r, h, op in chunk:
        w = max(len(r), len(h), 1)
        a += (r or '-').ljust(w); b += (h or '-').ljust(w); o += mk[op].ljust(w)
    lines.extend(['REF: ' + a, 'HYP: ' + b, 'OPS: ' + o, ''])
for r, h, op in ops:
    chunk.append((r, h, op)); rc += bool(r)
    if rc >= 80: flush(); chunk = []; rc = 0
if chunk: flush()
hdr = f"match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} ref_chars={n} CER={cer:.4f}\n('-' = empty side; Devanagari spans may misalign visually)\n\n"
open(f'{D}/char_map.txt', 'w', encoding='utf-8').write(hdr + '\n'.join(lines))
print(hdr)
for x in recs:
    if x['op'] != 'match': print(x)
