import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))
def norm(s, lower):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower() if lower else s
R = norm(open(f'{D}/reference.txt', encoding='utf-8').read(), True).split()
H = norm(open(f'{D}/whisper.txt', encoding='utf-8').read(), False).split()
eq = lambda a, b: a == b.lower()   # all hyp words except the trailing Devanagari are Latin; sound-equality judged = case-insensitive spelling
n, m = len(R), len(H)
dp = [[0]*(m+1) for _ in range(n+1)]
for i in range(n+1): dp[i][0] = i
for j in range(m+1): dp[0][j] = j
for i in range(1, n+1):
    for j in range(1, m+1):
        dp[i][j] = min(dp[i-1][j-1] + (0 if eq(R[i-1], H[j-1]) else 1), dp[i-1][j]+1, dp[i][j-1]+1)
ops = []; i, j = n, m
while i or j:
    if i and j and dp[i][j] == dp[i-1][j-1] + (0 if eq(R[i-1], H[j-1]) else 1):
        ops.append([R[i-1], H[j-1], 'match' if eq(R[i-1], H[j-1]) else 'sub']); i -= 1; j -= 1
    elif j and dp[i][j] == dp[i][j-1] + 1:
        ops.append(['', H[j-1], 'ins']); j -= 1
    else:
        ops.append([R[i-1], '', 'del']); i -= 1
ops.reverse()
WHY = {
 ('streambox','Stringbox'): "different word by sound: 'stream' vs 'string' (Whisper mishearing of brand name)",
 ('you','Youre'): "ref token is truncated \"You'\" (contraction suffix lost in text stream) -> 'you'; audio/Whisper has 'you're' -> different word form",
 ('i','Im'): "ref token is truncated \"I'\" -> 'i'; Whisper has \"I'm\" -> different word form",
 ('that','Thats'): "ref token is truncated \"That'\" -> 'that'; Whisper has \"That's\" -> different word form",
 ('122024','2024'): "ref '12,2024' collapses to one token '122024' after dropping punctuation; Whisper splits '12' '2024' -> ins '12' + sub",
 ('122025','2025'): "ref '12,2025' collapses to one token '122025' after dropping punctuation; Whisper splits '12' '2025' -> ins '12' + sub",
}
out = []
for k, (r, h, op) in enumerate(ops):
    if op == 'match': w = 'same word (case-insensitive, same script)'
    elif op == 'sub': w = WHY.get((r, h), 'different word')
    elif op == 'ins':
        if h == '12': w = "first half of Whisper's split of the ref date token (day '12'); see next sub"
        elif any('ऀ' <= c <= 'ॿ' for c in h): w = "trailing Devanagari 'करते हैं' (karte hain) not in reference: Whisper hallucination/extra speech after end"
        else: w = 'extra hyp word'
    else: w = 'ref word not in hyp'
    out.append({'i': k, 'ref': r, 'hyp': h, 'op': op, 'why': w})
assert ' '.join(o['ref'] for o in out if o['ref']) == ' '.join(R)
assert ' '.join(o['hyp'] for o in out if o['hyp']) == ' '.join(H)
with open(f'{D}/word_map.jsonl', 'w', encoding='utf-8') as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + '\n')
c = {k: sum(o['op'] == k for o in out) for k in ('match','sub','del','ins')}
wer = (c['sub']+c['del']+c['ins'])/len(R)
json.dump({**c, 'ref_words': len(R), 'hyp_words': len(H), 'wer': round(wer, 6),
  'notes': "punctuation (Unicode P*) dropped; ref lowercased; hyp compared case-insensitively. Reference text stream truncates contractions (You'/I'/That') so they become different word forms vs Whisper's you're/I'm/that's (counted sub). Dates '12,2024' are one ref token vs two hyp tokens (sub+ins)."},
  open(f'{D}/word_metrics.json', 'w'), ensure_ascii=False, indent=2)
w = max(len(o['ref']) for o in out) + 2
with open(f'{D}/word_map.txt', 'w', encoding='utf-8') as f:
    f.write(f"{'i':>3}  {'ref':<{w}}| {'hyp':<14}| op\n")
    for o in out: f.write(f"{o['i']:>3}  {o['ref']:<{w}}| {o['hyp']:<14}| {o['op']}\n")
print(c, len(R), len(H), wer)
for o in out:
    if o['op'] != 'match': print(o)
