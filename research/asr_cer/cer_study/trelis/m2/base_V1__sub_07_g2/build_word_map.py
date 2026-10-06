import json, difflib
from normalize import norm
R = norm(open('reference.txt', encoding='utf-8').read()).split()
H = norm(open('whisper.txt', encoding='utf-8').read()).split()
# sound-equivalence / substitution decisions taken from char_map.jsonl
SPECIAL = {  # ref word -> list of (hyp words, op, why)
    'streambox': ('string box', 'sub', "char map: 'ea' /i:/ -> /I/ and /m/ -> /ng/ (stream->string); sound differs"),
    'varun': ('वारुण', 'match', 'char map: same sound across scripts (Devanagari transliteration)'),
    'alright': ('all right', 'match', 'char map: same sound, word-break/spelling variant'),
    '12': ('twelfth', 'match', "char map: 12 spoken as ordinal 'twelfth' - same number"),
    '2024': ('twenty twenty four', 'match', 'char map: 2024 read as twenty twenty four - same number'),
    '2025': ('twenty twenty five', 'match', 'char map: 2025 read as twenty twenty five - same number'),
}
TRUNC = {('you', 'youre'): "char map: audio says you're; reference truncated to you' (ins r,e)",
         ('i', 'im'): "char map: audio says I'm; reference truncated to I' (ins m)",
         ('that', 'thats'): "char map: audio says that's; reference truncated to That' (ins s)"}
rows = []; j = 0
for w in R:
    if w in SPECIAL:
        hw, op, why = SPECIAL[w]
        n = len(hw.split()); assert H[j:j+n] == hw.split(), (w, H[j:j+n]); j += n
    elif H[j] == w:
        hw, op, why = w, 'match', 'identical'; j += 1
    elif (w, H[j]) in TRUNC:
        hw, op, why = H[j], 'sub', TRUNC[(w, H[j])]; j += 1
    else:
        raise SystemExit(f'unaligned {w} {H[j]}')
    rows.append({'i': len(rows), 'ref': w, 'hyp': hw, 'op': op, 'why': why})
assert j == len(H)
# verification
assert ' '.join(r['ref'] for r in rows if r['ref']).split() == R
assert ' '.join(r['hyp'] for r in rows if r['hyp']).split() == H
c = {k: sum(r['op'] == k for r in rows) for k in ('match', 'sub', 'del', 'ins')}
m = {'pair': 'base_V1__sub_07_g2', 'ref_words': len(R), 'hyp_words': len(H), **c,
     'wer': round((c['sub'] + c['del'] + c['ins']) / len(R), 6),
     'normalisation': 'normalize.norm (NFC, lowercase, apostrophes deleted, other punctuation->space so 12,2024 -> 12 2024), whitespace tokens',
     'verified': 'joined ref/hyp fields reproduce normalised word sequences'}
with open('word_map.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
json.dump(m, open('word_metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
with open('word_map.txt', 'w', encoding='utf-8') as f:
    f.write(f"{'i':>4}  {'op':<5} {'ref':<12} {'hyp':<22} why\n")
    for r in rows: f.write(f"{r['i']:>4}  {r['op']:<5} {r['ref']:<12} {r['hyp']:<22} {r['why']}\n")
    f.write(json.dumps(m, ensure_ascii=False) + '\n')
print(m)
