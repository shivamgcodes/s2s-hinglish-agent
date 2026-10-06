#!/usr/bin/env python3
"""Character-level cross-script alignment map: reference (Roman) vs Whisper (Latin + Devanagari).

Latin prefix (both sides Latin): case-insensitive Levenshtein alignment.
Devanagari tail: hand-judged per-word sound alignment (TAIL table below), each Devanagari
char assigned to exactly one step so the hyp concatenation reproduces the normalised hypothesis.
"""
import json, re, unicodedata, os
HERE = os.path.dirname(os.path.abspath(__file__))

def strip_punct(s):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()

REF = strip_punct(open(os.path.join(HERE, 'reference.txt'), encoding='utf-8').read().lower())
HYP = strip_punct(open(os.path.join(HERE, 'whisper.txt'), encoding='utf-8').read())  # case kept; compared case-insensitively

# ---- split into Latin prefix and Devanagari tail ----
SPLIT_REF = 'there '
r_cut = REF.index('he wait for you there ') + len('he wait for you there ')
h_cut = HYP.index('इस')
ref_lat, ref_tail = REF[:r_cut], REF[r_cut:]
hyp_lat, hyp_tail = HYP[:h_cut], HYP[h_cut:]

def lev_align(a, b):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0] = i
    for j in range(1, m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i-1] == b[j-1].lower() else 1
            D[i][j] = min(D[i-1][j-1] + c, D[i-1][j] + 1, D[i][j-1] + 1)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (0 if a[i-1] == b[j-1].lower() else 1):
            ok = a[i-1] == b[j-1].lower()
            out.append((a[i-1], b[j-1], 'match' if ok else 'sub', '' if ok else 'different letter'))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i-1][j] + 1:
            out.append((a[i-1], '', 'del', 'not spoken/transcribed')); i -= 1
        else:
            out.append(('', b[j-1], 'ins', 'extra in whisper')); j -= 1
    return out[::-1]

steps = lev_align(ref_lat, hyp_lat)

# ---- Devanagari tail: hand-judged sound alignment ----
M, S, D, I = 'match', 'sub', 'del', 'ins'
SP = (' ', ' ', M, '')
TAIL = [
  # its ~ इस
  ('i','इ',M,''), ('t','',D,'t not heard in इस'), ('s','स',M,''), SP,
  # rajesh ~ राजश
  ('r','र',M,''), ('a','ा',M,'aa matra'), ('j','ज',M,''), ('e','',D,'e vowel missing (राजश has no े matra)'),
  ('s','श',M,'sh ~ श'), ('h','',M,'part of sh digraph ~ श'), SP,
  # no ~ नो
  ('n','न',M,''), ('o','ो',M,''), SP,
  # problem ~ प्रॉब्लम
  ('p','प्',M,'p + virama (conjunct)'), ('r','र',M,''), ('o','ॉ',M,'candra-o for English o'),
  ('b','ब्',M,'b + virama'), ('l','ल',M,''), ('e','',M,'schwa = inherent vowel of ल (standard spelling)'),
  ('m','म',M,''), SP,
  # have ~ हव
  ('h','ह',M,''), ('a','',M,'inherent vowel of ह'), ('v','व',M,''), ('e','',M,'silent e'), SP,
  # a ~ (nothing)
  ('a','',D,'article a not transcribed'), (' ','',D,'space of dropped word'),
  # great ~ ग्रेट
  ('g','ग्',M,'g + virama'), ('r','र',M,''), ('e','े',M,'ea ~ े'), ('a','',M,'part of ea digraph ~ े'),
  ('t','ट',M,''), SP,
  # day ~ दे
  ('d','द',M,''), ('a','े',M,'ay ~ े'), ('y','',M,'part of ay digraph ~ े'), SP,
  # aarti ~ आती
  ('a','आ',M,'aa ~ आ'), ('a','',M,'part of aa ~ आ'), ('r','',D,'r dropped (Aati)'),
  ('t','त',M,''), ('i','ी',M,''), SP,
  # thank ~ थैंक
  ('t','थ',M,'th ~ थ'), ('h','',M,'part of th ~ थ'), ('a','ै',M,'English a ~ ै'), ('n','ं',M,'anusvara n'),
  ('k','क',M,''), SP,
  # you ~ यू
  ('y','य',M,''), ('o','ू',M,'ou ~ ू'), ('u','',M,'part of ou ~ ू'), SP,
  # you ~ यो
  ('y','य',M,''), ('o','ो',M,''), ('u','',D,'yo instead of you: u sound missing'), SP,
  # welcome ~ वेलकम
  ('w','व',M,'w ~ व'), ('e','े',M,''), ('l','ल',M,''), ('c','क',M,'c (k sound) ~ क'),
  ('o','',M,'schwa = inherent vowel of क'), ('m','म',M,''), ('e','',M,'silent e'), SP,
  # have ~ हव
  ('h','ह',M,''), ('a','',M,'inherent vowel of ह'), ('v','व',M,''), ('e','',M,'silent e'), SP,
  ('a','',D,'article a not transcribed'), (' ','',D,'space of dropped word'),
  ('g','ग्',M,'g + virama'), ('r','र',M,''), ('e','े',M,'ea ~ े'), ('a','',M,'part of ea digraph ~ े'),
  ('t','ट',M,''), SP,
  ('d','द',M,''), ('a','े',M,'ay ~ े'), ('y','',M,'part of ay digraph ~ े'),
]
steps += TAIL

# ---- verify ----
ref_cat = ''.join(s[0] for s in steps); hyp_cat = ''.join(s[1] for s in steps)
assert ref_cat == REF, ('REF mismatch', ref_cat, REF)
assert hyp_cat == HYP, ('HYP mismatch', hyp_cat, HYP)
for s in steps:
    assert (s[2] == 'del') == (s[1] == '' and s[2] != 'match') or s[2] in ('match', 'ins', 'sub')
    assert not (s[2] == 'ins' and s[0]) and not (s[2] == 'sub' and (not s[0] or not s[1]))

with open(os.path.join(HERE, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
    for i, (r, h, op, why) in enumerate(steps):
        f.write(json.dumps({'i': i, 'ref': r, 'hyp': h, 'op': op, 'why': why}, ensure_ascii=False) + '\n')

cnt = {k: sum(1 for s in steps if s[2] == k) for k in ('match', 'sub', 'del', 'ins')}
N = len(REF)
cer = (cnt['sub'] + cnt['del'] + cnt['ins']) / N
metrics = dict(cnt, ref_chars=N, hyp_chars=len(HYP), cer=round(cer, 4),
               ref_concat_ok=True, hyp_concat_ok=True,
               notes='Ref: lowercase, P* punctuation dropped, whitespace collapsed (spaces are units). '
                     'Hyp: P* punctuation dropped, whitespace collapsed, case kept (Latin compared case-insensitively). '
                     'Latin prefix aligned by Levenshtein; Devanagari tail aligned by hand-judged sound per word. '
                     'Digraph/silent letters (sh, th, ea, ay, ou, silent e) and inherent schwa count as match with empty hyp.')
json.dump(metrics, open(os.path.join(HERE, 'char_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ---- human-readable map ----
mark = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
def w(s):  # display width of a hyp span (combining marks have zero width)
    return sum(0 if unicodedata.category(c) in ('Mn', 'Mc') else 1 for c in s)
lines, cur, nref = [], [[], [], []], 0
for r, h, op, _ in steps:
    r_d = r if r else '-'; h_d = h if h else '-'
    width = max(1, w(h_d))
    cur[0].append(r_d.replace(' ', '_') + ' ' * (width - 1))
    cur[1].append(h_d.replace(' ', '_') + ' ' * (width - w(h_d)))
    cur[2].append(mark[op] + ' ' * (width - 1))
    if r: nref += 1
    if nref >= 80:
        lines.append('REF ' + ''.join(cur[0])); lines.append('HYP ' + ''.join(cur[1])); lines.append('OP  ' + ''.join(cur[2])); lines.append('')
        cur, nref = [[], [], []], 0
if cur[0]:
    lines.append('REF ' + ''.join(cur[0])); lines.append('HYP ' + ''.join(cur[1])); lines.append('OP  ' + ''.join(cur[2]))
hdr = f"CER = {cer:.4f}  (S={cnt['sub']} D={cnt['del']} I={cnt['ins']} / N={N})  '_'=space '-'=empty\n\n"
open(os.path.join(HERE, 'char_map.txt'), 'w', encoding='utf-8').write(hdr + '\n'.join(lines) + '\n')
print(json.dumps(metrics, ensure_ascii=False, indent=1))
