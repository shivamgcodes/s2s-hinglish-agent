#!/usr/bin/env python3
"""Cross-script character alignment: Roman Hinglish reference vs Whisper (Devanagari+Latin) hypothesis.
Each hyp char is romanised into phonetic units (required / optional). A weighted Levenshtein aligns
reference chars to roman units; hyp chars are then attributed to alignment steps."""
import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))

def norm(s, lower):
    s = ''.join(' ' if unicodedata.category(c).startswith('P') or c in '/' else c for c in s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower() if lower else s

ref = norm(open(f'{D}/reference.txt', encoding='utf-8').read(), True)
hyp = norm(open(f'{D}/whisper.txt', encoding='utf-8').read(), False)
# NB: punctuation is replaced by space then whitespace collapsed; for "7:15", "3,000", "IN-402" join instead
def norm2(s, lower):
    s = re.sub(r'(?<=\w)[:,\-](?=\w)', '', s)
    return norm(s, lower)
ref = norm2(open(f'{D}/reference.txt', encoding='utf-8').read(), True)
hyp = norm2(open(f'{D}/whisper.txt', encoding='utf-8').read(), False)

CONS = {'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'n','च':'ч','छ':'чh','ज':'j','झ':'jh','ञ':'n',
 'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
 'प':'p','फ':'f','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh',
 'स':'s','ह':'h'}
VOW = {'अ':'a','आ':'aa','इ':'i','ई':'II','उ':'u','ऊ':'uu','ए':'e','ऐ':'ai','ओ':'o','औ':'au','ऑ':'o','ऋ':'ri'}
MAT = {'ा':'aa','ि':'i','ी':'II','ु':'u','ू':'uu','े':'e','ै':'ai','ो':'o','ौ':'au','ॉ':'o','ृ':'ri'}
SIGN = {'ं':'N','ँ':'N','ः':'h','्':'','़':''}

# roman units: list of (char, hyp_index, optional)
units = []
for k, c in enumerate(hyp):
    if c in CONS:
        r = CONS[c]
        for j, ch in enumerate(r): units.append((ch, k, False))
        nxt = hyp[k+1] if k+1 < len(hyp) else ''
        if nxt == '़' and k+2 < len(hyp): nxt = hyp[k+2]
        if nxt not in MAT and nxt != '्':
            units.append(('a', k, True))          # inherent schwa: optional
    elif c in VOW or c in MAT:
        r = (VOW.get(c) or MAT.get(c))
        if r == 'II': r = 'Ii'
        if r == 'uu': r = 'Uu'
        for j, ch in enumerate(r): units.append((ch, k, j > 0))   # 2nd letter of long vowel optional
    elif c in SIGN:
        for ch in SIGN[c]: units.append((ch, k, True))
    else:
        units.append((c.lower(), k, False))

EQ = [set('vw'), set('ck'), set('cs'), set('zj'), set('sj'), set('iI'), set('ou'), set('oU'), set('uU'), set('ae'), set('nN'), set('mN')]
def same(a, b):
    if a == b: return True
    if a == ' ' or b == ' ': return False
    return any(a in g and b in g for g in EQ)
VOWELS='aeiou'
def same_ctx(i, u):
    a = ref[i]
    if same(a, u): return True
    if a == 't' and i+1 < n and ref[i+1] == 'h' and u in 'dt': return True   # th ~ द/थ/त
    if a == 'e' and u == 'I' and ((i+1 < n and ref[i+1] in 'ea') or (i and ref[i-1] == 'e')): return True  # ee/ea ~ ी
    return False
def silent(i):
    a = ref[i]; p = ref[i-1] if i else ''; q = ref[i+1] if i+1 < n else ' '
    if a == ' ' or a.isdigit(): return None
    if a == p: return 'doubled letter (spelling convention)'
    if a == 'h' and p in 'tgl': return 'digraph/silent h'
    if a == 'h' and p in VOWELS and q == ' ': return 'word-final silent h'
    if a == 'a' and p == 'e' and q != ' ': return "'ea' digraph = one vowel"
    if a == 'g' and q == 'h' and p == 'i': return "silent gh ('igh' = /ai/)"
    return None
n, m = len(ref), len(units)
INF = 10**9
def relax(dp, bp, i, j, c, b):
    if c < dp[i][j]: dp[i][j] = c; bp[i][j] = b
def align(r0, r1, u0, u1):
    forced = (u0 == u1)  # whole ref span unheard: every char is a real deletion
    N, M = r1-r0, u1-u0
    dp = [[INF]*(M+1) for _ in range(N+1)]; bp = [[None]*(M+1) for _ in range(N+1)]
    dp[0][0] = 0
    for ii in range(N+1):
        for jj in range(M+1):
            v = dp[ii][jj]
            if v == INF: continue
            i, j = r0+ii, u0+jj
            if ii < N and jj < M:
                relax(dp, bp, ii+1, jj+1, v + (0 if same_ctx(i, units[j][0]) else (101 if units[j][2] else 100)), ('M', 1))
                if ref[i] == 'i' and units[j][0] == 'a':   # diphthong 'i' = /ai/ (flight ~ लाइट)
                    if jj+1 < M and units[j+1][0] == 'i': relax(dp, bp, ii+1, jj+2, v, ('M', 2))
                    if jj+2 < M and units[j+1][0] == 'a' and units[j+1][2] and units[j+2][0] == 'i': relax(dp, bp, ii+1, jj+3, v, ('M', 3))
            if ii < N:
                relax(dp, bp, ii+1, jj, v + (0 if (silent(i) and not forced) else 100), ('D', 0))
            if jj < M:
                relax(dp, bp, ii, jj+1, v + (0 if units[j][2] else 100), ('I', 1))
    ii, jj = N, M; p = []
    while ii or jj:
        o = bp[ii][jj]; p.append(o)
        if o[0] == 'M': ii -= 1; jj -= o[1]
        elif o[0] == 'D': ii -= 1
        else: jj -= 1
    return p[::-1]

# Manual anchors (judgement after listening-free reading of both texts): Whisper skipped the reference
# span 'done ji the refund ... 3000 hai ' entirely, and appended a hallucinated tail after 'नहीं'.
def u_index(h):  # first unit index whose hyp char >= h
    for k, u in enumerate(units):
        if u[1] >= h: return k
    return m
R1 = ref.index('done ji the refund'); R2 = ref.index('ji you are welcome')
H1 = hyp.index('जी यू आर वेलकम'); H2 = hyp.index(' एक मिनट होके')
assert hyp[H1-1] == ' ' and ref[R1-1] == ' '
segs = [(0, R1, 0, u_index(H1)), (R1, R2, u_index(H1), u_index(H1)), (R2, n, u_index(H1), u_index(H2)), (n, n, u_index(H2), m)]
path = []
for sg in segs: path += align(*sg)
# Build steps
steps = []  # dict ref, units(list), op
owner = {}  # hyp idx -> step idx
pending_skip = []
i = j = 0
for o in path:
    if o[0] == 'M':
        us = units[j:j+o[1]]
        ok = same_ctx(i, us[0][0]) or o[1] > 1
        why = ''
        if o[1] > 1: why = "diphthong: 'i' = /ai/"
        elif ok and ref[i] != us[0][0] and not (ref[i] == ' '):
            why = f"same sound: ref '{ref[i]}' ~ '{us[0][0]}'"
        steps.append({'ref': ref[i], 'u': us, 'op': 'match' if ok else 'sub', 'why': why}); i += 1; j += o[1]
    elif o[0] == 'D':
        w = silent(i) if not (R1 <= i < R2) else None
        prev = steps[-1] if steps else None
        # a silent/doubled letter only counts as matched if the letter it leans on was heard
        k2 = len(steps)-1
        while k2 >= 0 and steps[k2]['op'] in ('skip', 'ins'): k2 -= 1
        if w and (k2 < 0 or steps[k2]['op'] == 'del'): w = None
        steps.append({'ref': ref[i], 'u': [], 'op': 'match' if w else 'del', 'why': w or ''}); i += 1
    else:
        u = units[j]
        if u[2]: steps.append({'ref': None, 'u': [u], 'op': 'skip'})
        else: steps.append({'ref': '', 'u': [u], 'op': 'ins', 'why': ''})
        j += 1
# attribute hyp chars: first non-skip step touching it owns it; skip-only chars go to the step owning
# another unit of the same char, else the previous real step (or next if none)
real = [k for k, s in enumerate(steps) if s['op'] != 'skip']
for k, s in enumerate(steps):
    if s['op'] == 'skip': continue
    for u in s['u']:
        owner.setdefault(u[1], k)
for k, s in enumerate(steps):
    if s['op'] != 'skip': continue
    h = s['u'][0][1]
    if h in owner: continue
    prev = [r for r in real if r < k]
    nxt = [r for r in real if r > k]
    owner[h] = prev[-1] if prev else nxt[0]
for h in range(len(hyp)):
    if h not in owner:
        # chars with no units (e.g. virama/nukta): attach to owner of previous char
        owner[h] = owner.get(h-1, real[0])
hyp_of = {}
for h in range(len(hyp)): hyp_of.setdefault(owner[h], []).append(h)

out = []
for k in real:
    s = steps[k]
    hs = ''.join(hyp[h] for h in sorted(hyp_of.get(k, [])))
    out.append({'ref': s['ref'], 'hyp': hs, 'op': s['op'], 'why': s.get('why',''), 'u': ''.join(u[0] for u in s['u'])})
# ins steps whose hyp char was already owned by another step -> merge their cost into a 'sub'? keep, flag
json.dump(out, open(f'{D}/raw_steps.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
open(f'{D}/norm_ref.txt','w',encoding='utf-8').write(ref)
open(f'{D}/norm_hyp.txt','w',encoding='utf-8').write(hyp)
print(len(ref), len(hyp))

# ---------------- outputs ----------------
SKIPPED = ref[R1:R2].strip()
rows = []
for k, x in enumerate(out):
    why = x['why']
    if x['op'] == 'sub' and not why:
        why = f"different sound: ref '{x['ref']}' vs hyp '{x['hyp']}' (~'{x['u'].replace(chr(1095),'ch')}')"
        if x['ref'] == ' ' or x['hyp'] == ' ': why = 'word-boundary mismatch: ' + why
    if x['op'] == 'del' and not why:
        why = "part of phrase Whisper skipped entirely: '" + SKIPPED + "'" if R1 <= sum(len(o['ref']) for o in out[:k]) < R2 \
              else ("missing word boundary in Whisper" if x['ref'] == ' ' else "sound not present in Whisper")
    if x['op'] == 'ins' and not why:
        why = "Whisper tail with no reference (hallucinated/extra speech after 'nahi')" if sum(len(o['hyp']) for o in out[:k]) >= H2 \
              else ("extra word boundary in Whisper" if x['hyp'] == ' ' else "extra sound in Whisper")
    rows.append({'i': k, 'ref': x['ref'], 'hyp': x['hyp'], 'op': x['op'], 'why': why})

assert ''.join(r['ref'] for r in rows) == ref, 'ref reconstruction failed'
assert ''.join(r['hyp'] for r in rows) == hyp, 'hyp reconstruction failed'
with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

from collections import Counter
c = Counter(r['op'] for r in rows)
S, Dl, I, M = c['sub'], c['del'], c['ins'], c['match']
metrics = {'pair': 'V1_B__air_16_g4', 'ref_chars': len(ref), 'hyp_chars': len(hyp), 'match': M, 'sub': S, 'del': Dl, 'ins': I,
           'errors': S + Dl + I, 'cer': round((S + Dl + I) / len(ref), 4),
           'silent_letter_matches': sum(1 for r in rows if r['op'] == 'match' and r['hyp'] == ''),
           'notes': ["CER = (sub+del+ins)/ref_chars; spaces are units.",
                     f"Whisper skipped reference span ({R2-R1} chars) '{SKIPPED}' -> all deletions.",
                     f"Whisper appended '{hyp[H2:].strip()}' after 'nahi' -> insertions.",
                     "Matches include conventional Hinglish/English spelling: inherent schwa, aa~ा, ee/ea~ी, oo~ु/ू, th~द, c~क/स, s~ज (/z/), w~व, igh~ाइ, silent/doubled letters."]}
json.dump(metrics, open(f'{D}/char_metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

def dw(s): return sum(0 if unicodedata.category(ch) == 'Mn' else 1 for ch in s)
lines = []; blk = []; cnt = 0
def flush():
    a = b = cc = ''
    for r in blk:
        rr = r['ref'] or '·'; hh = r['hyp'] or '·'
        w = max(dw(rr), dw(hh), 1)
        a += rr + ' ' * (w - dw(rr)); b += hh + ' ' * (w - dw(hh))
        cc += {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}[r['op']] + ' ' * (w - 1)
    lines.extend(['REF: ' + a, 'HYP: ' + b, 'OP:  ' + cc, ''])
for r in rows:
    blk.append(r); cnt += len(r['ref'])
    if cnt >= 80 and r['ref'] == ' ': flush(); blk = []; cnt = 0
if blk: flush()
hdr = [f"V1_B__air_16_g4 char map  ('·' = empty side; markers: ' ' match, S sub, D del, I ins)",
       f"ref_chars={len(ref)} match={M} sub={S} del={Dl} ins={I} CER={metrics['cer']}", '']
open(f'{D}/char_map.txt', 'w', encoding='utf-8').write('\n'.join(hdr + lines))
print(metrics)
