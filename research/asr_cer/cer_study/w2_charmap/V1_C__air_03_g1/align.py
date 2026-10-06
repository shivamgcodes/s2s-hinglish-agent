#!/usr/bin/env python3
"""Cross-script (Roman ref vs Devanagari/Latin hyp) char-level alignment by sound."""
import json, re, unicodedata, os, sys
D = os.path.dirname(os.path.abspath(__file__))

def is_punct(ch):
    return unicodedata.category(ch).startswith('P') or unicodedata.category(ch).startswith('S')

def norm(s, lower):
    s = ''.join(ch for ch in s if not is_punct(ch))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower() if lower else s

REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read(), True)
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read(), False)

O = 'opt'
CONS = {  # consonant -> list of (accepted letters, optional?)
 'क':[('kcq',0)], 'ख':[('k',0),('h',1)], 'ग':[('g',0)], 'घ':[('g',0),('h',1)], 'ङ':[('n',0)],
 'च':[('c',0),('h',1)], 'छ':[('c',0),('h',1)], 'ज':[('jz',0)], 'झ':[('jz',0),('h',1)], 'ञ':[('n',0)],
 'ट':[('t',0)], 'ठ':[('t',0),('h',1)], 'ड':[('d',0)], 'ढ':[('d',0),('h',1)], 'ण':[('n',0)],
 'त':[('t',0)], 'थ':[('t',0),('h',1)], 'द':[('d',0)], 'ध':[('d',0),('h',1)], 'न':[('n',0)],
 'प':[('p',0)], 'फ':[('fp',0),('h',1)], 'ब':[('b',0)], 'भ':[('b',0),('h',1)], 'म':[('m',0)],
 'य':[('y',0)], 'र':[('r',0)], 'ल':[('l',0)], 'व':[('vw',0)], 'श':[('s',0),('h',1)], 'ष':[('s',0),('h',1)],
 'स':[('sc',0)], 'ह':[('h',0)],
}
NUKTA = {'क':'q','ज':'z','फ':'f','ड':'r','ढ':'r','ग':'g','ख':'k'}
VOW = {'अ':[('a',0)], 'आ':[('a',0),('a',1)], 'इ':[('ie',0)], 'ई':[('ie',0),('ie',1)], 'उ':[('uo',0)],
 'ऊ':[('uo',0),('uo',1)], 'ए':[('e',0)], 'ऐ':[('ae',0),('i',1)], 'ओ':[('o',0)], 'औ':[('ao',0),('u',1)],
 'ऑ':[('o',0)], 'ऋ':[('r',0),('i',1)]}
MATRA = {'ा':[('a',0),('a',1)], 'ि':[('ie',0)], 'ी':[('ie',0),('ie',1)], 'ु':[('uo',0)], 'ू':[('uo',0),('uo',1)],
 'े':[('e',0)], 'ै':[('ae',0),('i',1)], 'ो':[('o',0)], 'ौ':[('ao',0),('u',1)], 'ॉ':[('o',0)], 'ृ':[('r',0),('i',1)]}
VIRAMA, NUK = '्', '़'
MODS = {'ं':[('nm',1)], 'ँ':[('n',1)], 'ः':[('h',1)]}

def units(h):
    """split hyp into units: (text, rom list of (letters, optional))"""
    out, i = [], 0
    while i < len(h):
        c = h[i]
        if c in CONS:
            txt = c; i += 1; rom = list(CONS[c])
            if i < len(h) and h[i] == NUK:
                txt += NUK; i += 1; rom = [(NUKTA.get(c, rom[0][0]), 0)]
            if i < len(h) and h[i] == VIRAMA:
                txt += VIRAMA; i += 1
            elif i < len(h) and h[i] in MATRA:
                rom += MATRA[h[i]]; txt += h[i]; i += 1
            else:
                rom += [('a', 1)]  # inherent schwa (optional)
            while i < len(h) and h[i] in MODS:
                rom += MODS[h[i]]; txt += h[i]; i += 1
            out.append((txt, rom))
        elif c in VOW:
            txt = c; i += 1; rom = list(VOW[c])
            while i < len(h) and h[i] in MODS:
                rom += MODS[h[i]]; txt += h[i]; i += 1
            out.append((txt, rom))
        elif c == ' ':
            out.append((c, [(' ', 0)])); i += 1
        else:
            out.append((c, [(c.lower(), 0)])); i += 1
    return out

U = units(HYP)
R = []  # flat romanized positions: (letters, opt, unit index)
for ui, (t, rom) in enumerate(U):
    nreq = max(1, sum(1 for _, o in rom if not o))
    for L, o in rom:
        R.append((L, o, ui, 0.0 if o else 0.6))

n, m = len(REF), len(R)
INF = 1e18
dp = [[INF]*(m+1) for _ in range(n+1)]
bt = [[None]*(m+1) for _ in range(n+1)]
dp[0][0] = 0
for i in range(n+1):
    row, prow = dp[i], dp[i-1] if i else None
    for j in range(m+1):
        if i == 0 and j == 0: continue
        best, b = INF, None
        if i and j:
            L = R[j-1][0]; c = REF[i-1]
            cost = 0 if c in L else 1
            v = prow[j-1] + cost
            if v < best: best, b = v, 'M'
        if i:
            v = prow[j] + 1
            if v < best: best, b = v, 'D'
        if j:
            v = row[j-1] + R[j-1][3]
            if v < best: best, b = v, 'I'
        row[j] = best; bt[i][j] = b

# backtrace
path = []
i, j = n, m
while i or j:
    b = bt[i][j]; path.append((b, i-1 if b in 'MD' else None, j-1 if b in 'MI' else None))
    if b == 'M': i -= 1; j -= 1
    elif b == 'D': i -= 1
    else: j -= 1
path.reverse()

# overrides: optional manual list of (ref_index -> forced op) not needed
steps = []
emitted = set()
aligned_units = set(R[j][2] for b, i, j in path if b == 'M')
for b, ri, rj in path:
    if b == 'D':
        steps.append({'ref': REF[ri], 'hyp': '', 'op': 'del'})
        continue
    ui = R[rj][2]
    if b == 'M':
        c = REF[ri]; L = R[rj][0]
        hyp = '' if ui in emitted else U[ui][0]
        emitted.add(ui)
        op = 'match' if c in L else 'sub'
        steps.append({'ref': c, 'hyp': hyp, 'op': op, '_u': ui, '_L': L})
    else:  # skipped rom char
        if ui in aligned_units or ui in emitted:
            continue
        emitted.add(ui)
        steps.append({'ref': '', 'hyp': U[ui][0], 'op': 'ins'})

# why fields
for k, s in enumerate(steps):
    s['i'] = k
    u = s.pop('_u', None); L = s.pop('_L', None)
    if s['op'] == 'del':
        s['why'] = 'ref sound not present in Whisper transcript'
    elif s['op'] == 'ins':
        s['why'] = 'Whisper span with no counterpart in ref' if s['hyp'] != ' ' else 'extra word boundary in Whisper'
    elif s['op'] == 'sub':
        s['why'] = f"ref '{s['ref']}' vs akshara '{U[u][0]}' (sound '{L[0]}'): different sound"
    else:
        t = U[u][0]
        if t.isascii():
            s['why'] = ''
        elif s['hyp'] == '':
            s['why'] = f"continuation of '{t}' (same sound '{L[0]}')"
        else:
            s['why'] = f"'{s['ref']}' ~ '{t}'"
steps = [{'i': s['i'], 'ref': s['ref'], 'hyp': s['hyp'], 'op': s['op'], 'why': s['why']} for s in steps]
json.dump({'REF': REF, 'HYP': HYP}, open(f'{D}/normalised.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(f'{D}/char_map.auto.jsonl', 'w', encoding='utf-8') as f:
    for s in steps: f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(len(steps), 'steps; ref chars', len(REF))
