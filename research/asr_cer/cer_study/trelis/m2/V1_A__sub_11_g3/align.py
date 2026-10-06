#!/usr/bin/env python3
"""Sound-based character alignment of Roman-Hinglish reference vs mixed-script Whisper hypothesis.

Ref units  = normalised reference characters (spaces included).
Hyp units  = Latin chars / spaces individually; Devanagari split into consonant(+nukta/virama),
             vowel sign, independent vowel, anusvara/chandrabindu; a few special multi-char units
             (spelled-out number words, the letter "G" pronounced "jee").
A hyp unit may 'match' a span of 1-3 ref chars when that span is an accepted romanisation of its sound.
Sub/del/ins each cost 1 and touch exactly one ref char (sub/del) or one hyp unit (ins).
"""
import json, re, unicodedata, os

D = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower()
    s = ''.join(c if (unicodedata.category(c)[0] in 'LMN' or c.isspace()) else ' ' for c in s)
    return re.sub(r'\s+', ' ', s).strip()


CONS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh',
    'न': 'n', 'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l',
    'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
}
CONS_ALT = {'व': ['w'], 'फ': ['f'], 'थ': ['t'], 'ठ': ['t']}
MATRA = {
    'ा': ['a', 'aa'], 'ि': ['i'], 'ी': ['i', 'ee', 'ii'], 'ु': ['u'], 'ू': ['u', 'oo'],
    'े': ['e', 'ay'], 'ै': ['ai', 'e', 'ae'], 'ो': ['o'], 'ौ': ['au', 'o'], 'ृ': ['ri'],
}
INDEP = {
    'अ': ['a'], 'आ': ['aa', 'a'], 'इ': ['i'], 'ई': ['i', 'ee'], 'उ': ['u'], 'ऊ': ['u', 'oo'],
    'ए': ['e', 'ye'], 'ऐ': ['ai', 'e'], 'ओ': ['o'], 'औ': ['au'],
}
NASAL = {'ं': ['n', 'm'], 'ँ': ['n']}


def hyp_units(h):
    """Return list of (text, accepted_romanisations, kind)."""
    units = []
    # special multi-char units first (sound-equivalent to ref spans)
    specials = [
        ('four hundred ', ['4'], 'number-word: four hundred = digit 4 of 499'),
        ('ninety ', ['9'], 'number-word: ninety = digit 9 (tens) of 499'),
        ('nine ', ['9'], 'number-word: nine = digit 9 (units) of 499'),
    ]
    i = 0
    while i < len(h):
        hit = False
        for txt, acc, why in specials:
            if h.startswith(txt, i):
                units.append((txt, acc, why)); i += len(txt); hit = True; break
        if hit:
            continue
        c = h[i]
        if c in CONS:
            txt = c; j = i + 1
            while j < len(h) and h[j] in '़्':
                txt += h[j]; j += 1
            base = [CONS[c]] + CONS_ALT.get(c, [])
            nxt = h[j] if j < len(h) else ''
            acc = list(base)
            if '्' not in txt and nxt not in MATRA:
                acc += [b + 'a' for b in base]  # inherent schwa (optional by schwa deletion)
            units.append((txt, acc, 'dev-consonant')); i = j
        elif c in MATRA:
            units.append((c, MATRA[c], 'dev-vowel-sign')); i += 1
        elif c in INDEP:
            units.append((c, INDEP[c], 'dev-vowel')); i += 1
        elif c in NASAL:
            units.append((c, NASAL[c], 'dev-nasal')); i += 1
        elif c == 'g' and (i + 1 == len(h) or h[i + 1] == ' ') and (i == 0 or h[i - 1] == ' '):
            units.append((c, ['ji', 'g'], 'latin-letter G (pronounced "jee")')); i += 1
        else:
            units.append((c, [c], 'latin')); i += 1
    return units


def align(r, U):
    n, m = len(r), len(U)
    INF = 10 ** 9
    EPS = 1e-6
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            v = dp[i][j]
            if v >= INF:
                continue
            # del
            if i < n and v + 1 + EPS * (n - i) < dp[i + 1][j]:
                dp[i + 1][j] = v + 1 + EPS * (n - i); bp[i + 1][j] = ('del', i, j, 1)
            # ins (tie-break: prefer insertions/deletions later in the string, so trailing
            # extra hypothesis text is inserted as a block instead of scattering ref matches)
            if j < m and v + 1 + EPS * (m - j) < dp[i][j + 1]:
                dp[i][j + 1] = v + 1 + EPS * (m - j); bp[i][j + 1] = ('ins', i, j, 0)
            if j < m:
                for k in (1, 2, 3):
                    if i + k <= n and r[i:i + k] in U[j][1]:
                        if v < dp[i + k][j + 1] or (v == dp[i + k][j + 1] and bp[i + k][j + 1][0] != 'match'):
                            dp[i + k][j + 1] = v; bp[i + k][j + 1] = ('match', i, j, k)
                if i < n and (r[i] == ' ') == (U[j][0] == ' ') and v + 1 < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = v + 1; bp[i + 1][j + 1] = ('sub', i, j, 1)
    steps = []
    i, j = n, m
    while i or j:
        op, pi, pj, k = bp[i][j]
        if op == 'match':
            steps.append(('match', r[pi:pi + k], U[pj][0], U[pj][2]))
        elif op == 'sub':
            steps.append(('sub', r[pi], U[pj][0], U[pj][2]))
        elif op == 'del':
            steps.append(('del', r[pi], '', ''))
        else:
            steps.append(('ins', '', U[pj][0], U[pj][2]))
        i, j = pi, pj
    return round(dp[n][m]), steps[::-1]


def why(op, ref, hyp, kind):
    if op == 'match':
        if kind.startswith('number-word') or kind.startswith('latin-letter'):
            return kind
        if kind.startswith('dev'):
            return f'{kind} "{hyp}" sounds "{ref}"'
        return 'same character/sound' if ref != ' ' else 'word boundary'
    if op == 'sub':
        return f'different sound: ref "{ref}" vs hyp "{hyp}"'
    if op == 'del':
        return f'ref "{ref}" not heard in hypothesis'
    return f'hyp "{hyp}" has no counterpart in reference'


ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read())
U = hyp_units(hyp)
assert ''.join(u[0] for u in U) == hyp
cost, steps = align(ref, U)

rows = []
for n_, (op, rr, hh, kind) in enumerate(steps):
    rows.append({'i': n_, 'ref': rr, 'hyp': hh, 'op': op, 'why': why(op, rr, hh, kind)})

with open(os.path.join(D, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
    for r_ in rows:
        f.write(json.dumps(r_, ensure_ascii=False) + '\n')
print('cost', cost, 'ref_len', len(ref))
