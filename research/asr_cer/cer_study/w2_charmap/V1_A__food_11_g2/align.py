#!/usr/bin/env python3
"""Character-level cross-script alignment: Roman Hinglish reference vs Whisper (Devanagari+Latin).

Hyp units = individual code points. Each Devanagari code point carries a set of
Roman spellings (sound variants). A hyp unit may cover 0..3 reference chars.
Cost of a unit covering ref substring s = len(s) - best LCS(s, variant)  (ref chars
not spelled by the unit); a unit covering 0 ref chars costs 1 (insertion) unless it
is silent (virama/nukta), which is free and merged into the previous step.
Within a unit's span, the code point(s) are recorded on the FIRST ref char of the
span; further ref chars of the same sound get hyp '' and op match ("continuation").
"""
import json, os, re, sys, unicodedata

D = os.path.dirname(os.path.abspath(__file__))

def strip_punct(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('P'))

def norm_ref(s):
    return re.sub(r'\s+', ' ', strip_punct(s.lower())).strip()

def norm_hyp(s):
    return re.sub(r'\s+', ' ', strip_punct(s)).strip()

CONS = {
 'क':['k','c','q','ck'], 'ख':['kh'], 'ग':['g'], 'घ':['gh'], 'ङ':['n'],
 'च':['ch','c'], 'छ':['chh','ch'], 'ज':['j','z','g'], 'झ':['jh'], 'ञ':['n'],
 'ट':['t'], 'ठ':['th'], 'ड':['d'], 'ढ':['dh'], 'ण':['n'],
 'त':['t'], 'थ':['th'], 'द':['d','th'], 'ध':['dh'], 'न':['n'],
 'प':['p'], 'फ':['ph','f'], 'ब':['b'], 'भ':['bh'], 'म':['m'],
 'य':['y'], 'र':['r'], 'ल':['l'], 'व':['v','w'], 'श':['sh','s'], 'ष':['sh'],
 'स':['s','c'], 'ह':['h'], 'ज़':['z'], 'फ़':['f'],
}
MATRA = {
 'ा':['aa','a'], 'ि':['i','e'], 'ी':['ee','i','y'], 'ु':['u','oo','o'], 'ू':['oo','u','ou'],
 'े':['e','ay','a'], 'ै':['ai','e','a'], 'ो':['o'], 'ौ':['au','o','ou'], 'ृ':['ri'],
 'ॉ':['o','aw','au','a'], 'ं':['n','m'], 'ँ':['n'], 'ः':['h'],
}
VOW = {
 'अ':['a','u'], 'आ':['aa','a'], 'इ':['i','e'], 'ई':['ee','i'], 'उ':['u','oo'], 'ऊ':['oo','u'],
 'ए':['e','a'], 'ऐ':['ai','e','a'], 'ओ':['o'], 'औ':['au','o'], 'ऑ':['o','aw','au'], 'ऋ':['ri'],
}
SILENT = {'्', '़'}

def units(h):
    """List of (text, variants). Devanagari consonants get inherent-'a' variants when
    not followed by a matra/virama."""
    out = []
    i = 0
    while i < len(h):
        c = h[i]
        nxt = h[i+1] if i+1 < len(h) else ''
        if c in CONS:
            base = CONS[c]
            # nukta forms
            if nxt == '़':
                k = c + nxt
                if k in CONS:
                    base = CONS[k]
            j = i+1
            while j < len(h) and h[j] in SILENT:
                j += 1
            follow = h[j] if j < len(h) else ''
            vs = list(base) + [b[0] + b for b in base if len(b) == 1]  # doubled letters: ll, ff, tt
            has_matra = follow in MATRA and follow not in ('ं', 'ँ', 'ः')
            if not has_matra and '्' not in h[i+1:j]:
                vs += [b + 'a' for b in base] + [b + 'e' for b in base] + [b + 'u' for b in base]
            out.append((c, vs))
        elif c in MATRA:
            out.append((c, MATRA[c]))
        elif c in VOW:
            out.append((c, VOW[c]))
        elif c in SILENT:
            out.append((c, None))
        elif c == ' ':
            out.append((c, [' ']))
        else:
            out.append((c, [c.lower()]))
        i += 1
    return out

def lcs_align(s, v):
    """Return (matches, per-ref-char flags list: 'm','s','d') maximising matches; ref chars
    aligned to an unequal variant char are 's', unaligned are 'd'."""
    n, m = len(s), len(v)
    # DP maximise matches, tie-break: prefer subs over dels (more aligned pairs)
    best = [[None]*(m+1) for _ in range(n+1)]
    for a in range(n+1):
        for b in range(m+1):
            if a == 0:
                best[a][b] = (0, 0, [])
                continue
            cands = []
            # ref char a-1 deleted
            x = best[a-1][b]; cands.append((x[0], x[1], x[2]+['d']))
            if b > 0:
                x = best[a-1][b-1]
                if s[a-1] == v[b-1]:
                    cands.append((x[0]+1, x[1]+1, x[2]+['m']))
                else:
                    cands.append((x[0], x[1]+1, x[2]+['s']))
                x = best[a][b-1]; cands.append(x)  # free variant extra
            best[a][b] = max(cands, key=lambda t: (t[0], t[1]))
    return best[n][m]

def cls(c):
    return 'space' if c == ' ' else ('digit' if c.isdigit() else 'letter')

def span_cost(s, vs):
    # a substitution across classes (space/digit/letter) is not a sound confusion: disallow
    if all(cls(s[0]) != cls(v[0]) for v in vs):
        return 10**6, None, None
    bestr = None
    for v in vs:
        if cls(s[0]) != cls(v[0]):
            continue
        if s[0] != v[0]:
            # the span must begin with the unit's own leading sound; otherwise a plain substitution
            r = (0, len(s), ['s'] + ['d'] * (len(s) - 1))
        else:
            r = lcs_align(s, v)
        key = (len(s) - r[0], -r[1])
        if bestr is None or key < bestr[0]:
            bestr = (key, r, v)
    if bestr is None:
        return 10**6, None, None
    return bestr[0][0], bestr[1][2], bestr[2]

def align(ref, hyp, nomatch=False):
    """nomatch=True: segment judged to have no sound correspondence (hallucination/unspoken);
    letters may only be substituted, never matched (spaces may still pair with spaces)."""
    U = units(hyp)
    n, m = len(ref), len(U)
    INF = 10**9
    dp = [[INF]*(m+1) for _ in range(n+1)]
    bp = [[None]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    cache = {}
    for i in range(n+1):
        for j in range(m+1):
            if dp[i][j] >= INF:
                continue
            cur = dp[i][j]
            # delete ref char
            if i < n and cur + 1 < dp[i+1][j]:
                dp[i+1][j] = cur + 1; bp[i+1][j] = ('del', i, j, None)
            if j < m:
                txt, vs = U[j]
                if vs is None:  # silent mark: free
                    if cur < dp[i][j+1]:
                        dp[i][j+1] = cur; bp[i][j+1] = ('silent', i, j, None)
                    continue
                # insertion
                if cur + 1 < dp[i][j+1]:
                    dp[i][j+1] = cur + 1; bp[i][j+1] = ('ins', i, j, None)
                for k in (1, 2, 3):
                    if i + k > n:
                        break
                    s = ref[i:i+k]
                    if ' ' in s and k > 1:
                        break
                    key = (s, j)
                    if key not in cache:
                        cache[key] = span_cost(s, vs)
                    c, flags, v = cache[key]
                    if flags is None:
                        continue
                    if nomatch and s != ' ':
                        if k > 1:
                            break
                        c, flags = 1, ['s']
                    if k > 1 and flags.count('m') < k:  # multi-char spans only when all spelled
                        continue
                    if cur + c < dp[i+k][j+1] or (cur + c == dp[i+k][j+1] and bp[i+k][j+1][0] in ('del','ins') ):
                        dp[i+k][j+1] = cur + c; bp[i+k][j+1] = ('span', i, j, (k, flags, v))
    # backtrack
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        kind, pi, pj, info = bp[i][j]
        if kind == 'del':
            steps.append(dict(ref=ref[pi], hyp='', op='del'))
        elif kind == 'ins':
            steps.append(dict(ref='', hyp=U[pj][0], op='ins'))
        elif kind == 'silent':
            steps.append(dict(ref=None, hyp=U[pj][0], op='silent'))
        else:
            k, flags, v = info
            grp = []
            for t in range(k):
                op = {'m': 'match', 's': 'sub', 'd': 'del'}[flags[t]]
                grp.append(dict(ref=ref[pi+t], hyp=U[pj][0] if t == 0 else '', op=op, var=v,
                                cont=(t > 0)))
            steps.extend(reversed(grp))
        i, j = pi, pj
    steps.reverse()
    # merge silent marks into previous step's hyp
    merged = []
    for s in steps:
        if s['op'] == 'silent':
            if merged:
                merged[-1]['hyp'] += s['hyp']
            else:
                merged.append(dict(ref='', hyp=s['hyp'], op='ins'))
            continue
        merged.append(s)
    # a 'del' inside a span that got '' hyp: fine. A 'sub'/'del' continuation is honest.
    return merged

def main():
    ref = norm_ref(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
    hyp = norm_hyp(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read())
    steps = align(ref, hyp)
    json.dump(dict(ref=ref, hyp=hyp, steps=steps), open(os.path.join(D, 'auto_align.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)
    print(len(ref), len(hyp), len(steps))

if __name__ == '__main__':
    main()
