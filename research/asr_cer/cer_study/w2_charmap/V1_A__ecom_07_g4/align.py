#!/usr/bin/env python3
"""Cross-script char-level alignment: Roman Hinglish reference vs Devanagari/Latin Whisper hyp.
DP with phonetic compatibility table; manual overrides applied afterwards in fix.py."""
import json, re, unicodedata, sys, os
D = os.path.dirname(os.path.abspath(__file__))
def norm_ref(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation (hyphen in H-12 too)
    return re.sub(r"\s+", " ", s).strip()
def norm_hyp(s):
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()
GLUE = {"\u094d", "\u093c"}  # halant, nukta glue to previous char
def to_units(h):
    units = []
    for ch in h:
        if ch in GLUE and units: units[-1] += ch
        else: units.append(ch)
    return units
ROM = {
 'क':'kcq','ख':'k','ग':'g','घ':'g','ङ':'n','च':'c','छ':'c','ज':'jz','झ':'j','ञ':'n',
 'ट':'t','ठ':'t','ड':'dr','ढ':'d','ण':'n','त':'t','थ':'t','द':'d','ध':'d','न':'n',
 'प':'p','फ':'fp','ब':'b','भ':'b','म':'m','य':'y','र':'r','ल':'l','व':'vw','श':'s','ष':'s','स':'sc','ह':'h',
 'अ':'a','आ':'a','इ':'iey','ई':'iey','उ':'uo','ऊ':'uo','ए':'eay','ऐ':'ae','ओ':'o','औ':'ao','ऑ':'oa',
 'ा':'a','ि':'iey','ी':'iey','ु':'u','ू':'uo','े':'ea','ै':'ae','ो':'o','ौ':'oa','ॉ':'oa','ं':'nm','ँ':'nm','ः':'h',
}
ASP = set('खघछझठढथधफभशषच')
CONS = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
VOW = set('अआइईउऊएऐओऔऑािीुूेैोौॉ')
def rom(u):
    b = u[0]
    if b.isascii(): return b.lower()
    return ROM.get(b, '')
def compat(c, u):
    if c == ' ' or u == ' ': return c == u
    return c in rom(u)
def empty_ok(REF, i, prev):
    """REF[i] consumes no hyp unit but is still a match given last hyp unit prev.
    Only called when the previous step was a matching diag (attached to a hyp unit)."""
    c = REF[i]; nxt = REF[i+1] if i+1 < len(REF) else ' '; prv = REF[i-1] if i else ' '
    if prev is None or prev == ' ': return None
    b = prev[0]
    if b.isascii(): return None
    hal = '\u094d' in prev
    if c == 'h' and b in ASP and not hal: return 'aspirate digraph'
    if c == 'a' and b in CONS and not hal: return 'inherent vowel'
    if c == 'e' and nxt == 'r' and b in CONS and not hal: return 'schwa -er ~ inherent vowel'
    if c == 'c' and nxt == 'k': return 'ck digraph'
    if c == 'e' and nxt == ' ' and prv not in 'aeiou ' and b in CONS: return 'silent final e'
    if c in 'aeiou' and b in VOW:
        if c in rom(prev): return 'long-vowel double spelling'
        if b in 'ैऐ' and c == 'i': return 'ai ~ ai matra'
        if b in 'ौऔ' and c == 'u': return 'au ~ au matra'
    if c == 'y' and b in 'ेए' and prv == 'a': return 'ay digraph'
    return None
def align(REF, units):
    n, m = len(REF), len(units)
    INF = float('inf')
    # state s: 1 if last step was a matching diag (hyp unit attached), else 0
    dp = [[[INF, INF] for _ in range(m+1)] for _ in range(n+1)]
    bt = [[[None, None] for _ in range(m+1)] for _ in range(n+1)]
    dp[0][0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            for s_ in (0, 1):
                cur = dp[i][j][s_]
                if cur == INF: continue
                if i < n and j < m:
                    ok = compat(REF[i], units[j]); ns = 1 if ok else 0
                    v = cur + (0 if ok else 1)
                    if v < dp[i+1][j+1][ns]: dp[i+1][j+1][ns] = v; bt[i+1][j+1][ns] = ('diag', s_, None)
                if i < n:
                    why = empty_ok(REF, i, units[j-1] if j else None) if s_ == 1 else None
                    v = cur + (0 if why else 1) + 0.001
                    if v < dp[i+1][j][0]: dp[i+1][j][0] = v; bt[i+1][j][0] = ('up', s_, why)
                if j < m:
                    v = cur + 1 + 0.001
                    if v < dp[i][j+1][0]: dp[i][j+1][0] = v; bt[i][j+1][0] = ('left', s_, None)
    steps = []
    i, j = n, m; s_ = 0 if dp[n][m][0] <= dp[n][m][1] else 1
    while i or j:
        t, ps, why = bt[i][j][s_]
        if t == 'diag':
            r, h = REF[i-1], units[j-1]
            steps.append([r, h, 'match' if compat(r, h) else 'sub', '' if compat(r,h) else 'different sound']); i -= 1; j -= 1
        elif t == 'up':
            steps.append([REF[i-1], '', 'match' if why else 'del', why or 'not in hypothesis']); i -= 1
        else:
            steps.append(['', units[j-1], 'ins', 'extra in hypothesis']); j -= 1
        s_ = ps
    steps.reverse()
    return steps
