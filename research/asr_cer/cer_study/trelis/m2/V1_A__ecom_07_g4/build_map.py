#!/usr/bin/env python3
"""Build sound-aligned char map for V1_A__ecom_07_g4 (ref Roman Hinglish vs Trelis Whisper hyp)."""
import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))

def norm(s):
    s = s.lower()
    s = ''.join(ch for ch in s if not unicodedata.category(ch).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()

REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read())
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read())

M, S, Dl, I = 'match', 'sub', 'del', 'ins'
# explicit piece lists: (ref_piece, hyp_piece, op, why). Multi-char ref piece -> first char carries hyp, rest get "" with same op.
def P(*items): return ('X', list(items))
DEV = {
 'shopkart': P(('s','श',M,'sh~श'),('h','',M,'sh digraph = श'),('o','ा',S,'o vs aa'),('p','प',M,''),('k','क',M,''),('a','ा',M,''),('r','र',M,''),('t','',Dl,'final t missing')),
 'rohit': P(('r','र',M,''),('o','ो',M,''),('h','ह',M,''),('i','ि',M,''),('t','त',M,'')),
 'main': P(('m','म',M,''),('a','ै',M,'ai~ै'),('i','',M,'ai digraph = ै'),('n','ं',M,'nasal')),
 'aapki': P(('a','आ',M,'aa~आ'),('a','',M,'aa digraph'),('p','प',M,''),('k','क',M,''),('i','ी',M,'')),
 'kaise': P(('k','क',M,''),('a','ै',M,'ai~ै'),('i','',M,'ai digraph'),('s','स',M,''),('e','े',M,'')),
 'kar': P(('k','क',M,''),('a','',M,'inherent schwa of क'),('r','र',M,'')),
 'sakta': P(('s','स',M,''),('a','',M,'inherent schwa'),('k','क',M,''),('t','त',M,''),('a','ा',M,'')),
 'hoon': P(('h','ह',M,''),('o','ू',M,'oo~ू'),('o','',M,'oo digraph'),('n','ँ',M,'nasal')),
 'ji': P(('j','ज',M,''),('i','ी',M,'')),
 'aapka': P(('a','आ',M,'aa~आ'),('a','',M,'aa digraph'),('p','प',M,''),('k','क',M,''),('a','ा',M,'')),
 'bataiye': P(('b','ब',M,''),('a','',M,'inherent schwa'),('t','त',M,''),('a','ा',M,''),('i','इ',M,''),('y','',M,'glide between i and e, same pronunciation'),('e','ए',M,'')),
 'aur': P(('a','औ',M,'au~औ'),('u','',M,'au digraph'),('r','र',M,'')),
 'ek': P(('e','ए',M,''),('k','क',M,'')),
 'rohini': P(('r','र',M,''),('o','ो',M,''),('h','ह',M,''),('i','ि',M,''),('n','न',M,''),('i','ी',M,'')),
 'delhi': P(('d','द',M,''),('e','ि',S,'e vs i'),('l','ल्',M,''),('h','ल',S,'h vs geminate l'),('i','ी',M,'')),
 'lekin': P(('l','ल',M,''),('e','े',M,''),('k','क',M,''),('i','ि',M,''),('n','न',M,'')),
 'ko': P(('k','क',M,''),('o','ो',M,'')),
 'hai': P(('h','ह',M,''),('a','ै',M,'ai~ै'),('i','',M,'ai digraph')),
}
BOAT = P(('b','ब',M,''),('o','ु',S,'o vs u'),('a','',Dl,'oa digraph /o/ not realised'),('t','थ्',S,'t vs aspirated th'),
         (' ','',Dl,'hyp merged words'),('r','र',M,''),('o','ॉ',M,''),('c','क',M,''),('k','',M,'ck digraph = single k'),
         ('e','',M,'vowel carried by क inherent schwa'),('r','र्',M,''),('z','स',S,'z vs s'))
N450 = P(('4','four hundred',M,'same number 450'),('5',' fifty',M,''),('0','',M,'part of fifty'))
N25 = P(('2','twenty',M,'tens part spoken'),('5','',Dl,'25 vs 20: units missing'))
EC = P(('e','a',S,'EC vs AC'),('c','c',M,''),('3',' three',M,''),('1',' one',M,''),('4',' four',M,''),('7',' seven',M,''))
H12 = P(('h','is',S,'H vs is'),('1',' twelve',M,'same number 12'),('2','',M,'part of twelve'))
N15 = P(('1','fifteen',M,'same number 15'),('5','',M,'part of fifteen'))
JIG = P(('j','g',M,'letter g pronounced jee = ji'),('i','',M,'vowel in g name'))

# word-level alignment entries, in order
rw = REF.split(' ')
hw = HYP.split(' ')
entries = []  # each: ('A', refword, hypword) roman auto, or ('X', pieces, nref, nhyp)
ri = hi = 0
special = {('boat',):(BOAT,2,1), ('450',):(N450,1,3), ('25',):(N25,1,1), ('ec3147',):(EC,1,5),
           ('h12',):(H12,1,2), ('15',):(N15,1,1)}
while ri < len(rw):
    w = rw[ri]
    if (w,) in special:
        p, nr, nh = special[(w,)]
        entries.append(('X', p[1], nr, nh)); ri += nr; hi += nh; continue
    if w == 'ji' and hw[hi] == 'g':
        entries.append(('X', JIG[1], 1, 1)); ri += 1; hi += 1; continue
    if w in DEV and any('ऀ' <= c <= 'ॿ' for c in hw[hi]):
        entries.append(('X', DEV[w][1], 1, 1)); ri += 1; hi += 1; continue
    entries.append(('A', w, hw[hi])); ri += 1; hi += 1
assert hi == len(hw), (hi, len(hw))

def lev(a, b):
    n, m = len(a), len(b)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(a[i-1] != b[j-1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1]+(a[i-1] != b[j-1]):
            out.append((a[i-1], b[j-1], M if a[i-1]==b[j-1] else S)); i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j]+1:
            out.append((a[i-1], '', Dl)); i -= 1
        else:
            out.append(('', b[j-1], I)); j -= 1
    return out[::-1]

steps = []
for k, e in enumerate(entries):
    if k: steps.append((' ', ' ', M, 'word boundary'))
    if e[0] == 'A':
        if e[1] != e[2] and any('ऀ' <= c <= 'ॿ' for c in e[2]):
            raise SystemExit(f'unhandled devanagari {e}')
        for r, h, op in lev(e[1], e[2]):
            why = '' if op == M else ('different sound' if op == S else ('missing in hyp' if op == Dl else 'extra in hyp'))
            steps.append((r, h, op, why))
    else:
        for r, h, op, why in e[1]:
            if len(r) <= 1:
                steps.append((r, h, op, why))
            else:
                steps.append((r[0], h, op, why))
                for c in r[1:]: steps.append((c, '', op, why))

with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for i, (r, h, op, why) in enumerate(steps):
        f.write(json.dumps({'i': i, 'ref': r, 'hyp': h, 'op': op, 'why': why}, ensure_ascii=False) + '\n')
print('steps', len(steps))
