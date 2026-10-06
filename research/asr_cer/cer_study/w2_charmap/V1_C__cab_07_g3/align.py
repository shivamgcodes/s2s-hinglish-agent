#!/usr/bin/env python3
"""Character-level cross-script (Roman ref vs Devanagari/Latin hyp) alignment.
Phonetic-equivalence DP; outputs char_map.jsonl, char_metrics.json, char_map.txt.
Manual overrides (judgement calls) are applied from overrides.json if present."""
import json, re, unicodedata, os, sys

D = os.path.dirname(os.path.abspath(__file__))

def norm(s, lower):
    s = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower() if lower else s

REF = norm(open(f'{D}/reference.txt', encoding='utf-8').read(), True)
HYP = norm(open(f'{D}/whisper.txt', encoding='utf-8').read(), False)

# Devanagari -> plausible Roman letters
# aspirates list only their first Roman letter; the following 'h' is absorbed by free_ok
CONS = {
 'क':'kcq','ख':'k','ग':'g','घ':'g','ङ':'n','च':'c','छ':'c','ज':'jz','झ':'j','ञ':'n',
 'ट':'t','ठ':'t','ड':'d','ढ':'d','ण':'n','त':'t','थ':'t','द':'d','ध':'d','न':'n',
 'प':'p','फ':'pf','ब':'b','भ':'b','म':'m','य':'y','र':'r','ल':'l','व':'vw','श':'s',
 'ष':'s','स':'sc','ह':'h'}
VOW = {'अ':'au','आ':'a','इ':'i','ई':'ie','उ':'u','ऊ':'uo','ए':'e','ऐ':'ae','ओ':'o','औ':'ao','ऑ':'o','ऋ':'r'}
MATRA = {'ा':'a','ि':'i','ी':'iey','ु':'u','ू':'uo','े':'e','ै':'ae','ो':'o','ौ':'ao','ॉ':'o','ृ':'r','ं':'nm','ँ':'n'}
COMBINE = {'्','़'}

# hypothesis units: merge virama/nukta into preceding char
units = []
for c in HYP:
    if c in COMBINE and units:
        units[-1] += c
    else:
        units.append(c)

def base(u): return u[0]
def is_cons(u): return base(u) in CONS
def letters(u):
    b = base(u)
    if b in CONS: return CONS[b] + ('z' if b=='ज' and '़' in u else '') + ('f' if b=='फ' else '')
    if b in VOW: return VOW[b]
    if b in MATRA: return MATRA[b]
    return b.lower()

def is_latin_unit(u): return u.isascii()

ROMAN_CONS = set('bcdfghjklmnpqrstvwxyz')

def sub_cost(i, j):
    """cost of aligning REF[i] with units[j]"""
    r, u = REF[i], units[j]
    if r == ' ' or u == ' ':
        return (0, 'match') if r == u else (1, 'sub')
    if is_latin_unit(u):
        return (0, 'match') if r == u.lower() else (1, 'sub')
    if r in letters(u):
        return (0, 'match')
    return (1, 'sub')

def free_ok(i, j):
    """REF[i] may map to '' as match (absorbed by previous hyp unit)."""
    r = REF[i]
    if i == 0 or j == 0: return False
    p, pu = REF[i-1], units[j-1]
    if is_latin_unit(pu): return False
    nxt = units[j] if j < len(units) else ' '
    bare = is_cons(pu) and '्' not in pu and not (base(nxt) in MATRA and base(nxt) not in ('ं','ँ'))
    if r == 'a' and p in ROMAN_CONS and bare: return 'inherent vowel of ' + pu
    if r == 'e' and p in ROMAN_CONS and bare and i+1 < len(REF) and REF[i+1] == 'r': return "English 'er' schwa = inherent vowel of " + pu
    if r == 'e' and p in ROMAN_CONS and (i+1 == len(REF) or REF[i+1] == ' ') and not is_latin_unit(pu): return "silent final 'e'"
    if r == 'u' and p == 'o' and base(pu) in ('ो','ओ'): return "'ou' spelled by " + pu
    if r == 'a' and p == 'a' and base(pu) in ('ा','आ'): return "'aa' spelled by " + pu
    if r == 'h' and p in 'kgcjtdpbs' and base(pu) in ('ख','घ','छ','झ','ठ','ढ','थ','ध','फ','भ','श','ष'): return 'aspirate/digraph in ' + pu
    if r == p and r in 'eoi' and base(pu) in MATRA or (r==p and r in 'eoi' and base(pu) in VOW): return 'doubled vowel spelled by ' + pu
    if r == 'o' and p == 'o' and base(pu) in ('ू','ऊ'): return "'oo' spelled by " + pu
    if r == 'i' and p == 'e' and base(pu) in ('ी','ई'): return "'ee' spelled by " + pu
    if r == 'e' and p == 'e' and base(pu) in ('ी','ई'): return "'ee' spelled by " + pu
    if r == 'i' and p == 'a' and base(pu) in ('ै','ऐ','े','ए'): return "'ai' spelled by " + pu
    if r == 'e' and p == 'a' and base(pu) in ('ै','ऐ','े','ए'): return "'ae' spelled by " + pu
    if r == 'u' and p == 'a' and base(pu) in ('ौ','औ'): return "'au' spelled by " + pu
    if r == 'u' and p in ROMAN_CONS and bare: return "English 'u' (schwa) = inherent vowel of " + pu
    if r == 'k' and p == 'c' and base(pu) == 'क': return "'ck' spelled by " + pu
    if r == p and r in ROMAN_CONS and not is_latin_unit(pu) and r in letters(pu): return 'doubled consonant spelled by single ' + pu
    return False

# word index of each ref char
WORD = []
for wi, wd in enumerate(REF.split(' ')):
    WORD += [wd]*len(wd) + [' ']
WORD = WORD[:len(REF)]
# (ref word, ref char) -> hyp spans that spell the same SOUND (letter names, numbers, 'th', 'ow')
# (ref word, ref char) -> [(hyp span, op, why)]  -- cross-script judgement calls
SPECIAL = {
 ('igi','i'):[('आई','match',"letter name I")], ('igi','g'):[('जी','match',"letter name G")],
 ('dl','d'):[('डी','match',"letter name D")], ('dl','l'):[('एल','match',"letter name L")],
 ('3','3'):[('थ्री','match',"digit 3 spoken 'three'")], ('1','1'):[('वन','match',"digit 1 spoken 'one'")],
 ('01','0'):[('जी रो','match',"digit 0 spoken 'zero' (z~ज)")],
 ('01','1'):[('वा','sub',"digit 1 'one' heard as 'वा' (final n missing)")],
 ('2','2'):[('दू','sub',"digit 2 'two' heard as 'दू' (t vs d)")],
 ('the','t'):[('द','match',"English 'th' in 'the' ~ द")],
 ('now','o'):[('ा','match',"'ow' in now = /aa-o/: o~ा")], ('now','w'):[('ओ','match',"'ow' in now: w~ओ")],
 ('driver','i'):[('ाइ','match',"'i' in driver = /aai/ ~ ाइ")],
 ('aerocity','a'):[('ए','match',"'ae' in aero ~ ए")],
 ('airport','a'):[('ए','match',"'ai' in airport = /e/ ~ ए")],
}
SILENT = {('building','u'):"silent 'u' in building", ('done','o'):"'o' in done = schwa, inherent vowel of ड"}
N, M = len(REF), len(units)
INF = 1e9
cost = [[INF]*(M+1) for _ in range(N+1)]
bp = [[None]*(M+1) for _ in range(N+1)]
cost[0][0] = 0
for i in range(N+1):
    for j in range(M+1):
        c0 = cost[i][j]
        if c0 >= INF: continue
        if i < N and j < M:
            c, op = sub_cost(i, j)
            if c0 + c < cost[i+1][j+1]:
                cost[i+1][j+1] = c0 + c; bp[i+1][j+1] = (i, j, op, '')
        if i < N:
            for sp, sop, swhy in SPECIAL.get((WORD[i], REF[i]), []):
                k, acc = j, ''
                while k < M and len(acc) < len(sp): acc += units[k]; k += 1
                sc = 0 if sop == 'match' else 1
                if acc == sp and c0 + sc < cost[i+1][k]:
                    cost[i+1][k] = c0 + sc; bp[i+1][k] = (i, j, sop, swhy)
            if (WORD[i], REF[i]) in SILENT and c0 + 0.001 < cost[i+1][j]:
                cost[i+1][j] = c0 + 0.001; bp[i+1][j] = (i, j, 'match', SILENT[(WORD[i], REF[i])])
            if WORD[i] == 'the' and REF[i] == 'h' and j and base(units[j-1]) == 'द':
                if c0 + 0.001 < cost[i+1][j]:
                    cost[i+1][j] = c0 + 0.001; bp[i+1][j] = (i, j, 'match', "'th' spelled by द")
            fr = free_ok(i, j)
            c, op, why = (0.001, 'match', fr) if fr else (1, 'del', '')
            if c0 + c < cost[i+1][j]:
                cost[i+1][j] = c0 + c; bp[i+1][j] = (i, j, op, why)
        if j < M:
            if c0 + 1 < cost[i][j+1]:
                cost[i][j+1] = c0 + 1; bp[i][j+1] = (i, j, 'ins', '')
steps = []
i, j = N, M
while (i, j) != (0, 0):
    pi, pj, op, why = bp[i][j]
    steps.append({'ref': REF[pi:i], 'hyp': ''.join(units[pj:j]), 'op': op, 'why': why})
    i, j = pi, pj
steps.reverse()

# manual overrides: list of {"i": idx, ...fields}
ov_path = f'{D}/overrides.json'
if os.path.exists(ov_path):
    for o in json.load(open(ov_path, encoding='utf-8')):
        steps[o['i']].update({k: v for k, v in o.items() if k != 'i'})

for k, s in enumerate(steps):
    s['i'] = k
    if not s['why'] and s['op'] != 'match':
        if s['op'] == 'sub': s['why'] = f"'{s['ref']}' vs '{s['hyp']}': different sound"
        elif s['op'] == 'del': s['why'] = f"'{s['ref']}' not spoken/transcribed"
        else: s['why'] = f"'{s['hyp']}' has no reference counterpart"
    if not s['why'] and s['op'] == 'match' and s['hyp'] and not s['hyp'].isascii() and s['ref'] != ' ':
        s['why'] = f"'{s['ref']}' ~ '{s['hyp']}'"

with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for s in steps:
        f.write(json.dumps({k: s[k] for k in ('i','ref','hyp','op','why')}, ensure_ascii=False) + '\n')

# verify
assert ''.join(s['ref'] for s in steps) == REF, 'ref concat mismatch'
assert ''.join(s['hyp'] for s in steps) == HYP, 'hyp concat mismatch'
for s in steps:
    assert (s['op']=='ins') == (s['ref']=='') and (s['op']=='del') <= (s['hyp']=='')
    assert len(s['ref']) <= 1

cnt = {o: sum(1 for s in steps if s['op']==o) for o in ('match','sub','del','ins')}
n = len(REF)
cer = (cnt['sub']+cnt['del']+cnt['ins'])/n
metrics = {'pair':'V1_C__cab_07_g3','ref_chars':n,'hyp_chars':len(HYP),**cnt,
           'errors':cnt['sub']+cnt['del']+cnt['ins'],'cer':round(cer,4),
           'normalised_reference':REF,'normalised_hypothesis':HYP,
           'checks':{'ref_concat_exact':True,'hyp_concat_exact':True}}
json.dump(metrics, open(f'{D}/char_metrics.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)

# human readable
def w(s): return max(1, len(s['ref']), sum(1 for c in s['hyp'] if unicodedata.category(c) not in ('Mn','Mc')))
lines = []
blk = []; rc = 0
def flush():
    if not blk: return
    r = h = o = ''
    for s in blk:
        width = w(s)
        r += (s['ref'] or '·').ljust(width)
        hv = s['hyp'] or '·'
        h += hv + ' '*(width - sum(1 for c in hv if unicodedata.category(c) not in ('Mn','Mc')))
        o += {'match':' ','sub':'S','del':'D','ins':'I'}[s['op']].ljust(width)
    lines.extend(['REF: '+r, 'HYP: '+h, 'OP:  '+o, ''])
for s in steps:
    blk.append(s); rc += len(s['ref'])
    if rc >= 80 and s['ref'] == ' ':
        flush(); blk = []; rc = 0
flush()
hdr = f"V1_C__cab_07_g3  ref_chars={n} match={cnt['match']} sub={cnt['sub']} del={cnt['del']} ins={cnt['ins']} CER={cer:.4f}\n('·' = empty side; widths approximate for Devanagari)\n\n"
open(f'{D}/char_map.txt','w',encoding='utf-8').write(hdr + '\n'.join(lines))
print(json.dumps({k: metrics[k] for k in ('ref_chars','match','sub','del','ins','cer')}))
