#!/usr/bin/env python3
"""Character-level cross-script (Roman ref vs Devanagari/Latin hyp) alignment + CER.

Hyp is split into units: Latin chars / digits / spaces individually; Devanagari
consonant(+nukta)(+virama) as one unit (inherent vowel implied when no matra/virama
follows), each matra its own unit, independent vowels their own unit; anusvara /
chandrabindu are attached to the preceding unit (optional 'n'/'m'). Each unit has a
set of acceptable Roman spellings (sound-equivalent). A DP aligns ref chars to units:
match (a ref segment of 1..4 chars spelling the unit's sound; first char carries the
unit, the rest carry '' and are also 'match'), sub (1 ref char <-> 1 unit), del, ins.
"""
import json, os, re, unicodedata, itertools

D = os.path.dirname(os.path.abspath(__file__))

def norm(s, lower):
    s = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)
    if lower:
        s = s.lower()
    return re.sub(r'\s+', ' ', s).strip()

CONS = {
 'क': ['k','c','ck','q'], 'ख': ['kh','k'], 'ग': ['g','gh'], 'घ': ['gh','g'], 'ङ': ['n'],
 'च': ['ch','c'], 'छ': ['chh','ch'], 'ज': ['j','z','g','s'], 'झ': ['jh','j'], 'ञ': ['n'],
 'ट': ['t'], 'ठ': ['th','t'], 'ड': ['d'], 'ढ': ['dh','d'], 'ण': ['n'],
 'त': ['t','th'], 'थ': ['th','t'], 'द': ['d','th','dh'], 'ध': ['dh','d'], 'न': ['n'],
 'प': ['p'], 'फ': ['ph','f'], 'ब': ['b'], 'भ': ['bh','b'], 'म': ['m'],
 'य': ['y','i'], 'र': ['r'], 'ल': ['l'], 'व': ['v','w','wh'], 'श': ['sh','s','ti','tio','ci'], 'ष': ['sh'],
 'स': ['s','c','z'], 'ह': ['h'],
 'क़': ['q','k'], 'ख़': ['kh'], 'ग़': ['g'], 'ज़': ['z','j'], 'ड़': ['r','d'], 'ढ़': ['rh'],
 'फ़': ['f','ph'], 'य़': ['y'],
}
MATRA = {'ा': ['aa','a'], 'ि': ['i','e','y'], 'ी': ['ee','i','ea','y','ey','ie','e'],
         'ु': ['u','o','oo'], 'ू': ['oo','u','ou'], 'े': ['e','ay','ai','a','ey','ei'],
         'ै': ['ai','e','a','ay'], 'ो': ['o','oh','ou','oa','ow'], 'ौ': ['au','ou','o','aw'],
         'ॉ': ['o','au','a','aw'], 'ृ': ['ri','ru'], 'ॅ': ['e','a']}
INDEP = {'अ': ['a','u'], 'आ': ['aa','a'], 'इ': ['i','e'], 'ई': ['ee','i'], 'उ': ['u','o'],
         'ऊ': ['oo','u'], 'ए': ['e','a','ay'], 'ऐ': ['ai','e','a'], 'ओ': ['o'],
         'औ': ['au','ou','o'], 'ऑ': ['o','au','a'], 'ऋ': ['ri']}
LETTER = {'a': ['ए'], 'b': ['बी'], 'c': ['सी'], 'd': ['डी'], 'e': ['ई'], 'f': ['एफ़', 'एफ'],
          'g': ['जी'], 'h': ['एच'], 'j': ['जे'], 'k': ['के'], 'l': ['एल'], 'm': ['एम'],
          'n': ['एन'], 'p': ['पी'], 'r': ['आर'], 's': ['एस'], 't': ['टी'], 'v': ['वी']}
INHERENT = ['a', '', 'e', 'u', 'o']  # schwa / English short vowels (done, correct)
NUKTA, VIRAMA, NASAL = '़', '्', {'ं', 'ँ'}

def units(h):
    out, i = [], 0
    while i < len(h):
        c = h[i]
        if c in NASAL and out:
            span, v = out[-1]
            out[-1] = (span + c, sorted({x + n for x in v for n in ['', 'n', 'm']}))
            i += 1; continue
        if c + (h[i+1] if i+1 < len(h) else '') in CONS and i+1 < len(h) and h[i+1] == NUKTA:
            base, j = c + NUKTA, i + 2
        elif c in CONS:
            base, j = c, i + 1
        else:
            base, j = None, i + 1
        if base:
            r = CONS[base]
            r = r + [x + x[-1] for x in r]  # doubled spelling: ll, tt, ss ...
            if j < len(h) and h[j] == VIRAMA:
                out.append((base + VIRAMA, r)); i = j + 1; continue
            if j < len(h) and h[j] in MATRA:
                out.append((base, r)); i = j; continue
            out.append((base, sorted({x + v for x in r for v in INHERENT})))
            i = j; continue
        if c == 'ा' and i+1 < len(h) and h[i+1] in 'इई':
            out.append((c + h[i+1], ['aai', 'ai', 'i', 'y', 'ay', 'aay'])); i += 2; continue
        if c in MATRA:
            out.append((c, MATRA[c]))
        elif c in INDEP:
            out.append((c, INDEP[c]))
        else:
            out.append((c, [c.lower()]))
        i += 1
    return out

def align(ref, U):
    n, m = len(ref), len(U)
    in_code = [False]*n
    for mt in re.finditer(r'\S+', ref):
        if re.search(r'\d', mt.group()) and re.search(r'[a-z]', mt.group()):
            for q in range(mt.start(), mt.end()):
                in_code[q] = ref[q].isalpha()
    INF = 10**9
    dp = [[INF]*(m+1) for _ in range(n+1)]
    bp = [[None]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            cur = dp[i][j]
            if cur >= INF: continue
            def relax(a, b, c, op):
                if c < dp[a][b] or (c == dp[a][b] and bp[a][b][0] not in ('match','lmatch') and op[0] in ('match','lmatch')):
                    dp[a][b] = c; bp[a][b] = op
            if i < n and in_code[i]:
                for k in range(1, 4):
                    if j + k <= m:
                        sp = ''.join(u[0] for u in U[j:j+k])
                        if any(sp == x for x in sum(LETTER.values(), [])):
                            ok = sp in LETTER.get(ref[i], [])
                            relax(i+1, j+k, cur + (0 if ok else 1), ('lmatch' if ok else 'lsub', i, j, k))
            if j < m:
                vs = U[j][1]
                for L in range(1, 5):
                    if i + L <= n and ref[i:i+L] in vs:
                        relax(i+L, j+1, cur, ('match', i, j, L))
                if i < n:
                    relax(i+1, j+1, cur+1, ('sub', i, j, 1))
                relax(i, j+1, cur+1, ('ins', i, j, 0))
            if i < n:
                relax(i+1, j, cur+1, ('del', i, j, 1))
    steps, i, j = [], n, m
    while i or j:
        op = bp[i][j]; steps.append(op); i, j = op[1], op[2]
    return steps[::-1]

def main():
    ref = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read(), True)
    hyp = norm(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read(), False)
    U = units(hyp)
    assert ''.join(u[0] for u in U) == hyp
    rows = []
    for op, i, j, L in align(ref, U):
        if op == 'match':
            seg, span = ref[i:i+L], U[j][0]
            cross = not (span.isascii())
            for k in range(L):
                why = ''
                if k == 0 and cross:
                    why = f"'{seg}' ~ '{span}'"
                elif k > 0:
                    why = f"part of '{seg}' ~ '{span}'"
                rows.append(dict(ref=ref[i+k], hyp=span if k == 0 else '', op='match', why=why))
        elif op in ('lmatch', 'lsub'):
            sp = ''.join(u[0] for u in U[j:j+L])
            rows.append(dict(ref=ref[i], hyp=sp, op='match' if op == 'lmatch' else 'sub',
                             why=(f"letter '{ref[i]}' spoken by name ~ '{sp}'" if op == 'lmatch'
                                  else f"letter '{ref[i]}' vs spoken letter-name '{sp}' (different letter)")))
        elif op == 'sub':
            rows.append(dict(ref=ref[i], hyp=U[j][0], op='sub', why=f"'{ref[i]}' vs '{U[j][0]}' different sound"))
        elif op == 'del':
            rows.append(dict(ref=ref[i], hyp='', op='del', why=f"'{ref[i]}' not spoken/transcribed"))
        else:
            rows.append(dict(ref='', hyp=U[j][0], op='ins', why=f"extra '{U[j][0]}' in hyp"))
    for k, r in enumerate(rows):
        r['i'] = k
    rows = [{k: r[k] for k in ('i', 'ref', 'hyp', 'op', 'why')} for r in rows]
    assert ''.join(r['ref'] for r in rows) == ref, 'ref concat mismatch'
    assert ''.join(r['hyp'] for r in rows) == hyp, 'hyp concat mismatch'
    with open(os.path.join(D, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    c = {o: sum(r['op'] == o for r in rows) for o in ('match', 'sub', 'del', 'ins')}
    N = len(ref)
    met = dict(pair='V1_A__food_12_g3', ref_chars=N, hyp_chars=len(hyp), hyp_units=len(U),
               steps=len(rows), **c, cer=round((c['sub']+c['del']+c['ins'])/N, 4),
               normalised_reference=ref, normalised_hypothesis=hyp,
               concat_check={'ref': True, 'hyp': True})
    json.dump(met, open(os.path.join(D, 'char_metrics.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    # human-readable map: blocks of ~80 ref chars; each column = one step, padded
    mark = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
    lines, blk, cnt = [], [], 0
    def flush():
        if not blk: return
        cols = [(r['ref'] or '-', r['hyp'] or '-', mark[r['op']]) for r in blk]
        w = [max(len(a), len(b), 1) for a, b, _ in cols]
        lines.append('REF ' + '|'.join(a.ljust(x) for (a, _, _), x in zip(cols, w)))
        lines.append('HYP ' + '|'.join(b.ljust(x) for (_, b, _), x in zip(cols, w)))
        lines.append('OP  ' + '|'.join(o.ljust(x) for (_, _, o), x in zip(cols, w)))
        lines.append('')
    for r in rows:
        blk.append(r); cnt += len(r['ref'])
        if cnt >= 80 and r['ref'] == ' ':
            flush(); blk, cnt = [], 0
    flush()
    hdr = (f"V1_A__food_12_g3 char map  (columns separated by '|', '-' = empty)\n"
           f"ref_chars={N} match={c['match']} sub={c['sub']} del={c['del']} ins={c['ins']} "
           f"CER={met['cer']}\n\n")
    open(os.path.join(D, 'char_map.txt'), 'w', encoding='utf-8').write(hdr + '\n'.join(lines))
    print(json.dumps({k: met[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'cer')}))

if __name__ == '__main__':
    main()
