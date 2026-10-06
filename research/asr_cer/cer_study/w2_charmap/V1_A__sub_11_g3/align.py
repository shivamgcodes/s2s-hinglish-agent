#!/usr/bin/env python3
"""Auto cross-script char alignment: Roman Hinglish reference vs Devanagari/Latin Whisper hyp.
Produces char_map_auto.jsonl; manual overrides in overrides.json are applied by finalize.py."""
import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))

def norm_ref(s):
    s = s.lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()

def norm_hyp(s):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()

CONS = {'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'n','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'n',
        'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
        'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh',
        'ष':'sh','स':'s','ह':'h','ड़':'r','ढ़':'rh'}
VOW = {'अ':'a','आ':'aa','इ':'i','ई':'ii','उ':'u','ऊ':'uu','ए':'e','ऐ':'ai','ओ':'o','औ':'au','ऑ':'o','ऋ':'ri'}
MAT = {'ा':'aa','ि':'i','ी':'ii','ु':'u','ू':'uu','े':'e','ै':'ai','ो':'o','ौ':'au','ॉ':'o','ृ':'ri'}
# second letter of long vowels / diphthong tail is optional
def roman_units(h):
    """list of (roman_char, optional, hyp_index)"""
    out = []
    for k, c in enumerate(h):
        nxt = h[k+1] if k+1 < len(h) else ''
        nxt2 = h[k+2] if k+2 < len(h) else ''
        if c in CONS:
            r = CONS[c]
            for ch in r: out.append((ch, False, k))
            follow = nxt if nxt != '़' else nxt2
            if follow not in MAT and follow != '्':
                out.append(('a', True, k))  # inherent vowel, optional (schwa deletion)
        elif c in VOW or c in MAT:
            r = (VOW.get(c) or MAT.get(c))
            out.append((r[0], False, k))
            for ch in r[1:]: out.append((ch, True, k))
        elif c == 'ं': out.append(('n', False, k))
        elif c == 'ँ': out.append(('n', True, k))
        elif c == 'ः': out.append(('h', True, k))
        elif c in '़्': pass
        else:
            out.append((c.lower(), False, k))
    return out

VOWELS = set('aeiou')
NEAR = {('c','k'),('c','s'),('q','k'),('x','k'),('z','j'),('s','j'),('f','p'),('w','v'),
        ('y','i'),('i','y'),('e','y'),('t','d'),('d','t')}
def cost(r, m):
    if r == m: return 0.0
    if (r, m) in NEAR: return 0.3
    if r in VOWELS and m in VOWELS: return 0.4
    return 1.0

def align(R, U):
    n, m = len(R), len(U)
    INF = 1e9
    dp = [[INF]*(m+1) for _ in range(n+1)]
    bp = [[None]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            v = dp[i][j]
            if v >= INF: continue
            if i < n and j < m:
                c = cost(R[i], U[j][0])
                if R[i] == ' ' or U[j][0] == ' ':
                    c = 0 if R[i] == U[j][0] else 1.0
                if v + c < dp[i+1][j+1]: dp[i+1][j+1] = v + c; bp[i+1][j+1] = ('MS', c)
            if i < n and v + 1 < dp[i+1][j]: dp[i+1][j] = v + 1; bp[i+1][j] = ('D',)
            if j < m:
                c = 0.01 if U[j][1] else 1.0
                if v + c < dp[i][j+1]: dp[i][j+1] = v + c; bp[i][j+1] = ('K',) if U[j][1] else ('I',)
    ops = []; i, j = n, m
    while i or j:
        b = bp[i][j]
        if b[0] == 'MS': ops.append(('M' if b[1] < 1 else 'S', i-1, j-1)); i -= 1; j -= 1
        elif b[0] == 'D': ops.append(('D', i-1, None)); i -= 1
        else: ops.append((b[0], None, j-1)); j -= 1
    return ops[::-1]

def main():
    ref = norm_ref(open(f'{D}/reference.txt').read())
    hyp = norm_hyp(open(f'{D}/whisper.txt').read())
    U = roman_units(hyp)
    ops = align(ref, U)
    steps = []; owner = {}; pending = []
    def roman_of(k): return ''.join(u[0] for u in U if u[2] == k)
    for op, i, j in ops:
        hk = U[j][2] if j is not None else None
        if op in 'MS' or op == 'D':
            st = {'ref': ref[i], 'hyp_idx': [], 'op': {'M':'match','S':'sub','D':'del'}[op], 'rom': U[j][0] if j is not None else ''}
            steps.append(st)
            for p in pending: owner[p] = len(steps)-1; st['hyp_idx'].append(p)
            pending = []
            if op != 'D' and hk not in owner:
                owner[hk] = len(steps)-1; st['hyp_idx'].append(hk)
        elif op == 'I':
            if hk not in owner:
                if steps and steps[-1]['op'] == 'ins' and steps[-1]['hyp_idx'] and steps[-1]['hyp_idx'][-1] == hk-1 and hyp[hk] != ' ' and hyp[hk-1] != ' ':
                    st = steps[-1]
                else:
                    st = {'ref': '', 'hyp_idx': [], 'op': 'ins', 'rom': ''}; steps.append(st)
                owner[hk] = steps.index(st) if st is not steps[-1] else len(steps)-1
                st['hyp_idx'].append(hk); st['rom'] += U[j][0]
            else:
                steps.append({'ref': '', 'hyp_idx': [], 'op': 'ins', 'rom': U[j][0], 'inner': True})
        else:  # K optional skip
            if hk not in owner:
                if steps: owner[hk] = len(steps)-1; steps[-1]['hyp_idx'].append(hk)
                else: pending.append(hk)
    # attach romanless chars (virama, nukta) to owner of previous char
    for k in range(len(hyp)):
        if k not in owner:
            o = owner[k-1]; owner[k] = o; steps[o]['hyp_idx'].append(k)
    out = []
    for s, st in enumerate(steps):
        idx = sorted(st['hyp_idx'])
        h = ''.join(hyp[k] for k in idx)
        why = ''
        if st['op'] == 'sub': why = f"ref '{st['ref']}' vs hyp sound '{st['rom']}'"
        elif st['op'] == 'del': why = 'no corresponding sound in hyp'
        elif st['op'] == 'ins': why = (f"extra sound '{st['rom']}' inside hyp akshara" if st.get('inner') else f"hyp '{h}' (~{st['rom']}) has no ref counterpart")
        elif st['ref'] != st['rom'] and st['ref'] != ' ': why = f"near-equivalent: ref '{st['ref']}' ~ '{st['rom']}'"
        out.append({'i': s, 'ref': st['ref'], 'hyp': h, 'op': st['op'], 'why': why})
    with open(f'{D}/char_map_auto.jsonl', 'w') as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + '\n')
    assert ''.join(o['ref'] for o in out) == ref
    assert ''.join(o['hyp'] for o in out) == hyp, 'hyp mismatch'
    print('ok', len(out), {k: sum(o['op']==k for o in out) for k in ['match','sub','del','ins']})

if __name__ == '__main__':
    main()
