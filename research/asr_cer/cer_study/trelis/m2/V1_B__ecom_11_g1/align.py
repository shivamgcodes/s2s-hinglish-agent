#!/usr/bin/env python3
"""Sound-based character alignment of reference (Roman Hinglish) vs Whisper hypothesis
(mixed Roman/Devanagari) for pair V1_B__ecom_11_g1.

Units:
  ref  = characters of normalised reference (spaces are units)
  hyp  = Roman chars / spaces individually; Devanagari split into sub-akshara units:
         consonant(+nukta)(+virama) | vowel sign (matra) | anusvara/chandrabindu | independent vowel.
Ops: match (same sound), sub (different sound), del (ref unit with no hyp), ins (hyp unit with no ref).
Mostly an automatic DP with phonetic readings for Devanagari units; number regions are
aligned by hand (MANUAL) because digits vs number words need word-level judgement.
"""
import json, re, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))


def normalise(s):
    s = s.lower()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(' ')
        elif cat[0] in ('L', 'N', 'M'):
            out.append(ch)
        # punctuation / symbols dropped
    return re.sub(r' +', ' ', ''.join(out)).strip()


# ---------- Devanagari readings ----------
CONS = {
    'क': ['k'], 'ख': ['kh'], 'ग': ['g'], 'घ': ['gh'], 'च': ['ch'], 'छ': ['chh', 'ch'],
    'ज': ['j', 'z'], 'झ': ['jh'], 'ट': ['t'], 'ठ': ['th'], 'ड': ['d'], 'ढ': ['dh'],
    'ण': ['n'], 'त': ['t'], 'थ': ['th'], 'द': ['d'], 'ध': ['dh'], 'न': ['n'], 'प': ['p'],
    'फ': ['ph', 'f'], 'ब': ['b'], 'भ': ['bh'], 'म': ['m'], 'य': ['y'], 'र': ['r'], 'ल': ['l'],
    'व': ['v', 'w'], 'श': ['sh', 's'], 'ष': ['sh'], 'स': ['s'], 'ह': ['h'],
}
NUKTA_CONS = {'ज': ['z'], 'फ': ['f'], 'ड': ['r', 'd'], 'ढ': ['rh'], 'क': ['q', 'k'], 'ख': ['kh'], 'ग': ['g']}
MATRA = {'ा': ['aa', 'a'], 'ि': ['i'], 'ी': ['ee', 'i', 'y'], 'ु': ['u'], 'ू': ['oo', 'u'],
         'े': ['e'], 'ै': ['ai', 'ae', 'e'], 'ो': ['o'], 'ौ': ['au', 'o'], 'ृ': ['ri']}
INDEP = {'अ': ['a'], 'आ': ['aa', 'a'], 'इ': ['i'], 'ई': ['ee', 'i'], 'उ': ['u'], 'ऊ': ['oo', 'u'],
         'ए': ['e'], 'ऐ': ['ai', 'e'], 'ओ': ['o'], 'औ': ['au', 'o']}
NASAL = {'ं': ['n', 'm'], 'ँ': ['n']}
VIRAMA, NUKTA = '्', '़'
VOWELS = set('aeiou')


def is_deva(ch):
    return 'ऀ' <= ch <= 'ॿ'


def hyp_units(h):
    """Split hypothesis into units, returning list of (text, readings, is_vowelish)."""
    units = []
    i = 0
    while i < len(h):
        ch = h[i]
        if not is_deva(ch):
            units.append((ch, [ch], ch in VOWELS))
            i += 1
            continue
        if ch in CONS:
            txt = ch
            base = CONS[ch]
            j = i + 1
            if j < len(h) and h[j] == NUKTA:
                txt += NUKTA
                base = NUKTA_CONS.get(ch, base)
                j += 1
            if j < len(h) and h[j] == VIRAMA:
                txt += VIRAMA
                j += 1
                reads = list(base)
            elif j < len(h) and h[j] in MATRA:
                reads = list(base)
            else:  # inherent schwa: optional (schwa deletion)
                reads = list(base) + [b + 'a' for b in base]
            units.append((txt, reads, False))
            i = j
        elif ch in MATRA:
            units.append((ch, MATRA[ch], True)); i += 1
        elif ch in INDEP:
            units.append((ch, INDEP[ch], True)); i += 1
        elif ch in NASAL:
            units.append((ch, NASAL[ch], False)); i += 1
        else:
            raise ValueError('unhandled devanagari char %r' % ch)
    return units


def dp_align(ref, hyp):
    """Weighted edit-distance alignment ref chars vs hyp units with phonetic readings."""
    U = hyp_units(hyp)
    n, m = len(ref), len(U)
    INF = float('inf')
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            c = D[i][j]
            if c == INF:
                continue
            def relax(ni, nj, cost, op):
                if c + cost < D[ni][nj] - 1e-9:
                    D[ni][nj] = c + cost
                    B[ni][nj] = (i, j, op)
            if i < n:
                relax(i + 1, j, 1.0, 'del')
            if j < m:
                relax(i, j + 1, 1.0, 'ins')
            if j < m:
                txt, reads, vow = U[j]
                for k in range(1, 5):
                    if i + k > n:
                        break
                    span = ref[i:i + k]
                    if span in reads:
                        relax(i + k, j + 1, 0.0, 'match')
                if i < n:
                    a, b = ref[i], txt
                    pen = 0.0 if ((a == ' ') == (b == ' ')) else 1.5  # never sub space<->letter (del+ins instead)
                    if a != ' ' and b != ' ' and (a in VOWELS) != vow:
                        pen += 0.005  # prefer consonant<->consonant / vowel<->vowel subs
                    relax(i + 1, j + 1, 1.0 + 0.01 + pen, 'sub')
                    # vowel digraph vs single vowel unit counts as one sound substitution
                    if vow and i + 2 <= n and all(x in VOWELS for x in ref[i:i + 2]):
                        relax(i + 2, j + 1, 1.0 + 0.02, 'sub')
    steps = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = B[i][j]
        r = ref[pi:i]
        h = ''.join(U[x][0] for x in range(pj, j))
        steps.append({'ref': r, 'hyp': h, 'op': op})
        i, j = pi, pj
    steps.reverse()
    return steps


# ---------- segmentation plan ----------
# Each entry: ('auto', ref_chunk, hyp_chunk) or ('manual', [(ref, hyp, op, why), ...])
NUM = 'number region aligned by hand'
PLAN = [
    ('auto', 'hello thank you for calling bazaarly main pooja main koi kaise help kar sakti hoon ji the order is for boat rockerz ',
             'hello thank you for calling बेजारली में पूजा में कोई कैसे help कर सकती हो जी the order is for boot rockers '),
    ('manual', [('4', 'four', 'match', '4 = four'),
                ('', ' hydro', 'ins', 'spurious word "hydro" (likely misheard "hundred" implied by 450); counted as one inserted word-unit'),
                ('50', ' fifty', 'match', '50 = fifty')]),
    ('auto', ' headphones midnight black colour ji bilkul the order is for lokesh srivastava on ',
             ' headphones midnight black colour जी बिलकुल the order is पर लोकेश श्रीवास्तवा on '),
    ('manual', [('ph', 'f', 'match', '"ph" in phone is the f sound'),
                ('o', 'o', 'match', 'same vowel'),
                ('n', 'u', 'sub', 'phone heard as four: n -> u'),
                ('e', 'r', 'sub', 'phone heard as four: e -> r'),
                (' ', ' ', 'match', 'space'),
                ('9', 'nine', 'match', '9 = nine'),
                ('8', ' एक', 'sub', '8 transcribed as एक (one): different number'),
                ('0', ' zero', 'match', '0 = zero'),
                ('0', ' zero', 'match', '0 = zero'),
                ('0', ' zero', 'match', '0 = zero'),
                (' ', ' ', 'match', 'space between digit groups'),
                ('0', 'zero', 'match', '0 = zero'),
                ('1', ' one', 'match', '1 = one'),
                ('2', ' two', 'match', '2 = two'),
                ('9', ' nine', 'match', '9 = nine'),
                ('9', ' nine', 'match', '9 = nine')]),
    ('auto', ' ji the order is for boat rockerz ', ' जी the order is पर boat rockers '),
    ('manual', [('4', 'four', 'match', '4 = four'),
                ('', ' hydro', 'ins', 'spurious word "hydro" (likely misheard "hundred"); one inserted word-unit'),
                ('50', ' fifty', 'match', '50 = fifty')]),
    ('auto', ' and status shipped hain ji the delivery address is h', ' and status shipped है जी the delivery address hh'),
    ('manual', [('12', ' twelve', 'match', '12 = twelve')]),
    ('auto', ' sector ', ' sector '),
    ('manual', [('15', 'fifteen', 'match', '15 = fifteen')]),
    ('auto', ' rohini ', ' रोहिनी '),
    ('manual', [('de', 'दि', 'match', 'Delhi is the conventional Roman spelling of दिल्ली'),
                ('lhi', 'ल्ली', 'match', 'Delhi is the conventional Roman spelling of दिल्ली')]),
    ('auto', ' sahi hai ji please remind me gaadi ka ek ji bilkul the delivery is scheduled for friday ',
             ' सही है जी please mind make आधीकारीक जी बिलकुल the delivery is scheduled for friday '),
    ('manual', [('25', 'twenty five', 'match', '25 = twenty five')]),
    ('auto', ' october haan ji the address is correct just the order for boat rockerz ',
             ' october हाँ जी the address is correct just the order for boot rockers '),
    ('manual', [('4', 'for', 'match', '"for" is a homophone of four: same sound, same number'),
                ('', ' hydro', 'ins', 'spurious word "hydro" (likely misheard "hundred"); one inserted word-unit'),
                ('50', ' fifty', 'match', '50 = fifty')]),
    ('auto', ' is friday ', ' is friday '),
    ('manual', [('25', 'twenty five', 'match', '25 = twenty five')]),
    ('auto', ' ji thank you for calling bazaarly have a good din ji',
             ' जी thank you for calling बाजारली have a good दिन जी'),
    ('tail', ' एक और फिर आए है समय जी life है कम भी'),
]


def why_for(s):
    r, h, op = s['ref'], s['hyp'], s['op']
    if op == 'match':
        if r == h:
            return 'identical'
        return 'same sound: %r ~ %r' % (r, h)
    if op == 'sub':
        return 'different sound: %r vs %r' % (r, h)
    if op == 'del':
        return 'ref %r not heard in hypothesis' % r
    return 'hypothesis adds %r not in reference' % h


def main():
    ref = normalise(open(os.path.join(HERE, 'reference.txt'), encoding='utf-8').read())
    hyp = normalise(open(os.path.join(HERE, 'whisper.txt'), encoding='utf-8').read())
    steps = []
    rp = hp = 0
    for seg in PLAN:
        if seg[0] == 'auto':
            _, rc, hc = seg
            assert ref[rp:rp + len(rc)] == rc, ('ref mismatch at', rp, ref[rp:rp + len(rc)], rc)
            assert hyp[hp:hp + len(hc)] == hc, ('hyp mismatch at', hp, hyp[hp:hp + len(hc)], hc)
            for s in dp_align(rc, hc):
                s['why'] = why_for(s)
                steps.append(s)
            rp += len(rc); hp += len(hc)
        elif seg[0] == 'tail':
            hc = seg[1]
            assert hyp[hp:hp + len(hc)] == hc
            for u in hyp_units(hc):
                steps.append({'ref': '', 'hyp': u[0], 'op': 'ins',
                              'why': 'trailing hallucinated speech after the call ends; not in reference'})
            hp += len(hc)
        else:
            for r, h, op, why in seg[1]:
                assert ref[rp:rp + len(r)] == r, ('ref manual mismatch', rp, ref[rp:rp + 10], r)
                assert hyp[hp:hp + len(h)] == h, ('hyp manual mismatch', hp, hyp[hp:hp + 10], h)
                steps.append({'ref': r, 'hyp': h, 'op': op, 'why': why})
                rp += len(r); hp += len(h)
    assert rp == len(ref) and hp == len(hyp), (rp, len(ref), hp, len(hyp))
    for k, s in enumerate(steps):
        s['i'] = k
    with open(os.path.join(HERE, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
        for s in steps:
            f.write(json.dumps({'i': s['i'], 'ref': s['ref'], 'hyp': s['hyp'], 'op': s['op'], 'why': s['why']},
                               ensure_ascii=False) + '\n')
    with open(os.path.join(HERE, 'normalised.json'), 'w', encoding='utf-8') as f:
        json.dump({'ref': ref, 'hyp': hyp}, f, ensure_ascii=False, indent=1)
    print('steps', len(steps))


if __name__ == '__main__':
    main()
