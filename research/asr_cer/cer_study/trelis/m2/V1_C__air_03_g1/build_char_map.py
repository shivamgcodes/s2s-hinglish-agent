#!/usr/bin/env python3
"""Build a sound-aligned character map for V1_C__air_03_g1 (reference Roman Hinglish
vs Trelis Whisper hypothesis, mixed Roman/Devanagari)."""
import json, os, re, unicodedata

D = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(' ')
        elif cat[0] in 'LMN':
            out.append(ch)
        else:  # punctuation / symbols dropped
            continue
    return re.sub(r' +', ' ', ''.join(out)).strip()


REF = norm(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
HYP = norm(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read())

VIRAMA = '्'


def aksharas(s):
    """Split a string into units: Devanagari aksharas, single other chars."""
    units = []
    for ch in s:
        if units and (unicodedata.category(ch) in ('Mn', 'Mc') or units[-1].endswith(VIRAMA)) \
                and 'ऀ' <= ch <= 'ॿ':
            units[-1] += ch
        else:
            units.append(ch)
    return units


steps = []


def add(r, h, op, why):
    steps.append({'ref': r, 'hyp': h, 'op': op, 'why': why})


def auto(r, h):
    """Levenshtein char alignment for same-script (Roman) spans."""
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + (r[i - 1] != h[j - 1]), dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, out = n, m, []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            same = r[i - 1] == h[j - 1]
            out.append((r[i - 1], h[j - 1], 'match' if same else 'sub',
                        'same letter' if same else 'different sound'))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            out.append((r[i - 1], '', 'del', 'missing in hypothesis'))
            i -= 1
        else:
            out.append(('', h[j - 1], 'ins', 'extra in hypothesis'))
            j -= 1
    for t in reversed(out):
        add(*t)


def man(pairs):
    for p in pairs:
        add(*p)


def ins_run(h, why):
    for u in aksharas(h):
        add('', u, 'ins', why)


M = 'match'
S = 'sub'
SCH = 'inherent schwa of preceding consonant'

auto('hello thank you for calling ', 'hello thank you for coming ')
man([('u', 'उ', M, 'u'), ('', 'र्', 'ins', 'extra r'), ('d', 'द', M, 'd'),
     ('a', 'ा', M, 'aa = ा'), ('a', '', M, 'second a of aa digraph, long vowel present in ा'),
     ('n', 'ं', M, 'nasal n = anusvara'), ('', 'श', 'ins', 'extra sh')])
auto(' air this is ', ' dear this is ')
man([('a', 'अ', M, 'a'), ('d', 'द', M, 'd'), ('i', 'ि', M, 'i'), ('t', 'त', M, 't'), ('i', 'ि', M, 'i')])
auto(' ', ' ')
man([('m', 'म', M, 'm'), ('a', 'ै', M, 'ai = ै'), ('i', '', M, 'second letter of ai digraph, sound in ै'),
     ('n', 'ं', M, 'nasal = anusvara')])
auto(' ', ' ')
man([('a', 'आ', M, 'aa = आ'), ('a', '', M, 'second a of aa, long vowel in आ'), ('p', 'प', M, 'p'),
     ('k', 'क', M, 'k'), ('i', 'ी', M, 'i = ी')])
auto(' ', ' ')
man([('k', 'क', M, 'k'), ('a', 'ै', M, 'ai = ै'), ('i', '', M, 'second letter of ai digraph, sound in ै'),
     ('s', 'स', M, 's'), ('e', 'े', M, 'e = े')])
auto(' help ', ' help ')
man([('k', 'क', M, 'k'), ('a', '', M, SCH + ' क'), ('r', 'र', M, 'r')])
auto(' ', ' ')
man([('s', 'स', M, 's'), ('a', '', M, SCH + ' स'), ('k', 'क', M, 'k'), ('t', 'त', M, 't'), ('i', 'ी', M, 'i = ी')])
auto(' ', ' ')
man([('h', 'ह', M, 'h'), ('o', 'ू', M, 'oo = ू'), ('o', '', M, 'second o of oo digraph, sound in ू'),
     ('n', 'ँ', M, 'nasal = chandrabindu')])
auto(' ', ' ')
JI = [('j', 'ज', M, 'j'), ('i', 'ी', M, 'i = ी')]
man(JI)
auto(' no problem please tell me ', ' no problem please tell me ')
AAPKA = [('a', 'आ', M, 'aa = आ'), ('a', '', M, 'second a of aa, long vowel in आ'), ('p', 'प', M, 'p'),
         ('k', 'क', M, 'k'), ('a', 'ा', M, 'a = ा')]
man(AAPKA)
auto(' booking id ', ' booking id ')
man([('y', 'य', M, 'y'), ('a', 'ा', M, 'a = ा')])
auto(' pnr number ', ' pnr number ')
man([('b', 'ब', M, 'b'), ('a', '', M, SCH + ' ब'), ('t', 'त', M, 't'), ('a', 'ा', M, 'a = ा'),
     ('i', 'इ', M, 'i = इ'), ('y', '', M, 'y glide between इ and ए is audible in बताइए'), ('e', 'ए', M, 'e = ए')])
auto(' ', ' ')
man(JI)
auto(' ', ' ')
man(AAPKA)
auto(' flight ua ', ' flight ua ')
man([('4', 'four', M, 'same number 402: four'), ('0', ' hundred', M, 'same number 402: hundred'),
     ('2', ' two', M, 'same number 402: two')])
auto(' is from delhi to mumbai on ', ' is from delhi to mumbai on ')
man([('2', 'twenty', M, 'same number 22: twenty'), ('2', ' two', M, 'same number 22: two')])
auto(' october ', ' october ')
man(JI)
auto(' your gate is ', ' your gate is ')
man([('1', 'fif', M, 'same number 15: fif-'), ('5', 'teen ', M, 'same number 15: -teen (space is number-word spacing)'),
     ('c', 'c', M, 'c')])
auto(' and seat ', ' and seat ')
man([('1', 'twi', S, '12f (twelve eff) heard as twilling: not the same number'),
     ('2', 'lli', S, '12f heard as twilling'), ('f', 'ng', S, 'f (eff) heard as -ng')])
auto(' ', ' ')
AUR = [('a', 'औ', M, 'au = औ'), ('u', '', M, 'second letter of au digraph, sound in औ'), ('r', 'र', M, 'r')]
man(AUR)
auto(' a meal preference vegetarian ', ' eat meal preference vegetarian ')
man([('h', 'ह', M, 'h'), ('a', 'ै', M, 'ai = ै'), ('i', '', M, 'second letter of ai digraph, sound in ै')])
auto(' ', ' ')
man(JI)
auto(' boarding starts at ', ' boarding starts at ')
man([('6', 'six', M, 'same number 6:30: six'), ('3', ' thir', M, 'same number 30: thir-'),
     ('0', 'ty', M, 'same number 30: -ty')])
auto(' ', ' ')
man([('p', 'पी', M, 'p spoken pee = पी'), ('m', '', 'del', 'em of pm not in hypothesis')])
auto(' ', ' ')
man(AUR)
auto(' flight departure ', ' flight departure ')
man([('7', 'seven', M, 'same number 7:15: seven'), ('1', ' fif', M, 'same number 15: fif-'),
     ('5', 'teen', M, 'same number 15: -teen')])
auto(' pm from terminal ', ' pm from terminal ')
man([('3', 'three', M, 'same number 3')])
auto(' ', ' ')
man(JI)
auto(' the current terminal is terminal ', ' the current terminal is terminal ')
man([('3', 'three', M, 'same number 3')])
auto(' which is listed on your record ', ' which is listed on your data ')
man([('a', 'अ', M, 'a'), ('c', 'च्', M, 'c of ch = च्'), ('h', '', M, 'h of ch digraph, sound in च्'),
     ('h', 'छ', M, 'aspirated chh = छ'), ('a', 'ा', M, 'a = ा')])
auto(' ', ' ')
man(JI)
auto(' thank you for the info ', ' thank you for the info ')
man([('l', 'ल', M, 'l'), ('i', 'े', S, 'i heard as e'), ('n', 'न', M, 'n'), ('a', 'ा', M, 'a = ा'),
     ('', ' ', 'ins', 'extra word break'), ('t', 'थ', S, 't heard as aspirated th'),
     ('', 'ा', 'ins', 'extra vowel aa after थ')])
auto(' ', ' ')
man([('h', 'ह', M, 'h'), ('o', 'ू', M, 'oo = ू'), ('o', '', M, 'second o of oo digraph, sound in ू'),
     ('n', 'ँ', M, 'nasal = chandrabindu')])
auto(' ', ' ')
man([('e', 'ए', M, 'e = ए'), ('k', 'क', M, 'k')])
auto(' minute ', ' minute ')
man(JI)
auto(' have a safe journey thank you for calling ', ' have a safe journey thank you for calling ')
man([('u', 'उ', M, 'u'), ('d', 'द', M, 'd'), ('a', 'ा', M, 'aa = ा'),
     ('a', '', M, 'second a of aa, long vowel in ा'), ('n', 'ं', M, 'nasal n = anusvara'),
     ('', 'शि', 'ins', 'extra shi'), (' ', '', 'del', 'word break missing (udaansh-iye run together)'),
     ('a', 'य', S, 'air heard as -ye'), ('i', 'े', S, 'air heard as -ye'), ('r', '', 'del', 'r of air missing')])
ins_run(HYP[len(''.join(s['hyp'] for s in steps)):], 'trailing hallucinated speech not in reference')

for k, s in enumerate(steps):
    s['i'] = k
steps = [{'i': s['i'], 'ref': s['ref'], 'hyp': s['hyp'], 'op': s['op'], 'why': s['why']} for s in steps]

cref = ''.join(s['ref'] for s in steps)
chyp = ''.join(s['hyp'] for s in steps)
assert cref == REF, ('REF mismatch', cref[:len(os.path.commonprefix([cref, REF])) + 20])
assert chyp == HYP, ('HYP mismatch', chyp[:len(os.path.commonprefix([chyp, HYP])) + 20])
assert all(len(s['ref']) <= 1 for s in steps)
for s in steps:
    assert (s['op'] == 'ins') == (s['ref'] == '')
    assert s['op'] != 'del' or s['hyp'] == ''

with open(os.path.join(D, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
    for s in steps:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

cnt = {o: sum(1 for s in steps if s['op'] == o) for o in ('match', 'sub', 'del', 'ins')}
metrics = {
    'pair': 'V1_C__air_03_g1',
    'ref_chars': len(REF),
    'hyp_units': len(steps) - cnt['del'],
    **cnt,
    'errors': cnt['sub'] + cnt['del'] + cnt['ins'],
    'cer': round((cnt['sub'] + cnt['del'] + cnt['ins']) / len(REF), 4),
    'normalised_reference': REF,
    'normalised_hypothesis': HYP,
}
json.dump(metrics, open(os.path.join(D, 'char_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# char_map.txt: blocks of ref/hyp/op lines
W = 40
lines = []
for b in range(0, len(steps), W):
    blk = steps[b:b + W]
    cells = []
    for s in blk:
        r = s['ref'].replace(' ', '_') or '-'
        h = s['hyp'].replace(' ', '_') or '-'
        o = {'match': '.', 'sub': 'S', 'del': 'D', 'ins': 'I'}[s['op']]
        cells.append((r, h, o))
    lines.append('ref: ' + '|'.join(c[0] for c in cells))
    lines.append('hyp: ' + '|'.join(c[1] for c in cells))
    lines.append('op:  ' + '|'.join(c[2] for c in cells))
    lines.append('')
lines.append('Legend: . match, S sub, D del, I ins; _ = space, - = empty')
lines.append(json.dumps({k: metrics[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'errors', 'cer')}))
open(os.path.join(D, 'char_map.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(json.dumps({k: metrics[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'errors', 'cer')}))
