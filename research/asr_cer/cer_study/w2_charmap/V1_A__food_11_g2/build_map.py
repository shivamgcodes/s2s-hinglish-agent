#!/usr/bin/env python3
"""Build the final char map: the automatic phonetic DP (align.py) runs inside segments whose
boundaries were set by listening-by-reading judgement (which ref words Whisper actually
rendered), so that letters are only 'matched' within the corresponding words.
Writes char_map.jsonl, char_map.txt, char_metrics.json and verifies concatenation."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import align, norm_ref, norm_hyp, D

ref = norm_ref(open(os.path.join(D, 'reference.txt'), encoding='utf-8').read())
hyp = norm_hyp(open(os.path.join(D, 'whisper.txt'), encoding='utf-8').read())

# (mode, ref_piece, hyp_piece) in order. modes: dp | nomatch | del | ins | manual
SEG = [
 ('dp', 'hello thank you for calling tiffingo this is arjun main aapki kaise help kar sakta hoon ji ',
        'Hello thank you for calling Tiffinbo This is Arjun मैं आपकी कैसे हेल्ब कर सकता हूँ जी '),
 # Whisper repeated the greeting ("aapki kaise") where the model said "ji the order ... status"
 ('nomatch', 'ji the order fd2130 is still preparing aur status ', 'आपकी कैसे '),
 ('dp', 'update kar deta ', 'हेल्ब कर सकता '),
 ('del', 'haan ji abhi update ', ''),
 ('dp', 'hoon ', 'हूँ '),
 ('del', 'yaa ', ''),
 ('dp', 'delivery sahi the order includes kashmiri biryani aur', 'दिलिवरी से दा आर्डर इंक्रूट्स कश्मीरी बिर्यानी और'),
 ('dp', ' ', ' बर्हानी '),   # extra word heard after "aur"
 ('dp', 'ji the order ', 'जी दा आर्डर '),
 ('manual', 'fd2130', 'एफटी 2130'),
 ('dp', ' includes kashmiri biryani burhalia raita aur coca cola ji bilkul the coca cola is included in the delivery price of ',
        ' इंक्रूट्स कश्मीरी बिर्यानी बर्हालिया रेहिता और कोका कोला जी बिल्कुल द कोका कोला इस इंप्रूडेड इन द डिलिवरी प्राइस आफ '),
 ('del', 'rs ', ''), ('dp', '720', '720'), ('ins', '', ' रुपी'),
 ('dp', ' ji the total fare is ', ' जी द टोटल फैर इस '),
 ('del', 'rs ', ''), ('dp', '720', '720'), ('ins', '', ' रुपी'),
 ('dp', ' aur the order is updated in progress ka', ' और द आर्डर इस अपिडेल इन प्रोग्रेस का'),
 ('ins', '', ' है'),
 ('dp', ' ji the order total is ', ' जी द आउडर टोटल इस '),
 ('del', 'rs ', ''), ('dp', '720', '720'), ('ins', '', ' रुपिये'),
 ('dp', ' aur status abhi update kar deta hoon ji thank you for calling tiffingo have a good day ji',
        ' और स्टेटस अभी अप्टेल कर दीता हूँ जी थैंक यू फ़ॉर कॉलिंग टिफिंगो हैव अ गुड देजी'),
 # trailing Whisper hallucination after the call ended
 ('ins', '', ' इबीजी इनमे आप ही कराई ही सही है थैंक यू जाइर कू'),
]

assert ''.join(s[1] for s in SEG) == ref, 'segment ref pieces do not rebuild ref'
assert ''.join(s[2] for s in SEG) == hyp, 'segment hyp pieces do not rebuild hyp'

DEV = lambda t: any('ऀ' <= c <= 'ॿ' for c in t)

steps = []
rpos = 0; hpos = 0
for mode, r, h in SEG:
    if mode in ('dp', 'nomatch'):
        st = align(r, h, nomatch=(mode == 'nomatch'))
        for s in st:
            s['seg'] = mode
        steps += st
    elif mode == 'del':
        steps += [dict(ref=c, hyp='', op='del', seg='del') for c in r]
    elif mode == 'ins':
        steps += [dict(ref='', hyp=c, op='ins', seg='ins') for c in h]  # one step per code point
    elif mode == 'manual':
        steps += [
            dict(ref='f', hyp='एफ', op='match', why="letter F spoken 'ef' ~ एफ"),
            dict(ref='d', hyp='टी', op='sub', why="letter D ('dee') heard as 'tee' टी"),
            dict(ref='', hyp=' ', op='ins', why='whisper split FD / 2130 with a space'),
        ] + [dict(ref=c, hyp=c, op='match') for c in '2130']

# merge any ins of a lone combining mark into the previous step (keeps aksharas whole when possible)
# (not needed for correctness; concatenation is what matters)

# word context for 'why'
def word_at(text, idx):
    if idx < 0 or idx >= len(text) or text[idx] == ' ':
        return None
    a = text.rfind(' ', 0, idx) + 1
    b = text.find(' ', idx)
    return text[a: b if b != -1 else len(text)]

out = []
ri = hi = 0
for n, s in enumerate(steps):
    r, h, op = s['ref'], s['hyp'], s['op']
    rw = word_at(ref, ri) if r else None
    hw = word_at(hyp, hi) if h else None
    why = s.get('why', '')
    if not why:
        if op == 'match':
            if s.get('cont'):
                why = f"part of '{ref[ri-1] if False else ''}{s.get('var','')}' sound already given by previous hyp span"
            elif DEV(h):
                v = s.get('var', '')
                if v and v != r:
                    why = f"'{r}' ~ {h} ({v}) in {rw}~{hw}"
        elif op == 'sub':
            if s.get('seg') == 'nomatch':
                why = f"whisper repeated greeting ({hw}) where model said '{rw}'"
            elif r == ' ' or h == ' ':
                why = 'word boundary mismatch'
            else:
                why = f"'{r}' in {rw} heard as {h} ({s.get('var','')}) in {hw}"
        elif op == 'del':
            if s.get('cont'):
                why = f"'{r}' not voiced in {hw if hw else 'hyp'} for {rw}"
            elif r == ' ':
                why = 'space missing in whisper (words merged/dropped)'
            elif s.get('seg') == 'del' and rw == 'rs':
                why = "'Rs' voiced after the number as रुपी/रुपिये (reordered)"
            elif s.get('seg') == 'del':
                why = f"word '{rw}' not in whisper"
            elif s.get('seg') == 'nomatch':
                why = f"'{rw}' unheard: whisper repeated greeting 'आपकी कैसे' here"
            else:
                why = f"'{r}' of {rw} not in whisper"
        elif op == 'ins':
            if h == ' ':
                why = 'extra word boundary in whisper'
            elif s.get('seg') == 'ins' and hw and hw.startswith('रुप'):
                why = f"{hw} = spoken 'Rs', said after the number (reordered; ref 'rs' counted as del)"
            elif s.get('seg') == 'ins' and hi > hyp.find('देजी'):
                why = f"trailing whisper hallucination after call end ({hw})"
            elif s.get('seg') == 'ins':
                why = f"extra word {hw} in whisper"
            else:
                why = f"extra sound {h} in {hw} (ref {rw or 'none'})"
    # conventional Hindi spellings of English 'order'/'of' write the vowel as आ (आर्डर, आफ)
    if op == 'sub' and r == 'o' and h == 'आ' and rw in ('order', 'of'):
        op = 'match'; why = f"conventional Hindi spelling {hw} for '{rw}' (o~आ)"
    # fix continuation text
    if op == 'match' and s.get('cont'):
        why = f"continuation of '{s.get('var','')}' spelled by previous hyp span"
    out.append(dict(i=n, ref=r, hyp=h, op=op, why=why))
    ri += len(r); hi += len(h)

R = ''.join(o['ref'] for o in out); H = ''.join(o['hyp'] for o in out)
assert R == ref, 'ref concat mismatch'
assert H == hyp, 'hyp concat mismatch'

from collections import Counter
c = Counter(o['op'] for o in out)
N = len(ref)
cer = (c['sub'] + c['del'] + c['ins']) / N
with open(os.path.join(D, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + '\n')
metrics = dict(pair='V1_A__food_11_g2', ref_chars=N, hyp_chars=len(hyp), steps=len(out),
               match=c['match'], sub=c['sub'], del_=c['del'], ins=c['ins'],
               errors=c['sub'] + c['del'] + c['ins'], cer=round(cer, 4),
               ref_normalised=ref, hyp_normalised=hyp,
               method='phonetic DP (align.py) within judgement-anchored word segments (build_map.py)')
metrics['del'] = metrics.pop('del_')
json.dump(metrics, open(os.path.join(D, 'char_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# human-readable: blocks of ~80 reference chars; each step padded to its display width
import unicodedata
def dw(t):
    return sum(0 if unicodedata.combining(ch) or unicodedata.category(ch) == 'Mc' and False else 1 for ch in t)
lines = []
L1 = L2 = L3 = ''; rc = 0
def flush():
    global L1, L2, L3, rc
    if L1 or L2:
        lines.extend(['REF: ' + L1, 'HYP: ' + L2, 'OP:  ' + L3, ''])
    L1 = L2 = L3 = ''; rc = 0
for o in out:
    r, h = o['ref'], o['hyp']
    w = max(len(r), len(h), 1)
    L1 += r.ljust(w) if r else '·'.ljust(w) if o['op'] == 'ins' else ''.ljust(w)
    L2 += h.ljust(w) if h else ('·'.ljust(w) if o['op'] == 'del' else ''.ljust(w))
    L3 += {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}[o['op']] * w
    rc += len(r)
    if rc >= 80 and r == ' ':
        flush()
flush()
hdr = [f"V1_A__food_11_g2 char map  |  ref chars {N}  match {c['match']}  sub {c['sub']}  del {c['del']}  ins {c['ins']}  CER {cer:.4f}",
       "Columns are per alignment step (Devanagari spans may render narrower than they are padded). '·' = empty side.", '']
open(os.path.join(D, 'char_map.txt'), 'w', encoding='utf-8').write('\n'.join(hdr + lines))
print(json.dumps({k: metrics[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'errors', 'cer')}))
print('concat checks OK')
