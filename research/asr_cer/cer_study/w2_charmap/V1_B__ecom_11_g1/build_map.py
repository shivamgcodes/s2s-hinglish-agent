#!/usr/bin/env python3
"""Character-level cross-script alignment: Roman Hinglish reference vs Devanagari Whisper hyp.
Word groups are paired by hand (sound judgement); inside each group a phonetic DP aligns
reference letters to hypothesis akshara pieces. Numeral groups are aligned by hand."""
import json, re, unicodedata, os
D = os.path.dirname(os.path.abspath(__file__))

def norm_ref(s):
    s = s.lower()
    s = ''.join(c if (c.isalnum() or c.isspace()) else '' for c in s)
    return re.sub(r'\s+', ' ', s).strip()

def norm_hyp(s):
    out = []
    for c in s:
        cat = unicodedata.category(c)
        if cat.startswith('P') or cat.startswith('S'):
            continue
        out.append(c)
    return re.sub(r'\s+', ' ', ''.join(out)).strip()

REF = norm_ref(open(f'{D}/reference.txt', encoding='utf-8').read())
HYP = norm_hyp(open(f'{D}/whisper.txt', encoding='utf-8').read())

CONS = {'क':'kcq','ख':'','ग':'g','घ':'g','च':'c','छ':'c','ज':'jz','झ':'jz','ट':'t','ठ':'t',
        'ड':'d','ढ':'d','ण':'n','त':'t','थ':'t','द':'d','ध':'d','न':'n','प':'p','फ':'fp','ब':'b',
        'भ':'b','म':'m','य':'yi','र':'r','ल':'l','व':'vw','श':'s','ष':'s','स':'sc','ह':'h'}
NUKTA_CONS = {'ज':'z','फ':'f','क':'q','ड':'r','ढ':'r'}
VOW = {'अ':'aeuo','आ':'ao','इ':'iy','ई':'iey','उ':'uo','ऊ':'uo','ए':'eay','ऐ':'aei','ओ':'o','औ':'oau','ऑ':'oa',
       'ा':'ao','ि':'iy','ी':'iey','ु':'uo','ू':'uow','े':'eai','ै':'aei','ो':'o','ौ':'oau','ॉ':'oa',
       'ं':'nm','ँ':'n'}
VIRAMA, NUKTA = '्', '़'
MATRAS = set('ािीुूेैोौॉ')

def tokens(word):
    """list of dicts: hyp(text), acc(set of ref letters accepted), opt(bool, zero-cost skip)"""
    t, i = [], 0
    while i < len(word):
        c = word[i]
        if c in CONS:
            txt, acc = c, CONS[c]
            j = i + 1
            if j < len(word) and word[j] == NUKTA:
                txt += NUKTA; acc = NUKTA_CONS.get(c, acc) + acc; j += 1
            vir = j < len(word) and word[j] == VIRAMA
            if vir:
                txt += VIRAMA; j += 1
            t.append(dict(hyp=txt, acc=acc, opt=False))
            if not vir and not (j < len(word) and word[j] in MATRAS):
                t.append(dict(hyp='', acc='aeuo', opt=True, inherent=True))
            i = j
        elif c in VOW:
            t.append(dict(hyp=c, acc=VOW[c], opt=False)); i += 1
        elif c == ' ':
            t.append(dict(hyp=' ', acc=' ', opt=False)); i += 1
        else:  # digits / latin
            t.append(dict(hyp=c, acc=c.lower(), opt=False)); i += 1
    return t

VOWELS = set('aeiou')
def ext_cost(ref, i):
    """cost of absorbing ref[i] with no hyp (as a spelling-convention match); None => real deletion"""
    c = ref[i]; p = ref[i-1] if i > 0 else ''
    nxt = ref[i+1] if i + 1 < len(ref) else ' '
    if c == ' ' or p in ('', ' '):
        return None
    if c == p: return (0.1, f"doubled letter '{p}{c}' = one sound")
    if c == 'h' and p in 'tscpkgbdw': return (0.1, f"digraph '{p}h' = one sound")
    if c == 'k' and p == 'c': return (0.1, "'ck' = one k sound")
    if c in VOWELS and p in VOWELS: return (0.3, f"vowel digraph '{p}{c}' = one vowel")
    if c == 'y' and p in VOWELS: return (0.3, f"'{p}y' = one vowel")
    if c == 'e' and nxt == ' ' : return (0.3, "silent final e")
    if c == 'e' and nxt == 's' and (i + 2 >= len(ref) or ref[i+2] == ' '): return (0.3, "silent e in -es")
    return None

def dp_align(ref, hyp):
    toks = tokens(hyp)
    n, m = len(ref), len(toks)
    INF = 1e9
    C = [[INF]*(m+1) for _ in range(n+1)]; B = [[None]*(m+1) for _ in range(n+1)]
    C[0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            if C[i][j] >= INF: continue
            cur = C[i][j]
            if i < n and j < m:
                ok = ref[i] in toks[j]['acc']
                if ref[i] == ' ' or toks[j]['hyp'] == ' ':
                    cost = 0 if ok else 3
                elif toks[j].get('inherent') and not ok:
                    cost = 99  # never 'substitute' against an empty inherent vowel; that is a deletion
                else:
                    cost = 0 if ok else 1
                if cur + cost < C[i+1][j+1]:
                    C[i+1][j+1] = cur + cost; B[i+1][j+1] = ('diag', ok)
            if i < n:
                e = ext_cost(ref, i)
                cost = e[0] if e else 1
                if cur + cost < C[i+1][j]:
                    C[i+1][j] = cur + cost; B[i+1][j] = ('ref', e)
            if j < m:
                cost = 0 if toks[j]['opt'] else 1
                if cur + cost < C[i][j+1]:
                    C[i][j+1] = cur + cost; B[i][j+1] = ('tok', None)
    steps, i, j = [], n, m
    while i or j:
        kind, info = B[i][j]
        if kind == 'diag':
            tk = toks[j-1]
            if info:
                why = 'inherent vowel of preceding consonant' if tk.get('inherent') else ''
                steps.append([ref[i-1], tk['hyp'], 'match', why])
            else:
                steps.append([ref[i-1], tk['hyp'], 'sub', f"'{ref[i-1]}' vs '{tk['hyp'] or '(inherent a)'}': different sound"])
            i -= 1; j -= 1
        elif kind == 'ref':
            if info: steps.append([ref[i-1], '', 'match', info[1]])
            else: steps.append([ref[i-1], '', 'del', f"no sound for '{ref[i-1]}' in hypothesis"])
            i -= 1
        else:
            tk = toks[j-1]
            if tk['hyp']:
                steps.append(['', tk['hyp'], 'ins', f"extra '{tk['hyp']}' not in reference"])
            j -= 1
    return steps[::-1]

# ---- hand-made word pairing (by sound) -------------------------------------------------
M = 'match'
SPECIAL = {
 '450a': [['4','फोर',M,'four'], ['',' हा','ins',"' हाइडो ' spurious/garbled word (hundred?) not in reference"],
          ['','इ','ins','part of spurious हाइडो'], ['','डो','ins','part of spurious हाइडो'],
          ['5',' 5',M,'digit (word gap folded)'], ['0','0',M,'digit']],
 '450b': [['4','फोर',M,'four'], ['',' हा','ins',"' हाइडो ' spurious/garbled word (hundred?) not in reference"],
          ['','इ','ins','part of spurious हाइडो'], ['','डो','ins','part of spurious हाइडो'],
          ['5',' फिफ़टी',M,"fifty (spelled number; covers '50')"], ['0','',M,"part of 'fifty'"]],
 'phone': [['9','नाइन',M,'nine'], ['8',' एक','sub',"heard 'ek', reference 'eight'"],
           ['0',' जीरो',M,'zero (word gap folded)'], ['0',' जीरो',M,'zero'], ['0',' जीरो',M,'zero'],
           [' ',' ',M,''], ['0','जीरो',M,'zero'],
           ['',' जी','ins','extra fifth zero'], ['','रो','ins','extra fifth zero'],
           ['1',' वन',M,'one'], ['2',' तू',M,'two'], ['9',' नाइन',M,'nine'], ['9',' नाइन',M,'nine']],
 'h12': [['h','एच',M,"letter name 'aitch'"], ['1',' ट्वे',M,"'twe-' of twelve (12 read as one number)"],
         ['2','ल','sub',"'ट्वेल'=twel: final 've' of twelve missing"]],
 '15':  [['1','फिफ्टीन',M,"fifteen (spelled number covers '15')"], ['5','',M,"part of 'fifteen'"]],
 '25w': [['2','ट्रिंकी','sub',"'trinki' is not 'twenty'"], ['5',' फाइव',M,'five']],
}
P = [('hello','हेलो'),('thank','थैंक'),('you','यू'),('for','पार'),('calling','कलीन'),('bazaarly','बेजारली'),
 ('main','में'),('pooja','पूजा'),('main','में'),('koi','कोई'),('kaise','कैसे'),('help','हेल्प'),('kar','कर'),
 ('sakti','सकती'),('hoon','हो'),('ji','जी'),('the','दा'),('order','आर्डर'),('is','इस'),('for','पार'),
 ('boat','बुट'),('rockerz','रकर्स'),('450','@450a'),('headphones','हेटफोन्स'),('midnight','मिदनाइट'),
 ('black','ब्रैक'),('colour','कलर'),('ji','जी'),('bilkul','बिलकुल'),('the','दा'),('order','आर्डर'),('is','इस'),
 ('for','पार'),('lokesh','लोखिश'),('srivastava','श्रीवास्तावा'),('on','अन'),('phone','फोन'),
 ('98000 01299','@phone'),('ji','जी'),('the','दा'),('order','आर्डर'),('is','इस'),('for','पार'),('boat','बोर्ट'),
 ('rockerz','रोकर्स'),('450','@450b'),('and','एन'),('status','स्टेटस'),('shipped','शिप्ट'),('hain','है'),
 ('ji','जी'),('the','दे'),('delivery','डिलिवरी'),('address','अड्रेस'),('is','इच'),('h12','@h12'),
 ('sector','सिक्टर'),('15','@15'),('rohini','रोहीनी'),('delhi','दिली'),('sahi','सही'),('hai','है'),('ji','जी'),
 ('please','प्लीज'),('remind','माइंड'),('me','में'),('gaadi','कारी'),('ka','का'),('ek','एक'),('ji','जी'),
 ('bilkul','बिलकूल'),('the','दे'),('delivery','डिलिवरी'),('is','इच'),('scheduled','शिरूल'),('for','पार'),
 ('friday','फ्राइडे'),('25','@25w'),('october','अक्टूबर'),('haan','हाँ'),('ji','जी'),('the','दा'),
 ('address','अड्रेस'),('is','इच'),('correct','करेक'),('just','जस्ट'),('the','दा'),('order','ओर्डर'),('for','पार'),
 ('boat','बूट'),('rockerz','रॉकर्स'),('450','@450a'),('is','इस'),('friday','प्राइडे'),('25','25'),('ji','जी'),
 ('thank','थैंक'),('you','यू'),('for','पार'),('calling','कुलिंग'),('bazaarly','बजारली'),('have','हैव'),('a','अ'),
 ('good','गुद'),('din','दिन'),('ji','जी'),('','एक और फिर आई है समेजी लाइफ हे कंबी')]

OVR = json.load(open(f'{D}/overrides.json', encoding='utf-8')) if os.path.exists(f'{D}/overrides.json') else {}

steps = []
for k, (r, h) in enumerate(P):
    if k:
        if r: steps.append([' ', ' ', M, '', k])
        else: steps.append(['', ' ', 'ins', 'hallucinated tail after end of reference', k])
    if str(k) in OVR and not str(k).startswith('_'):
        grp = [list(x) for x in OVR[str(k)]]
    elif h.startswith('@'):
        grp = [list(x) for x in SPECIAL[h[1:]]]
    elif not r:
        grp = [['', tk['hyp'], 'ins', 'hallucinated tail after end of reference'] for tk in tokens(h) if tk['hyp']]
    else:
        grp = dp_align(r, h)
    grp = [s[:4] + [k] for s in grp]
    steps.extend(grp)

ref_cat = ''.join(s[0] for s in steps); hyp_cat = ''.join(s[1] for s in steps)
assert ref_cat == REF, ('REF mismatch', ref_cat[:200], REF[:200])
assert hyp_cat == HYP, ('HYP mismatch', hyp_cat, HYP)
for s in steps:
    assert (s[2] == 'ins') == (s[0] == '') and (s[2] == 'del') <= (s[1] == '')

with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for i, s in enumerate(steps):
        f.write(json.dumps(dict(i=i, ref=s[0], hyp=s[1], op=s[2], why=s[3], group=s[4]), ensure_ascii=False) + '\n')

cnt = {o: sum(1 for s in steps if s[2] == o) for o in ('match', 'sub', 'del', 'ins')}
N = len(REF)
assert cnt['match'] + cnt['sub'] + cnt['del'] == N
cer = (cnt['sub'] + cnt['del'] + cnt['ins']) / N
metrics = dict(pair='V1_B__ecom_11_g1', ref_chars=N, hyp_chars=len(HYP), **cnt,
               errors=cnt['sub']+cnt['del']+cnt['ins'], cer=round(cer, 4),
               ins_counting='one ins step per hypothesis akshara (word-gap spaces folded into neighbouring span where noted)',
               normalised_reference=REF, normalised_hypothesis=HYP)
json.dump(metrics, open(f'{D}/char_metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# human-readable: ref / hyp / op lines, each column padded to width of the wider cell
mark = {'match': ' ', 'sub': 'S', 'del': 'D', 'ins': 'I'}
def w(s): return sum(0 if unicodedata.category(c) in ('Mn', 'Mc') else 1 for c in s)
lines, block, rc = [], [], 0
def flush():
    if not block: return
    a = b = c = ''
    for r, h, o in block:
        r = r.replace(' ', '_') if r else '·'; h = h.replace(' ', '_') if h else '·'
        wd = max(w(r), w(h), 1)
        a += r + ' ' * (wd - w(r)) + '|'; b += h + ' ' * (wd - w(h)) + '|'; c += mark[o] + ' ' * (wd - 1) + '|'
    lines.extend(['REF ' + a, 'HYP ' + b, 'OP  ' + c, ''])
    block.clear()
for s in steps:
    block.append((s[0], s[1], s[2])); rc += len(s[0])
    if rc >= 80 and s[0] == ' ':
        flush(); rc = 0
flush()
hdr = f"V1_B__ecom_11_g1 char map  ('_'=space, '·'=empty, ops: ' ' match S sub D del I ins)\n" \
      f"ref_chars={N} match={cnt['match']} sub={cnt['sub']} del={cnt['del']} ins={cnt['ins']} CER={cer:.4f}\n\n"
open(f'{D}/char_map.txt', 'w', encoding='utf-8').write(hdr + '\n'.join(lines))
print(json.dumps({k: v for k, v in metrics.items() if not k.startswith('normalised')}, ensure_ascii=False))
