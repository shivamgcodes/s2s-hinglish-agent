#!/usr/bin/env python3
"""Hand-specified sound-based character alignment for V1_C__cab_07_g3.

Units: reference = characters of normalised reference (spaces are units).
Hypothesis Devanagari is split into consonant(+halant), vowel-sign (matra),
independent vowel and anusvara units; Roman hyp = characters.

Token syntax in word specs (space-separated, '_' means the empty string,
'~' stands for a literal space inside a field):
  r>h     match       r>h!   sub       r>-   del       ->h   ins
Identical words are expanded automatically into char matches.
"""
import json, re, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(t):
    t = t.lower()
    t = ''.join(c if unicodedata.category(c)[0] in 'LMN' or c.isspace() else ' ' for c in t)
    return re.sub(r'\s+', ' ', t).strip()


REF = norm(open(os.path.join(HERE, 'reference.txt'), encoding='utf-8').read())
HYP = norm(open(os.path.join(HERE, 'whisper.txt'), encoding='utf-8').read())

steps = []


def add(r, h, op, why):
    steps.append({"ref": r, "hyp": h, "op": op, "why": why})


def same(word, why="identical spelling, same sound"):
    for c in word:
        add(c, c, "match", why)


def space(why="word boundary"):
    add(" ", " ", "match", why)


def spec(s, note=""):
    for tok in s.split():
        tok = tok.replace('~', ' ')
        if tok.startswith('->'):
            h = tok[2:]
            add("", h, "ins", f"inserted hyp unit '{h}' has no ref counterpart" + (f" ({note})" if note else ""))
            continue
        r, h = tok.split('>', 1)
        if h == '-':
            add(r, "", "del", f"ref '{r}' not heard in hyp" + (f" ({note})" if note else ""))
        elif h.endswith('!'):
            h = h[:-1]
            add(r, h, "sub", f"ref '{r}' vs hyp '{h}' are different sounds" + (f" ({note})" if note else ""))
        else:
            h = '' if h == '_' else h
            if h == '':
                w = f"ref '{r}' sound genuinely carried by preceding hyp unit" + (f" ({note})" if note else "")
            elif h == r:
                w = "same character, same sound" + (f" ({note})" if note else "")
            else:
                w = f"ref '{r}' = hyp '{h}' same sound" + (f" ({note})" if note else "")
            add(r, h, "match", w)


def words(s):
    for i, w in enumerate(s.split()):
        if i:
            space()
        same(w)


# ---------------------------------------------------------------- alignment
spec("h>h e>a! l>g! l>g! o>l! ->e", "hello heard as 'haggle'")
space(); words("thank you for calling"); space()
RIDENOW = "r>r i>i ->g ->h d>t! e>~! n>n o>o w>w"
spec(RIDENOW, "brand 'ridenow' heard as 'right now'; 'gh' silent extra letters")
space(); words("this is"); space()
spec("d>द i>ि v>व् y>य a>ा", "divya = दिव्या")
space()
spec("m>म a>ै i>_ n>ं", "main = मैं; 'ai' is the ै vowel, n = anusvara")
space()
spec("a>आ a>_ p>प k>क i>ी", "aapki = आपकी; 'aa' long vowel in आ")
space()
spec("k>क a>ै i>_ s>स e>े", "kaise = कैसे; 'ai' is ै")
space(); words("help"); space()
spec("k>क a>_ r>र", "kar = कर; 'a' is inherent schwa of क, pronounced")
space()
spec("s>स a>_ k>क t>त i>ी", "sakti = सकती; 'a' inherent schwa of स, pronounced")
space()
spec("h>ह o>ो! o>- n>-", "hoon (/huː~/) heard as हो (/hoː/): vowel differs, nasal n missing")
space()
spec("j>ज i>ी", "ji = जी")
space(); words("sorry"); space()
spec("l>ल e>े k>क i>ि n>न", "lekin = लेकिन")
space(); words("this is only for ride"); space()
spec("r>r d>b!", "booking id letter D heard as B")
spec("4>~four 6>~six 1>~one 6>~six", "digit read as the same number word; leading space separates spoken digits")
space()
spec("a>आ a>_ p>प k>क a>ा", "aapka = आपका")
space(); words("address correct"); space()
spec("s>स e>े t>ठ!", "set heard as सेठ: aspirated retroflex ठ vs plain t")
space()
spec("k>- i>- j>ज i>ी", "kiji heard as just जी; 'ki' dropped")
space()
CHECKING = "c>t! h>- e>a! c>- k>k i>i n>n g>g"
spec(CHECKING, "checking heard as 'taking'; ch->t, e->a, ck digraph extra c not voiced")
space(); words("just a sec"); space()
spec("e>ए k>क", "ek = एक")
space(); words("minute done"); space()
spec("j>g i>_", "ji heard as letter 'g', pronounced /dʒiː/: same sound")
space(); words("the pickup is now igi airport terminal"); space()
spec("3>three", "same number as word")
space()
DELHI = "d>द e>ि l>ल् h>ल! i>ी"
spec(DELHI, "delhi = दिल्ली conventional; ref silent 'h' vs geminate ल counted as sub")
space()
spec(CHECKING, "checking heard as 'taking'; ch->t, e->a, ck digraph extra c not voiced")
space(); words("just a"); space()
spec("s>s e>i! c>c ->k", "sec heard as 'sick'")
space()
spec("e>e ->i ->g ->h k>t!", "ek heard as 'eight'")
space(); words("minute"); space()
spec("j>ज i>ी", "ji = जी")
space()
spec("b>ब i>ि l>ल k>क u>ु l>ल", "bilkul = बिलकुल")
space(); words("the drop is"); space()
spec("u>a ->~ p>p ->r d>e! a>t! t>t e>y! ->~ ->a d>i! ->r",
     "updated heard as 'a pretty air'")
space(); words("aerocity"); space()
spec(DELHI, "delhi = दिल्ली conventional; ref silent 'h' vs geminate ल counted as sub")
space(); words("building"); space()
spec("1>one", "same number as word"); space(); spec("2>two", "same number as word")
space()
spec("j>ज i>ी", "ji = जी")
space(); words("your driver is"); space()
spec("s>स a>_ n>ं d>द e>ी e>_ p>प", "sandeep = संदीप; 'ee' is long ी, 'a' inherent of स")
space()
spec("a>o! u>u r>r", "aur heard as English 'our'")
space(); words("car is"); space()
spec("w>b! h>- i>i t>k! e>e", "white heard as 'bike'")
space(); words("maruti swift dl"); space()
spec("0>zero 1>~one", "digits read as same number words")
space()
spec("k>क o>ो i>ई", "koi = कोई")
space()
spec("b>ब a>ा a>_ t>त", "baat = बात; 'aa' is ा")
space()
spec("n>न a>_ h>ह i>ीं", "nahi = नहीं conventional; 'a' inherent of न, nasal ं conventional")
space()
spec("j>ज i>ी", "ji = जी")
space(); words("thank you for calling"); space()
spec(RIDENOW, "brand 'ridenow' heard as 'right now'; 'gh' silent extra letters")
spec("->~ ->j ->a ->k ->e", "extra word 'jake'")
space(); words("have a nice"); space()
spec("d>d i>a! n>y!", "din heard as 'day'")
spec("->~ ->t ->a ->k ->e ->~ ->y ->o ->u ->~ ->t ->a ->k ->e ->~ ->म ->ि ->ल ->त ->े ->~ ->m ->a ->k ->e ->s",
     "trailing hallucinated words 'take you take मिलते makes'")

# ---------------------------------------------------------------- verify + write
cr = ''.join(s['ref'] for s in steps)
ch = ''.join(s['hyp'] for s in steps)
assert cr == REF, ("REF mismatch", next(i for i in range(min(len(cr), len(REF))) if cr[i] != REF[i]) if cr != REF[:len(cr)] else len(cr), cr[-40:])
assert ch == HYP, ("HYP mismatch", [i for i in range(min(len(ch), len(HYP))) if ch[i] != HYP[i]][:1], ch[-60:])
for s in steps:
    assert s['op'] in ('match', 'sub', 'del', 'ins')
    if s['op'] == 'del': assert s['ref'] and not s['hyp']
    if s['op'] == 'ins': assert s['hyp'] and not s['ref']
    if s['op'] in ('match', 'sub'): assert s['ref']
    assert len(s['ref']) <= 1

with open(os.path.join(HERE, 'char_map.jsonl'), 'w', encoding='utf-8') as f:
    for i, s in enumerate(steps):
        f.write(json.dumps({"i": i, **s}, ensure_ascii=False) + "\n")

cnt = {k: sum(1 for s in steps if s['op'] == k) for k in ('match', 'sub', 'del', 'ins')}
N = len(REF)
assert cnt['match'] + cnt['sub'] + cnt['del'] == N
metrics = {
    "pair": "V1_C__cab_07_g3",
    "ref_chars": N, "hyp_units": len(HYP),
    "match": cnt['match'], "sub": cnt['sub'], "del": cnt['del'], "ins": cnt['ins'],
    "errors": cnt['sub'] + cnt['del'] + cnt['ins'],
    "cer": round((cnt['sub'] + cnt['del'] + cnt['ins']) / N, 4),
    "normalised_reference": REF, "normalised_hypothesis": HYP,
}
json.dump(metrics, open(os.path.join(HERE, 'char_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# char_map.txt: blocks of ref/hyp/op lines
def cell(x): return x if x else '·'
lines = []
for b in range(0, len(steps), 40):
    blk = steps[b:b + 40]
    w = [max(len(cell(s['ref'])), len(cell(s['hyp'])), 1) for s in blk]
    sym = {'match': '.', 'sub': 'S', 'del': 'D', 'ins': 'I'}
    lines.append("ref: " + "|".join(cell(s['ref']).replace(' ', '_').ljust(x) for s, x in zip(blk, w)))
    lines.append("hyp: " + "|".join(cell(s['hyp']).replace(' ', '_').ljust(x) for s, x in zip(blk, w)))
    lines.append("op : " + "|".join(sym[s['op']].ljust(x) for s, x in zip(blk, w)))
    lines.append("")
lines.append(json.dumps({k: metrics[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'errors', 'cer')}))
open(os.path.join(HERE, 'char_map.txt'), 'w', encoding='utf-8').write("\n".join(lines) + "\n")
print(json.dumps({k: metrics[k] for k in ('ref_chars', 'match', 'sub', 'del', 'ins', 'errors', 'cer')}))
