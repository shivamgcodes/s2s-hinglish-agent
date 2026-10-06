#!/usr/bin/env python3
"""Sound-based character alignment: reference (Roman Hinglish) vs Trelis Whisper hypothesis.

Items are aligned word-by-word (manually paired). Roman/Roman pairs are aligned
with char Levenshtein (or a manual map where sound demands it); Devanagari
hypothesis words get manual sound maps. Hyp units are single Devanagari code
points (consonant / matra / nasal / virama-bearing consonant), or a whole word
for spelled-out letters / numbers.
"""
import json, re, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(c if (c.isalnum() or unicodedata.category(c) in ("Mn", "Mc") or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def lev(a, b):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            op = "match" if a[i - 1] == b[j - 1] else "sub"
            out.append((a[i - 1], b[j - 1], op, "same letter" if op == "match" else f"'{a[i-1]}' heard as '{b[j-1]}'"))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append((a[i - 1], "", "del", f"'{a[i-1]}' not heard"))
            i -= 1
        else:
            out.append(("", b[j - 1], "ins", f"extra '{b[j-1]}'"))
            j -= 1
    return out[::-1]


OPS = {"m": "match", "s": "sub", "d": "del", "i": "ins"}


def M(*steps):
    """manual steps: (ref, hyp, op[, why])"""
    out = []
    for st in steps:
        r, h, o = st[:3]
        why = st[3] if len(st) > 3 else {
            "m": "same sound" if h else "sound carried by previous hyp unit (digraph/inherent vowel)",
            "s": f"'{r}' heard as '{h}'", "d": f"'{r}' sound absent", "i": f"extra '{h}'"}[o]
        out.append((r, h, OPS[o], why))
    return out


A = lev  # auto Roman/Roman

JI = M(("j", "ज", "m"), ("i", "ी", "m"))
HOON = M(("h", "ह", "m"), ("o", "ू", "m"), ("o", "", "m", "oo digraph = ू"), ("n", "ँ", "m", "final n = nasalisation"))
AAPKA = M(("a", "आ", "m"), ("a", "", "m", "aa digraph = आ"), ("p", "प", "m"), ("k", "क", "m"), ("a", "ा", "m"))
ABHI = M(("a", "अ", "m"), ("b", "भ", "m"), ("h", "", "m", "bh digraph = भ"), ("i", "ी", "m"))
KAR = M(("k", "क", "m"), ("a", "", "m", "inherent a of क"), ("r", "र", "m"))
HAI = M(("h", "ह", "m"), ("a", "ै", "m"), ("i", "", "m", "ai digraph = ै"))
THEEK = M(("t", "ठ", "m", "th = ठ"), ("h", "", "m", "th digraph = ठ"), ("e", "ी", "m"), ("e", "", "m", "ee digraph = ी"), ("k", "क", "m"))
BILKUL1 = M(("b", "ब", "m"), ("i", "ि", "m"), ("l", "ल", "m"), ("k", "क", "m"), ("u", "ु", "m"), ("l", "ल", "m"))
BILKUL2 = M(("b", "ब", "m"), ("i", "ि", "m"), ("l", "ल्", "m"), ("k", "क", "m"), ("u", "ु", "m"), ("l", "ल", "m"))

items = [
    ("hello", "hello", A), ("thank", "thank", A), ("you", "you", A), ("for", "for", A), ("calling", "calling", A),
    M(("b", "ड", "s", "b heard as retroflex d"), ("i", "ि", "s", "ai (bite) heard as short i"), ("t", "प", "s", "t heard as p"),
      ("e", "", "d", "silent e; ai quality not heard"), ("w", "", "d", "w absent"), ("a", "े", "m", "ay = e"), ("y", "", "m", "ay digraph = े")),
    ("this", "this", A), ("is", "is", A),
    M(("k", "क", "m"), ("a", "ा", "m", "kaavya a~ा"), ("v", "व", "m"), ("y", "", "d", "y of -vya absent"), ("a", "ा", "m", "final a~ा")),
    M(("m", "म", "m"), ("a", "ै", "m"), ("i", "", "m", "ai digraph = ै"), ("n", "ं", "m", "final n = anusvara")),
    M(("a", "आ", "m"), ("a", "", "m", "aa digraph = आ"), ("p", "प", "m"), ("k", "क", "m"), ("i", "ी", "m")),
    M(("k", "क", "m"), ("a", "ै", "m"), ("i", "", "m", "ai digraph = ै"), ("s", "स", "m"), ("e", "े", "m")),
    ("help", "help", A), KAR,
    M(("s", "स", "m"), ("a", "", "m", "inherent a of स"), ("k", "क", "m"), ("t", "त", "m"), ("i", "ी", "m")),
    HOON, JI, ("your", "your", A), ("order", "order", A),
    M(("f", "if ", "m", "letter F spoken 'ef' ~ 'if'"), ("d", "the ", "s", "letter D 'dee' heard as 'the'"),
      ("2", "two ", "m", "same number"), ("2", "two ", "m", "same number"), ("4", "four ", "m", "same number"), ("3", "three", "m", "same number")),
    M(("f", "फ", "m"), ("", "ॉ", "i", "vowel o inserted before r (metathesis fro->for)"), ("r", "र्", "m"),
      ("o", "", "d", "o after r absent (moved before r)"), ("m", "म", "m")),
    M(("b", "ब", "m"), ("e", "े", "m"), ("h", "ह", "m"), ("r", "र", "m"), ("o", "ू", "m", "ou = oo"), ("u", "", "m", "ou digraph = ू"),
      ("z", "ष", "s", "z heard as sh")),
    ("is", "is", A), ("out", "out", A), ("for", "for", A), ("delivery", "delivery", A), ("and", "and", A),
    AAPKA, ("eta", "eta", A), ABHI,
    M(("1", "trail", "s", "12 ('twelve') heard as 'trail' - not the same number"), ("2", "", "d", "rest of number not recognised")),
    M(("m", "म", "m"), ("i", "े", "s", "i heard as e"), ("n", "ं", "m", "n as nasal"), ("u", "", "d"), ("t", "", "d"), ("e", "", "d"), ("s", "", "d")),
    JI, ("the", "the", A), ("instruction", "instruction", A), ("is", "is", A), ("leave", "leave", A), ("at", "out", A),
    ("main", "main", A), ("gate", "gate", A), ("which", "which", A), ("is", "is", A), ("currently", "currently", A), ("set", "set", A),
    JI, BILKUL1, ("please", "please", A),
    M(("c", "t", "s", "ch heard as t"), ("h", "", "d", "ch affricate absent"), ("e", "a", "s", "e heard as ei (take)"),
      ("c", "k", "m", "ck = k"), ("k", "", "m", "ck digraph = k"), ("", "e", "i", "extra (silent) e in 'take'")),
    ("and", "and", A), ("confirm", "confirm", A), ("the", "the", A), ("correct", "correct", A), ("instruction", "instruction", A),
    ("then", "then", A), ("we", "we", A), ("can", "can", A), ("update", "update", A), ("it", "it", A), ("let", "let", A),
    ("me", "me", A), ("check", "check", A), ("just", "just", A), ("now", "now", A),
    M(("h", "ह", "m"), ("a", "ा", "m"), ("a", "", "m", "aa digraph = ा"), ("n", "ँ", "m", "final n = nasalisation")),
    JI, ABHI, ("update", "update", A), KAR,
    M(("d", "द", "m"), ("e", "े", "m"), ("t", "त", "m"), ("i", "ी", "m")), HOON,
    M(("d", "ध", "s", "d heard as aspirated dh"), ("o", "", "m", "short u/a vowel = inherent a of ध"), ("n", "न", "m"),
      ("e", "", "d", "silent e")),
    JI, ("the", "the", A), ("instruction", "instruction", A), ("is", "is", A), ("updated", "updated", A),
    ("ring", "bring", A), ("bell", "bill", A), ("and", "in", A), ("hand", "hand", A), ("over", "over", A),
    ("personalna", "persona", A),
    THEEK, HAI, M(("y", "य", "m"), ("a", "ा", "m")),
    M(("m", "म", "m"), ("u", "ु", "m"), ("s", "स्", "m"), ("k", "क", "m"), ("u", "ु", "m"), ("r", "र", "m"), ("e", "े", "m"),
      ("i", "", "m", "ei digraph = े"), ("n", "न", "m")),
    AAPKA, ("delivery", "delivery", A), BILKUL2, ("safe", "safe", A), M(("h", "ह", "m"), ("o", "ो", "m")),
    M(("t", "द", "s", "th (ठ) heard as d"), ("h", "ह", "m", "aspiration heard as separate ह"), ("e", "ि", "m", "ee~i"),
      ("e", "", "m", "ee digraph = ि"), ("k", "क", "m")),
    HAI, JI, ("the", "the", A), ("request", "request", A), ("is", "is", A), ("submitted", "submit", A),
    JI, ("thank", "thank", A), ("you", "you", A), ("for", "for", A), ("calling", "calling", A), ("biteway", "bitpay", A),
    ("have", "have", A), ("a", "a", A), ("nice", "nice", A), ("din", "day", A),
    M(("e", "ए", "m"), ("n", "न", "m"), ("d", "", "d", "final d absent")), ("song", "song", A),
    M(("j", "ज", "m"), ("i", "ी", "m"), (" ", "", "d", "word break absent (ji+t merged)"), ("t", "त", "m", "t of thank"),
      ("", " ", "i", "word break inserted"), ("h", "ह", "m", "h of th heard as ह"), ("a", "ै", "s", "a heard as ai"),
      ("n", "", "d"), ("k", "", "d")),
]

steps = []
for k, it in enumerate(items):
    if k:
        steps.append((" ", " ", "match", "word boundary"))
    if isinstance(it, tuple):
        r, h, f = it
        steps += f(r, h)
    else:
        steps += it

ref_n = norm(open(os.path.join(HERE, "reference.txt")).read())
hyp_n = norm(open(os.path.join(HERE, "whisper.txt")).read())
R = "".join(s[0] for s in steps)
H = "".join(s[1] for s in steps)
assert all(len(s[0]) <= 1 for s in steps), "ref unit >1 char"
if R != ref_n or H != hyp_n:
    for name, a, b in (("REF", R, ref_n), ("HYP", H, hyp_n)):
        if a != b:
            i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(name, "MISMATCH at", i, repr(a[i-20:i+20]), "vs", repr(b[i-20:i+20]))
    raise SystemExit(1)

cnt = {"match": 0, "sub": 0, "del": 0, "ins": 0}
with open(os.path.join(HERE, "char_map.jsonl"), "w") as f:
    for i, (r, h, o, w) in enumerate(steps):
        cnt[o] += 1
        f.write(json.dumps({"i": i, "ref": r, "hyp": h, "op": o, "why": w}, ensure_ascii=False) + "\n")
nref = len(ref_n)
assert nref == cnt["match"] + cnt["sub"] + cnt["del"]
cer = (cnt["sub"] + cnt["del"] + cnt["ins"]) / nref
metrics = {"pair": "V1_A__food_12_g3", "ref_chars": nref, "hyp_units": sum(1 for s in steps if s[1]),
           "match": cnt["match"], "sub": cnt["sub"], "del": cnt["del"], "ins": cnt["ins"], "cer": round(cer, 4),
           "ref_normalised": ref_n, "hyp_normalised": hyp_n, "concat_verified": True}
json.dump(metrics, open(os.path.join(HERE, "char_metrics.json"), "w"), ensure_ascii=False, indent=2)

# char_map.txt: blocks of ref/hyp/op lines
W = 60
lines = []
sym = {"match": "|", "sub": "S", "del": "D", "ins": "I"}
for b in range(0, len(steps), W):
    blk = steps[b:b + W]
    cw = [max(1, len(r), len(h)) for r, h, _, _ in blk]
    lines.append("ref: " + " ".join((r or "-").replace(" ", "_").ljust(w) for (r, _, _, _), w in zip(blk, cw)))
    lines.append("hyp: " + " ".join((h or "-").replace(" ", "_").ljust(w) for (_, h, _, _), w in zip(blk, cw)))
    lines.append("op:  " + " ".join(sym[o].ljust(w) for (_, _, o, _), w in zip(blk, cw)))
    lines.append("")
lines.append(f"ref_chars={nref} match={cnt['match']} sub={cnt['sub']} del={cnt['del']} ins={cnt['ins']} CER={cer:.4f}")
open(os.path.join(HERE, "char_map.txt"), "w").write("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in metrics.items() if "normalised" not in k}))
