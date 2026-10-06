#!/usr/bin/env python3
"""Sound-based character alignment: reference (Roman Hinglish) vs Whisper hypothesis (mixed script).

Reference units = characters of normalised ref. Hypothesis units = Roman characters, spaces,
Devanagari aksharas, plus a few merged spoken-form units (spelled digits/letters, the currency
phrase "seven hundred twenty रुपए" == "rs 720").

Each hyp unit gets romanisations as PHONEME TOKEN tuples (e.g. ली -> ("l","ee")). A ref span
matches a token only if it spells that whole token (so ref 'e' never half-matches 'ee').
DP cost = (edits, -matched_ref_chars). The text is cut into anchored segments; a few garbled
regions are aligned by hand (MANUAL) so the map follows sound rather than letter coincidence.
"""
import json, re, itertools, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))

def normalise(s):
    s = unicodedata.normalize("NFC", s).lower()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(" ")
        elif cat[0] in "LN" or cat in ("Mn", "Mc"):
            out.append(ch)
        else:
            out.append(" ")  # punctuation incl. danda
    return re.sub(r" +", " ", "".join(out)).strip()

# ---------------- Devanagari ----------------
C = "[क-हक़-य़]"
AKS = re.compile("(?:" + C + "़?्)*" + C + "़?[ा-ौॢॣ्]?[ऀ-ं]*"
                 "|[ऄ-औ][ऀ-ं]*")
CONS = {"क": ["k", "q", "c"], "ख": ["kh"], "ग": ["g"], "घ": ["gh"], "च": ["ch", "c"], "छ": ["chh", "ch"],
        "ज": ["j", "z"], "झ": ["jh"], "ट": ["t"], "ठ": ["th"], "ड": ["d"], "ढ": ["dh"], "ण": ["n"],
        "त": ["t"], "थ": ["th"], "द": ["d"], "ध": ["dh"], "न": ["n"], "प": ["p"], "फ": ["ph", "f", "ff"],
        "ब": ["b"], "भ": ["bh"], "म": ["m"], "य": ["y"], "र": ["r"], "ल": ["l", "ll"], "व": ["v", "w"],
        "श": ["sh"], "ष": ["sh"], "स": ["s", "ss"], "ह": ["h"]}
MATRA = {"ा": ["aa", "a"], "ि": ["i"], "ी": ["i", "ee"], "ु": ["u"], "ू": ["u", "oo"], "े": ["e", "ay"],
         "ै": ["ai", "e", "ae"], "ो": ["o"], "ौ": ["au", "o"], "ृ": ["ri"], "्": [""]}
INDEP = {"अ": ["a"], "आ": ["aa", "a"], "इ": ["i"], "ई": ["i", "ee"], "उ": ["u"], "ऊ": ["u", "oo"],
         "ए": ["e"], "ऐ": ["ai"], "ओ": ["o"], "औ": ["au"]}
NASAL = ["n", "m", ""]

def rom_akshara(a, prev_i=False):
    nas = any(ch in "ऀँं" for ch in a)
    core = "".join(ch for ch in a if ch not in "ऀँं़")
    if core[0] in INDEP:
        cons, vs = [], INDEP[core[0]]
    else:
        cons, vs = [], ["a", ""]  # inherent schwa may be dropped (schwa deletion)
        for ch in core:
            if ch in CONS:
                cons.append(CONS[ch])
            elif ch in MATRA:
                vs = MATRA[ch]
    out = set()
    for cs in itertools.product(*cons):
        for v in vs:
            for n in (NASAL if nas else [""]):
                toks = tuple(t for t in list(cs) + [v, n] if t)
                if toks:
                    out.add(toks)
    if prev_i and core.startswith("य"):  # 'ia' ~ 'iya' conventional spelling
        out |= {t[1:] for t in out if t[0] == "y" and len(t) > 1}
    return out

# ---------------- hypothesis units ----------------
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "eighty": "80"}
LETTER = {"g": [("g",), ("j", "i"), ("j", "ee")], "f": [("f",)], "d": [("d",)]}
CURRENCY = "seven hundred twenty रुपए"

def is_spelled(w):
    return w in NUMW or w in LETTER

def hyp_units(h):
    """split a hyp chunk (may start/end with a space) into units."""
    units = []
    toks = re.findall(r" |[^ ]+", h.replace(CURRENCY, "\x00"))
    prev_word = None
    k = 0
    while k < len(toks):
        t = toks[k]
        if t == " ":
            nxt = toks[k + 1] if k + 1 < len(toks) else None
            if prev_word and nxt and is_spelled(prev_word) and is_spelled(nxt):
                # space between spelled-out digits/letters: attached to next word, not a sound
                base = [(NUMW[nxt],)] if nxt in NUMW else LETTER[nxt]
                units.append({"text": " " + nxt, "roms": set(base) | {(" ",) + b for b in base}, "kind": "spelled"})
                prev_word = nxt; k += 2; continue
            units.append({"text": " ", "roms": {(" ",)}, "kind": "space"}); k += 1; continue
        if t == "\x00":
            units.append({"text": CURRENCY, "roms": {("rs 720",)}, "kind": "currency"})
            prev_word = CURRENCY; k += 1; continue
        if is_spelled(t):
            base = [(NUMW[t],)] if t in NUMW else LETTER[t]
            units.append({"text": t, "roms": set(base), "kind": "spelled"})
            prev_word = t; k += 1; continue
        pos, prev_i = 0, False
        while pos < len(t):
            m = AKS.match(t, pos)
            if m and m.end() > pos:
                a = m.group(0)
                units.append({"text": a, "roms": rom_akshara(a, prev_i), "kind": "deva"})
                prev_i = any(x in a for x in "िीइई")
                pos = m.end()
            else:
                units.append({"text": t[pos], "roms": {(t[pos],)}, "kind": "roman"})
                prev_i = False
                pos += 1
        prev_word = t; k += 1
    assert "".join(u["text"] for u in units) == h, (h, units)
    return units

# ---------------- inner: ref span vs token tuple ----------------
def inner(ref, toks):
    """every token consumed by: exact match of an equal ref span, or sub vs a ref span (len<=len(tok),
    cost = span len); ref chars may also be deleted (cost 1). Returns ((cost,-matches), per-char ops)."""
    n, m = len(ref), len(toks)
    INF = (10 ** 9, 0)
    D = {(0, 0): (0, 0)}; B = {}
    for a in range(n + 1):
        for b in range(m + 1):
            if (a, b) not in D:
                continue
            c0, mt = D[(a, b)]
            cands = []
            if a < n and ref[a] != " ":
                cands.append((a + 1, b, (c0 + 1, mt), ["del"]))
            if b < m:
                tok = toks[b]
                L = len(tok)
                if ref[a:a + L] == tok:
                    cands.append((a + L, b + 1, (c0, mt - L), ["match"] * L))
                for l in range(1, L + 1):
                    span = ref[a:a + l]
                    if len(span) == l and span != tok and " " not in span and " " not in tok:
                        cands.append((a + l, b + 1, (c0 + l, mt), ["sub"] * l))
            for na, nb, cost, ops in cands:
                if (na, nb) not in D or cost < D[(na, nb)]:
                    D[(na, nb)] = cost; B[(na, nb)] = (a, b, ops)
    if (n, m) not in D:
        return None
    ops, cur = [], (n, m)
    while cur != (0, 0):
        a, b, o = B[cur]
        ops = o + ops; cur = (a, b)
    return D[(n, m)], ops

def group_cost(refsub, unit):
    best = None
    for toks in sorted(unit["roms"], key=lambda t: (-len("".join(t)), t)):
        res = inner(refsub, toks)
        if res and (best is None or res[0] < best[0]):
            best = (res[0], res[1], "".join(toks))
    return best

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def align(ref, units):
    n, m = len(ref), len(units)
    D = {(0, 0): (0, 0)}; B = {}
    def relax(key, cost, bp):
        if key not in D or cost < D[key]:
            D[key] = cost; B[key] = bp
    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in D:
                continue
            d = D[(i, j)]
            if j < m:
                relax((i, j + 1), add(d, (1, 0)), ("ins", i, j, None))
            if i < n:
                relax((i + 1, j), add(d, (1, 0)), ("del", i, j, None))
            if i < n and j < m:
                u = units[j]
                maxk = 6 if u["kind"] == "currency" else 5
                for k in range(1, min(maxk, n - i) + 1):
                    g = group_cost(ref[i:i + k], u)
                    if g is None or g[0][1] == 0 and k > 1:
                        continue
                    relax((i + k, j + 1), add(d, g[0]), ("grp", i, j, g))
                if ref[i] != " " and u["text"] != " ":
                    relax((i + 1, j + 1), add(d, (1, 0)), ("sub1", i, j, None))
    steps, cur = [], (n, m)
    while cur != (0, 0):
        kind, pi, pj, g = B[cur]
        i, j = cur
        if kind == "del":
            steps.append([{"ref": ref[pi], "hyp": "", "op": "del", "why": "not heard in hypothesis"}])
        elif kind == "ins":
            steps.append([{"ref": "", "hyp": units[pj]["text"], "op": "ins", "why": "extra in hypothesis"}])
        elif kind == "sub1":
            steps.append([{"ref": ref[pi], "hyp": units[pj]["text"], "op": "sub", "why": "different sound"}])
        else:
            _, ops, r = g
            u, sub = units[pj], ref[pi:i]
            grp, placed = [], False
            for ch, op in zip(sub, ops):
                hyp = ""
                if not placed and op != "del":
                    hyp, placed = u["text"], True
                if u["kind"] == "currency":
                    why = "'rs 720' read aloud as 'seven hundred twenty rupaye' (same amount+currency; spoken order)"
                elif op == "match":
                    why = "same" if sub == u["text"] else f"'{sub}'~'{u['text']}' (heard '{r.strip()}')"
                elif op == "sub":
                    why = f"'{sub}' vs '{u['text']}' (heard '{r.strip()}'): different sound"
                else:
                    why = "not heard in hypothesis"
                grp.append({"ref": ch, "hyp": hyp, "op": op, "why": why})
            steps.append(grp)
        cur = (pi, pj)
    return [s for grp in steps[::-1] for s in grp]

# ---------------- segments ----------------
def S(r, h, op, why):
    return {"ref": r, "hyp": h, "op": op, "why": why}

MANUAL = {
    # "abhi update hoon" heard as "अभी अपठे रखू हूँ"
    "update1": [S("u", "अ", "match", "'u' of 'update' is /ʌ/ = अ"), S("p", "प", "match", "p = प"),
                S("d", "ठे", "sub", "d vs ठ: different consonant"),
                S("a", "", "match", "'a' (/eɪ/) ~ vowel े of ठे"),
                S("t", "", "del", "t not heard"), S("e", "", "del", "silent e: no sound in hyp"),
                S(" ", " ", "match", "same"),
                S("", "र", "ins", "extra word रखू"), S("", "खू", "ins", "extra word रखू"),
                S("", " ", "ins", "extra word रखू")],
    # "is updated in" heard as "is appear in"
    "updated": [S("u", "a", "match", "/ʌ/ ~ /ə/ of 'appear'"),
                S("p", "pp", "match", "single /p/ (doubled spelling)"),
                S("d", "e", "sub", "d vs e: different sound"), S("a", "a", "match", "same"),
                S("t", "r", "sub", "t vs r"), S("e", "", "del", "not heard"), S("d", "", "del", "final 'ed' not heard")],
    # "abhi update kar" heard as "अभी uptail कर"
    # "burhalia raita" -> "बरहालिया रेहिता": raita heard as 'rehita'
    "raita": [S("r", "रे", "match", "r = र of रे"), S("a", "", "sub", "vowel a/ai vs े (e): different vowel"),
              S("i", "हि", "sub", "हि = extra h + i; consonant h not in ref"), S("t", "ता", "match", "'ta'~'ता'"),
              S("a", "", "match", "'ta'~'ता' (vowel of ता)")],
    "update2": [S("u", "u", "match", "same"), S("p", "p", "match", "same"), S("d", "t", "sub", "d vs t"),
                S("a", "ai", "match", "/eɪ/ ~ 'ai'"), S("t", "l", "sub", "t vs l"),
                S("e", "", "del", "silent e: no sound in hyp")],
}

def manual_steps(key):
    return [dict(s) for s in MANUAL[key]]

def all_ins(h, why):
    return [{"ref": "", "hyp": u["text"], "op": "ins", "why": why} for u in hyp_units(h)]

def main():
    ref = normalise(open(os.path.join(HERE, "reference.txt"), encoding="utf-8").read())
    hyp = normalise(open(os.path.join(HERE, "whisper.txt"), encoding="utf-8").read())

    def cut(text, marker, start):
        p = text.index(marker, start)
        return p
    # anchored segments: (ref_piece, hyp_piece, mode)
    r1 = ref.index("abhi update hoon") + len("abhi ")
    h1 = hyp.index("अभी अपठे") + len("अभी ")
    r2 = r1 + len("update ")
    h2 = h1 + len("अपठे रखू ")
    r3 = ref.index("biryani aur ji") + len("biryani aur ")
    h3 = hyp.index("बिरयानी और बरहानी") + len("बिरयानी और ")
    h4 = h3 + len("बरहानी ")
    r5 = ref.index("is updated in") + len("is ")
    h5 = hyp.index("is appear in") + len("is ")
    r6 = r5 + len("updated")
    h6 = h5 + len("appear")
    r7 = ref.index("abhi update kar") + len("abhi ")
    h7 = hyp.index("अभी uptail") + len("अभी ")
    r8 = r7 + len("update")
    h8 = h7 + len("uptail")
    h9 = hyp.index("good day जी") + len("good day जी")
    ra = ref.index("raita"); ha = hyp.index("रेहिता")
    segs = [
        (ref[:r1], hyp[:h1], "auto"),
        (ref[r1:r2], hyp[h1:h2], "update1"),
        (ref[r2:r3], hyp[h2:h3], "auto"),
        ("", hyp[h3:h4], "ins:extra word बरहानी"),
        (ref[r3:ra], hyp[h4:ha], "auto"),
        (ref[ra:ra + 5], hyp[ha:ha + len("रेहिता")], "raita"),
        (ref[ra + 5:r5], hyp[ha + len("रेहिता"):h5], "auto"),
        (ref[r5:r6], hyp[h5:h6], "updated"),
        (ref[r6:r7], hyp[h6:h7], "auto"),
        (ref[r7:r8], hyp[h7:h8], "update2"),
        (ref[r8:], hyp[h8:h9], "auto"),
        ("", hyp[h9:], "ins:trailing hallucinated speech after closing"),
    ]
    assert "".join(s[0] for s in segs) == ref and "".join(s[1] for s in segs) == hyp
    steps = []
    for r, h, mode in segs:
        if mode == "auto":
            part = align(r, hyp_units(h))
        elif mode.startswith("ins:"):
            assert r == ""
            part = all_ins(h, mode[4:])
        else:
            part = manual_steps(mode)
            assert "".join(s["ref"] for s in part) == r, (mode, r)
            assert "".join(s["hyp"] for s in part) == h, (mode, h)
        steps.extend(part)
    with open(os.path.join(HERE, "char_map.jsonl"), "w", encoding="utf-8") as f:
        for k, s in enumerate(steps):
            f.write(json.dumps({"i": k, "ref": s["ref"], "hyp": s["hyp"], "op": s["op"], "why": s["why"]},
                               ensure_ascii=False) + "\n")
    open(os.path.join(HERE, "ref_norm.txt"), "w", encoding="utf-8").write(ref)
    open(os.path.join(HERE, "hyp_norm.txt"), "w", encoding="utf-8").write(hyp)
    print("steps", len(steps))

if __name__ == "__main__":
    main()
