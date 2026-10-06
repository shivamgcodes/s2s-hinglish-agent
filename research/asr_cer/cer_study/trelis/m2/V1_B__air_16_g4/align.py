#!/usr/bin/env python3
"""Sound-based character alignment of reference (Roman Hinglish) vs Whisper hypothesis
(Roman/Devanagari mixed).  Weighted edit-distance DP where a 'match' step may pair
1..4 reference chars with one hypothesis unit when they spell the same sound
(e.g. 'bh'~भ, 'aa'~ा, 'ka'~क with inherent schwa), plus a short list of exact
same-number equivalences (digits vs number words).

Reference units = characters.  Hypothesis units = Roman chars / spaces / digits,
and Devanagari split into consonant(+nukta)(+virama) | matra | vowel | nasal sign.
"""
import json, re, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))


def normalise(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in s)
    # colon / comma inside numbers ("7:15", "3,000") are dropped, not spaced
    return re.sub(r"\s+", " ", s).strip()


def normalise_ref(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"(?<=\d)[:,](?=\d)", "", s)
    return normalise(s)


def normalise_hyp(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"(?<=\d)[:,](?=\d)", "", s)
    return normalise(s)


CONS = {
    "क": ["k", "c", "q"], "ख": ["kh"], "ग": ["g"], "घ": ["gh"], "ङ": ["n"],
    "च": ["ch", "c"], "छ": ["chh", "ch"], "ज": ["j"], "झ": ["jh"], "ञ": ["n"],
    "ट": ["t"], "ठ": ["th"], "ड": ["d"], "ढ": ["dh"], "ण": ["n"],
    "त": ["t"], "थ": ["th"], "द": ["d"], "ध": ["dh"], "न": ["n"],
    "प": ["p"], "फ": ["ph", "f"], "ब": ["b"], "भ": ["bh"], "म": ["m"],
    "य": ["y"], "र": ["r"], "ल": ["l"], "व": ["v", "w"],
    "श": ["sh", "s"], "ष": ["sh", "s"], "स": ["s"], "ह": ["h"],
    "क़": ["q", "k"], "ख़": ["kh"], "ग़": ["g"], "ज़": ["z", "j"], "फ़": ["f", "ph"],
    "ड़": ["r", "d"], "ढ़": ["rh"],
}
MATRA = {
    "ा": ["aa", "a"], "ि": ["i"], "ी": ["ee", "i"], "ु": ["u"], "ू": ["oo", "u"],
    "े": ["e", "ay"], "ै": ["ai", "ae", "e"], "ो": ["o"], "ौ": ["au", "o"], "ृ": ["ri"],
    "ं": ["n", "m"], "ँ": ["n"], "ः": ["h"], "ॅ": ["e"], "ॉ": ["o"],
}
VOWEL = {
    "अ": ["a"], "आ": ["aa", "a"], "इ": ["i"], "ई": ["ee", "i"], "उ": ["u"],
    "ऊ": ["oo", "u"], "ए": ["e", "ye", "ay"], "ऐ": ["ai", "e"], "ओ": ["o"], "औ": ["au", "o"],
    "ऋ": ["ri"],
}
NUKTA, VIRAMA = "़", "्"

# same-number / same-sound phrase equivalences (ref string, hyp string)
PHRASES = [
    ("891", "eight nine one"),
    ("402", "four hundred two"),
    ("715", "seven fifteen"),
    ("22", "twenty two"),
    ("3000", "three thousand"),
    ("wo", "व"),  # 'woh' ~ वह : वह is pronounced 'voh'
]


# "koi baat nahi" ends the reference; everything after "नहीं" in the hypothesis
# (एक मेनत रुके किसी फ़ायदा में चही) is hallucinated trailing speech.
ANCHORS = [("koi baat nahi", "कोई बात नहीं")]


def hyp_units(h):
    units, i = [], 0
    while i < len(h):
        ch = h[i]
        if "ऀ" <= ch <= "ॿ" and unicodedata.category(ch) == "Lo" and ch not in VOWEL:
            u = ch
            i += 1
            while i < len(h) and h[i] in (NUKTA, VIRAMA):
                u += h[i]
                i += 1
            units.append(u)
        else:
            units.append(ch)
            i += 1
    return units


def unit_options(u, nxt=""):
    """Return list of ref strings this unit matches (sound-equivalent).
    A consonant carries an inherent 'a' only when no matra/virama follows it."""
    base = unicodedata.normalize("NFC", u.replace(VIRAMA, ""))
    if base in CONS:
        opts = list(CONS[base])
        if VIRAMA not in u and nxt not in MATRA:
            opts += [o + "a" for o in opts]  # inherent schwa
        return opts
    if u in MATRA:
        return MATRA[u]
    if u in VOWEL:
        return VOWEL[u]
    return [u]  # roman / digit / space: identity


def align(ref, hu):
    """Primary cost = edit count (sub/del/ins = 1, match = 0).
    Secondary cost (tie-break only, never changes the edit count):
      +2 each time an edit run opens after a match (keeps gaps contiguous/local),
      +1 per sub (prefer del+match over sub when equal), +3 for sub space<->non-space."""
    n, m = len(ref), len(hu)
    INF = (10 ** 9, 0)
    D = [[[INF, INF] for _ in range(m + 1)] for _ in range(n + 1)]
    B = [[[None, None] for _ in range(m + 1)] for _ in range(n + 1)]
    D[0][0][0] = (0, 0)
    opts = [unit_options(u, hu[k + 1] if k + 1 < len(hu) else "") for k, u in enumerate(hu)]
    phrase_units = [(r, hyp_units(p)) for r, p in PHRASES]
    for i in range(n + 1):
        for j in range(m + 1):
            for st in (0, 1):
                d = D[i][j][st]
                if d == INF:
                    continue
                cands = []
                if i < n:
                    cands.append((i + 1, j, 1, 0, "del"))
                if j < m:
                    cands.append((i, j + 1, 1, 0, "ins"))
                if i < n and j < m:
                    for o in opts[j]:
                        if o and ref.startswith(o, i):
                            cands.append((i + len(o), j + 1, 0, 0, "match"))
                    sp = (ref[i] == " ") != (hu[j] == " ")
                    cands.append((i + 1, j + 1, 1, 1 + (3 if sp else 0), "sub"))
                    for r, pu in phrase_units:
                        if ref.startswith(r, i) and hu[j:j + len(pu)] == pu:
                            cands.append((i + len(r), j + len(pu), 0, 0, "match"))
                for ni, nj, c, sec, op in cands:
                    nst = 0 if op == "match" else 1
                    if nst == 1 and st == 0:
                        sec += 2
                    nd = (d[0] + c, d[1] + sec)
                    if nd < D[ni][nj][nst]:
                        D[ni][nj][nst] = nd
                        B[ni][nj][nst] = (i, j, st, op)
    st = 0 if D[n][m][0] <= D[n][m][1] else 1
    cost = D[n][m][st]
    steps, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, pst, op = B[i][j][st]
        steps.append((ref[pi:i], "".join(hu[pj:j]), op))
        i, j, st = pi, pj, pst
    return steps[::-1], cost


def why(r, h, op):
    if op == "match":
        if r == h:
            return "identical"
        if any(c.isdigit() for c in r):
            return f"same number: {r} spoken as '{h}'"
        return f"same sound: '{r}' ~ {h}"
    if op == "sub":
        return f"different sound: '{r}' vs '{h}'"
    if op == "del":
        return f"ref '{r}' not heard in hypothesis"
    return f"hyp '{h}' extra (not in reference)"


def main():
    ref = normalise_ref(open(os.path.join(HERE, "reference.txt"), encoding="utf-8").read())
    hyp = normalise_hyp(open(os.path.join(HERE, "whisper.txt"), encoding="utf-8").read())
    # Segment anchors: alignment is run independently between anchors so that a
    # hallucinated hypothesis tail cannot "borrow" isolated letters from far away.
    steps, cost, ri, hi = [], [0, 0], 0, 0
    for ra, ha in ANCHORS:
        re_ = ref.index(ra, ri) + len(ra)
        he_ = hyp.index(ha, hi) + len(ha)
        st, c = align(ref[ri:re_], hyp_units(hyp[hi:he_]))
        steps += st; cost[0] += c[0]; cost[1] += c[1]
        ri, hi = re_, he_
    st, c = align(ref[ri:], hyp_units(hyp[hi:]))
    steps += st; cost[0] += c[0]; cost[1] += c[1]
    rows = [{"i": k, "ref": r, "hyp": h, "op": op, "why": why(r, h, op)} for k, (r, h, op) in enumerate(steps)]
    with open(os.path.join(HERE, "char_map.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    open(os.path.join(HERE, "ref_norm.txt"), "w", encoding="utf-8").write(ref)
    open(os.path.join(HERE, "hyp_norm.txt"), "w", encoding="utf-8").write(hyp)
    print("cost", cost, "steps", len(rows))


if __name__ == "__main__":
    main()
