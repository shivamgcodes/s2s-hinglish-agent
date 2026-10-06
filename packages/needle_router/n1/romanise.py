# Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
# research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle/romanise.py (verbatim).
#
# Deterministic romanisation of Devanagari tokens in Trelis ASR output -> typed Roman Hinglish (N1 rendering b).
#
# 1. LEXICON (romanise_lexicon.json), mined from the parallel text in calls.jsonl, TRAIN scenarios only
#    (holdout.json test_scenarios excluded), agent+customer turns, V1+V3:
#    text_tts (Devanagari, IDs kept in Latin) vs text_roman. Both sides are split on whitespace and
#    stripped of punctuation; a turn is used only if token counts are equal AND every Latin token on the
#    tts side equals (case-insensitively) the roman token at the same position. Each Devanagari key
#    (normalised, see key()) maps to its most frequent lower-cased roman form; ties -> alphabetically first.
# 2. FALLBACK for lexicon misses: rule-based character transliteration (tables below):
#    consonant + inherent 'a', matras replace the 'a', virama removes it, word-final schwa deleted,
#    anusvara/chandrabindu -> 'n' ('m' before p/b/m), visarga -> 'h', nukta forms ज़->z फ़->f क़->q ख़->kh ग़->gh
#    ड़->r ढ़->rh; Devanagari digits -> ASCII digits.
# 3. Latin tokens (English words, spelled-out numbers like "twenty", "F D one zero") are left unchanged.
#    Devanagari number words become words via the lexicon/fallback (एक -> ek), never digits.
#    Danda । -> '.'; punctuation attached to a token is kept around the romanised core.
# Fallback vowel/consonant choices (ा->a, ी/ई->i, ऊ->u, व->w, फ->f) were picked by exact-match on the
# TRAIN lexicon entries (type level); typed Hinglish rarely doubles vowels (aapka, pareshan, abhi).
# Output case: lower-case for romanised tokens (Latin tokens keep their ASR case).
#
# Usage: python romanise.py build <calls_dir_with_V1_V3> <holdout.json>      -> writes romanise_lexicon.json + eval
#        python romanise.py apply <asr_turns_raw.jsonl> <asr_turns.jsonl>   -> adds asr_roman
import json, re, sys, os, unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_PATH = os.path.join(HERE, "romanise_lexicon.json")

DEV = re.compile(r"[ऀ-ॿ]")
PUNCT = ".,?!;:\"'()[]{}-–—…।॥"

def key(w):
    """Lookup key: NFC, chandrabindu->anusvara, nukta kept (a second, nukta-stripped key is tried on miss)."""
    w = unicodedata.normalize("NFC", w)
    w = unicodedata.normalize("NFD", w)  # split precomposed nukta letters (ज़ = ज + ़)
    w = w.replace("ँ", "ं")
    return unicodedata.normalize("NFC", w)

def key_nonukta(w):
    w = unicodedata.normalize("NFD", key(w)).replace("़", "")
    return unicodedata.normalize("NFC", w)

def strip_punct(t):
    i, j = 0, len(t)
    while i < j and t[i] in PUNCT: i += 1
    while j > i and t[j - 1] in PUNCT: j -= 1
    return t[:i], t[i:j], t[j:]

# ---------------- rule-based fallback ----------------
CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "w",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
NUKTA = {"ज": "z", "फ": "f", "क": "q", "ख": "kh", "ग": "gh", "ड": "r", "ढ": "rh"}
VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
       "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "oo", "ृ": "ri",
         "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
VIRAMA, NUK, ANU, CBINDU, VISARGA = "्", "़", "ं", "ँ", "ः"
DIG = {chr(0x0966 + i): str(i) for i in range(10)}

def fallback(w):
    s = unicodedata.normalize("NFD", w)
    u = []  # units: [kind, text, vowel]; kind C (vowel None=inherent, ''=virama, else matra), V, X (other)
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in CONS:
            r = CONS[ch]
            if i + 1 < len(s) and s[i + 1] == NUK:
                r = NUKTA.get(ch, r); i += 1
            u.append(["C", r, None])
        elif ch in MATRA and u and u[-1][0] == "C" and u[-1][2] is None:
            u[-1][2] = MATRA[ch]
        elif ch == VIRAMA and u and u[-1][0] == "C":
            u[-1][2] = ""
        elif ch in VOW:
            u.append(["V", VOW[ch], None])
        elif ch in (ANU, CBINDU):
            nxt = s[i + 1] if i + 1 < len(s) else ""
            u.append(["X", "m" if nxt in ("प", "फ", "ब", "भ", "म") else "n", None])
        elif ch == VISARGA:
            u.append(["X", "h", None])
        elif ch in DIG:
            u.append(["X", DIG[ch], None])
        elif ch == NUK or ch in MATRA:
            pass
        else:
            u.append(["X", ch, None])
        i += 1
    def has_v(x):  # unit carries a pronounced vowel
        return x[0] == "V" or (x[0] == "C" and x[2] not in ("", "-"))
    # schwa deletion, right to left: word-final inherent a dropped (if the word has another vowel);
    # medial inherent a dropped in V C[a] C V context (bilakul -> bilkul, samajhana -> samajhna)
    nv = sum(1 for x in u if has_v(x))
    for k in range(len(u) - 1, -1, -1):
        x = u[k]
        if x[0] != "C" or x[2] is not None: continue
        if k == len(u) - 1 or all(y[0] == "X" for y in u[k + 1:]):
            if nv > 1 and not (k + 1 < len(u) and u[k + 1][0] == "X"): x[2] = "-"; nv -= 1
            continue
        if 0 < k < len(u) - 1 and has_v(u[k - 1]) and u[k + 1][0] == "C" and has_v(u[k + 1]):
            x[2] = "-"; nv -= 1
    out = []
    for kind, t, v in u:
        out.append(t)
        if kind == "C": out.append("a" if v is None else ("" if v in ("", "-") else v))
    return "".join(out)

# ---------------- lexicon ----------------
def build(calls_dir, holdout):
    held = set(json.load(open(holdout))["test_scenarios"])
    cnt = defaultdict(Counter); n_turns = n_used = 0
    evalpairs = []  # held-out (key, roman) pairs for romaniser evaluation
    for V in ["V1", "V3"]:
        for l in open(os.path.join(calls_dir, V, "calls.jsonl")):
            c = json.loads(l)
            for t in c["turns"]:
                a, b = t["text_tts"].split(), t["text_roman"].split()
                n_turns += 1
                if len(a) != len(b): continue
                pairs = []; ok = True
                for x, y in zip(a, b):
                    x, y = strip_punct(x)[1], strip_punct(y)[1]
                    if not x or not y: ok = False; break
                    if DEV.search(x):
                        if DEV.search(y): ok = False; break
                        # mixed-script tokens (e.g. IDs like एफडी1565) are never looked up: romanise() only
                        # passes pure Devanagari runs, so they are not added to the lexicon
                        if re.search(r"[0-9A-Za-z]", x) or re.search(r"[0-9]", y): continue
                        pairs.append((x, y.lower()))
                    elif x.lower() != y.lower(): ok = False; break
                if not ok: continue
                n_used += 1
                if c["scenario_id"] in held:
                    evalpairs.extend(pairs)
                else:
                    for x, y in pairs: cnt[key(x)][y] += 1
    lex = {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, v in sorted(cnt.items())}
    json.dump(lex, open(LEX_PATH, "w"), ensure_ascii=False, indent=0, sort_keys=True)
    print(f"turns {n_turns} aligned {n_used} lexicon_entries {len(lex)}")
    return evalpairs

def evaluate(evalpairs, lex):
    # held-out aligned pairs (gold text_tts tokens, not ASR) vs gold text_roman tokens
    hit = hit_ok = miss = miss_ok = 0
    for x, y in evalpairs:
        r, src = rom_core(x, lex)
        if src == "lex": hit += 1; hit_ok += r == y
        else: miss += 1; miss_ok += r == y
    print(f"heldout gold tokens {len(evalpairs)}: lexicon hits {hit} ({hit_ok/max(hit,1):.3f} exact), "
          f"fallback {miss} ({miss_ok/max(miss,1):.3f} exact), overall exact {(hit_ok+miss_ok)/max(len(evalpairs),1):.3f}")

def rom_core(core, lex):
    k = key(core)
    if k in lex: return lex[k], "lex"
    k2 = key_nonukta(core)
    if k2 in LEX_NN: return LEX_NN[k2], "lex"
    return fallback(core), "rule"

LEX_NN = {}
def load_lex():
    lex = json.load(open(LEX_PATH))
    LEX_NN.clear()
    for k in sorted(lex):  # nukta-stripped alias; first key in sorted order wins (deterministic)
        LEX_NN.setdefault(key_nonukta(k), lex[k])
    return lex

def romanise(text, lex, stats=None):
    out = []
    for tok in text.split():
        pre, core, post = strip_punct(tok)
        post = post.replace("।", ".").replace("॥", ".")
        pre = pre.replace("।", ".")
        if core and DEV.search(core):
            # a token may mix scripts (rare); romanise Devanagari runs only
            parts = re.split(r"([ऀ-ॿ]+)", core)
            rs = []
            for p in parts:
                if p and DEV.search(p):
                    r, src = rom_core(p, lex); rs.append(r)
                    if stats is not None: stats[src] += 1
                else: rs.append(p)
            core = "".join(rs)
        elif stats is not None and core: stats["latin"] += 1
        out.append(pre + core + post)
    return " ".join(out)

if __name__ == "__main__":
    if sys.argv[1] == "build":
        ev = build(sys.argv[2], sys.argv[3]); evaluate(ev, load_lex())
    elif sys.argv[1] == "apply":
        lex = load_lex(); st = Counter(); rows = []
        for l in open(sys.argv[2]):
            r = json.loads(l); r["asr_roman"] = romanise(r["asr_text"], lex, st); rows.append(r)
        with open(sys.argv[3], "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tot = st["lex"] + st["rule"]
        print(f"rows {len(rows)} tokens latin {st['latin']} devanagari {tot} lexicon {st['lex']} ({st['lex']/max(tot,1):.3f}) rule {st['rule']}")
