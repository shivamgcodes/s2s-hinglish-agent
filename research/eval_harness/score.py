"""Score test runs (spec section 6 metrics 1-6). CPU only; merges .asr.json / .judge.json when present.

  python score.py TAG [TAG ...]
Per run -> tests/out/<TAG>/runs.csv (+ runs.jsonl with lists). Aggregate -> tests/out/<TAG>/scores.csv and
scores.json: for each variant, per seed the mean over calls, then mean +- std (population std, ddof=0) over seeds.
Also writes tests/out/<TAG>/gate0_monologues.txt and tests/out/<TAG>/streams.txt (verbatim text streams).

Time base: model text frame j <-> input time (j + skipped_steps) / 12.5 s; script times from inputs/<V>/<call>.meta.json
(already shifted to the 2 s-lead input).
Metrics (per run):
 m1_hindi_recall     fraction of scripted agent Hindi word types (hindi_share 'H' tokens of the agent text_roman)
                     that occur as tokens in the model text stream (lower-cased tokens).
 m1_novel_hindi      count of Hindi word types (hindi_share 'H') the model used that are not in the agent script
                     or the role prompt.
 m1_model_hindi_share hindi_share.share() of the model text stream.
 m2_vf_clauses       heuristic: clauses (split on . , ? ! ;) of >= 3 tokens ending in a Hindi verb/auxiliary AND
                     containing a Hindi postposition; m2_vf_frac = that / clauses >= 3 tokens.
 m2_judge_r2 / m2_judge_vf   Gemma judge (judge.py) if run; else empty.
 m3_check_rate       per scripted write: a check-line (regex/fuzzy on the 5 phrasings) starting in the window
                     [end of last customer turn before the scripted check_line, start of first customer turn
                     after the confirm turn]; m3_check_sil_rate also needs >= 0.6 s (8 frames) of PAD/EPAD after
                     the check-line's last piece (piece level: check_silence_after).
 m4_read_rate        per scripted 'read' agent turn: share of its facts (digit-bearing tokens, and content words
                     >= 4 letters that occur in the role prompt's Information slot) found in the model text
                     (numbers normalised: spoken English/Hindi number words -> digits, fact = concatenation of a contiguous run of
                     whole model tokens).
 m4_value_rate       pooled share of the digit-bearing facts only (numbers, IDs, times, prices) found.
 m4_invented         count of numbers / month names in the model text not traceable to the role prompt or a
                     customer turn (n traceable if equal to a prompt/customer number, or, for >= 3 digits, a
                     substring of one); 'one' / 1 is exempt. m4_flag = m4_invented > 0.
 m5_greeting         greeting regex matched in model text before first customer turn start + 1.0 s.
 m5_max_3gram / m5_degenerate   max count of any word 3-gram; degenerate if > 4.
 m6_cer_raw / m6_cer_fold   Whisper(hi) transcript of the model wav vs the model's own text stream. Devanagari is
                     romanised with a crude table; raw = CER on lower-case letters+digits; fold = CER after
                     dropping vowels/h/y, merging aspirates and repeats. PROXY for intelligibility only.
"""
import csv
import difflib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[2]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_GEN_DIR') or str(_REPO / 'research/data_gen/gen'))
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
from tcommon import FRAME_RATE, INPUTS, OUT, jdump, jload, segments, test_meta_path, text_from_tokens  # noqa: E402

try:
    import hindi_share as HS
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"score.py needs packages/hinglish_text/hindi_share.py: {e}")

# ------------------------------------------------------------------ text helpers
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words(s):
    return WORD_RE.findall(s.lower().replace("’", "'"))


ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
ONES.update({"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "saat": 7,
             "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
             "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19})
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
TENS.update({"bees": 20, "tees": 30, "chalis": 40, "pachas": 50, "saath": 60})
MULT = {"hundred": 100, "sau": 100, "thousand": 1000, "hazaar": 1000, "hazar": 1000, "lakh": 100000}
AMBIG_NUM = {"do", "das", "char", "saath", "teen"}  # Hindi number words that are also common other words
MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec").split()


def normalise_numbers(text):
    """Return (normalised token list, list of number strings). Spoken numbers -> digit strings; 'five forty' ->
    '5','40' (adjacent, so alnum-compression gives '540'); 'five hundred and forty' -> '540'."""
    toks = words(text)
    out, nums = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit() or (t not in AMBIG_NUM and (t in ONES or t in TENS)) or (t in MULT and out and out[-1].isdigit()):
            chunk, kind, j = None, None, i
            while j < len(toks):
                w = toks[j]
                if w == "and" and chunk is not None and j + 1 < len(toks) and (toks[j + 1] in ONES or toks[j + 1] in TENS):
                    j += 1
                    continue
                if w.isdigit():
                    if chunk is not None:
                        out.append(str(chunk)); nums.append(str(chunk)); chunk = None
                    out.append(w); nums.append(w); kind = "digits"; j += 1
                    continue
                if w in ONES and (w not in AMBIG_NUM or chunk is not None):
                    v = ONES[w]
                    if chunk is None or kind in ("unit", "teen", "digits") or (kind == "tens" and v >= 10):
                        if chunk is not None:
                            out.append(str(chunk)); nums.append(str(chunk))
                        chunk, kind = v, ("teen" if v >= 10 else "unit")
                    else:
                        chunk, kind = chunk + v, ("teen" if v >= 10 else "unit")
                elif w in TENS:
                    v = TENS[w]
                    if chunk is None or kind in ("unit", "teen", "tens", "digits"):
                        if chunk is not None:
                            out.append(str(chunk)); nums.append(str(chunk))
                        chunk = v
                    else:
                        chunk += v
                    kind = "tens"
                elif w in MULT and (chunk is not None or (out and out[-1].isdigit())):
                    if chunk is None:
                        chunk = int(out.pop()); nums.pop()
                    if MULT[w] > chunk:
                        chunk *= MULT[w]
                    kind = "mult"
                else:
                    break
                j += 1
            if chunk is not None:
                out.append(str(chunk)); nums.append(str(chunk))
            i = max(j, i + 1)
            continue
        out.append(t)
        i += 1
    return out, nums


def compress(toks):
    return "".join(toks)


# ------------------------------------------------------------------ metric 3 patterns
CHECK_TEMPLATES = ["ek minute rukiye main check karke batati hoon", "ek minute rukiye main check karke batata hoon",
                   "theek hai ek second main dekh leti hoon", "theek hai ek second main dekh leta hoon",
                   "haan ji abhi update kar deti hoon ek minute", "haan ji abhi update kar deta hoon ek minute",
                   "sure let me check that for you bas ek second", "checking just a sec ek minute"]
CHECK_RE = re.compile(r"\b(ek (minute|min|second|sec|pal)|just a (sec|second|moment|minute)|one (sec|second|moment|"
                      r"minute)|let me (check|see|look|update|quickly)|check(ing)?\b.*\b(karke|kar)|dekh (leti|leta|"
                      r"lete|lu|loon)|rukiye|hold on|bas ek|update kar (deti|deta|dete)|give me a (sec|second|moment)|"
                      r"(i'll|i will) (check|look|update))\b")


def is_check_line(text):
    t = " ".join(words(text))
    if CHECK_RE.search(t):
        return True, 1.0
    best = max(difflib.SequenceMatcher(None, t, c).ratio() for c in CHECK_TEMPLATES)
    return best >= 0.6, round(best, 3)


def check_silence_after(tokens, s, max_more_words=7):
    """Silence (s) after the check-line inside segment s, measured at piece level (segments() merges pieces across
    gaps < 1.2 s, so the segment end is not the check-line end). Start at the first piece where the text so far
    reads as a check-line; walk on to the first sentence end or PAD/EPAD run >= 8 frames and return the gap after
    that piece. More than max_more_words further words without such a gap -> 0.0 (speech ran on)."""
    pi = [i for i in range(s["start_f"], len(tokens)) if tokens[i] not in ("PAD", "EPAD")]
    txt, m = "", None
    for n, i in enumerate(pi):
        if i > s["end_f"]:
            break
        txt += tokens[i]
        if is_check_line(txt)[0]:
            m = n
            break
    if m is None:
        return 0.0
    more = 0
    for n in range(m, len(pi)):
        nxt = pi[n + 1] if n + 1 < len(pi) else len(tokens)
        gap = nxt - pi[n] - 1
        if gap >= 8 or tokens[pi[n]].rstrip().endswith((".", "?", "!")):
            return gap / FRAME_RATE
        if n + 1 < len(pi) and tokens[nxt].startswith(" "):
            more += 1
            if more > max_more_words:
                return 0.0
    return 0.0


def has_fact(fact, toks, max_span=12):
    """fact matches the concatenation of a contiguous run of whole tokens (a phone/ID read in chunks still matches;
    '15' no longer matches inside '9800001523')."""
    for i in range(len(toks)):
        acc = ""
        for t in toks[i:i + max_span]:
            acc += t
            if acc == fact:
                return True
            if not fact.startswith(acc):
                break
    return False


GREET_RE = re.compile(r"\b(hello|hi|hey|namaste|namaskar|good (morning|afternoon|evening)|thank(s| you) for calling|"
                      r"welcome|this is|my name is|speaking)\b")

# ------------------------------------------------------------------ metric 2 heuristic
POSTP = {"ko", "se", "mein", "me", "liye", "wala", "wali", "wale", "ka", "ki", "ke", "par", "pe", "tak"}
VERB_END = {"hai", "hain", "hoon", "hun", "hu", "ho", "tha", "thi", "the", "hoga", "hogi", "honge", "raha", "rahi", "rahe",
            "gaya", "gayi", "gaye", "diya", "di", "diye", "liya", "li", "liye", "kiya", "kiye", "chahiye", "sakta",
            "sakti", "sakte", "dijiye", "kijiye", "batati", "batata", "bataiye", "karti", "karta", "karte", "deti",
            "deta", "dete", "leti", "leta", "lete", "jayega", "jaayega", "jayegi", "jaayegi", "milega", "milegi",
            "hua", "hui", "hue", "rukiye", "dekhiye", "boliye", "suniye", "karenge", "karega", "karegi", "denge"}
VERB_SUFFIX = re.compile(r"(ta|ti|te|unga|ungi|enge|ega|egi|iye|iyega|ogi|oge|aaya|aayi|aaye)$")


def vf_clauses(text):
    hindi, _ = HS._lexicons()
    n3, vf = 0, 0
    for cl in re.split(r"[.,?!;]", text.lower()):
        ws = words(cl)
        if len(ws) < 3:
            continue
        n3 += 1
        last = ws[-1]
        verb = last in VERB_END or (last in hindi and VERB_SUFFIX.search(last))
        if verb and any(w in POSTP for w in ws[:-1]):
            vf += 1
    return vf, n3


# ------------------------------------------------------------------ metric 6
_C = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
      "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
      "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
      "ष": "sh", "स": "s", "ह": "h", "ळ": "l"}
_NUK = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_V = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
      "औ": "au", "ऑ": "o", "ऍ": "e"}
_M = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
      "ॉ": "o", "ॅ": "e"}
_DD = str.maketrans("०१२३४५६७८९", "0123456789")


def deva_to_latin(s):
    s = unicodedata.normalize("NFD", s).translate(_DD)
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch in _C:
            nuk = i + 1 < len(s) and s[i + 1] == "़"
            out.append(_NUK.get(ch, _C[ch]) if nuk else _C[ch])
            i += 2 if nuk else 1
            nxt = s[i] if i < len(s) else ""
            if nxt in _M:
                out.append(_M[nxt]); i += 1
            elif nxt == "्":
                i += 1
            elif "ऀ" <= nxt <= "ॿ" and nxt not in "ंँः":
                out.append("a")
            elif nxt in "ंँः":
                out.append("a")
            # word-final: schwa deleted
            continue
        if ch in _V:
            out.append(_V[ch])
        elif ch in "ंँ":
            out.append("n")
        elif ch == "ः":
            out.append("h")
        elif ch in "़्":
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def raw_norm(s):
    return "".join(c for c in deva_to_latin(s).lower() if c.isalnum() and c.isascii())


def fold(s):
    s = raw_norm(s)
    for a, b in (("chh", "C"), ("ch", "C"), ("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("bh", "b"), ("jh", "j"), ("c", "k"), ("C", "c"), ("w", "v"), ("z", "j"), ("q", "k"),
                 ("x", "ks")):
        s = s.replace(a, b)
    s = re.sub(r"[aeiouhy]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def cer(ref, hyp):
    return (lev(ref, hyp) / len(ref)) if ref else (0.0 if not hyp else 1.0)


# ------------------------------------------------------------------ per-run scoring
def score_run(tokens, rmeta, tmeta, asr, judge):
    skipped = rmeta.get("skipped_steps", 0)
    tsec = lambda f: (f + skipped) / FRAME_RATE  # noqa: E731
    text = text_from_tokens(tokens)
    segs = segments(tokens)
    r = {"frames": len(tokens), "n_words": len(words(text))}
    # ---- m1
    excl = HS.build_exclude(tmeta.get("record"), [tmeta.get("agent_name") or "", tmeta.get("brand") or ""])
    agent_lines = [t["text_roman"] for t in tmeta["turns"] if t["speaker"] == "agent"]
    script_h, script_all = set(), set()
    for ln in agent_lines:
        for tok, lab in HS.classify_line(ln, excl):
            script_all.add(tok.lower())
            if lab == "H":
                script_h.add(tok.lower())
    model_set = set(words(text))
    model_h = {tok.lower() for tok, lab in HS.classify_line(text, excl) if lab == "H"} if text else set()
    r["m1_hindi_recall"] = round(len(script_h & model_set) / len(script_h), 4) if script_h else None
    r["m1_script_hindi_types"] = len(script_h)
    prompt_words = set(words(tmeta.get("role_prompt", "")))
    novel = model_h - script_all - prompt_words
    r["m1_novel_hindi"] = len(novel)
    r["m1_novel_hindi_list"] = sorted(novel)
    r["m1_model_hindi_share"] = HS.share([text], excl) if text else 0.0
    # ---- m2
    vf, n3 = vf_clauses(text)
    r["m2_vf_clauses"], r["m2_vf_frac"] = vf, (round(vf / n3, 4) if n3 else 0.0)
    if judge and "r2" in judge:
        r["m2_judge_r2"] = int(bool(judge["r2"]))
        r["m2_judge_vf"] = judge.get("hindi_verb_final_clauses")
        r["m2_judge_example"] = judge.get("example", "")
    # ---- m3
    turns = tmeta["turns"]
    writes = tmeta.get("writes") or []
    checks, checks_sil, w_detail = 0, 0, []
    for w in writes:
        ci = w.get("confirm_turn_idx")
        if ci is None or ci >= len(turns):
            continue
        chk = [t for t in turns[:ci] if "check_line" in t["tags"]]
        anchor = chk[-1]["idx"] if chk else ci
        prev_c = [t for t in turns[:anchor] if t["speaker"] == "customer"]
        w0 = prev_c[-1]["end"] if prev_c else 0.0
        nxt_c = [t for t in turns[ci + 1:] if t["speaker"] == "customer"]
        w1 = nxt_c[0]["start"] if nxt_c else 1e9
        found, sil_ok, best = False, False, None
        for k, s in enumerate(segs):
            if not (w0 - 0.3 <= tsec(s["start_f"]) <= w1):
                continue
            ok, sc = is_check_line(s["text"])
            if ok:
                gap = check_silence_after(tokens, s)
                found = True
                best = {"text": s["text"], "t": round(tsec(s["start_f"]), 2), "silence_after_s": round(gap, 2)}
                if gap >= 0.6:
                    sil_ok = True
                    break
        checks += found
        checks_sil += sil_ok
        w_detail.append({"tool": w.get("tool"), "window": [round(w0, 2), round(w1, 2)], "found": best})
    r["m3_n_writes"] = len(w_detail)
    r["m3_check_rate"] = round(checks / len(w_detail), 4) if w_detail else None
    r["m3_check_sil_rate"] = round(checks_sil / len(w_detail), 4) if w_detail else None
    r["m3_detail"] = w_detail
    # ---- m4
    prompt = tmeta.get("role_prompt", "")
    info = prompt.split("Information:", 1)[-1]
    info_toks, _ = normalise_numbers(info)
    info_set = set(info_toks)
    model_toks, model_nums = normalise_numbers(text)
    stop = {"your", "with", "from", "this", "that", "have", "order", "will", "been", "there", "they", "what", "please",
            "thank", "minutes", "minute", "hello", "sure", "okay", "about", "just", "status"}
    hindi, _ = HS._lexicons()
    reads, fact_hits, fact_n, read_detail = [], 0, 0, []
    val_hits, val_n = 0, 0  # digit-bearing facts only (values; generic words like "flight" excluded)
    for t in turns:
        if t["speaker"] != "agent" or "read" not in t["tags"]:
            continue
        toks, nums = normalise_numbers(t["text_roman"])
        facts = set()
        for tok in re.findall(r"[A-Za-z]*\d(?:\w|:\d)*", t["text_roman"]):  # 6:40 stays one fact
            ft, _ = normalise_numbers(tok)
            facts.add(compress(ft))
        for tok in toks:
            if len(tok) >= 4 and not tok.isdigit() and tok in info_set and tok not in stop and tok not in hindi:
                facts.add(tok)
        facts = {f for f in facts if f}
        if not facts:
            continue
        hit = [f for f in facts if has_fact(f, model_toks)]
        fact_hits += len(hit)
        fact_n += len(facts)
        val_n += sum(any(ch.isdigit() for ch in f) for f in facts)
        val_hits += sum(any(ch.isdigit() for ch in f) for f in hit)
        reads.append(len(hit) / len(facts))
        read_detail.append({"turn": t["idx"], "facts": sorted(facts), "hit": sorted(hit)})
    r["m4_n_reads"] = len(reads)
    r["m4_read_rate"] = round(sum(reads) / len(reads), 4) if reads else None
    r["m4_fact_rate"] = round(fact_hits / fact_n, 4) if fact_n else None
    r["m4_value_rate"] = round(val_hits / val_n, 4) if val_n else None
    r["m4_detail"] = read_detail
    allowed = set(re.findall(r"\d+", " ".join(normalise_numbers(prompt)[0])))
    for t in turns:
        if t["speaker"] == "customer":
            allowed |= set(re.findall(r"\d+", " ".join(normalise_numbers(t["text_roman"])[0])))
    # n traceable: exact match of a prompt/customer number; numbers of >= 3 digits may also be a substring
    # (phone/ID read in chunks); 1 is exempt ("one second").
    # group consecutive numeric tokens ("five forty" -> 5 40, a phone read digit by digit -> 9 8 7 ...);
    # a group is traceable if its joined digits are, or every member is.
    ok = lambda n: n == "1" or n in allowed or (len(n) >= 3 and any(n in a for a in allowed))  # noqa: E731
    groups_, cur = [], []
    for tok in model_toks + [""]:
        if tok.isdigit():
            cur.append(tok)
        elif cur:
            groups_.append(cur)
            cur = []
    invented = [" ".join(g) for g in groups_ if not ok("".join(g)) and not all(ok(n) for n in g)]
    prompt_low = prompt.lower() + " " + " ".join(t["text_roman"].lower() for t in turns if t["speaker"] == "customer")
    inv_months = [m for m in set(words(text)) if m in MONTHS and m != "may" and m not in prompt_low]
    r["m4_invented"] = len(invented) + len(inv_months)
    r["m4_flag"] = int(r["m4_invented"] > 0)
    r["m4_invented_list"] = invented + inv_months
    # ---- m5
    first_c = next((t for t in turns if t["speaker"] == "customer"), None)
    g_end = (first_c["start"] + 1.0) if first_c else 6.0
    early = " ".join(s["text"] for s in segs if tsec(s["start_f"]) <= g_end)
    r["m5_greeting"] = int(bool(GREET_RE.search(" ".join(words(early)))))
    r["m5_greeting_text"] = early[:200]
    ws = words(text)
    grams = Counter(zip(ws, ws[1:], ws[2:]))
    r["m5_max_3gram"] = max(grams.values()) if grams else 0
    r["m5_degenerate"] = int(r["m5_max_3gram"] > 4)
    # ---- m6
    if asr is not None:
        ref_raw, hyp_raw = raw_norm(text), raw_norm(asr.get("text", ""))
        r["m6_cer_raw"] = round(cer(ref_raw, hyp_raw), 4)
        r["m6_cer_fold"] = round(cer(fold(text), fold(asr.get("text", ""))), 4)
        r["m6_asr_text"] = asr.get("text", "")[:400]
    r["text"] = text
    return r


NUM_KEYS = ["n_words", "m1_hindi_recall", "m1_novel_hindi", "m1_model_hindi_share", "m2_vf_clauses", "m2_vf_frac",
            "m2_judge_r2", "m2_judge_vf", "m3_check_rate", "m3_check_sil_rate", "m4_read_rate", "m4_fact_rate", "m4_value_rate",
            "m4_invented", "m4_flag", "m5_greeting", "m5_max_3gram", "m5_degenerate", "m6_cer_raw", "m6_cer_fold"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows):
    """rows of one (tag, variant): per seed mean over calls, then mean/std over seeds."""
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    out = {}
    for k in NUM_KEYS:
        per_seed = [mean([r.get(k) for r in rs]) for rs in by_seed.values()]
        per_seed = [x for x in per_seed if x is not None]
        if not per_seed:
            continue
        m = sum(per_seed) / len(per_seed)
        sd = math.sqrt(sum((x - m) ** 2 for x in per_seed) / len(per_seed))
        out[k] = {"mean": round(m, 4), "std": round(sd, 4), "n_seeds": len(per_seed),
                  "n_calls": len({r["call_id"] for r in rows})}
    return out


def gate0_dump(tag_dir):
    lines = []
    for m in sorted((tag_dir / "gate0").glob("*.meta.json")):
        meta = jload(m)
        toks = jload(m.with_name(m.name.replace(".meta.json", ".json")))
        lines.append(f"--- {meta['name']}  seed {meta['seed']}  voice {meta['voice']}  {meta['duration_s']:.1f} s")
        lines.append(f"PROMPT: {meta['prompt']}")
        ev = {e["start_frame"]: e for e in meta.get("events", [])}
        segs = segments(toks)
        items = [(s["start_f"], f"MODEL  {s['text']}") for s in segs] + \
                [(f, f"USER   <clip {e['clip']}, {e['released_by']}>") for f, e in ev.items()]
        for f, txt in sorted(items):
            lines.append(f"  [{f / FRAME_RATE:6.2f} s]  {txt}")
        lines.append("")
    if lines:
        (tag_dir / "gate0_monologues.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    for tag in sys.argv[1:]:
        tdir = OUT / tag
        rows = []
        for m in sorted(tdir.rglob("*.meta.json")):
            rel = m.relative_to(tdir)
            if rel.parts[0] in ("gate0", "logs"):
                continue
            variant = rel.parts[0]
            rmeta = jload(m)
            stem = m.name[:-len(".meta.json")]
            call_id = stem.rsplit("_s", 1)[0]
            tm = test_meta_path(variant, call_id, m)  # = INPUTS/<V>/<call>.meta.json unless a vad-mode .tmeta.json
            if not tm.exists():
                print(f"[score] no test meta for {rel}", flush=True)
                continue
            tokens = jload(m.with_name(stem + ".json"))
            asr_p, j_p = m.with_name(stem + ".asr.json"), m.with_name(stem + ".judge.json")
            r = score_run(tokens, rmeta, jload(tm), jload(asr_p) if asr_p.exists() else None,
                          jload(j_p) if j_p.exists() else None)
            r = {"tag": tag, "variant": variant, "call_id": call_id, "seed": rmeta["seed"], **r}
            rows.append(r)
        with open(tdir / "runs.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cols = ["tag", "variant", "call_id", "seed"] + NUM_KEYS + ["m3_n_writes", "m4_n_reads"]
        with open(tdir / "runs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        agg = {}
        for v in sorted({r["variant"] for r in rows}):
            agg[v] = aggregate([r for r in rows if r["variant"] == v])
        jdump(tdir / "scores.json", {"tag": tag, "aggregation": "per seed: mean over calls; then mean and population "
                                     "std over seeds", "variants": agg})
        with open(tdir / "scores.csv", "w", newline="") as f:
            f.write("# per seed: mean over calls; then mean +- population std over seeds. m6 is a proxy.\n")
            w = csv.writer(f)
            w.writerow(["tag", "variant", "metric", "mean", "std", "n_seeds", "n_calls"])
            for v, d in agg.items():
                for k, s in d.items():
                    w.writerow([tag, v, k, s["mean"], s["std"], s["n_seeds"], s["n_calls"]])
        with open(tdir / "streams.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"--- {r['variant']}/{r['call_id']} seed {r['seed']}\n{r['text']}\n\n")
        gate0_dump(tdir)
        print(f"[score] {tag}: {len(rows)} runs -> {tdir}/scores.csv", flush=True)
        for v, d in agg.items():
            print(f"  {v}: " + ", ".join(f"{k}={s['mean']}±{s['std']}" for k, s in d.items()), flush=True)


if __name__ == "__main__":
    main()
