"""Spoken-form normalisation for Kokoro input, the CER reference, and MMS alignment text.

Stdlib only (imported from venv-tts, venv-asr and venv-pp).

One tokenizer, two output scripts:
  script="deva"  -> Devanagari spoken form (fed to Kokoro, and used as CER reference/hypothesis)
  script="roman" -> romanised spoken tokens [a-z'] (fed to MMS_FA)
Numbers are spoken as English number words (Hinglish call-centre style), written in Devanagari for
Kokoro and phonetically romanised for MMS:
  phones / IDs / plates / flight numbers / >=4-digit numbers / leading-zero numbers -> digit by digit
  1-3 digit numbers (incl. "1,250") -> cardinal ("five hundred forty")
  Rs/₹/INR/रु amounts -> cardinal + "rupaye"
  6:40 -> "six forty", 6:05 -> "six oh five", 6:00 -> "six"
  AM/PM, all-caps 2-4 letter tokens (UPI, DL, AB, PNR) and letters inside IDs -> letter names
  dd/mm/yyyy -> "eighteen October twenty twenty six"; year after a month name -> year words
The text fields (text_tts / text_roman) are never rewritten; this is applied right before use.
"""
import re
import unicodedata

# canonical -> (devanagari, roman)
W = {
    "zero": ("ज़ीरो", "zero"), "one": ("वन", "van"), "two": ("टू", "tu"), "three": ("थ्री", "thri"),
    "four": ("फ़ोर", "for"), "five": ("फ़ाइव", "faiv"), "six": ("सिक्स", "siks"), "seven": ("सेवन", "seven"),
    "eight": ("एट", "et"), "nine": ("नाइन", "nain"), "ten": ("टेन", "ten"), "eleven": ("इलेवन", "ilevan"),
    "twelve": ("ट्वेल्व", "twelv"), "thirteen": ("थर्टीन", "thartin"), "fourteen": ("फ़ोर्टीन", "fortin"),
    "fifteen": ("फ़िफ़्टीन", "fiftin"), "sixteen": ("सिक्सटीन", "sikstin"), "seventeen": ("सेवनटीन", "seventin"),
    "eighteen": ("एटीन", "etin"), "nineteen": ("नाइनटीन", "naintin"), "twenty": ("ट्वेंटी", "twenti"),
    "thirty": ("थर्टी", "tharti"), "forty": ("फ़ोर्टी", "forti"), "fifty": ("फ़िफ़्टी", "fifti"),
    "sixty": ("सिक्सटी", "siksti"), "seventy": ("सेवनटी", "seventi"), "eighty": ("एटी", "eti"),
    "ninety": ("नाइनटी", "nainti"), "hundred": ("हंड्रेड", "handred"), "thousand": ("थाउज़ेंड", "thauzend"),
    "lakh": ("लाख", "laakh"), "crore": ("करोड़", "karod"), "point": ("पॉइंट", "point"),
    "percent": ("परसेंट", "parsent"), "rupaye": ("रुपये", "rupaye"), "paise": ("पैसे", "paise"),
    "oh": ("ओ", "o"), "to": ("टू", "tu"), "plus": ("प्लस", "plas"),
    "first": ("फ़र्स्ट", "farst"), "second": ("सेकंड", "sekand"), "third": ("थर्ड", "thard"),
    "fifth": ("फ़िफ़्थ", "fifth"), "eighth": ("एट्थ", "eith"), "ninth": ("नाइंथ", "nainth"),
    "twelfth": ("ट्वेल्फ़्थ", "twelfth"), "twentieth": ("ट्वेंटिएथ", "twentieth"), "thirtieth": ("थर्टिएथ", "thartieth"),
    # review fix: every other word ordinal() can emit for 0-99 (was KeyError on "4th" -> plan/asr crash)
    "zeroth": ("ज़ीरोथ", "zeroth"), "fourth": ("फ़ोर्थ", "forth"), "sixth": ("सिक्स्थ", "siksth"),
    "seventh": ("सेवंथ", "seventh"), "tenth": ("टेंथ", "tenth"), "eleventh": ("इलेवंथ", "ilevanth"),
    "thirteenth": ("थर्टींथ", "thartinth"), "fourteenth": ("फ़ोर्टींथ", "fortinth"),
    "fifteenth": ("फ़िफ़्टींथ", "fiftinth"), "sixteenth": ("सिक्सटींथ", "sikstinth"),
    "seventeenth": ("सेवनटींथ", "seventinth"), "eighteenth": ("एटींथ", "etinth"), "nineteenth": ("नाइनटींथ", "naintinth"),
    "fortieth": ("फ़ोर्टिएथ", "fortieth"), "fiftieth": ("फ़िफ़्टिएथ", "fiftieth"), "sixtieth": ("सिक्सटिएथ", "sikstieth"),
    "seventieth": ("सेवनटिएथ", "seventieth"), "eightieth": ("एटिएथ", "etieth"), "ninetieth": ("नाइनटिएथ", "naintieth"),
    "at": ("ऐट", "aet"), "dot": ("डॉट", "dot"),  # emails
}
LETTERS = {
    "A": ("ए", "e"), "B": ("बी", "bi"), "C": ("सी", "si"), "D": ("डी", "di"), "E": ("ई", "i"), "F": ("एफ़", "ef"),
    "G": ("जी", "ji"), "H": ("एच", "ech"), "I": ("आई", "aai"), "J": ("जे", "je"), "K": ("के", "ke"), "L": ("एल", "el"),
    "M": ("एम", "em"), "N": ("एन", "en"), "O": ("ओ", "o"), "P": ("पी", "pi"), "Q": ("क्यू", "kyu"), "R": ("आर", "aar"),
    "S": ("एस", "es"), "T": ("टी", "ti"), "U": ("यू", "yu"), "V": ("वी", "vi"), "W": ("डब्ल्यू", "dablyu"),
    "X": ("एक्स", "eks"), "Y": ("वाई", "vaai"), "Z": ("ज़ेड", "zed"),
}
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
MONTHS_ROMAN = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "may": "may", "jun": "june",
                "jul": "july", "aug": "august", "sep": "september", "sept": "september", "oct": "october",
                "nov": "november", "dec": "december"}
MONTH_FULL = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
MONTH_DEVA = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
MONTH_DEVA_SET = set(MONTH_DEVA) | {"फरवरी", "सितम्बर", "अक्तूबर", "नवम्बर", "दिसम्बर", "ऑक्टोबर", "सेप्टेंबर",
                                    "नवेंबर", "डिसेंबर", "जैनुअरी", "फ़ेब्रुअरी", "अगस्ट", "अप्रिल", "जुलाय", "मे"}
CURRENCY = {"₹", "rs", "rs.", "inr", "रु", "रु.", "रू", "रू."}
RUPEE_WORDS = {"रुपये", "रुपए", "रुपया", "रुपयों", "रुपिये", "रूपये", "रूपए", "रुपय", "रूपया",  # review: Whisper spellings
               "rupaye", "rupees", "rupee", "rupay", "rupaiye", "rupye", "rupiya"}
AMPM = {"am", "pm", "a.m.", "p.m.", "a.m", "p.m"}
NO_SPELL = {"OK", "TV", "AC"}  # all-caps words left alone (OK) -- TV/AC still spelled? keep simple: spelled below
NO_SPELL = {"OK"}
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
PUNCT = ",.?!।॥;:\"'()[]{}…—–-‘’“”/"
LATIN_RE = re.compile(r"[A-Za-z]")
DEVA_RE = re.compile(r"[ऀ-ॿ]")


def cardinal(n):
    n = int(n)
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [ONES[n // 100], "hundred"] + (cardinal(n % 100) if n % 100 else [])
    if n < 100000:
        return cardinal(n // 1000) + ["thousand"] + (cardinal(n % 1000) if n % 1000 else [])
    if n < 10000000:
        return cardinal(n // 100000) + ["lakh"] + (cardinal(n % 100000) if n % 100000 else [])
    return cardinal(n // 10000000) + ["crore"] + (cardinal(n % 10000000) if n % 10000000 else [])


def ordinal(n):
    w = cardinal(n)
    last = w[-1]
    special = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
               "nine": "ninth", "twelve": "twelfth", "twenty": "twentieth", "thirty": "thirtieth"}
    w[-1] = special.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return w


def year_words(y):
    y = int(y)
    if 2000 <= y <= 2009:
        return cardinal(y)
    return cardinal(y // 100) + (["hundred"] if y % 100 == 0 else (["oh"] + cardinal(y % 100) if y % 100 < 10 else cardinal(y % 100)))


def digits(s):
    return [ONES[int(c)] for c in s if c.isdigit()]


def number_words(core):
    """1-3 digit (or comma-grouped) -> cardinal; >=4 digits or leading zero -> digit by digit."""
    plain = core.replace(",", "")
    if "," in core and plain.isdigit():
        return cardinal(plain)
    if len(plain) > 1 and plain.startswith("0"):
        return digits(plain)
    if len(plain) >= 4:
        return digits(plain)
    return cardinal(plain)


def spell_id(core):
    out = []
    for run in re.findall(r"[A-Za-z]+|\d+", core):
        if run.isdigit():
            out += cardinal(run) if (len(run) <= 2 and not (len(run) == 2 and run[0] == "0")) else digits(run)
        else:
            out += [("L", ch.upper()) for ch in run]
    return out


def amount_words(num):
    num = num.replace(",", "")
    if "." in num:
        r, p = num.split(".", 1)
        w = cardinal(r or "0") + ["rupaye"]
        if p.strip("0"):
            w += cardinal(p[:2].ljust(2, "0")) + ["paise"]
        return w
    return cardinal(num) + ["rupaye"]


def _split_punct(tok):
    i, j = 0, len(tok)
    while i < j and tok[i] in PUNCT:
        i += 1
    while j > i and tok[j - 1] in PUNCT:
        j -= 1
    # keep a trailing '.' that belongs to "Rs." / "p.m."
    return tok[:i], tok[i:j], tok[j:]


def _is_month(core):
    c = core.lower().rstrip(".")
    return c in MONTHS_ROMAN or c in MONTH_FULL or core in MONTH_DEVA_SET


def _is_num(core):
    return bool(re.fullmatch(r"\d[\d,]*(\.\d+)?", core))


def groups(text):
    """Split text into groups: dict(orig, lead, trail, words) where words is a list of canonical items:
    str (key of W), ("L", letter), ("RAW", token), ("MONTH", month_index)."""
    toks = text.translate(DEVA_DIGITS).split()
    parts = [_split_punct(t) for t in toks]
    out = []
    i = 0
    while i < len(parts):
        lead, core, trail = parts[i]
        nxt = parts[i + 1][1] if i + 1 < len(parts) else ""
        prev = parts[i - 1][1] if i > 0 else ""
        g = {"orig": toks[i], "lead": lead, "trail": trail, "words": []}
        lc = core.lower()
        lcp = (core + trail[:1]).lower() if trail[:1] == "." else lc
        m_attached = re.fullmatch(r"(₹|rs\.?|inr|रु\.?|रू\.?)(\d[\d,]*(?:\.\d+)?)", core, flags=re.I)
        m_time = re.fullmatch(r"(\d{1,2}):(\d{2})(am|pm)?", core, flags=re.I)
        m_date = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", core)
        m_ord = re.fullmatch(r"(\d{1,2})(st|nd|rd|th|थ|स्ट|न्ड|र्ड)", core, flags=re.I)  # review: + Devanagari suffix
        m_email = re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+", core)
        m_pct = re.fullmatch(r"(\d+(?:\.\d+)?)%", core)
        m_range = re.fullmatch(r"(\d{1,3})-(\d{1,3})", core)
        if not core:
            g["words"] = []
        elif m_email:
            # review fix: "bhatia.home57@gmail.com" -> bhatia dot home fifty seven at gmail dot com (one group;
            # letter runs stay as written -- Kokoro reads Latin names acceptably)
            g["words"] = []
            for run in re.findall(r"[A-Za-z]+|\d+|[@.]", core):
                g["words"] += (["at"] if run == "@" else ["dot"] if run == "." else
                               number_words(run) if run.isdigit() else [("RAW", run)])
        elif m_attached:
            g["words"] = amount_words(m_attached.group(2))
            if nxt.lower() in RUPEE_WORDS:
                g["words"] = g["words"][:-1] if "paise" not in g["words"] else g["words"]
        elif (lc in CURRENCY or lcp in CURRENCY) and _is_num(nxt):
            # "Rs 540" -> one group spoken "five hundred forty rupaye"
            lead2, core2, trail2 = parts[i + 1]
            g = {"orig": toks[i] + " " + toks[i + 1], "lead": lead, "trail": trail2,
                 "words": amount_words(core2)}
            i += 1
        elif m_time:
            h, mm = int(m_time.group(1)), int(m_time.group(2))
            w = cardinal(h) + ([] if mm == 0 else (["oh"] + cardinal(mm) if mm < 10 else cardinal(mm)))
            if m_time.group(3):
                w += [("L", c) for c in m_time.group(3).upper()]
            g["words"] = w
        elif m_date:
            d, mo, y = int(m_date.group(1)), int(m_date.group(2)), m_date.group(3)
            if 1 <= mo <= 12:
                y = int(y) + (2000 if len(y) == 2 else 0)
                g["words"] = cardinal(d) + [("MONTH", mo - 1)] + year_words(y)
            else:
                g["words"] = digits(core)
        elif m_ord:
            g["words"] = ordinal(m_ord.group(1))
        elif m_pct:
            a = m_pct.group(1)
            g["words"] = (cardinal(a.split(".")[0]) + ["point"] + digits(a.split(".")[1]) if "." in a
                          else cardinal(a)) + ["percent"]
        elif m_range:
            g["words"] = cardinal(m_range.group(1)) + ["to"] + cardinal(m_range.group(2))
        elif _is_num(core):
            if nxt.lower() in RUPEE_WORDS:
                g["words"] = amount_words(core)[:-1] if "." not in core else amount_words(core)
            elif "." in core:
                a, b = core.replace(",", "").split(".")
                g["words"] = cardinal(a) + ["point"] + digits(b)
            elif len(core) == 4 and core[:2] in ("19", "20") and _is_month(prev):
                g["words"] = year_words(core)
            else:
                g["words"] = number_words(core)
        elif lc.replace(".", "") in {"am", "pm"} and (_is_num(prev) or re.fullmatch(r"\d{1,2}:\d{2}", prev)):
            g["words"] = [("L", c) for c in lc.replace(".", "").upper()]
        elif re.fullmatch(r"\+\d+", core):
            g["words"] = ["plus"] + digits(core)
        elif re.fullmatch(r"[xX]\d{1,2}", core):
            g["words"] = cardinal(core[1:])  # review fix: quantity "x2" is spoken "two" (text_tts has "2")
        elif re.fullmatch(r"[A-Za-z0-9-]+", core) and re.search(r"\d", core) and re.search(r"[A-Za-z]", core):
            g["words"] = spell_id(core)
        elif re.search(r"\d", core) and DEVA_RE.search(core):
            # review fix: Devanagari letters + digits ("एच-12", "टी3", "एम35") -- digits were reaching Kokoro raw
            # (read as Hindi numbers). Same digit rule as spell_id, so it matches roman "H-12" / "T3".
            g["words"] = []
            for run in re.findall(r"\d+|[^\d-]+", core):
                g["words"] += (spell_id(run) if run.isdigit() else [("RAW", run)])
        elif re.fullmatch(r"[A-Z]{2,4}", core) and core not in NO_SPELL:
            g["words"] = [("L", c) for c in core]
        elif re.fullmatch(r"[A-Za-z]{3,9}\.?", core) and lc.rstrip(".") in MONTHS_ROMAN and LATIN_RE.search(core):
            g["words"] = [("MONTH", MONTH_FULL.index(MONTHS_ROMAN[lc.rstrip(".")]))]
        else:
            g["words"] = [("RAW", core)]
        out.append(g)
        i += 1
    return out


def _render(item, script):
    k = 0 if script == "deva" else 1
    if isinstance(item, str):
        return W[item][k]
    kind, val = item
    if kind == "L":
        return LETTERS[val][k]
    if kind == "MONTH":
        return MONTH_DEVA[val] if script == "deva" else MONTH_FULL[val]
    # RAW
    if script == "deva":
        return val
    r = re.sub(r"[^a-z']", "", val.lower())
    return r


def tts_text(text):
    """Devanagari spoken form for Kokoro (punctuation kept for prosody)."""
    out = []
    for g in groups(text):
        body = " ".join(x for x in (_render(w, "deva") for w in g["words"]) if x)
        if body or g["lead"] or g["trail"]:
            out.append(g["lead"] + body + g["trail"])
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def roman_words(text):
    """[(orig_word, [roman spoken tokens])] for MMS alignment; groups with no tokens are merged into the
    previous group (or next, at the start), so every returned entry has >= 1 token."""
    res = []
    pending = ""
    for g in groups(text):
        toks = []
        for w in g["words"]:
            r = _render(w, "roman")
            toks += [t for t in r.split() if t]
        if not toks:
            if res:
                res[-1] = (res[-1][0] + " " + g["orig"], res[-1][1])
            else:
                pending = (pending + " " + g["orig"]).strip()
            continue
        orig = (pending + " " + g["orig"]).strip() if pending else g["orig"]
        pending = ""
        res.append((orig, toks))
    if pending and res:
        res[-1] = (res[-1][0] + " " + pending, res[-1][1])
    return res


def cer_norm(text):
    """Text -> comparison string: spoken form, NFD nukta removed, chandrabindu->anusvara, no spaces/punct."""
    t = tts_text(text)
    t = unicodedata.normalize("NFD", t).replace("़", "")
    t = unicodedata.normalize("NFC", t).replace("ँ", "ं").replace("‍", "").replace("‌", "")
    t = t.lower()
    return "".join(c for c in t if ("ऀ" <= c <= "ॿ" and c not in "।॥") or c.isalnum())


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    r, h = cer_norm(ref), cer_norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)



# ---- script-independent comparison (Whisper often writes English words in Latin script) ----
_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
         "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
         "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
         "ष": "sh", "स": "s", "ह": "h", "ळ": "l"}
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
        "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ॉ": "o", "ॅ": "e"}


def deva_to_latin(text):
    t = unicodedata.normalize("NFD", text)
    out, i = [], 0
    while i < len(t):
        c = t[i]
        if c in _CONS:
            nuk = i + 1 < len(t) and t[i + 1] == "़"
            out.append(_NUKTA.get(c, _CONS[c]) if nuk else _CONS[c])
            i += 2 if nuk else 1
            nx = t[i] if i < len(t) else ""
            if nx in _MATRA:
                out.append(_MATRA[nx])
                i += 1
            elif nx == "्":
                i += 1
            elif nx and ("ऀ" <= nx <= "ॿ") and nx not in "।॥":
                out.append("a")
            # word-final schwa dropped
            continue
        if c in _VOW:
            out.append(_VOW[c])
        elif c in "ंँ":
            out.append("n")
        elif c == "ः":
            out.append("h")
        elif c == "़" or c == "्":
            pass
        else:
            out.append(c)
        i += 1
    return "".join(out)


def latin_fold(text):
    """Spoken form -> rough phonetic Latin key (aspiration, vowel length, doubled letters folded)."""
    t = deva_to_latin(tts_text(text)).lower()
    t = re.sub(r"[^a-z]", "", t)
    t = t.replace("chh", "C").replace("ch", "C").replace("sh", "S").replace("ph", "f").replace("c", "k")
    t = t.replace("C", "c").replace("S", "s")
    t = re.sub(r"([kgtdpbj])h", r"\1", t)
    for a, b in (("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"), ("ee", "i"), ("oo", "u"), ("ai", "e"),
                 ("au", "o"), ("aa", "a"), ("y", "i")):
        t = t.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", t)


def cer_latin(ref, hyp):
    r, h = latin_fold(ref), latin_fold(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def clean_hyp(hyp):
    """Whisper output quirks before normalisation: 6.40 PM -> 6:40 PM; digit groups joined (98765-43210)."""
    h = re.sub(r"\b(\d{1,2})\.(\d{2})(?=\s*(?:[AaPp]\.?[Mm]|पीएम|एएम|पी एम|ए एम))", r"\1:\2", hyp)
    return re.sub(r"(?<=\d)[\s-]+(?=\d)", "", h)


def cer_best(ref, hyp):
    """min(Devanagari CER, folded-Latin CER): Whisper's script choice for English words must not cause rejects."""
    hyp = clean_hyp(hyp)
    return min(cer(ref, hyp), cer_latin(ref, hyp))

if __name__ == "__main__":
    tests = [
        "आपका ऑर्डर A1234 है, अमाउंट Rs 540 और ETA 15 मिनट।",
        "फ़ोन नंबर 9876543210 पे कॉल करें, फ़्लाइट 6E 2134 शाम 6:40 PM पे है।",
        "गाड़ी का नंबर DL 01 AB 4821 है, सीट 18A, रिन्यूअल 18 अक्टूबर 2026 को।",
        "₹1,250 का रिफ़ंड, डेट 12/10/2026, 540 रुपये, 6:05 बजे, 2.5 किलो, 10% ऑफ़",
        "Your order A1234 of Rs 540 is out, call 98765 43210, flight AI-302 at 6:40 PM on 18 Oct, UPI se.",
    ]
    for t in tests:
        print(t)
        print("  TTS :", tts_text(t))
        print("  ROM :", roman_words(t))
        print("  CER :", cer_norm(t)[:80])
