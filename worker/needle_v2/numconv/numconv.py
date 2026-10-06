"""S2S serverless worker: verbatim copy of needle_v2/numconv/numconv.py.
numconv: spoken numbers in Trelis Whisper-Hinglish output -> digits.  Pure Python, no deps.

    from numconv import convert
    convert("number nine eight seven three zero seven five eight nine three hai")
        -> "number 9873075893 hai"

Handles (N2, 2026-10-05):
  * English number words (Latin script): zero..nineteen, twenty..ninety, hundred/thousand/lakh/crore/million,
    "double X" / "triple X", "oh"/"o" as zero between digits, "and" inside a compositional number, "point".
  * The same English words as Trelis writes them in Devanagari (टू थ्री त्री एट वन फोर नाइन सेवन सिक्स ट्वेंटी ...).
  * Hindi number words 0-99 + sau/hazaar/lakh/crore, Devanagari and romanised spellings (with common variants),
    dedh/dhai/saadhe/sawa before a multiplier.
  * Digits already present (Latin or Devanagari digits), comma groups (1,85,000), mixed runs ("AC8 nine eight zero").
  * Composition: a run is split into compositional groups (twenty one = 21, eight hundred thirty eight = 838,
    one lakh twenty two thousand four hundred ten = 122410); groups are concatenated (one eight zero two = 1802;
    phones digit by digit = 10 contiguous digits). Adjacent digit tokens totalling 10 digits (98730 75893, or a
    phone split by a comma) are joined.
  * ID prefixes: Latin capitals (FD, "F D", TX, gate "D"), glued forms (IDAC -> ID AC, PNRWT -> PNR WT,
    MW-3730 -> MW3730), Devanagari letter names (एफडी, आर डी, पीएनआरएफएन -> PNR FN, के Z -> KZ) directly before a
    number are joined onto it: "FD four one two four" -> "FD4124".
  * Times: "seven thirty pm" -> "7:30 pm".  Seat/flat suffix letter: "seat fourteen B" -> "seat 14B".
  * Narrow email fix: "eight" right before a mail-domain word -> "at" ("forty two eight yahoo" -> "42 at yahoo").

Ambiguity rules (Hindi ek/do/teen/char/saat/nau/das are common words; English "one" is a pronoun):
  * a run with >= 2 number tokens converts, EXCEPT a bare approximate pair of small Hindi words (do teen,
    teen chaar, ek do, das bees ...), which stays as words (the references keep them as words);
  * a single English (Latin) number word converts, except "one";
  * a single "one", Hindi word or Devanagari-English word converts only with context: a unit/currency after it
    (rupees, minutes, GB, %, pm, baje ...), a context word before it (Rs, flat, sector, gate, terminal, seat, ...)
    or an ID prefix before it;
  * "ek/one minute", "ek/one second" (check-line phrases) never convert;
  * "ek saath" (together) never converts.
"""
import re
import unicodedata

__all__ = ["convert", "extract_numbers"]

# ---------------------------------------------------------------- lexicons
EN_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9}
EN_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
           "eighty": 80, "ninety": 90}
EN_MULT = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
           "crore": 10 ** 7, "crores": 10 ** 7, "million": 10 ** 6}

# English number words as Trelis writes them in Devanagari (keys are fold()-ed, see below)
DEVA_EN = {
    "ज़ीरो": 0, "जीरो": 0, "ज़ेरो": 0, "जेरो": 0, "वन": 1, "टू": 2, "थ्री": 3, "त्री": 3, "थ्रि": 3, "फोर": 4,
    "फ़ोर": 4, "फाइव": 5, "फ़ाइव": 5, "फाईव": 5, "सिक्स": 6, "सेवन": 7, "सेवेन": 7, "एट": 8, "एइट": 8, "नाइन": 9,
    "नाईन": 9, "टेन": 10, "इलेवन": 11, "ट्वेल्व": 12, "थर्टीन": 13, "फोर्टीन": 14, "फिफ्टीन": 15, "सिक्सटीन": 16,
    "सेवनटीन": 17, "एटीन": 18, "नाइनटीन": 19, "ट्वेंटी": 20, "ट्वेन्टी": 20, "थर्टी": 30, "फोर्टी": 40,
    "फ़ोर्टी": 40, "फिफ्टी": 50, "फ़िफ़्टी": 50, "सिक्सटी": 60, "सेवेंटी": 70, "सेवनटी": 70, "एटी": 80, "नाइंटी": 90,
    "नाइनटी": 90, "हंड्रेड": 100, "हन्ड्रेड": 100, "थाउज़ेंड": 1000, "थाउजेंड": 1000, "थाउसेंड": 1000,
    "डबल": "double", "ट्रिपल": "triple",
}
# fused Trelis forms seen in V4 ("नाइनतीन" = nineteen)
DEVA_EN.update({"नाइनतीन": 19, "तेन": 10})

HI_ROMAN = {
    0: "shunya shoonya sunya", 1: "ek", 2: "do", 3: "teen tin", 4: "char chaar chaar", 5: "paanch panch paach panj",
    6: "chhe chhah chah chhah chhey chheh chhai", 7: "saat", 8: "aath aat", 9: "nau naw", 10: "das dus dass",
    11: "gyarah gyara gyaarah gyaraah igyarah", 12: "barah baarah bara baara", 13: "terah tera",
    14: "chaudah chaudha chauda choudah", 15: "pandrah pandra pandhrah pondrah", 16: "solah sola",
    17: "satrah satra sattrah", 18: "atharah athara attharah atharaah", 19: "unnis unees unnees unis",
    20: "bees bis", 21: "ikkis ikkees", 22: "baais bais baees", 23: "teis teyis teish teees tees_",
    24: "chaubis chaubees chobis", 25: "pachchis pachees pacchis pachis pachhis", 26: "chhabbis chabbees chhabis chabbis",
    27: "sattais sattaees sataees sattaais", 28: "atthais athais atthaees athaees", 29: "untees untis unatis",
    30: "tees tis", 31: "ikattis iktees ikattees iktis", 32: "battis battees", 33: "taintis tetees taitees tentis",
    34: "chautis chauntees chontis chautees", 35: "paintis paintees pentis", 36: "chhattis chattees chattis chhattees",
    37: "saintis saintees", 38: "adtis artees adtees arthees", 39: "untalis unchalis unchaalees untaalees",
    40: "chalis chaalees chaalis chalees", 41: "iktalis iktaalees iktalees", 42: "bayalis bayaalees bayalees",
    43: "tentalis taitaalees taitalis", 44: "chavalis chauvaalees chauvalis", 45: "paintalis paintaalees pentalis",
    46: "chhiyalis chhiyaalees chiyalis", 47: "saintalis saintaalees", 48: "adtalis artaalees adtaalees",
    49: "unchas unchaas", 50: "pachas pachaas pachaas", 51: "ikyavan ikyaavan ikyawan", 52: "bavan baavan bawan",
    53: "tirpan tirepan", 54: "chauvan chauvan chouvan", 55: "pachpan", 56: "chhappan chappan",
    57: "sattavan sattaavan", 58: "atthavan atthaavan athavan", 59: "unsath unsaath", 60: "saath sath",
    61: "iksath iksaath", 62: "basath baasath", 63: "tirsath tiresath", 64: "chausath chaunsath",
    65: "painsath pensath", 66: "chhiyasath chhiyaasath chiyasath", 67: "sadsath sarsath", 68: "adsath arsath",
    69: "unhattar", 70: "sattar", 71: "ikhattar ikahattar", 72: "bahattar", 73: "tihattar", 74: "chauhattar",
    75: "pachhattar pachattar", 76: "chhihattar chihattar", 77: "sathattar satattar sathhattar",
    78: "athhattar athattar", 79: "unasi unaasi unyasi", 80: "assi assee", 81: "ikyasi ikyaasi", 82: "bayasi bayaasi",
    83: "tirasi tiraasi", 84: "chaurasi chauraasi", 85: "pachasi pachaasi", 86: "chhiyasi chhiyaasi",
    87: "sattasi sataasi sattaasi", 88: "athasi atthaasi atthasi", 89: "navasi nawaasi navaasi",
    90: "nabbe nabbe nabe", 91: "ikyanve ikyaanve", 92: "banve baanve", 93: "tiranve tiraanve", 94: "chauranve",
    95: "pachanve pachaanve", 96: "chhiyanve chhiyaanve", 97: "sattanve sattaanve", 98: "atthanve atthaanve anthanve",
    99: "ninyanve ninyaanve",
}
HI_DEVA = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच पाँच पाच", 6: "छह छः छे छै छेह", 7: "सात", 8: "आठ", 9: "नौ",
    10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह पन्द्रह", 16: "सोलह", 17: "सत्रह सतरह",
    18: "अठारह अट्ठारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस अठाईस", 29: "उनतीस उन्तीस", 30: "तीस", 31: "इकतीस इकत्तीस",
    32: "बत्तीस", 33: "तैंतीस तेंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस",
    39: "उनतालीस उन्तालीस", 40: "चालीस", 41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस चौवालीस",
    45: "पैंतालीस", 46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास", 51: "इक्यावन",
    52: "बावन", 53: "तिरपन तिरेपन", 54: "चौवन", 55: "पचपन", 56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
    60: "साठ", 61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ",
    69: "उनहत्तर", 70: "सत्तर", 71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी उनासी", 80: "अस्सी", 81: "इक्यासी", 82: "बयासी",
    83: "तिरासी", 84: "चौरासी", 85: "पचासी", 86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे", 96: "छियानवे", 97: "सत्तानवे",
    98: "अट्ठानवे", 99: "निन्यानवे",
}
HI_MULT_ROMAN = {"sau": 100, "so_": 100, "hazaar": 1000, "hazar": 1000, "hajar": 1000, "hajaar": 1000,
                 "lakh": 100000, "lakh_": 100000, "crore": 10 ** 7, "karod": 10 ** 7, "karor": 10 ** 7}
HI_MULT_DEVA = {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000, "करोड़": 10 ** 7, "करोड": 10 ** 7}
# fractional prefixes, only valid directly before a multiplier (dedh sau = 150) or (saadhe/sawa) before N + mult
HI_FRAC_FIXED = {"dedh": 1.5, "derh": 1.5, "dedh_": 1.5, "dhai": 2.5, "dhaai": 2.5, "ढाई": 2.5, "डेढ़": 1.5, "डेढ": 1.5}
HI_FRAC_ADD = {"saadhe": 0.5, "sadhe": 0.5, "saade": 0.5, "साढ़े": 0.5, "साढे": 0.5, "sawa": 0.25, "sava": 0.25,
               "सवा": 0.25}


def fold(w):
    """Lookup key: NFC, lowercase, nukta stripped, chandrabindu -> anusvara."""
    w = unicodedata.normalize("NFD", w.lower())
    w = w.replace("़", "").replace("ँ", "ं")
    return unicodedata.normalize("NFC", w)


# kind: 'u' unit 0-9, 't' teen 10-19 (English), 'x' tens 20..90 (English), 'h' Hindi 0-99 (complete), 'm' multiplier
# lang: 'en' Latin English, 'de' Devanagari English, 'hi' Hindi (Devanagari or roman), 'hr' Hindi roman
LEX = {}


def _add(word, val, lang):
    if isinstance(val, str):
        LEX[fold(word)] = (val, val, lang)
        return
    if val >= 100:
        kind = "m"
    elif lang in ("hi", "hr"):
        kind = "u" if val < 10 else "h"
    elif val < 10:
        kind = "u"
    elif val < 20:
        kind = "t"
    else:
        kind = "x"
    LEX[fold(word)] = (kind, val, lang)


for _w, _v in EN_UNITS.items():
    _add(_w, _v, "en")
for _w, _v in EN_TEENS.items():
    _add(_w, _v, "en")
for _w, _v in EN_TENS.items():
    _add(_w, _v, "en")
for _w, _v in EN_MULT.items():
    _add(_w, _v, "en")
_add("double", "double", "en")
_add("triple", "triple", "en")
for _w, _v in DEVA_EN.items():
    _add(_w, _v, "de")
for _v, _ws in HI_ROMAN.items():
    for _w in _ws.split():
        if not _w.endswith("_"):
            _add(_w, _v, "hr")
for _v, _ws in HI_DEVA.items():
    for _w in _ws.split():
        _add(_w, _v, "hi")
for _w, _v in HI_MULT_ROMAN.items():
    if not _w.endswith("_"):
        LEX[fold(_w)] = ("m", _v, "hr")
for _w, _v in HI_MULT_DEVA.items():
    LEX[fold(_w)] = ("m", _v, "hi")
# English 'lakh'/'crore' are shared with Hindi; keep them English-kind (always numeric, never ambiguous)
for _w in ("lakh", "lakhs", "lac", "lacs", "crore", "crores"):
    LEX[_w] = ("m", EN_MULT[_w], "en")
FRAC_FIXED = {fold(k): v for k, v in HI_FRAC_FIXED.items() if not k.endswith("_")}
FRAC_ADD = {fold(k): v for k, v in HI_FRAC_ADD.items()}

# Hindi words that are common non-number words in roman script: a Latin "do"/"no"/"sat" is English; these roman
# spellings count only next to another roman-Hindi number token.
ROMAN_WEAK = {"do", "tin", "char", "sath", "saath", "das", "bees", "tees", "bis", "tis", "sola", "bara", "tera",
              "aat", "sau"}

# ---------------------------------------------------------------- context
UNIT_AFTER = {fold(w) for w in (
    "rs rs. rupees rupee rupaye rupaye rupay rupaiya rupiya rupya inr ₹ रुपये रुपए रुपया रुपय रूपये रूपए रुपैया "
    "minute minutes min mins मिनट second seconds sec secs सेकंड सेकेंड hour hours hr hrs x "
    "gb mb kb जीबी एमबी % percent percentage परसेंट प्रतिशत pm am p.m. a.m. बजे baje baj "
    "km kms kilometer kilometers किलोमीटर kg kgs किलो kilo paisa paise पैसे points"
).split()}
CONTEXT_BEFORE = {fold(w) for w in (
    "rs rs. ₹ inr flat sector gate terminal seat row platform floor block tower plot house room coach berth "
    "ending pin pincode otp extension ext version no. flight cvv phase part stage level wing"
).split()}
# Hindi/Devanagari single words with these after them are still not converted (approximate / idiomatic)
NO_SINGLE_UNIT_FOR_HINDI = set()
CHECK_LINE_AFTER = {fold(w) for w in "minute min second sec मिनट सेकंड सेकेंड".split()}
MAIL_DOMAINS = {"gmail", "yahoo", "hotmail", "outlook", "icloud", "rediffmail", "rediff", "protonmail", "proton",
                "live", "ymail", "zoho", "aol", "msn", "googlemail", "jio", "jiomail"}

# mail-domain-like tokens as Trelis writes them (zohomail -> जोहोमेल / jihomel, gmail -> जीमेल ...)
_MAIL_LIKE_RE = re.compile(r"(mail|mel$|mil$|^zoh|^jih|मेल|मिल$|^जोह|^जीह|याहू|आउटलुक|क्लाउड|हॉटम|हाटम|हॉट)")


def mail_like(key):
    return bool(key) and (key in MAIL_DOMAINS or bool(_MAIL_LIKE_RE.search(key)))


# Devanagari letter names (for ID prefixes); longest match first in decomposition
DEVA_LETTERS = {
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H", "आई": "I", "जे": "J",
    "के": "K", "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P", "क्यू": "Q", "आर": "R", "एस": "S", "टी": "T",
    "यू": "U", "वी": "V", "डब्ल्यू": "W", "डबल्यू": "W", "एक्स": "X", "वाई": "Y", "जेड": "Z", "ज़ेड": "Z", "आडी": "RD",
}
DEVA_LETTERS = {fold(k): v for k, v in DEVA_LETTERS.items()}
_DL_KEYS = sorted(DEVA_LETTERS, key=len, reverse=True)
# single-letter Devanagari tokens that are everyday words: never an ID prefix on their own
DEVA_LETTER_WORDS = {fold(w) for w in "जी के ए ओ ई सी पी आई टी".split()}
LATIN_PREFIX_STOP = {"I", "ID", "PNR", "OTP", "GB", "MB", "KB", "AM", "PM", "OK", "CVV", "PIN", "UPI", "SIM", "ETA",
                     "EMI", "ATM", "KYC", "IFSC", "NEFT", "IMPS", "RTGS", "INR", "RS", "SMS", "DM", "TV", "AC_"}
# single English words that Trelis also writes for to/too, for, at, one (pronoun): need context to convert alone
EN_HOMOPHONE = {"one", "two", "four", "eight"}
TIME_AFTER = {fold(w) for w in "am pm a.m. p.m. aim m बजे baje एएम पीएम".split()}
# Devanagari/roman Hindi number words that are everyday words (ek = a, do = give, saat ~ saath, nau ~ now ...)
HI_WEAK = {fold(w) for w in "एक दो तीन चार सात नौ दस ek do teen tin char chaar saat nau das".split()}
APPROX_NEXT = {fold(w) for w in (
    "din dino days day minute minutes mins ghante ghanta ghanton hours hour hafte hafta weeks week mahine mahina "
    "months month saal years baar times log logon people दिन दिनों मिनट घंटे घंटा हफ्ते महीने साल बार लोग"
).split()}
SUFFIX_CTX = {fold(w) for w in "seat flat gate terminal row block tower platform berth coach house plot".split()}

PUNCT = ".,!?;:\"'()[]{}|।…“”‘’"
DIGIT_TRANS = str.maketrans("०१२३४५६७८९", "0123456789")
DEVA = re.compile(r"[ऀ-ॿ]")


def deva_letters(w):
    """Decompose a Devanagari token fully into letter names -> 'RD', else None."""
    k = fold(w)
    out, i = "", 0
    while i < len(k):
        for key in _DL_KEYS:
            if k.startswith(key, i):
                out += DEVA_LETTERS[key]
                i += len(key)
                break
        else:
            return None
    return out or None


# ---------------------------------------------------------------- tokenisation
class Tok:
    __slots__ = ("lead", "core", "trail", "key", "kind", "val", "lang", "digits", "glue_prefix", "out")

    def __init__(self, raw):
        core = raw.strip(PUNCT)
        if not core:
            self.lead, self.core, self.trail = raw, "", ""
        else:
            i = raw.find(core)
            self.lead, self.core, self.trail = raw[:i], core, raw[i + len(core):]
        self.key = fold(self.core)
        self.kind = self.val = self.lang = None
        self.digits = None        # digit string for digit tokens
        self.glue_prefix = None   # letters glued in front of digits (AC8 -> 'AC')
        self.out = None
        if not self.core:
            return
        c = self.core.translate(DIGIT_TRANS)
        m = re.fullmatch(r"(\d{1,3}(?:,\d{2,3})+|\d+)", c)
        if m:
            self.kind, self.lang, self.digits = "d", "dg", c.replace(",", "")
            return
        m = re.fullmatch(r"([A-Za-z]{1,4})-?(\d+)", c)
        if m and m.group(1).isupper():
            self.kind, self.lang, self.digits, self.glue_prefix = "d", "dg", m.group(2), m.group(1)
            return
        if self.key in LEX:
            self.kind, self.val, self.lang = LEX[self.key]
        elif self.key in FRAC_FIXED:
            self.kind, self.val, self.lang = "ff", FRAC_FIXED[self.key], "hi"
        elif self.key in FRAC_ADD:
            self.kind, self.val, self.lang = "fa", FRAC_ADD[self.key], "hi"
        elif self.key in ("oh", "o"):
            self.kind, self.val, self.lang = "oh", 0, "en"
        elif self.key == "and":
            self.kind, self.lang = "and", "en"
        elif self.key == "point":
            self.kind, self.lang = "point", "en"

    @property
    def raw(self):
        return self.lead + self.core + self.trail

    def is_num(self):
        return self.kind in ("u", "t", "x", "h", "m", "d", "double", "triple", "ff", "fa")


def _split_hyphens(text):
    """Split hyphenated tokens when a part is a number word or digits (twenty-five, NW-एट-सिक्स, do-teen)."""
    out = []
    for raw in text.split():
        if "-" in raw.strip("-"):
            parts = raw.split("-")
            if any(fold(p.strip(PUNCT)) in LEX or p.strip(PUNCT).translate(DIGIT_TRANS).isdigit() for p in parts) \
                    and all(parts) and not re.fullmatch(r"[A-Z]{1,4}-\d+[^\w]*", raw):
                out.extend(parts)
                continue
        out.append(raw)
    return out


_DE_KEYS = sorted((k for k, v in LEX.items() if v[2] == "de" and isinstance(v[1], int)), key=len, reverse=True)


def _split_glued_id(toks):
    """IDAC -> ID AC, PNRWT -> PNR WT (Trelis glues the label onto the prefix); एफडीएट -> एफडी एट."""
    out = []
    for t in toks:
        if DEVA.search(t.core) and t.key not in LEX and not t.lead:
            hit = next((k for k in _DE_KEYS if t.key.endswith(k) and len(t.key) > len(k)
                        and deva_letters(t.key[: -len(k)])), None)
            if hit:
                out.append(Tok(t.key[: -len(hit)]))
                out.append(Tok(t.key[-len(hit):] + t.trail))
                continue
        m = re.fullmatch(r"(ID|PNR)([A-Z]{1,3})", t.core)
        if m and not t.trail:
            out.append(Tok(t.lead + m.group(1)))
            out.append(Tok(m.group(2)))
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------- run parsing
def _parse_group(ts, i):
    """Parse one compositional number starting at ts[i]. Returns (digit_string, next_index) or None."""
    t = ts[i]
    if t.kind == "d":
        s = t.digits
        j = i + 1
        # '5 lakh', '2 hundred'
        if len(s) <= 3 and j < len(ts) and ts[j].kind == "m":
            return _parse_compound(ts, i)
        return s, j
    if t.kind in ("double", "triple"):
        n = 2 if t.kind == "double" else 3
        if i + 1 < len(ts) and ts[i + 1].kind in ("u", "oh") or (i + 1 < len(ts) and ts[i + 1].kind == "d"
                                                                 and len(ts[i + 1].digits) == 1):
            nxt = ts[i + 1]
            d = nxt.digits if nxt.kind == "d" else str(nxt.val)
            return d * n, i + 2
        return None
    if t.kind == "oh":
        return "0", i + 1
    if t.kind in ("u", "t", "x", "h", "m", "ff", "fa"):
        return _parse_compound(ts, i)
    return None


def _small(ts, i):
    """value 0-99 at ts[i] (tens+unit composed). Returns (value, next) or None."""
    t = ts[i]
    if t.kind == "d" and len(t.digits) <= 3:
        return int(t.digits), i + 1
    if t.kind in ("u", "t", "h"):
        return t.val, i + 1
    if t.kind == "x":
        if i + 1 < len(ts) and ts[i + 1].kind == "u" and ts[i + 1].val > 0:
            return t.val + ts[i + 1].val, i + 2
        return t.val, i + 1
    return None


def _segment(ts, j, allow_bare_mult):
    """One sub-multiplier segment: small | small hundred [and] [small] | dedh/dhai + mult | saadhe N + mult."""
    t = ts[j]
    if t.kind == "ff":
        if j + 1 < len(ts) and ts[j + 1].kind == "m":
            val, k = t.val, j + 1
        else:
            return None
    elif t.kind == "fa":
        sm = _small(ts, j + 1) if j + 1 < len(ts) else None
        if not sm or sm[1] >= len(ts) or ts[sm[1]].kind != "m":
            return None
        val, k = sm[0] + t.val, sm[1]
    elif t.kind == "m":
        if not allow_bare_mult:
            return None
        val, k = 1, j  # bare 'hundred' / 'thousand'
    else:
        sm = _small(ts, j)
        if sm is None:
            return None
        val, k = sm
    if k < len(ts) and ts[k].kind == "m" and ts[k].val == 100:
        val *= 100
        k += 1
        if k + 1 < len(ts) and ts[k].kind == "and" and ts[k + 1].kind in ("u", "t", "x", "h"):
            k += 1
        if k < len(ts) and ts[k].kind in ("u", "t", "x", "h"):
            sm = _small(ts, k)
            val += sm[0]
            k = sm[1]
    return val, k


def _parse_compound(ts, i):
    """Indian/western compositional number: segment [lakh|thousand|crore segment ...]."""
    total, last_mult, j, started = 0, None, i, False
    while j < len(ts):
        sg = _segment(ts, j, not started)
        if sg is None:
            break
        val, k = sg
        if k < len(ts) and ts[k].kind == "m" and ts[k].val > 100:
            mult = ts[k].val
            if last_mult is not None and mult >= last_mult:
                break
            total += val * mult
            last_mult, started = mult, True
            j = k + 1
            if j + 1 < len(ts) and ts[j].kind == "and" and ts[j + 1].kind in ("u", "t", "x", "h"):
                j += 1
            continue
        if last_mult is not None and ts[j].kind == "d":
            break  # 'one thousand 2023' style: leave the digits as their own group
        total += val
        j, started = k, True
        break
    if not started or total != int(total):
        return None
    return str(int(total)), j


def _parse_run(ts, i):
    """Parse a maximal run of number groups from ts[i]. Returns (groups, next_index, decimal_part or None)."""
    groups = []
    j = i
    while j < len(ts):
        t = ts[j]
        if t.kind == "oh":  # 'oh' counts only between numbers
            if not groups or j + 1 >= len(ts) or not ts[j + 1].is_num():
                break
        if t.kind == "and" or t.kind == "point":
            break
        if not (t.is_num() or t.kind == "oh"):
            break
        if t.kind == "d" and t.glue_prefix and groups:
            break
        g = _parse_group(ts, j)
        if g is None:
            break
        groups.append((g[0], j, g[1]))
        j = g[1]
        if ts[j - 1].trail and any(c in ts[j - 1].trail for c in ".?!।"):
            break
    dec = None
    if groups and j + 1 < len(ts) and ts[j].kind == "point" and not _id_prefix(ts, i) and not ts[i].glue_prefix:
        k, d = j + 1, ""
        while k < len(ts) and (ts[k].kind == "u" or (ts[k].kind == "d" and len(ts[k].digits) == 1)):
            d += ts[k].digits if ts[k].kind == "d" else str(ts[k].val)
            k += 1
        if d and len(d) <= 2:
            dec, j = d, k
    return groups, j, dec


def _approx_pair(ts, groups, next_key=None):
    """'do teen', 'teen chaar', 'ek do', 'das bees', 'two three' -> approximate count, keep as words."""
    if len(groups) != 2:
        return False
    a, b = groups
    if a[2] - a[1] != 1 or b[2] - b[1] != 1:
        return False
    ta, tb = ts[a[1]], ts[b[1]]
    if ta.kind == "d" or tb.kind == "d":
        return False
    if not (ta.lang in ("hi", "hr") and tb.lang in ("hi", "hr")):
        # REVIEW-numconv-3: English/Devanagari-English pair only right before a duration/count word
        # ('two three days', 'five ten minutes'); elsewhere two English words may be part of a digit string
        if not (ta.lang in ("en", "de") and tb.lang in ("en", "de") and next_key in APPROX_NEXT):
            return False
    va, vb = int(a[0]), int(b[0])
    if va == vb or va == 0:
        return True  # "do do" (give, repeated)
    pairs = {(10, 15), (10, 20), (15, 20), (20, 25), (20, 30), (25, 30), (30, 40), (40, 50), (50, 60), (5, 7),
             (10, 12), (100, 200), (200, 300), (8, 10), (5, 10)}
    return (vb == va + 1 and va < 10) or (va, vb) in pairs


def _join_groups(ts, groups, has_prefix):
    """Concatenate the run's groups (digit-by-digit phones/IDs, 'twenty twenty four'), but keep two separate
    amounts apart: after a word-compound with hundred/thousand ('one hundred forty nine | twenty GB'), and for a bare
    pair of 2-digit word compounds that is not a year ('ten | nineteen rupees', 'twenty five | thirty nine')."""
    def worded(g):
        return ts[g[1]].kind != "d"

    def has_mult(g):
        return any(ts[x].kind == "m" for x in range(g[1], g[2]))

    out = groups[0][0]
    for a, b in zip(groups, groups[1:]):
        sep = ""
        if not has_prefix and worded(a) and has_mult(a) and worded(b):
            sep = " "
        elif not has_prefix and len(groups) == 2 and worded(a) and worded(b) and len(a[0]) >= 2 \
                and len(b[0]) >= 2 and a[2] - a[1] >= 1 and not (a[0] in ("19", "20") and len(b[0]) == 2):
            sep = " "
        out += sep + b[0]
    return out


def _prev_core(ts, i):
    return ts[i - 1].key if i > 0 else None


def _next_core(ts, j):
    return ts[j].key if j < len(ts) else None


def _id_prefix(ts, i):
    """Letter tokens directly before index i forming an ID prefix. Returns (start_index, 'LETTERS') or None."""
    letters, k, parts = "", i - 1, []
    while k >= 0 and len(letters) < 5:
        t = ts[k]
        if t.trail and k != i - 1:
            break
        c = t.core
        if re.fullmatch(r"[A-Z]{1,3}", c) and c not in LATIN_PREFIX_STOP:
            parts.append(("L", c))
        elif DEVA.search(c) and deva_letters(c) and t.key not in LEX:
            dl = deva_letters(c)
            if dl.startswith("PNR") and len(dl) > 3:
                dl = "PNR " + dl[3:]
                parts.append(("D", dl))
                letters = dl + letters
                k -= 1
                break
            parts.append(("D", dl))
        else:
            break
        letters = parts[-1][1] + letters
        if t.lead:
            k -= 1
            break
        k -= 1
    if not parts:
        return None
    if len(letters.replace("PNR ", "")) > 4:
        return None
    # a lone everyday Devanagari word (जी, के, ए ...) is not a prefix
    if len(parts) == 1 and parts[0][0] == "D" and fold(ts[i - 1].core) in DEVA_LETTER_WORDS:
        return None
    return k + 1, letters


# ---------------------------------------------------------------- 'at' = eight (REVIEW-numconv-1)
_AT_STOP_AFTER = {fold(w) for w in "o'clock oclock clock baje बजे am pm a.m. p.m. एएम पीएम".split()}


def _digit_len(t):
    """Digits contributed by a single-digit-like token (one/2/double/triple/oh), else 0."""
    if t.kind == "u" and t.lang in ("en", "de"):
        return 1
    if t.kind == "d" and len(t.digits) == 1 and not t.glue_prefix:
        return 1
    if t.kind == "oh":
        return 1
    return 0


def _at_to_eight(ts):
    """Latin 'at' -> 8 when it sits inside a spoken digit string. Trelis writes 'eight' as 'at' in V4:
    'nine at seven three zero seven five eight nine three' (sandwich), 'at zero seven six zero nine zero seven nine four'
    (leading 'at's + single digits = exactly 10), 'EC at five zero five' (ID prefix before). Never before a mail word,
    never when the digits are followed by a time word or a tens word ('at seven thirty', 'at five baje')."""
    n, i = len(ts), 0
    while i < n:
        if ts[i].key != "at" or ts[i].lead:
            i += 1
            continue
        j = i
        while j < n and ts[j].key == "at" and not ts[j].trail:
            j += 1
        n_at = j - i
        if j >= n or j == i or mail_like(ts[j].key):
            i = max(j, i + 1)
            continue
        # digits after (doubles/triples count as their expansion)
        k, d_after = j, 0
        while k < n:
            t = ts[k]
            if t.kind in ("double", "triple") and k + 1 < n and _digit_len(ts[k + 1]):
                d_after += 2 if t.kind == "double" else 3
                k += 2
            elif _digit_len(t):
                d_after += 1
                k += 1
            else:
                break
            if ts[k - 1].trail and any(c in ts[k - 1].trail for c in ".?!।"):
                break
        if d_after == 0 or (k < n and (ts[k].key in _AT_STOP_AFTER or ts[k].kind in ("t", "x", "m"))):
            i = j
            continue
        # digits before
        b, d_before = i - 1, 0
        if i > 0 and not ts[i - 1].trail:
            while b >= 0 and _digit_len(ts[b]) and (b == i - 1 or not ts[b].trail):
                d_before += 1
                b -= 1
        total = d_before + n_at + d_after
        ok = (d_before and total >= 4) or (not d_before and n_at + d_after == 10) or \
             (not d_before and total >= 4 and _id_prefix(ts, i) is not None)
        if ok:
            for x in range(i, j):
                ts[x].kind, ts[x].val, ts[x].lang = "u", 8, "en"
                ts[x].core, ts[x].key = "eight", "eight"
        i = j



def convert(text):
    """Return text with spoken numbers replaced by digits (see module docstring)."""
    if not text:
        return text
    ts = [Tok(w) for w in _split_hyphens(text)]
    ts = _split_glued_id(ts)
    n = len(ts)
    # narrow email fix: 'eight yahoo' -> 'at yahoo'
    for k in range(n - 1):
        if ts[k].key in ("eight", fold("एट")) and mail_like(ts[k + 1].key) and not ts[k].trail:
            ts[k].kind, ts[k].core, ts[k].key = None, "at", "at"
    # Trelis writes spoken "eight" as "at" inside digit strings (nine at seven three ..., EC at five zero five)
    _at_to_eight(ts)
    # 'ek saath' (together) / roman weak words without a roman-Hindi neighbour are not numbers
    for k in range(n):
        t = ts[k]
        if t.key in ("saath", "sath", fold("साथ")) and k > 0 and ts[k - 1].key in ("ek", fold("एक")):
            t.kind = None
    # REVIEW-numconv-2: decide from the ORIGINAL kinds (an in-place left-to-right pass killed 'sau' in
    # 'ek hazaar do sau' after 'do' had been kept because of it -> '1002 sau'); 'sau' after a number word is kept.
    kinds0 = [t.kind for t in ts]
    kill = []
    for k in range(n):
        t = ts[k]
        if t.lang == "hr" and t.key in ROMAN_WEAK | {"do"} and kinds0[k] is not None:
            if t.key == "sau" and k > 0 and kinds0[k - 1] in ("u", "h", "t", "x", "d", "ff", "fa"):
                continue
            nb = [x for x in (k - 1, k + 1) if 0 <= x < n]
            ctx = (k + 1 < n and ts[k + 1].key in UNIT_AFTER and ts[k + 1].key not in CHECK_LINE_AFTER) or \
                (k > 0 and ts[k - 1].key in CONTEXT_BEFORE)
            if not ctx and not any(ts[x].lang == "hr" and kinds0[x] is not None and
                                   ts[x].key not in ROMAN_WEAK | {"do"} or kinds0[x] in ("m", "ff", "fa")
                                   for x in nb):
                kill.append(k)
    for k in kill:
        ts[k].kind = None
    for k in range(n):
        t = ts[k]
        if t.lang in ("hi", "hr") and t.kind in ("u", "h") and t.key in HI_WEAK:
            nb = [ts[x] for x in (k - 1, k + 1) if 0 <= x < n]
            if any(x.lang in ("en", "de", "dg") and x.kind != "m" and (x.is_num() or x.kind == "oh") for x in nb) and \
                    not any(x.lang in ("hi", "hr") and x.kind in ("u", "h", "m", "ff", "fa") for x in nb):
                t.kind = None
    out = []
    last_end = 0
    i = 0
    while i < n:
        t = ts[i]
        if not (t.is_num() and t.kind not in ("fa",) or t.kind in ("ff", "fa")):
            out.append(t.raw)
            i += 1
            continue
        groups, j, dec = _parse_run(ts, i)
        if not groups:
            out.append(t.raw)
            i += 1
            continue
        num_toks = sum(1 for x in ts[i:j] if x.is_num() or x.kind == "oh")
        first, last = ts[i], ts[j - 1]
        prev_k, next_k = _prev_core(ts, i), _next_core(ts, j)
        prefix = _id_prefix(ts, i) if not first.glue_prefix else None
        if prefix and prefix[0] < last_end:
            prefix = None
        convert_it = True
        only_digits = all(ts[x].kind == "d" for x in range(i, j))
        if only_digits and len(groups) == 1 and not prefix:
            convert_it = bool(first.glue_prefix)  # MW-3730 -> MW3730; plain digits: nothing to do
        elif num_toks >= 2 or (len(groups) == 1 and dec):
            if _approx_pair(ts, groups, next_k) and not prefix and prev_k not in CONTEXT_BEFORE:
                convert_it = False
        else:
            single = first
            ctx = (next_k in UNIT_AFTER) or (prev_k in CONTEXT_BEFORE) or prefix is not None
            if single.kind == "d":
                convert_it = True
            elif single.lang == "en" and single.key not in EN_HOMOPHONE:
                convert_it = True
            else:
                convert_it = ctx
            # inside an email: pandey three at hotmail -> pandey 3 at hotmail
            if not convert_it and single.lang in ("en", "de") and not last.trail and j + 1 < n and \
                    next_k in ("at", "@") and mail_like(ts[j + 1].key):
                convert_it = True
            if single.key in ("one", "ek", fold("एक"), "वन") and next_k in CHECK_LINE_AFTER:
                convert_it = False
        if not convert_it:
            out.extend(x.raw for x in ts[i:j])
            i = j
            continue
        digits = _join_groups(ts, groups, bool(prefix or first.glue_prefix))
        # time: h mm + am/pm
        gv = [g[0] for g in groups]
        if len(gv) == 3 and gv[1] == "0" and len(gv[2]) == 1:
            gv = [gv[0], "0" + gv[2]]  # eleven oh seven
        if next_k in TIME_AFTER and len(gv) == 2 and 1 <= int(gv[0]) <= 12 and len(gv[1]) == 2 and int(gv[1]) < 60:
            digits = gv[0] + ":" + gv[1]
        if dec:
            digits += "." + dec
        if first.glue_prefix:
            digits = first.glue_prefix + digits
        lead = first.lead
        if prefix:
            s, letters = prefix
            # drop the prefix tokens already emitted
            del out[len(out) - (i - s):]
            lead = ts[s].lead
            digits = letters + digits
        # suffix letter: 'seat fourteen B' -> 14B
        trail = last.trail
        if not trail and not prefix and j < n and re.fullmatch(r"[A-HJ-Z]", ts[j].core) and len(groups) == 1 \
                and any(ts[x].key in SUFFIX_CTX for x in range(max(0, i - 3), i)):
            digits += ts[j].core
            trail = ts[j].trail
            j += 1
        out.append(lead + digits + trail)
        i = last_end = j
    res = " ".join(out)
    return _join_phone(res)


def _join_phone(text):
    """Adjacent pure-digit tokens whose total is 10 digits (98730 75893 / 98730, 75893) -> one phone number."""
    toks = text.split(" ")
    out, i = [], 0
    while i < len(toks):
        m = re.fullmatch(r"(\d+)([,]?)", toks[i])
        if m:
            k, acc, ends = i, "", []
            while k < len(toks):
                mm = re.fullmatch(r"(\d+)([,]?)", toks[k]) if k == i or ends[-1] in ("", ",") else None
                if k > i:
                    mm = re.fullmatch(r"(\d+)([,.?!।]?)", toks[k])
                if not mm or len(acc) + len(mm.group(1)) > 10:
                    break
                acc += mm.group(1)
                ends.append(mm.group(2))
                k += 1
                if len(acc) == 10:
                    break
                if mm.group(2) not in ("", ","):
                    break
            if len(acc) == 10 and k - i >= 2:
                out.append(acc + ends[-1])
                i = k
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------- evaluation helper
def extract_numbers(text):
    """Digit strings in a (converted or reference) text: 1,85,000 -> 185000; '98730 75893' -> one 10-digit phone;
    digits inside IDs/emails count (FD4124 -> 4124); times split (7:30 -> 7, 30)."""
    t = text.translate(DIGIT_TRANS)
    t = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", t)
    t = re.sub(r"(?<=\d),(?=\d{2},)", "", t)
    t = re.sub(r"\b(\d{5})\s(\d{5})\b", r"\1\2", t)
    # spelled-out digits in a reference (9 8 7 3 0 8 4 7 6 9): a run of >= 4 single spaced digits is one number
    t = re.sub(r"\b\d(?: \d\b){3,}", lambda m: m.group(0).replace(" ", ""), t)
    return re.findall(r"\d+", t)
