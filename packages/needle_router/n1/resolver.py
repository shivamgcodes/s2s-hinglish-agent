"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle/resolver.py (verbatim).

N1 resolver: order_ref (the phrase the router copied from the customer turn) -> entity ID.

    resolve(order_ref, record, agent_type) -> {"id": str|None, "rule": str, "candidates": [...]}

The record is a data/records.json entry (or any dict with an explicit "entities" list, see
entities_from_record). Entities are built from the record's top-level primary_id / secondary_id
(the facts keys vary per record, so they are only used as free text):
  primary   : id = primary_id, text = all facts values, active = True
  secondary : id = secondary_id, text = every distractor string containing that id,
              active = False ("Older order", "Past ride", "Reference", "Return PNR" ...)
  extra     : any other distractor carrying an ID of the agent's ID shape (e.g. air_15 has both
              "Return flight UA 405" and "Reference HQ2297"), active = False
Chronology (for pehla/dusra/first/second): a past entity ("older/past/previous/expired/delivered/
completed/last year/reference/invoice") comes BEFORE the primary; a "return" booking comes AFTER it.

Rule precedence (first that yields exactly one entity wins):
  1. id        explicit ID in the phrase: compared after lower-casing, dropping spaces/punctuation,
               converting spoken digits (English, Hindi, "double"/"triple", spaced letters like
               "F D 1 5 6 5"); full ID, or the ID's digit block (>=3 digits) inside the phrase digits.
               Also the record's misread_id (a known mis-hearing of the primary) -> primary.
  2. name      brand / restaurant / product / plan / place / airline / route token match that is
               unique to one entity (tokens shared by all entities, e.g. the company brand, ignored).
  3. item      item-name match (items / product words) unique to one entity.
  4. ordinal   pehla/first/1st -> earliest; dusra/second/2nd -> 2nd earliest; teesra/third -> 3rd;
               latest/last/naya/new/current/abhi wala/recent -> the active entity (newest active
               when several are active); purana/old/older/previous/pichhla/past -> the past entity;
               return/wapsi (air) -> the return booking.
               Ordinal words followed by address/number/email/ghar/pata/status are NOT ordinals
               ("purana address", "naya number").
  5. default   no rule fired (empty phrase or no match): the single active entity if exactly one
               is active ("default_active"), the only entity if there is one ("default_single"),
               else id None, rule "ambiguous". A non-empty phrase that matched nothing gets the
               suffix "_unmatched" (e.g. "default_active_unmatched").
"""
import re

# ---------------------------------------------------------------- normalisation / spoken digits
_DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hindi (romanised, common spellings)
    "shunya": "0", "sunya": "0", "sifar": "0", "ek": "1", "do": "2", "teen": "3", "tin": "3",
    "char": "4", "chaar": "4", "paanch": "5", "panch": "5", "pach": "5", "chhe": "6", "chhah": "6",
    "chah": "6", "che": "6", "chheh": "6", "saat": "7", "sat": "7", "aath": "8", "ath": "8",
    "nau": "9", "nao": "9",
    # Devanagari digits handled char-wise below
}
_TENS = {"ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
         "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19"}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_REPEAT = {"double": 2, "triple": 3, "treble": 3}
_LETTER_WORDS = {"dee": "d", "eff": "f", "ef": "f", "see": "c", "cee": "c", "kay": "k", "jay": "j",
                 "pee": "p", "are": "r", "ar": "r", "aye": "a", "bee": "b", "en": "n", "em": "m",
                 "ex": "x", "why": "y", "zed": "z", "zee": "z", "tee": "t", "vee": "v", "ess": "s",
                 "aitch": "h", "kyu": "q", "queue": "q", "double-u": "w"}


def _tokens(s):
    s = (s or "").translate(_DEV_DIGITS).lower()
    return re.findall(r"[a-z0-9]+", s.replace("'", ""))


def spoken_to_compact(s):
    """Lower-case alnum string with spoken digits/letters turned into characters.

    'F D one five six five' -> 'fd1565'; 'double five' -> '55'; 'ek paanch chhe paanch' -> '1565'.
    Only converts number words when they sit in a run of >=2 digit-like tokens, so ordinary words
    ('do' = 'do it', 'ek' = 'one order') are not turned into digits on their own.
    """
    toks = _tokens(s)
    out, i = [], 0
    digitish = lambda t: (t in _DIGIT_WORDS or t in _TENS or t in _REPEAT or t.isdigit())
    while i < len(toks):
        # find a run of digit-like tokens
        j = i
        while j < len(toks) and digitish(toks[j]):
            j += 1
        run = toks[i:j]
        if len(run) >= 2 or (run and run[0].isdigit()):
            k = 0
            while k < len(run):
                t = run[k]
                if t in _REPEAT and k + 1 < len(run):
                    nxt = run[k + 1]
                    d = _DIGIT_WORDS.get(nxt, nxt if nxt.isdigit() else "")
                    out.append(d * _REPEAT[t]); k += 2; continue
                out.append(_DIGIT_WORDS.get(t) or _TENS.get(t) or (t if t.isdigit() else ""))
                k += 1
            i = j
            continue
        t = toks[i]
        out.append(_LETTER_WORDS.get(t, t))
        i += 1
    return "".join(out)


def _digits(s):
    return re.sub(r"\D", "", spoken_to_compact(s))


# ---------------------------------------------------------------- entities from a record
_PAST_CUES = re.compile(r"\b(older|old|past|previous|prior|expired|delivered|completed|last year|"
                        r"reference|invoice|dispute|trial|cancelled)\b", re.I)
_RETURN_CUE = re.compile(r"\breturn\b", re.I)
_ID_SHAPE = re.compile(r"\b[A-Z]{2}\d{4}\b")

_STOP = set("""a an the of to from for and or on in at by with order orders ride rides booking bookings
plan plans account flight flights ticket tickets pnr id ref reference wala wali wale vala vali wale ka ki ke
ko mera meri mere my our is hai tha thi rs x1 x2 x3 status delivered completed placed out delivery
eta minutes paid via by upi card credit debit cash october november september december june july may
older old past previous return new naya purana pehla dusra first second latest last current this that
ye yeh woh wo vo please sir maam ji haan""".split())


def _words(s):
    return [w for w in _tokens(s) if w not in _STOP and not w.isdigit() and len(w) > 1]


def entities_from_record(record):
    """List of entities: dicts with id, text, active, chrono (smaller = earlier)."""
    if record.get("entities"):
        ents = [dict(e) for e in record["entities"]]
        for n, e in enumerate(ents):
            e.setdefault("chrono", n)
            e.setdefault("active", True)
            e.setdefault("names", "")
            e.setdefault("items", "")
        return ents
    facts = record.get("facts") or {}
    prim_text = " ; ".join(f"{v}" for k, v in facts.items()
                           if isinstance(v, (str, int, float)) and k.lower() not in _ITEM_KEYS)
    ents = [{"id": record["primary_id"], "names": prim_text, "items": _items_of(facts),
             "active": True, "chrono": 0, "kind": "primary"}]
    sec = record.get("secondary_id")
    used = set()
    for d in record.get("distractors") or []:
        ids = _ID_SHAPE.findall(d)
        eid = sec if (sec and sec in d) else next((i for i in ids if i != record["primary_id"]), None)
        if not eid or eid == record["primary_id"]:
            continue
        if eid in used:
            ent = next(e for e in ents if e["id"] == eid)
            ent["names"] += " ; " + d
            continue
        used.add(eid)
        ret = bool(_RETURN_CUE.search(d))
        ents.append({"id": eid, "names": d.split(":", 1)[0], "items": d.split(":", 1)[1] if ":" in d else "",
                     "active": False, "chrono": 1 if ret else -1,
                     "kind": "return" if ret else "past"})
    # a secondary id that no distractor mentions still exists as a (text-less) past entity
    if sec and sec not in used and sec != record["primary_id"]:
        ents.append({"id": sec, "names": "", "items": "", "active": False, "chrono": -1, "kind": "past"})
    # ID-less distractors: a past-cue one ("previous plan: Basic Quarterly") is attached to the
    # single past entity; others ("Registered email ...", air_02 "Return flight SI 405" with no
    # PNR) have no ID to route to and are skipped.
    past = [e for e in ents if e.get("kind") == "past"]
    for d in record.get("distractors") or []:
        if not _ID_SHAPE.search(d) and _PAST_CUES.search(d) and not _RETURN_CUE.search(d) and len(past) == 1:
            past[0]["names"] += " ; " + d
    ents.sort(key=lambda e: e["chrono"])
    return ents


_ITEM_KEYS = ("items", "item", "affected_items")


def _items_of(facts):
    out = []
    for k, v in facts.items():
        if k.lower() in _ITEM_KEYS:
            out.append(str(v))
    return " ; ".join(out)


# ---------------------------------------------------------------- rules
_ORD = [
    ("first", r"\b(pehla|pehle|pahla|pahle|pehli|pahli|first|1st|1st one)\b"),
    ("second", r"\b(dusra|dusre|doosra|doosre|dusri|doosri|second|2nd)\b"),
    ("third", r"\b(teesra|teesre|tisra|tisre|teesri|third|3rd)\b"),
    ("return", r"\b(return|wapsi|vapsi|waapsi|wapas|vapas)\b"),
    ("latest", r"\b(latest|last|naya|naye|nayi|new|newest|current|recent|abhi\s+wala|abhi\s+wali|"
               r"active|ongoing|aaj\s+wala|aaj\s+wali|today'?s)\b"),
    ("past", r"\b(purana|purane|purani|old|older|oldest|previous|pichhla|pichhle|pichli|pichla|"
             r"pichhli|past|earlier|pehle\s+wala\s+purana)\b"),
]
# ordinal word directly followed by one of these is about a field, not about the entity
_FIELD_AFTER = re.compile(r"^\s*(wala\s+|wali\s+|vala\s+)?(address|addresses|number|phone|email|e-mail|"
                          r"mail|ghar|pata|status|time|location|pickup|drop|instruction|card)\b", re.I)


def _ordinal(ref):
    s = (ref or "").lower()
    for name, pat in _ORD:
        for m in re.finditer(pat, s):
            if _FIELD_AFTER.match(s[m.end():]):
                continue
            # "pehle" as "earlier/before" ("pehle wala") is fine as ordinal; "last year" is not
            if name == "latest" and m.group(0) == "last" and re.match(r"\s*(year|month|week|time)\b", s[m.end():]):
                continue
            return name
    return None


def _pick_ordinal(kind, ents):
    by_time = sorted(ents, key=lambda e: e["chrono"])
    if kind in ("first", "second", "third"):
        n = {"first": 0, "second": 1, "third": 2}[kind]
        return by_time[n] if n < len(by_time) and len(by_time) > 1 else None
    if kind == "return":
        r = [e for e in ents if e.get("kind") == "return"]
        return r[0] if len(r) == 1 else None
    if kind == "latest":
        act = [e for e in ents if e["active"]]
        if len(act) == 1:
            return act[0]
        return max(act or ents, key=lambda e: e["chrono"]) if len(ents) > 1 else None
    if kind == "past":
        past = [e for e in ents if e.get("kind") == "past" or (not e["active"] and e.get("kind") != "return")]
        if len(past) == 1:
            return past[0]
        if not past and len(ents) > 1:          # all active (N0 shape): the older one
            return min(ents, key=lambda e: e["chrono"])
        return None
    return None


def _match_id(ref, ents, record):
    comp = spoken_to_compact(ref)
    runs = set(re.findall(r"\d+", comp))   # whole digit groups only: a phone never matches an ID

    def hit(idv):
        idv = idv.lower()
        d = re.sub(r"\D", "", idv)
        return idv in comp or (len(d) >= 3 and d in runs)
    hits = [e for e in ents if hit(e["id"])]
    if len(hits) == 1:
        return hits[0]
    mis = record.get("misread_id")
    if mis and hit(mis):
        return next((e for e in ents if e["id"] == record.get("primary_id")), None)
    return None


def _stem(w):
    # light plural / Hinglish suffix folding: momos->momo, headphones->headphone
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _match_words(ref, ents, field):
    rw = {_stem(w) for w in _words(ref)}
    if not rw:
        return None
    sets = [{_stem(w) for w in _words(e[field])} for e in ents]
    common = set.intersection(*sets) if len(sets) > 1 else set()
    scores = []
    for e, s in zip(ents, sets):
        own = (s - common) if len(ents) > 1 else s
        scores.append(len(rw & own))
    best = max(scores)
    if best == 0 or scores.count(best) != 1:
        return None
    return ents[scores.index(best)]


def resolve(order_ref, record, agent_type=None):
    ents = entities_from_record(record)
    cands = [e["id"] for e in ents]
    ref = (order_ref or "").strip()
    if ref:
        e = _match_id(ref, ents, record)
        if e:
            return {"id": e["id"], "rule": "id", "candidates": cands}
        e = _match_words(ref, ents, "names")
        if e:
            return {"id": e["id"], "rule": "name", "candidates": cands}
        e = _match_words(ref, ents, "items")
        if e:
            return {"id": e["id"], "rule": "item", "candidates": cands}
        k = _ordinal(ref)
        if k:
            e = _pick_ordinal(k, ents)
            if e:
                return {"id": e["id"], "rule": f"ordinal:{k}", "candidates": cands}
    # unmatched non-empty phrase is flagged in the rule name so the server can log / confirm it
    tag = "_unmatched" if ref else ""
    act = [e for e in ents if e["active"]]
    if len(act) == 1:
        return {"id": act[0]["id"], "rule": "default_active" + tag, "candidates": cands}
    if len(ents) == 1:
        return {"id": ents[0]["id"], "rule": "default_single" + tag, "candidates": cands}
    return {"id": None, "rule": "ambiguous", "candidates": cands}


# N0 (needle-exp0) record in this resolver's shape: two ACTIVE orders, A1234 placed earlier.
N0_RECORD = {"entities": [
    {"id": "A1234", "names": "Biryani Blues", "items": "biryani", "active": True, "chrono": 0},
    {"id": "B5678", "names": "Domino's", "items": "pizza", "active": True, "chrono": 1},
]}
