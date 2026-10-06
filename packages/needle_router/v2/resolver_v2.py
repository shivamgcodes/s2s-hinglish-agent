"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle_v2/schema/resolver_v2.py (path edit only: the runpod2
numconv dir /root/n2/numconv is replaced by the sibling ../numconv).

N2 resolver: a REF argument (what the model wrote) -> the exact record value, or ASK.

    resolve(kind, ref, record, agent_type=None)
        -> {"value": str|None, "rule": str, "ask": bool, "candidates": [str, ...]}
    candidates(kind, record) -> [Candidate]   (value + the record text around it)
    registered_address(record) -> str|None

Kinds (tools_v2.REF_KINDS):
  order ride booking account transaction charge   ID-shaped values [A-Z]{2}dddd in the record's information
                                                   (plus primary_id); context = the words around the ID
  plan           current plan (facts key with "plan") + previous plan (distractor "previous plan: X")
  card           card last 4 ("ending 8923", "Card 5567", facts.card_last4)
  service, pack  names parsed from the active-services / available-packs lists
  phone_on_file  {"registered": record.phone, "alternate": alternate number on record}

Rules, first that yields exactly one candidate wins:
  1. exact       the ref equals a candidate after normalisation (lower-case alnum; spoken digits -> digits via
                 N1 spoken_to_compact, and numconv when available). The trained (canonical) target hits this.
  2. id / digits ID kinds: a candidate ID inside the ref, or its digit block (>=3 digits) as a whole digit run;
                 card: the ref's last 4 digits; phone_on_file: the ref's digits are a suffix (>=4) of one number.
  3. n1          old 5 types, ID kinds only: N1 resolver.resolve(ref, record) when it fires a non-default rule
                 (name / item / ordinal) and its ID is a candidate. Keeps N1's spoken-phrase behaviour.
  4. words       content words of the ref (numbers must agree) unique to one candidate's context / name.
  5. ordinal     N1 _ordinal(): latest/current/naya -> the primary (active) one; purana/previous/pichhla ->
                 the single non-primary one; "registered"/"alternate" words for phone_on_file.
  6. default     empty ref: the only candidate, or the primary/current one for order ride booking account plan
                 card phone_on_file (N1 default_active). Empty ref with several candidates for transaction /
                 charge / service / pack -> ASK.
  7. otherwise   ASK. Exception (N1 compatibility): a non-ID phrase that matched nothing for order / ride /
                 booking / account resolves to the primary with rule "default_primary_unmatched".
An ID-shaped ref ([A-Z]{2}dddd) that is not in the record is always ASK (rule "unknown_id"): a hallucinated or
misheard ID must not be silently swapped for the primary.
"""
import re

import n1path  # noqa: F401  (puts N1 resolver on sys.path)
import resolver as n1  # N1 resolver.py

try:  # number converter from the sibling worker (optional here; the router applies it to the transcript)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "numconv"))
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "numconv"))
    import numconv as _numconv  # noqa: E402
except Exception:  # pragma: no cover
    _numconv = None

ID_RE = re.compile(r"\b[A-Z]{2}\d{4}\b")
ID_KINDS = {"order", "ride", "booking", "account", "transaction", "charge"}
OLD_ID_KINDS = {"order", "ride", "booking", "account"}
DEFAULT_PRIMARY_KINDS = {"order", "ride", "booking", "account", "plan", "card", "phone_on_file"}


# ----------------------------------------------------------------------------------------- text utils
def numconv_text(s):
    """numconv.convert(s) from the sibling worker (spoken numbers -> digits); identity if not importable."""
    if not s or _numconv is None:
        return s or ""
    return _numconv.convert(s)


_LEX = None
_DEV = re.compile(r"[\u0900-\u097F]")


def romanise_text(s):
    """N1 romanise.py (Trelis Devanagari -> typed Roman Hinglish); identity for Latin-only text."""
    global _LEX
    s = s or ""
    if not _DEV.search(s):
        return s
    import romanise as _rom  # N1 romanise.py (+ romanise_lexicon.json next to it)
    if _LEX is None:
        _LEX = _rom.load_lex()
    return _rom.romanise(s, _LEX)


def prepare(s):
    """The text normalisation the router applies to the transcript: numconv first (it reads Devanagari number
    words), then romanise what is left in Devanagari."""
    return romanise_text(numconv_text(str(s or "")))


def compact(s):
    """lower-case alnum, spoken digits -> digits ('F D one five six five' -> 'fd1565')."""
    return n1.spoken_to_compact(prepare(s))


def words(s):
    """content tokens: lower-case alnum words incl. numbers, minus N1 stop words, plus split letter/digit
    ('10gb' -> '10', 'gb')."""
    toks = []
    s = re.sub(r"(?<=\d),(?=\d)", "", prepare(s))      # Indian digit grouping: 4,127 / 1,45,000
    for t in n1._tokens(s):
        toks.extend(re.findall(r"[a-z]+|\d+", t))
    return [n1._stem(t) for t in toks if (t.isdigit() or (t not in n1._STOP and t not in _EXTRA_STOP and len(t) > 1))]


_EXTRA_STOP = set("""rs month mo gb mb day days validity calls sms unlimited service services pack packs plan
active available data booster wala wali wale karo kar kardo dijiye please band stop activate deactivate chahiye
transaction txn charge ref card ending number on at for""".split())


class Cand(dict):
    """{"value", "context", "primary": bool, "role"}"""


def _info(record):
    return str(record.get("information") or "")


# ----------------------------------------------------------------------------------------- candidates
# field separator: , ; : . followed by whitespace or end (keeps 'Rs 2,841', 'Rs 1,20,000', 'Rs 14.20' intact),
# or a joining ' and ' / ' aur '
_FIELD_SEP = re.compile(r"[,.;:](?=\s|$)|\s(?:and|aur)\s")


def _id_cands(record):
    info = _info(record)
    out, seen = [], set()
    ms = list(ID_RE.finditer(info))
    for i, m in enumerate(ms):
        v = m.group(0)
        if v in seen:
            continue
        seen.add(v)
        # REVIEW-schema-2: the text between two IDs is split at its LAST field separator: the part before it is
        # the earlier ID's item, the part after it the later ID's ("... (TL7413), 10 September Rs 799 (TL5911)",
        # "TX8032 Amazon India Rs 2,143 27 December and TX3938 Zomato ..."). Before, both IDs shared it.
        prev_end = ms[i - 1].end() if i > 0 else 0
        pre = info[max(prev_end, m.start() - 40):m.start()]
        pre = _FIELD_SEP.split(pre)[-1]              # words since the previous comma: "Older order", "baggage fee"
        end = ms[i + 1].start() if i + 1 < len(ms) else len(info)
        post = info[m.end():min(end, m.end() + 70)]
        if i + 1 < len(ms):
            seps = list(_FIELD_SEP.finditer(info[m.end():end]))
            if seps:
                post = info[m.end():m.end() + min(seps[-1].start(), 70)]
        out.append(Cand(value=v, context=(pre + " " + post).strip(), primary=(v == record.get("primary_id")),
                        role="id"))
    pid = record.get("primary_id")
    if pid and pid not in seen and ID_RE.fullmatch(pid):
        out.append(Cand(value=pid, context=" ".join(str(x) for x in (record.get("facts") or {}).values()),
                        primary=True, role="id"))
    return out


def _plan_cands(record):
    facts = record.get("facts") or {}
    cur, prev = [], []
    for k, v in facts.items():
        kl = k.lower()
        if "plan" in kl and not re.search(r"price|cost|amount|previous|prev|old|renew|date|cycle", kl) and \
                isinstance(v, str):
            cur.append(v.strip())
        elif re.search(r"previous.?plan|prev.?plan|old.?plan", kl) and isinstance(v, str):
            prev.append(v.strip())
    for d in record.get("distractors") or []:
        m = re.match(r"\s*(?:previous|prev|old|earlier)[ _]plan\s*:?\s*(.+?)\s*\.?$", d, re.I)
        if m:
            prev.append(m.group(1).strip())
    info = _info(record)
    if not cur:
        m = re.search(r"\b(?:current\s+)?plan\s*:?\s+([A-Z][\w\-+]*(?:\s+[A-Z0-9][\w\-+]*)*)", info)
        if m:
            cur.append(m.group(1))
    out = [Cand(value=v, context=v, primary=True, role="current") for v in dict.fromkeys(cur)]
    out += [Cand(value=v, context=v, primary=False, role="previous") for v in dict.fromkeys(prev) if v not in cur]
    return out


def _card_cands(record):
    info = _info(record)
    vals = re.findall(r"(?:ending|Card|card)\s+(?:no\.?\s+|number\s+)?(\d{4})\b", info)
    f = (record.get("facts") or {})
    for k, v in f.items():
        if re.search(r"last.?4|last.?four|card.?number", k, re.I) and re.fullmatch(r"\d{4}", str(v).strip()):
            vals.append(str(v).strip())
    vals = list(dict.fromkeys(vals))
    return [Cand(value=v, context=v, primary=(n == 0), role="card") for n, v in enumerate(vals)]


_LIST_KEY = {"service": r"(?:active\s+)?services?", "pack": r"(?:available\s+)?(?:data\s+)?packs?"}
_STOP_FIELD = re.compile(r"^(?:last|validity|valid|balance|main|data|email|refund|ref|previous|prev|sim|alternate|"
                         r"alt|registered|current|plan|expires|city|history|reference|disputed|failed)\b", re.I)


def _split_items(text):
    """'Cricket Alerts Rs 49/month, Astrology Daily (Rs 29/month), last recharge ...' -> names, stops at the
    first item that is another field."""
    text = re.sub(r"\([^)]*\)", "", text)
    out = []
    for part in re.split(r",\s*|\s+and\s+|;\s*|\.\s+", text):
        p = part.strip().rstrip(".")
        if not p:
            continue
        if not re.match(r"[A-Z]", p):
            break
        if _STOP_FIELD.match(p) and not re.match(r"\S+\s+[A-Z]", p):   # "data 1.4 GB" stops, "Data Booster" doesn't
            break
        p = re.split(r"\s+(?:for\s+)?Rs\.?\s*\d", p)[0]
        p = re.sub(r"\s+\d+\s*days.*$", "", p).strip()
        if p and len(p) <= 40:
            out.append(p)
    return out


def _list_cands(record, kind):
    names = []
    key = _LIST_KEY[kind]
    for k, v in (record.get("facts") or {}).items():
        kl = k.lower().replace("_", " ")
        if kind == "service" and re.search(r"\bservices?\b", kl) and "deactivat" not in kl and "stop" not in kl:
            names += _split_items(str(v))
        if kind == "pack" and re.search(r"available.*packs?|packs?\s+available", kl):
            names += _split_items(str(v))
    info = _info(record)
    for m in re.finditer(r"\b" + key + r"\s*:?\s+", info, re.I):
        if kind == "pack" and not re.search(r"available|data", m.group(0), re.I):
            continue  # "current pack X" is the plan, not an add-on pack
        names += _split_items(info[m.end():])
    names = list(dict.fromkeys(n for n in names if n))
    return [Cand(value=v, context=v, primary=False, role=kind) for v in names]


_ALT_RE = re.compile(r"(?:alternate|alt\.?)\s+(?:number|no\.?)(?:\s+on\s+record)?\s*:?\s*(\d{5}\s?\d{5}|\d{10})", re.I)


def _phone_cands(record):
    out = []
    if record.get("phone"):
        out.append(Cand(value=record["phone"], context="registered", primary=True, role="registered"))
    texts = [_info(record)] + list(record.get("distractors") or []) + \
        [f"{k} {v}" for k, v in (record.get("facts") or {}).items()]
    for t in texts:
        for m in _ALT_RE.finditer(t.replace("_", " ")):
            v = m.group(1)
            if all(re.sub(r"\D", "", v) != re.sub(r"\D", "", c["value"]) for c in out):
                out.append(Cand(value=v, context="alternate", primary=False, role="alternate"))
    return out


def candidates(kind, record):
    if kind in ID_KINDS:
        c = _id_cands(record)
        if kind == "transaction":
            pref = (record.get("primary_id") or "")[:2]
            c = [x for x in c if x["value"].startswith(pref)] or c
        return c
    if kind == "plan":
        return _plan_cands(record)
    if kind == "card":
        return _card_cands(record)
    if kind in ("service", "pack"):
        return _list_cands(record, kind)
    if kind == "phone_on_file":
        return _phone_cands(record)
    raise ValueError(f"unknown REF kind {kind!r}")


def registered_address(record):
    f = record.get("facts") or {}
    for k, v in f.items():
        if re.search(r"address", k, re.I) and not re.search(r"new|email", k, re.I) and isinstance(v, str):
            return v.strip()
    a = record.get("assigned") or {}
    for k in ("current_address", "registered_address"):
        if a.get(k):
            return a[k]
    return None


# ----------------------------------------------------------------------------------------- matching
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", compact(s))


def _digit_runs(s):
    return re.findall(r"\d+", compact(s))


def _unique(hits):
    hits = list({id(h): h for h in hits}.values())
    return hits[0] if len(hits) == 1 else None


_REG_WORDS = re.compile(r"\b(registered|register|primary|main|mera|meri|my|apna|own|same|isi|yahi|current|"
                        r"existing|on file|purana)\b", re.I)
_ALT_WORDS = re.compile(r"\b(alternate|alt|alternative|dusra|doosra|second|other|backup|daughter|son|wife|"
                        r"husband|family)\b", re.I)


def _out(value, rule, cands, ask=False):
    return {"value": value, "rule": rule, "ask": ask, "candidates": [c["value"] for c in cands]}


def resolve(kind, ref, record, agent_type=None):
    cands = candidates(kind, record)
    ref = "" if ref is None else str(ref).strip()
    nref = _norm(ref)
    if not cands:
        return _out(None, "no_candidates", cands, ask=True)

    if nref:
        # 1. exact
        hit = _unique([c for c in cands if _norm(c["value"]) == nref])
        if hit:
            return _out(hit["value"], "exact", cands)
        # 2. id / digits
        if kind in ID_KINDS:
            comp = compact(ref)
            runs = set(_digit_runs(ref))
            hit = _unique([c for c in cands if c["value"].lower() in comp])
            if not hit:
                hit = _unique([c for c in cands if re.sub(r"\D", "", c["value"]) in runs])
            if not hit and record.get("misread_id") and (record["misread_id"].lower() in comp):
                hit = _unique([c for c in cands if c["primary"]])
            if hit:
                return _out(hit["value"], "id", cands)
            if ID_RE.search(ref.upper().replace(" ", "")) and re.fullmatch(r"[a-z]{2}\d{4}", comp):
                return _out(None, "unknown_id", cands, ask=True)
        if kind == "card":
            d = re.sub(r"\D", "", compact(ref))
            if len(d) >= 4:
                hit = _unique([c for c in cands if c["value"] == d[-4:]])
                if hit:
                    return _out(hit["value"], "digits", cands)
                return _out(None, "unknown_card", cands, ask=True)
        if kind == "phone_on_file":
            d = re.sub(r"\D", "", compact(ref))
            if len(d) >= 4:
                hit = _unique([c for c in cands if re.sub(r"\D", "", c["value"]).endswith(d[-10:])
                               or (len(d) < 10 and re.sub(r"\D", "", c["value"]).endswith(d))])
                if hit:
                    return _out(hit["value"], "digits", cands)
                return _out(None, "unknown_number", cands, ask=True)
            if _ALT_WORDS.search(ref):
                hit = _unique([c for c in cands if c["role"] == "alternate"])
                return _out(hit["value"], "role:alternate", cands) if hit else \
                    _out(None, "no_alternate_on_record", cands, ask=True)
            if _REG_WORDS.search(ref) or re.search(r"\b(sim|number|phone)\b", ref, re.I):
                hit = _unique([c for c in cands if c["role"] == "registered"])
                if hit:
                    return _out(hit["value"], "role:registered", cands)
        # 3. N1 phrase rules (old types)
        if kind in OLD_ID_KINDS:
            r = n1.resolve(ref, record, agent_type)
            if r["id"] and not r["rule"].startswith(("default", "ambiguous")):
                hit = _unique([c for c in cands if c["value"] == r["id"]])
                if hit:
                    return _out(hit["value"], "n1:" + r["rule"], cands)
        # containment (plan/service/pack names said partially: "Cricket Alerts wala", "CineMax")
        if kind in ("plan", "service", "pack") and len(nref) >= 4:
            hit = _unique([c for c in cands if _norm(c["value"]) in nref or nref in _norm(c["value"])])
            if hit and _numbers_agree(ref, hit["value"]):
                return _out(hit["value"], "contains", cands)
        # 4. words
        hit = _match_words(ref, cands)
        if hit:
            return _out(hit["value"], "words", cands)
        # 5. ordinal
        k = n1._ordinal(ref)
        if k in ("latest",):
            hit = _unique([c for c in cands if c["primary"]])
            if hit:
                return _out(hit["value"], "ordinal:latest", cands)
        if k in ("past",):
            hit = _unique([c for c in cands if not c["primary"]])
            if hit:
                return _out(hit["value"], "ordinal:past", cands)
        if k == "return" and kind in ID_KINDS:
            hit = _unique([c for c in cands if re.search(r"\breturn\b", c["context"], re.I)])
            if hit:
                return _out(hit["value"], "ordinal:return", cands)
        # 7. unmatched
        # REVIEW-schema-1: an ID-shaped or digit-bearing ref that matched no candidate ("7138", "F D seven one
        # three eight", Devanagari digits of an ID not on the record) is a misheard/hallucinated ID -> ASK, never
        # the primary. Checked on compact(ref) (spoken digits converted). Amount/date refs have returned above.
        if kind in ID_KINDS and (re.search(r"[a-z]{2}\d{2,}", compact(ref)) or re.search(r"\d{3,}", compact(ref))):
            return _out(None, "unknown_id", cands, ask=True)
        if kind in OLD_ID_KINDS and not ID_RE.search(ref.upper()):
            hit = _unique([c for c in cands if c["primary"]])
            if hit:
                return _out(hit["value"], "default_primary_unmatched", cands)
        if len(cands) == 1 and kind in DEFAULT_PRIMARY_KINDS:
            return _out(cands[0]["value"], "default_single_unmatched", cands)
        return _out(None, "unmatched", cands, ask=True)

    # 6. empty ref
    if len(cands) == 1:
        return _out(cands[0]["value"], "default_single", cands)
    if kind in DEFAULT_PRIMARY_KINDS:
        hit = _unique([c for c in cands if c["primary"]])
        if hit:
            return _out(hit["value"], "default_primary", cands)
    return _out(None, "ambiguous", cands, ask=True)


def _numbers_agree(ref, value):
    a = set(re.findall(r"\d+", compact(ref)))
    b = set(re.findall(r"\d+", compact(value)))
    return not a or a <= b


def _match_words(ref, cands):
    rw = set(words(ref))
    if not rw:
        return None
    sets = [set(words(c["context"] + " " + c["value"])) for c in cands]
    common = set.intersection(*sets) if len(sets) > 1 else set()
    rnum = {w for w in rw if w.isdigit()}
    scores = []
    for c, s in zip(cands, sets):
        own = s - common if len(cands) > 1 else s
        if rnum and not (rnum & s):          # the number the customer said must be on this candidate
            scores.append(0)
            continue
        scores.append(len(rw & own))
    best = max(scores)
    if best == 0 or scores.count(best) != 1:
        return None
    return cands[scores.index(best)]


def resolve_address(value, record):
    """request_card_replacement.address: a new spoken address is passed through; omitted or 'registered /
    same address' -> the registered address on file; no address on file -> ASK."""
    v = (value or "").strip()
    if v and not re.fullmatch(r"(?i)\s*(registered|same|current|existing|old|purana|wahi|isi|on file)"
                              r"(\s+(wala|wali|address|pata|ghar))*\s*(address|pata)?\s*", v):
        return {"value": v, "rule": "new", "ask": False}
    a = registered_address(record)
    if a:
        return {"value": a, "rule": "registered_address", "ask": False}
    return {"value": None, "rule": "no_registered_address", "ask": True}
