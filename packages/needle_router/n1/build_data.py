"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle/build_data.py; edits: paths
from env (N1_SRC, HINGLISH_ROOT), hindi_share imported lazily (only the data build needs it; the runtime routers
use system_text / PINNED_DATE only, so the worker needs neither hindi_share nor its word lists).

N1 example construction (handoff N1 §2-3), renderings (a) text_roman and (c) synthetic noise.
Rendering (b) (ASR) is filled by another worker: every row carries renderings.b = None and the wav path(s).

Inputs (local copies of the pod files, see DECISIONS.md):
  src/V1_calls.jsonl, src/V3_calls.jsonl   <- /workspace/hinglish/data/{V1,V3}/calls.jsonl
  src/records.json, src/holdout.json        <- /workspace/hinglish/data/
  src/scenario_creation.json                <- /workspace/hinglish/guidance/
  hindi_share.py + lexicon/                 <- packages/hinglish_text (frozen counter, sha bf0c5834988a); the Google
                                               word lists it reads are NOT in the repo (env HINDI_SHARE_LEXICON)
Outputs:
  examples_raw.jsonl, DATA_STATS.md, tests_n0_relabelled.jsonl

Run: python3 build_data.py            (pure python, deterministic, no GPU, no network)
"""
import collections
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("N1_SRC") or os.path.join(HERE, "src")


class _LazyModule:
    """hindi_share, imported on first use: $HINDI_SHARE_DIR, packages/hinglish_text, or ./vendor (HF repo layout)."""
    _m = None

    def __getattr__(self, name):
        if _LazyModule._m is None:
            for d in (os.environ.get("HINDI_SHARE_DIR"), os.path.join(os.path.dirname(os.path.dirname(HERE)), "hinglish_text"),
                      os.path.join(HERE, "vendor")):
                if d and os.path.isfile(os.path.join(d, "hindi_share.py")):
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    break
            import hindi_share as m
            _LazyModule._m = m
        return getattr(_LazyModule._m, name)


class _LazySet:
    def __init__(self, fn):
        self._fn, self._s = fn, None

    def __contains__(self, x):
        if self._s is None:
            self._s = self._fn()
        return x in self._s


hindi_share = _LazyModule()

POD_DATA = os.path.join(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish"), "data")
PINNED_DATE = "date: 2026-10-04 Sun 10:00"
NEG_RATIO = 1.5
VARIANTS = ("V1", "V3")
EN_SHARE_MAX = 0.10  # hindi share < 0.10 -> "english", else "hinglish"


def h01(*parts):
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16) / 16 ** 12


# --------------------------------------------------------------------------- schema (new)
ID_ARGS = {"order_id", "reference_id", "ride_id", "pnr"}
ENTITY_ARGS = {"plan"}  # identifies the entity (subscription plan) -> becomes order_ref, value kept as gold
DEFAULT_REF = {"food_delivery_support": "my order", "ecommerce_support": "my order",
               "cab_ride_support": "my ride", "subscription_account_support": "my subscription",
               "airport_ticket_counter": "my booking"}

# tool intent keywords (used ONLY for: two-part fallback rule, intent-turn exclusion from negatives)
KW = {"cancel_order": r"cancel", "cancel_ride": r"cancel", "cancel_subscription": r"cancel|band kar|discontinue",
      "pause_subscription": r"pause|hold|rok", "request_ticket_cancellation": r"cancel",
      "request_refund": r"refund|paise wapas|paisa wapas|money back",
      "update_contact_number": r"\bnumber|contact|phone", "update_email_address": r"e-?mail",
      "change_delivery_address": r"address", "add_delivery_instruction": r"instruction|landmark|\bnote\b|rider ko|delivery (boy|partner|person) ko",
      "add_driver_instruction": r"instruction|\bnote\b|driver ko", "change_pickup_location": r"pick\s?-?up",
      "change_drop_location": r"\bdrop|destination"}

# agent turn that asks the customer for the value (two-part shape)
ASK_RE = re.compile(
    r"\?|\bbata(iye|ye|ie|o|\s*d(ijiye|o|ein|e))\b|\bbol(iye|o)\b|\btell me\b|\bshare\b|\bwhat\b|\bkya\b|"
    r"\bkaun\b|\bprovide\b|\blikhwa|\bdena chah|\bkahiye\b|\bgive me\b|\bplease (aap|apna|apni|apne|aapka|aapki|the|your)\b",
    re.I)


# --------------------------------------------------------------------------- helpers
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wav_path(V, call_id, i):
    return f"{POD_DATA}/{V}/work/utt/{call_id}/t{i:02d}.wav"


def digits(s):
    return re.sub(r"\D", "", s)


def toks(s):
    return [t.strip(".") for t in re.findall(r"[a-z0-9@.]+", s.lower()) if t.strip(".")]


def overlap(value, text):
    v = [t for t in toks(value) if len(t) > 1 or t.isdigit()]
    if not v:
        return 1.0
    if re.fullmatch(r"[\d\s+-]{8,}", value):  # phone: digit containment
        return 1.0 if digits(value) in digits(text) else 0.0
    T = set(toks(text))
    return sum(t in T for t in v) / len(v)


# --------------------------------------------------------------------------- order_ref extractor
STOP = set("""order orders from older old delivered rs the of and to on at in for with a an x1 x2 x3 x4 plan
plans pack gb black blue white silver red grey gray green set new previous past completed cancelled ride trip
flight return reference id account trial expired booking pnr""".split())
# never a single-word entity phrase on their own (allowed inside a longer phrase, e.g. "Premium Monthly")
GENERIC_SINGLE = set("""monthly annual yearly quarterly weekly january february march april may june july august
september october november december payment dispute premium""".split())
CITIES = _LazySet(lambda: set(hindi_share.PLACES) | {"delhi", "mumbai", "bangalore", "bengaluru", "jaipur", "goa", "pune",
                                                     "chennai", "kolkata", "hyderabad", "lucknow", "chandigarh", "ahmedabad"})
NAME_KEYS = ("restaurant", "product", "items", "item", "plan", "flight", "route", "store", "seller", "merchant",
             "airline", "brand")
IDRE = re.compile(r"\b([A-Z]{1,2})\s?(\d{4})\b")  # FD1565, AC 6424; N0 uses A1234
ORD_WORD = (r"pehl[ae]|pehli|pahl[ae]|dusr[ae]|doosr[ae]|dusri|first|second|latest|last|recent|current|purana|"
            r"purane|purani|older|old|naya|naye|nayi|new|abhi\s+wal[ae]|aaj\s+wal[ae]|wo|woh|vo|us|is|this|that")
ENT_NOUN = r"order|orders|ride|cab|booking|flight|ticket|plan|subscription|trip|parcel|delivery"
ORD_RE = re.compile(rf"\b(?:{ORD_WORD})\s+(?:wal[aei]|(?:\w+\s+)?(?:{ENT_NOUN}))\b", re.I)
ORD_STRICT = re.compile(r"^(?:is|us|wo|woh|vo|this|that)\b", re.I)  # demonstratives: only "<dem> <noun>"
POSS_MID = r"current|recent|latest|last|food|ek|booked|active|online|cab|upcoming|delhi|mumbai|purana|purani|old|new|naya|nayi"
POSS_RE = re.compile(rf"\b(?:my|mera|meri|mere|apna|apni|apne)\s+(?:(?:{POSS_MID})\s+)?(?:{ENT_NOUN})\b", re.I)
NEG_BEFORE = re.compile(r"(?:leave|chhodo|chodo|not|nahi|no)\s+(?:\w+\s+)?$", re.I)
WALA_RE = r"(?:\s+(?:ka|ki|ke))?\s+(?:wal[aei])\b"


def record_vocab(rec):
    """Entity-name phrases for this record: names of restaurants/products/items/plans/flights (+ distractor
    entities). Returns list of phrases (longest first). Agent brand, cities and address words excluded."""
    phrases = set()

    def add_name(s):
        s = re.sub(r"\bx\d+\b", " ", s)
        s = re.sub(r"\bRs\s*\d+", " ", s)
        for part in re.split(r"[,;:()]", s):
            words = [w for w in re.findall(r"[A-Za-z0-9!'&.-]+", part)]
            words = [w.strip(".") for w in words if w.strip(".")]
            n = len(words)
            for a in range(n):
                for b in range(a + 1, min(n, a + 4) + 1):
                    ph = words[a:b]
                    low = [w.lower() for w in ph]
                    if all(w in STOP or w.isdigit() or w in CITIES for w in low):
                        continue
                    if low[0] in STOP or low[-1] in STOP:
                        continue
                    if any(w in CITIES for w in low) and rec.get("agent_type") != "airport_ticket_counter":
                        continue
                    if len(ph) == 1 and (len(low[0]) < 3 or low[0].isdigit() or low[0] in GENERIC_SINGLE):
                        continue
                    if len(ph) == 1 and re.fullmatch(r"\d+[a-z]*", low[0]):
                        continue
                    phrases.add(" ".join(ph))

    for k, v in rec["facts"].items():
        kl = k.lower()
        if isinstance(v, str) and any(n in kl for n in NAME_KEYS) and "id" not in kl.split("_"):
            add_name(v)
    for d in rec.get("distractors", []):
        d = re.sub(r"\([^)]*\)", " ", d)
        if re.search(r"address|email|phone|seat|meal|baggage", d, re.I):
            continue
        m = re.search(r"\border\s+[A-Z]{2}\d{4}\s+from\s+([^:,]+?)\s*(?::|,|$)", d)
        if m:
            add_name(m.group(1))
        m = re.search(r"\border\s+[A-Z]{2}\d{4}(?:\s+from\s+[^:,]+?)?\s*:\s*([^,]+)", d)
        if m:
            add_name(m.group(1))
        m = re.search(r"\bplan:\s*([^,]+?)(?:\s+at\b|,|$)", d, re.I)
        if m:
            add_name(m.group(1))
    brand = rec["brand"].lower()
    out = [p for p in phrases if p.lower() != brand and brand not in p.lower().split()]
    return sorted(out, key=lambda p: (-len(p), p))


def skel(w):
    """Consonant skeleton for fuzzy name match: lower, drop apostrophes, z->s, ph->f, w->v, drop h, drop vowels
    and y after the first letter, collapse repeats."""
    w = re.sub(r"[^a-z0-9]", "", w.lower())
    if not w:
        return ""
    w = w.replace("ph", "f").replace("z", "s").replace("w", "v")
    w = w[0] + re.sub(r"[aeiouyh]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", w)


def fuzzy_ok(word, name_word):
    """Guard for a skeleton match: the spoken word is not a known Hindi word (galat/galti vs Galouti) and the
    normalised strings are >= 0.8 similar (difflib)."""
    import difflib
    a = re.sub(r"[^a-z0-9]", "", word.lower())
    b = re.sub(r"[^a-z0-9]", "", name_word.lower())
    if a in hindi_share._lexicons()[0]:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def mask_spans(text, spans):
    """Replace protected value spans (write args) with spaces of equal length (case-insensitive)."""
    masked = text
    for s in spans:
        if not s or len(s) < 4:
            continue
        for m in re.finditer(re.escape(s), masked, re.I):
            masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    return masked


def extract_order_ref(text, rec, agent_type, protect=()):
    """Returns (order_ref, rule, flags). Rules in priority order:
       id         explicit [A-Z]{2}dddd in the text (last one if several; flag multi_id)
       name       longest record entity phrase (restaurant/product/item/plan/flight/route words), extended by a
                  following 'wala/wale/wali' (or 'ka/ki/ke wala') -> rule name_wala
       ordinal    pehla/dusra/first/latest/last/purana/naya/current ... + wala|<entity noun>
       possessive my/mera/meri/mere/apna [+1 word] + order/ride/booking/flight/plan/subscription/...
       default    per agent type (my order / my ride / my subscription / my booking)
    """
    flags = []
    t = mask_spans(text, protect)
    t = hindi_share.PLATE_RE.sub(lambda m: " " * len(m.group(0)), t)  # vehicle plates are not entity IDs
    ids = list(IDRE.finditer(t))
    if ids:
        if len(ids) > 1:
            flags.append("multi_id")
        m = ids[-1]
        return text[m.start():m.end()], "id", flags
    low = t.lower()
    best = None
    for ph in record_vocab(rec):
        pl = ph.lower()
        if len(pl) < 3:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(pl) + r"(?:'s)?(?![a-z0-9])", low):
            cand = (m.end() - m.start(), -m.start(), m.start(), m.end())
            if best is None or cand[:2] > best[:2]:
                best = cand
        # loose plural / possessive: biryani -> biryanis? momo -> momos
    if best is None:  # fuzzy: consonant skeleton per word (dominos/dominoz ~ Domino's, biriyani ~ biryani)
        words = [(m.start(), m.end(), skel(m.group(0))) for m in re.finditer(r"[A-Za-z0-9'!]+", t)]
        for ph in record_vocab(rec):
            pw = [skel(w) for w in ph.split()]
            if any(len(x) < 3 or x.isdigit() for x in pw):
                continue
            n = len(pw)
            for a in range(len(words) - n + 1):
                win = words[a:a + n]
                if [w[2] for w in win] == pw and all(fuzzy_ok(t[w[0]:w[1]], p) for w, p in zip(win, ph.split())):
                    cand = (words[a + n - 1][1] - words[a][0], -words[a][0], words[a][0], words[a + n - 1][1])
                    if best is None or cand[:2] > best[:2]:
                        best = cand
        fuzzy = best is not None
    else:
        fuzzy = False
    if best is not None:
        s, e = best[2], best[3]
        w = re.match(WALA_RE, low[e:])
        rule = "name"
        if w:
            e += w.end()
            rule = "name_wala"
        if fuzzy:
            rule += "_fuzzy"
        return text[s:e], rule, flags
    for m in ORD_RE.finditer(t):
        span = text[m.start():m.end()]
        if ORD_STRICT.match(span) and re.search(r"wal[aei]$", span, re.I):
            continue  # "is wala" etc. too generic
        if NEG_BEFORE.search(t[:m.start()]):
            continue  # "leave the old plan": the customer is excluding that entity
        if re.match(r"(?:naya|naye|nayi|new)\s+(?:delivery)\b", span, re.I):
            continue
        if re.search(r"\bdelivery$", span, re.I):
            continue
        return span, ("demonstrative" if ORD_STRICT.match(span) else "ordinal"), flags
    m = POSS_RE.search(t)
    if m and not re.search(r"\bdelivery$", text[m.start():m.end()], re.I):
        return text[m.start():m.end()], "possessive", flags
    return DEFAULT_REF[agent_type], "default", flags


# --------------------------------------------------------------------------- system facts
def user_fact(rec):
    """Compact `user:` fact. Needle splits facts on ';' and detects a date key with (?<![A-Za-z])date:, so the
    value uses commas only, never 'xxx_date:' keys, never ISO stamps. Content = the record's Information prose
    (exactly what the role prompt carries) + customer phone."""
    info = rec["information"].replace(";", ",").strip().rstrip(".")
    info = re.sub(r"(?i)date:", "date", info)
    return f"user: customer phone {rec['phone']}, {info}"


def system_text(rec):
    s = f"{PINNED_DATE}; {user_fact(rec)}"
    assert s.count("date:") == 1 and not re.search(r"\d{4}-\d{2}-\d{2}", s[len(PINNED_DATE):]), s
    return s


# --------------------------------------------------------------------------- target build
def build_target(w, rec, agent_type, text):
    args = dict(w["args"])
    gold = {"entity_id": None, "entity_arg": None, "plan": None, "args_raw": dict(w["args"])}
    for k in list(args):
        if k in ID_ARGS:
            gold["entity_id"] = args.pop(k)
            gold["entity_arg"] = k
        elif k in ENTITY_ARGS:
            gold["plan"] = args.pop(k)
            gold["entity_arg"] = k
    if gold["entity_id"] is None:
        gold["entity_id"] = rec["primary_id"]
    gold["entity_is_primary"] = gold["entity_id"] == rec["primary_id"]
    if "phone" in args:
        args["phone"] = digits(args["phone"])
    value_args = [str(v) for k, v in w["args"].items() if k not in ID_ARGS and k not in ENTITY_ARGS]
    ref, rule, flags = extract_order_ref(text, rec, agent_type, protect=value_args)
    call = {"name": w["tool"], "arguments": {"order_ref": ref, **args}}
    return call, gold, rule, flags


# --------------------------------------------------------------------------- rendering (c)
ARTICLES = {"the", "a", "an"}
MISSPELL_OPS = [  # (name, regex, replacement) applied once to the chosen Hindi word
    ("aa->a", r"aa", "a"), ("ee->i", r"ee", "i"), ("oo->u", r"oo", "u"), ("ai$->e", r"ai$", "e"),
    ("h-drop", r"(?<=[kgcjtdpb])h", ""), ("final-a->aa", r"(?<=[^aeiou])a$", "aa"), ("w->v", r"w", "v"),
    ("v->w", r"v", "w"), ("z->j", r"z", "j"), ("i$->ee", r"(?<=[^aeiou])i$", "ee"), ("iye->ie", r"iye", "ie"),
    ("double-consonant", r"([bcdfgjklmnprstvz])(?=[aeiou])", r"\1\1"),
]


def protected_mask(text, protect):
    """Boolean per char: True inside a protected span (digits, IDs, emails, order_ref, arg values)."""
    mask = [False] * len(text)

    def mark(s, e):
        for k in range(s, e):
            mask[k] = True

    for m in re.finditer(r"\S*\d\S*|\S+@\S+", text):
        mark(m.start(), m.end())
    for p in protect:
        if p and len(p) >= 2:
            for m in re.finditer(re.escape(p), text, re.I):
                mark(m.start(), m.end())
    return mask


def render_c(text, example_id, protect, exclude):
    """Deterministic noise, seeded by sha256(example_id): (1) drop every article the/a/an, (2) merge one pair of
    adjacent words, (3) misspell one Hindi word (hindi_share label H). Protected spans are never touched."""
    ops = []
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    mask = protected_mask(text, protect)
    prot = [any(mask[s:e]) for _, s, e in words]
    core = [re.sub(r"^[^\w']+|[^\w']+$", "", w).lower() for w, _, _ in words]
    # (1) drop articles
    keep = []
    for k, (w, s, e) in enumerate(words):
        if core[k] in ARTICLES and not prot[k] and w.lower() == core[k]:
            ops.append(f"drop:{w}")
            continue
        keep.append(k)
    # (3) misspell one Hindi word (choose before merging so the label lookup sees whole words)
    labels = dict()
    for tok, lab in hindi_share.classify_line(text, exclude):
        labels.setdefault(tok.lower(), lab)
    out = {k: words[k][0] for k in keep}
    hi = [k for k in keep if not prot[k] and labels.get(core[k]) == "H" and len(core[k]) >= 3]
    if hi:
        r = h01(example_id, "c", "mis")
        start = int(r * len(hi))
        done = False
        for d in range(len(hi)):
            k = hi[(start + d) % len(hi)]
            w = words[k][0]
            m = re.match(r"^([^\w']*)([\w']+)(.*)$", w)
            if not m:
                continue
            pre, body, post = m.groups()
            o0 = int(h01(example_id, "c", "op", str(k)) * len(MISSPELL_OPS))
            for j in range(len(MISSPELL_OPS)):
                name, pat, rep = MISSPELL_OPS[(o0 + j) % len(MISSPELL_OPS)]
                nb = re.sub(pat, rep, body, count=1)
                if nb != body:
                    out[k] = pre + nb + post
                    ops.append(f"misspell:{body}->{nb}({name})")
                    done = True
                    break
            if done:
                break
        if not done:
            ops.append("misspell:none_applicable")
    else:
        ops.append("misspell:no_hindi_word")
    # (2) merge one adjacent pair (both unprotected, both alphabetic, first has no trailing punctuation)
    pairs = []
    for a, b in zip(keep, keep[1:]):
        wa, wb = out[a], out[b]
        if prot[a] or prot[b]:
            continue
        if not re.fullmatch(r"[A-Za-z']+", wa) or not re.match(r"[A-Za-z']+", wb):
            continue
        pairs.append((a, b))
    merged_into = {}
    if pairs:
        a, b = pairs[int(h01(example_id, "c", "merge") * len(pairs))]
        merged_into[a] = out[a] + out[b]
        ops.append(f"merge:{out[a]}+{out[b]}")
        skip_b = b
    else:
        skip_b = None
        ops.append("merge:none")
    res = []
    for k in keep:
        if k == skip_b:
            continue
        res.append(merged_into.get(k, out[k]))
    return " ".join(res), ops


# --------------------------------------------------------------------------- main construction
def twopart_assigned(scen_writes):
    """Re-implementation of gen/common.assignments() two-part set (keyed on call_id only)."""
    ECHO = {"cancel_order": "order_id", "change_delivery_address": "address", "add_delivery_instruction": "instruction",
            "update_contact_number": "phone", "request_refund": "reference_id", "cancel_ride": "ride_id",
            "change_pickup_location": "location", "change_drop_location": "location",
            "add_driver_instruction": "instruction", "cancel_subscription": "plan", "pause_subscription": "plan",
            "update_email_address": "email", "request_ticket_cancellation": "pnr"}
    ids = [f"{sid}_{g}" for sid in scen_writes for g in ("g1", "g2", "g3", "g4")]
    wids = [c for c in ids if scen_writes[c.rsplit("_", 1)[0]]
            and ECHO[scen_writes[c.rsplit("_", 1)[0]][0]] in ("address", "location", "instruction", "phone", "email")
            and c.rsplit("_", 1)[0] not in ("food_08", "ecom_13")]
    n_two = round(0.20 * sum(1 for c in ids if scen_writes[c.rsplit("_", 1)[0]]))
    return set(sorted(wids, key=lambda c: h01("twopart", c))[:n_two])


def input_block(T, i, prev_confirm, tool):
    """Customer turn indices forming the router input for the write whose value turn is i.
    Returns (indices, shape)."""
    block = [i]
    shape = ["single"]
    j = i - 1
    # consecutive customer turns (interruptions / back-to-back customer lines) belong to the same buffer
    while j > prev_confirm and T[j]["speaker"] == "customer":
        block.insert(0, j)
        j -= 1
        if "consecutive" not in shape:
            shape.append("consecutive")
    if j > prev_confirm + 1 and T[j]["speaker"] == "agent" \
            and not ({"check_line", "confirm_write", "greeting"} & set(T[j]["tags"])) \
            and T[j - 1]["speaker"] == "customer":
        asks = bool(ASK_RE.search(T[j]["text_roman"]))
        val_txt = " ".join(T[k]["text_roman"] for k in block)
        intent_prev = bool(re.search(KW[tool], T[j - 1]["text_roman"], re.I))
        intent_here = bool(re.search(KW[tool], val_txt, re.I))
        if asks or (intent_prev and not intent_here):
            k = j - 1
            add = [k]
            while k - 1 > prev_confirm and T[k - 1]["speaker"] == "customer":
                k -= 1
                add.insert(0, k)
            block = add + block
            shape[0] = "twopart_ask" if asks else "twopart_intentkw"
    return block, "+".join(shape)


def main():
    records = json.load(open(os.path.join(SRC, "records.json")))
    holdout = json.load(open(os.path.join(SRC, "holdout.json")))
    scen = json.load(open(os.path.join(SRC, "scenario_creation.json")))
    test_scen = set(holdout["test_scenarios"])
    test_calls = set(holdout["test_calls"])
    train_scen = set(holdout["train_scenarios"])
    scen_writes = {}
    scen_type = {}
    for a in scen["agents"]:
        for s in a["scenarios"]:
            scen_writes[s["id"]] = s.get("tools", [])
            scen_type[s["id"]] = a["agent_type"]
    # val scenarios: per agent type, 1 write-bearing training scenario (lowest h01('val', sid))
    val_scen = set()
    for at in sorted(set(scen_type.values())):
        cands = sorted([s for s in train_scen if scen_type[s] == at and scen_writes[s]], key=lambda s: h01("val", s))
        val_scen.add(cands[0])

    def split_of(sid):
        return "test" if sid in test_scen else ("val" if sid in val_scen else "train")

    tp_set = twopart_assigned(scen_writes)
    wavs = set(l.strip() for l in open(os.path.join(SRC, "utt_wav_list.txt")))

    rows = []
    neg_cands = []
    audit = collections.Counter()
    tp_check = collections.Counter()
    excluded_intent = []
    lowfit = []
    for V in VARIANTS:
        for c in load_jsonl(os.path.join(SRC, f"{V}_calls.jsonl")):
            sid, cid, at = c["scenario_id"], c["call_id"], c["agent_type"]
            rec = records[sid]
            T = c["turns"]
            exclude = hindi_share.build_exclude(c.get("record"), [c.get("agent_name", ""), c.get("brand", "")])
            sysx = system_text(rec)
            used = set()
            window = set()
            intent_turns = set()
            prev_confirm = -1
            per_write = []
            for k, w in enumerate(c["writes"]):
                i = w["confirm_turn_idx"] - 2
                assert T[i]["speaker"] == "customer" and "check_line" in T[i + 1]["tags"], (V, cid, k)
                block, shape = input_block(T, i, prev_confirm, w["tool"])
                for x in range(prev_confirm + 1, i):
                    if T[x]["speaker"] == "customer":
                        window.add(x)
                        if x not in block and re.search(KW[w["tool"]], T[x]["text_roman"], re.I):
                            intent_turns.add(x)
                used.update(block)
                if k == 0 and cid in tp_set:
                    tp_check["assigned_twopart_first_write"] += 1
                    tp_check["...detected_twopart" if shape.startswith("twopart") else "...NOT_detected"] += 1
                    if not shape.startswith("twopart"):
                        tp_check[f"missed:{V}:{cid}"] += 1
                per_write.append((k, w, block, shape))
                prev_confirm = w["confirm_turn_idx"]
            # multi-call: two writes whose input blocks share a customer turn
            shared = collections.Counter(x for _, _, b, _ in per_write for x in b)
            if any(v > 1 for v in shared.values()):
                audit["multi_call_shared_turn"] += 1
            for k, w, block, shape in per_write:
                text = " ".join(T[x]["text_roman"].strip() for x in block)
                call, gold, rule, flags = build_target(w, rec, at, text)
                ref = call["arguments"]["order_ref"]
                if rule != "default":
                    assert ref.lower() in text.lower(), (cid, ref, text)
                fit = {a: round(overlap(str(v), text), 2) for a, v in w["args"].items()
                       if a not in ID_ARGS and a not in ENTITY_ARGS}
                if any(v < 0.6 for v in fit.values()):
                    lowfit.append((V, cid, k, w["tool"], fit, text))
                share = hindi_share.share([text], exclude)
                ex_id = f"{V}:{cid}:w{k}"
                protect = [ref] + [str(v) for v in w["args"].values()]
                c_text, c_ops = render_c(text, ex_id, protect, exclude)
                rows.append({
                    "example_id": ex_id, "type": "positive", "call_id": cid, "variant": V, "scenario_id": sid,
                    "agent_type": at, "split": split_of(sid), "is_test_call": cid in test_calls,
                    "language": "english" if share < EN_SHARE_MAX else "hinglish", "hindi_share": share,
                    "turn_idx": block, "value_turn_idx": block[-1], "confirm_turn_idx": w["confirm_turn_idx"],
                    "shape": shape, "twopart_assigned_call": cid in tp_set, "write_idx": k,
                    "wav": [wav_path(V, cid, x) for x in block],
                    "system": sysx,
                    "renderings": {"a": text, "b": None, "c": c_text}, "c_ops": c_ops,
                    "answers": [call],
                    # tools.py convention (order_ref OPTIONAL, omitted when the customer did not name the entity)
                    "answers_omit_default": [{"name": call["name"], "arguments": {
                        k: v for k, v in call["arguments"].items() if not (k == "order_ref" and rule == "default")}}],
                    "order_ref_rule": rule, "order_ref_flags": flags,
                    "gold": gold, "arg_text_overlap": fit,
                    "args_grounded": all(v >= 0.6 for v in fit.values()),
                })
            for x, t in enumerate(T):
                if t["speaker"] != "customer" or x in used:
                    continue
                if x in intent_turns:
                    excluded_intent.append((V, cid, x, t["text_roman"]))
                    continue
                text = t["text_roman"].strip()
                ex_id = f"{V}:{cid}:t{x:02d}"
                share = hindi_share.share([text], exclude)
                c_text, c_ops = render_c(text, ex_id, [], exclude)
                neg_cands.append({
                    "example_id": ex_id, "type": "negative", "call_id": cid, "variant": V, "scenario_id": sid,
                    "agent_type": at, "split": split_of(sid), "is_test_call": cid in test_calls,
                    "language": "english" if share < EN_SHARE_MAX else "hinglish", "hindi_share": share,
                    "turn_idx": [x], "value_turn_idx": x, "confirm_turn_idx": None,
                    "shape": "single", "twopart_assigned_call": cid in tp_set, "write_idx": None,
                    "wav": [wav_path(V, cid, x)], "system": sysx,
                    "renderings": {"a": text, "b": None, "c": c_text}, "c_ops": c_ops,
                    "answers": [], "answers_omit_default": [], "order_ref_rule": None, "order_ref_flags": [], "gold": None,
                    "arg_text_overlap": None, "args_grounded": None,
                    "in_write_window": x in window,
                    "neg_kind": "greeting_or_first" if x <= 1 else ("after_write" if x > prev_confirm >= 0 else "other"),
                })
    # negative subsample 1:1.5 per (split, variant), seeded by sha256 of example_id
    npos = collections.Counter((r["split"], r["variant"]) for r in rows)
    by = collections.defaultdict(list)
    for r in neg_cands:
        by[(r["split"], r["variant"])].append(r)
    for key, lst in by.items():
        lst.sort(key=lambda r: h01("neg", r["example_id"]))
        n = round(NEG_RATIO * npos[key])
        for j, r in enumerate(lst):
            r["neg_selected"] = j < n
    for r in rows:
        r["neg_selected"] = None
    for r in rows + neg_cands:
        missing = [p for p in r["wav"] if p.replace(POD_DATA + "/", "") not in wavs]
        assert not missing, missing
    allrows = rows + neg_cands
    with open(os.path.join(HERE, "examples_raw.jsonl"), "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_stats(allrows, tp_check, excluded_intent, lowfit, val_scen, audit)
    relabel_n0()
    print("positives", len(rows), "negative candidates", len(neg_cands),
          "selected", sum(r["neg_selected"] for r in neg_cands))


# --------------------------------------------------------------------------- stats
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_stats(allrows, tp_check, excluded_intent, lowfit, val_scen, audit):
    pos = [r for r in allrows if r["type"] == "positive"]
    neg = [r for r in allrows if r["type"] == "negative"]
    sel = [r for r in neg if r["neg_selected"]]
    L = ["# N1 data stats (examples_raw.jsonl)", "",
         "Generated by build_data.py. One row per source example; each row carries renderings a (text_roman), "
         "b (ASR, null here, filled by the ASR worker) and c (synthetic noise). Counts are source examples "
         "(multiply by the number of renderings used).", ""]
    L.append(f"Positives: {len(pos)}. Negative candidates: {len(neg)}; selected at 1:{NEG_RATIO} per "
             f"(split, variant): {len(sel)}. Multi-call examples: 0 "
             f"(no two writes share a customer input turn in V1 or V3; shared-turn calls found: "
             f"{audit['multi_call_shared_turn']}).")
    L.append("")
    L.append(f"Val scenarios (1 write-bearing training scenario per agent type): {sorted(val_scen)}")
    L.append("")
    L.append("## By split x variant")
    rows = []
    for sp in ("train", "val", "test"):
        for V in VARIANTS:
            p = sum(1 for r in pos if r["split"] == sp and r["variant"] == V)
            nc = sum(1 for r in neg if r["split"] == sp and r["variant"] == V)
            ns = sum(1 for r in sel if r["split"] == sp and r["variant"] == V)
            sc = len({r["scenario_id"] for r in allrows if r["split"] == sp and r["variant"] == V})
            ca = len({r["call_id"] for r in allrows if r["split"] == sp and r["variant"] == V})
            rows.append((sp, V, sc, ca, p, nc, ns))
    L.append(md_table(["split", "variant", "scenarios", "calls", "positives", "neg candidates", "neg selected"], rows))
    L.append("")
    L.append("Test = all g1-g4 calls of the 25 held-out scenarios (holdout.json test_scenarios); the 25 "
             "holdout test_calls subset is flagged is_test_call: "
             f"{sum(1 for r in pos if r['is_test_call'])} positives, "
             f"{sum(1 for r in neg if r['is_test_call'])} negative candidates.")
    L.append("")
    L.append("## By tool x split (positives, V1+V3)")
    tc = collections.Counter((r["answers"][0]["name"], r["split"]) for r in pos)
    tools = sorted({t for t, _ in tc})
    L.append(md_table(["tool", "train", "val", "test", "total"],
                      [(t, tc[(t, "train")], tc[(t, "val")], tc[(t, "test")],
                        sum(tc[(t, s)] for s in ("train", "val", "test"))) for t in tools]))
    L.append("")
    L.append("## By agent type x split x type")
    ac = collections.Counter((r["agent_type"], r["split"], r["type"], bool(r["neg_selected"]) if r["type"] == "negative" else True) for r in allrows)
    ats = sorted({r["agent_type"] for r in allrows})
    L.append(md_table(["agent type", "train pos", "train neg sel", "val pos", "val neg sel", "test pos", "test neg sel"],
                      [(a, ac[(a, "train", "positive", True)], ac[(a, "train", "negative", True)],
                        ac[(a, "val", "positive", True)], ac[(a, "val", "negative", True)],
                        ac[(a, "test", "positive", True)], ac[(a, "test", "negative", True)]) for a in ats]))
    L.append("")
    L.append("## Language (hindi_share of the input text_roman < 0.10 -> english)")
    lc = collections.Counter((r["type"], r["variant"], r["language"]) for r in pos + sel)
    L.append(md_table(["type", "variant", "english", "hinglish"],
                      [(t, V, lc[(t, V, "english")], lc[(t, V, "hinglish")]) for t in ("positive", "negative") for V in VARIANTS]))
    L.append("(negatives: selected only)")
    L.append("")
    L.append("## Input shape (positives)")
    sc = collections.Counter((r["shape"], r["variant"]) for r in pos)
    L.append(md_table(["shape", "V1", "V3"], [(s, sc[(s, "V1")], sc[(s, "V3")]) for s in sorted({s for s, _ in sc})]))
    L.append("")
    L.append("Two-part check against the generator's assignment (gen/common.assignments re-implemented): "
             + ", ".join(f"{k}={v}" for k, v in sorted(tp_check.items())))
    L.append("")
    L.append("## order_ref rule histogram (positives)")
    rc = collections.Counter((r["order_ref_rule"], r["split"]) for r in pos)
    rules = sorted({k for k, _ in rc}, key=lambda k: -sum(rc[(k, s)] for s in ("train", "val", "test")))
    L.append(md_table(["rule", "train", "val", "test", "total"],
                      [(k, rc[(k, "train")], rc[(k, "val")], rc[(k, "test")], sum(rc[(k, s)] for s in ("train", "val", "test"))) for k in rules]))
    L.append("")
    fl = collections.Counter(f for r in pos for f in r["order_ref_flags"])
    L.append(f"Extractor flags: {dict(fl)}")
    L.append("")
    rbt = collections.Counter((r["order_ref_rule"], r["agent_type"]) for r in pos)
    L.append(md_table(["rule"] + ats, [(k, *[rbt[(k, a)] for a in ats]) for k in rules]))
    L.append("")
    refc = collections.Counter(r["answers"][0]["arguments"]["order_ref"] for r in pos)
    L.append("Most common order_ref values: " + ", ".join(f"`{k}` {v}" for k, v in refc.most_common(25)))
    L.append("")
    L.append(f"Gold entity id is NOT the record primary_id in {sum(1 for r in pos if not r['gold']['entity_is_primary'])} positives "
             "(refund/cancel on the secondary/distractor reference).")
    L.append("")
    L.append("## Negatives")
    nk = collections.Counter((r["neg_kind"], bool(r["neg_selected"])) for r in neg)
    L.append(md_table(["neg kind", "candidates", "selected"], [(k, nk[(k, True)] + nk[(k, False)], nk[(k, True)]) for k in sorted({k for k, _ in nk})]))
    L.append(f"\nIn a write window (between the previous confirm and the value turn): "
             f"{sum(1 for r in neg if r['in_write_window'])} candidates, {sum(1 for r in sel if r['in_write_window'])} selected.")
    L.append(f"\nCustomer turns excluded from negatives as un-joined intent turns (inside a write window, contain the "
             f"tool's intent keyword, not part of the input block): {len(excluded_intent)}.")
    L.append("")
    L.append("## Rendering (c) ops")
    oc = collections.Counter(o.split(":")[0] + (":" + o.split(":")[1] if o.split(":")[1] in ("none", "no_hindi_word", "none_applicable") else "")
                             for r in pos + sel for o in r["c_ops"])
    L.append(md_table(["op", "count"], sorted(oc.items())))
    L.append("")
    L.append(f"## Positives whose non-ID args have < 0.6 token overlap with the input text: {len(lowfit)}")
    for V, cid, k, tool, fit, text in lowfit:
        L.append(f"- {V} {cid} w{k} {tool} {fit}: {text}")
    open(os.path.join(HERE, "DATA_STATS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- N0 relabel
N0_REC = {  # N0 system facts as a pseudo-record for the extractor
    "agent_type": "food_delivery_support", "brand": "Swiggy-N0", "primary_id": "B5678",
    "facts": {"restaurant": "Biryani Blues", "restaurant_2": "Domino's", "items": "biryani, pizza"},
    "distractors": [],
}


def relabel_n0():
    src = os.path.join(SRC, "n0_tests.jsonl")  # copy of needle-exp0/tests.jsonl
    out = []
    for it in load_jsonl(src):
        calls = []
        for e in it["expected_calls"]:
            a = dict(e["arguments"])
            oid = a.pop("order_id")
            if "note" in a:
                a["instruction"] = a.pop("note")
            if "phone" in a:
                a["phone"] = digits(a["phone"])
            calls.append({"name": e["name"], "arguments": a, "expected_entity_id": oid})
        refs = []
        if calls:
            # one order_ref per call: the phrase in that call's clause; multi items split on ' and '/' aur '
            parts = [it["text"]]
            if len(calls) > 1:
                parts = re.split(r"\s+(?:and also|and|aur)\s+(?=\w)", it["text"], maxsplit=1)
                if len(parts) != len(calls):
                    parts = [it["text"]] * len(calls)
            for c, p in zip(calls, parts):
                protect = [v for k, v in c["arguments"].items()]
                ref, rule, flags = extract_order_ref(p, N0_REC, "food_delivery_support", protect=protect)
                if rule == "default" and len(calls) > 1:  # "... and raise a refund for it" -> same order as clause 1
                    ref, rule = refs[-1][0], "coref_previous_clause"
                refs.append((ref, rule))
        rec = {"id": it["id"], "text": it["text"], "lang": it["lang"], "kind": it["kind"],
               "system": f"{PINNED_DATE}; user: Customer has two active orders, order_id A1234 from Biryani Blues "
                         "placed 20 minutes ago (status preparing), order_id B5678 from Domino's placed 5 minutes ago "
                         "(status placed). Default address Sector 15, Faridabad",
               "expected_calls": [{"name": c["name"], "arguments": {"order_ref": r[0], **c["arguments"]}}
                                  for c, r in zip(calls, refs)],
               "expected_entity_ids": [c["expected_entity_id"] for c in calls],
               "order_ref_rules": [r[1] for r in refs],
               "n0_expected_calls": it["expected_calls"]}
        out.append(rec)
    with open(os.path.join(HERE, "tests_n0_relabelled.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
