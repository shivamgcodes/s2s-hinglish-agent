"""V4 variant (handoff D2 §4): the V3 recipe (novasynth guidance, Hinglish both sides, all-Devanagari text_tts) plus
per-call length bands, varied greeting/sign-off templates, read-tag validation, a no-recitation judge rule, a 'din' ban
and nukta checks in text_tts. Everything V4-specific lives here; generate.py / validate.py only call in when
variant == "V4", so V1/V2/V3/CONTROL/GATE1 prompts and checks are unchanged (verified by diffing --dry prompts and
check_call output before/after, NOTES "D2 / V4 generator").

Inputs (flags in generate.py / validate.py / gen_loop.py, defaults below):
  scenarios  guidance/scenario_creation_v2.json     records  data/V4/records.json
"""
import difflib
import json
import os
import re

_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
import statistics
import unicodedata

from common import DATA, ECHO_ARG, GEN, GUIDE, TOOL_ARGS, h01

V4_SCENARIOS = f"{GUIDE}/scenario_creation_v2.json"
V4_RECORDS = f"{DATA}/V4/records.json"
V4_OUT = f"{DATA}/V4"


# --------------------------------------------------------------------------- inputs
def load_scenarios(path=V4_SCENARIOS):
    """Same merge as common.load_scenarios() but from an explicit file (v2 adds agent types)."""
    if not os.path.exists(path):
        raise SystemExit(f"[V4] scenarios file missing: {path} (no fallback to the v1 file; pass --scenarios)")
    d = json.load(open(path))
    out = {}
    for a in d["agents"]:
        for s in a["scenarios"]:
            out[s["id"]] = {**s, "agent_type": a["agent_type"], "brand_context": a.get("brand_context", ""),
                            "static_read": a["tools"]["static_read"], "static_write_all": a["tools"]["static_write"],
                            "writes": s.get("tools") or []}
    missing = sorted({w for s in out.values() for w in s["writes"] if w not in TOOL_ARGS or w not in ECHO_ARG})
    if missing:
        raise SystemExit(f"[V4] write tools without common.TOOL_ARGS/ECHO_ARG entries: {missing} (add them first)")
    return out


def load_records(path=V4_RECORDS):
    if not os.path.exists(path):
        raise SystemExit(f"[V4] records file missing: {path} (no fallback to data/records.json; pass --records)")
    return json.load(open(path))


# --------------------------------------------------------------------------- call length (trainer window decision)
# Trainer duration_sec 140 s (D2 §4 preferred) -> real calls 70-130 s; if the trainer stays at 100 s -> 70-100 s.
# HINGLISH_V4_WINDOW = trainer window (140 default, or 100); decision recorded in NOTES "D2 / V4".
# The validator only has an ESTIMATE: on data/V3 real/estimate (f5 rates below) = 1.006 +/- 0.039 (max 1.135), so the
# estimate cap is real_max / 1.045 (mean + 1 sd): 130 -> 124 s, 100 -> 96 s. 130 s * (mean + 2 sd) would be 141 s.
WINDOW = int(os.environ.get("HINGLISH_V4_WINDOW", "140"))
if WINDOW not in (100, 140):
    raise SystemExit(f"HINGLISH_V4_WINDOW must be 140 or 100, got {WINDOW}")
REAL_MAX = 130 if WINDOW == 140 else 100
DUR_MAX = REAL_MAX
DUR = (70, int(REAL_MAX / 1.045))  # validated on est_duration (f5 rates)
TURNS = (10, 18)
SEC_PER_WORD = 0.432  # measured on data/V3: real call seconds / text_roman words over 287 calls (incl. gaps, lead)
# per-band target call seconds (inside DUR) -> word budgets in the prompt
BAND_SECONDS = {130: {"standard": (85, 110), "long": (95, 120), "mixed": (90, 115)},
                100: {"standard": (72, 88), "long": (78, 94), "mixed": (75, 92)}}[REAL_MAX]

# f5cs speaking rate in tts_word_units per second of turn audio (start..end, incl. intra-turn chunk gaps), measured
# on data/V3 stereo/*.json over all non-truncated turns; keyed by the Kokoro key stored in calls.jsonl `voices`
# (tts_backends.BACKEND_VOICES maps them to the f5 voices by gN).
F5_UNITS_PER_S = json.load(open(os.path.join(GEN, "f5_wps.json")))["units_per_s_by_kokoro_key"]

# --------------------------------------------------------------------------- length bands
BANDS = ("standard", "long", "mixed")
BAND_SPEC = {  # target agent / customer words, hard caps (agent, customer); hard floors stay 6 / 4 (as V3)
    "standard": {"agent": (10, 16), "customer": (7, 14), "cap": (18, 16), "turns": (15, 18)},
    "long": {"agent": (14, 22), "customer": (10, 18), "cap": (24, 20), "turns": (12, 16)},
    "mixed": {"agent_long": (15, 22), "agent_short": (6, 10), "customer_long": (12, 18), "customer_short": (4, 8),
              "cap": (24, 20), "turns": (14, 18)},
}
FLOOR = (6, 4)


def length_band(call_id):
    """40% standard / 40% long / 20% mixed by hash of call_id (reproducible)."""
    x = h01("v4_length_band", call_id)
    return "standard" if x < 0.4 else ("long" if x < 0.8 else "mixed")


def band_prompt(band):
    s = BAND_SPEC[band]
    lo, hi = BAND_SECONDS[band]
    wlo, whi = int(round(lo / SEC_PER_WORD, -1)), int(round(hi / SEC_PER_WORD, -1))
    t = s["turns"]
    common = (f" {t[0]}-{t[1]} turns in total. The whole call needs about {wlo}-{whi} words (about {lo}-{hi} seconds of "
              f"speech; calls outside {DUR[0]}-{DUR[1]} s are rejected). The greeting, check-lines, confirmations and sign-off "
              "keep their own length.")
    if band == "standard":
        return (f"LENGTH (this call: STANDARD lines): agent lines {s['agent'][0]}-{s['agent'][1]} words each (hard limits "
                f"{FLOOR[0]}-{s['cap'][0]}), customer lines {s['customer'][0]}-{s['customer'][1]} words each (hard limits "
                f"{FLOOR[1]}-{s['cap'][1]})." + common)
    if band == "long":
        return (f"LENGTH (this call: LONG lines): agent lines {s['agent'][0]}-{s['agent'][1]} words each (hard limits "
                f"{FLOOR[0]}-{s['cap'][0]}), customer lines {s['customer'][0]}-{s['customer'][1]} words each (hard limits "
                f"{FLOOR[1]}-{s['cap'][1]}). Make lines longer with natural phrasing (the customer explains, adds context "
                "or feelings; the agent acknowledges, restates the customer's question, or asks politely in full), "
                "NEVER by adding INFORMATION facts the customer did not ask about." + common)
    return (f"LENGTH (this call: MIXED lines): alternate long and short lines. Agent lines alternate between long "
            f"({s['agent_long'][0]}-{s['agent_long'][1]} words) and short ({s['agent_short'][0]}-{s['agent_short'][1]} "
            f"words); customer lines vary between long ({s['customer_long'][0]}-{s['customer_long'][1]}) and short "
            f"({s['customer_short'][0]}-{s['customer_short'][1]}). Plan the agent's own lines (not the greeting, "
            "check-lines, confirmations or sign-off) as long, short, long, short, ...: every second one is SHORT "
            f"({s['agent_short'][0]}-{s['agent_short'][1]} words, e.g. a quick acknowledgement plus the next question). "
            "Checked: at least 1 long and 1 short agent line, and at least 2 long and 2 short lines in the whole call. "
            f"Hard limits: agent {FLOOR[0]}-{s['cap'][0]}, customer {FLOOR[1]}-{s['cap'][1]} words. Long lines get longer "
            "with natural phrasing, never by adding INFORMATION facts the customer did not ask about." + common)


def _free_agent(turns):
    return [len(_words(t["text_roman"])) for t in turns if t["speaker"] == "agent" and not t.get("truncated")
            and not set(t["tags"]) & {"greeting", "check_line", "confirm_write", "signoff"}]


def _cust(turns):
    return [len(_words(t["text_roman"])) for t in turns if t["speaker"] == "customer" and not t.get("truncated")]


def _words(s):
    return [w for w in re.split(r"\s+", s.strip()) if re.search(r"[\wऀ-ॿ]", w)]


def band_checks(call):
    """(check, msg) list: hard line caps per band, and the band's shape on free lines."""
    F = []
    band = call.get("length_band")
    if band not in BAND_SPEC:
        return [("length_band", f"call has no valid length_band ({band})")]
    s = BAND_SPEC[band]
    for i, t in enumerate(call["turns"]):
        if "greeting" in (t.get("tags") or []):  # user 2026-10-04: fixed greeting template is exempt from line caps
            continue
        nw = len(_words(t["text_roman"]))
        lo, hi = (FLOOR[0], s["cap"][0]) if t["speaker"] == "agent" else (FLOOR[1], s["cap"][1])
        if nw > hi or (nw < lo and not t.get("truncated")):
            F.append(("line_words", f"turn {i} ({t['speaker']}) has {nw} words; need {lo}-{hi} ({band} band)"))
    fa, cu = _free_agent(call["turns"]), _cust(call["turns"])
    if not fa or not cu:
        return F
    ma, mc = statistics.median(fa), statistics.median(cu)
    if band == "standard":
        if not (9 <= ma <= 17):
            F.append(("length_band", f"standard band: median free agent line {ma} words; keep agent lines 10-16 words"))
        if not (6 <= mc <= 15):
            F.append(("length_band", f"standard band: median customer line {mc} words; keep customer lines 7-14 words"))
    elif band == "long":
        if ma < 14:
            F.append(("length_band", f"long band: median free agent line {ma} words < 14; agent lines need 14-22 words"))
        if mc < 10:
            F.append(("length_band", f"long band: median customer line {mc} words < 10; customer lines need 10-18 words"))
    else:
        # A call has only 2-4 free agent lines (greeting/check/confirm/sign-off excluded), so ">= 2 long AND >= 2 short
        # agent lines" was nearly unreachable (trial_gen_d2b: 1/3 mixed calls). Rule (D2 generator worker, 06:45 UTC):
        # free agent lines >= 1 long (>= 15) and >= 1 short (<= 11); over free agent + customer lines >= 2 long
        # (agent >= 15 / customer >= 12) and >= 2 short (agent <= 11 / customer <= 8).
        nl, ns = sum(n >= 15 for n in fa), sum(n <= 11 for n in fa)
        tl, ts = nl + sum(n >= 12 for n in cu), ns + sum(n <= 8 for n in cu)
        if nl < 1 or ns < 1 or tl < 2 or ts < 2:
            F.append(("length_band", f"mixed band: free agent lines {fa} ({nl} long >=15, {ns} short <=11), customer lines "
                      f"{cu}; need >= 1 long and >= 1 short agent line, and >= 2 long and >= 2 short lines in total "
                      "(customer long >= 12, short <= 8): alternate long and short lines"))
    return F


# --------------------------------------------------------------------------- greeting / sign-off templates
# Each greeting names brand + agent and has a gendered first-person verb (keeps gender_agent stable on no-write
# calls); none contains check/dekh/ek minute (read_no_check) or 'din'. {s}=sakti|sakta, {r}=rahi|raha.
GREETINGS = [
    "Hello, thank you for calling {brand}, this is {agent}. Main aapki kaise help kar {s} hoon?",
    "Namaste, {brand} mein aapka swagat hai, main {agent} bol {r} hoon. Bataiye, kya help chahiye?",
    "Hi, main {agent} {brand} customer support se baat kar {r} hoon. Boliye, kya madad kar {s} hoon?",
    "Hello ji, {brand} support mein aapka welcome hai, mera naam {agent} hai. Aaj main aapki kya help kar {s} hoon?",
    "Namaskar, {brand} se {agent} bol {r} hoon. Bataiye, main aapki kis cheez mein help kar {s} hoon?",
    "Hello, {brand} helpline, main {agent}. Bataiye, main aapke liye kya kar {s} hoon?",
]
# {sm}=sir|ma'am (customer gender), {p}=payi|paya (agent gender). No 'din', no promise, no new fact.
SIGNOFFS = [
    "Koi baat nahi {sm}, {brand} ko call karne ke liye thank you. Take care.",
    "Thank you {sm}, aur kuch help chahiye ho toh zaroor call kijiyega.",
    "Aapka bahut shukriya {sm}. Apna khayal rakhiye, bye bye.",
    "Ji {sm}, mujhe khushi hai ki main aapki help kar {p}. Thank you!",
    "Theek hai {sm}, thank you for your time. {brand} choose karne ke liye shukriya.",
    "Bilkul {sm}, aur koi sawaal ho toh dobara call kijiyega. Thank you, bye!",
]


def template_ids(call_id):
    return int(h01("v4_greeting", call_id) * len(GREETINGS)), int(h01("v4_signoff", call_id) * len(SIGNOFFS))


def greeting_text(k, brand, agent, agent_gender):
    f = agent_gender == "f"
    return GREETINGS[k].format(brand=brand, agent=agent, s="sakti" if f else "sakta", r="rahi" if f else "raha")


def signoff_text(k, brand, agent_gender, customer_gender):
    return SIGNOFFS[k].format(brand=brand, sm="ma'am" if customer_gender == "f" else "sir",
                              p="payi" if agent_gender == "f" else "paya")


def _sim(a, b):
    n = lambda x: re.sub(r"[^a-z0-9' ]", "", x.lower()).split()  # noqa: E731
    return difflib.SequenceMatcher(None, " ".join(n(a)), " ".join(n(b))).ratio()


def template_checks(call):
    F = []
    t = call["turns"]
    g = greeting_text(call["greeting_template"], call["brand"], call["agent_name"], call["agent_gender"])
    so = signoff_text(call["signoff_template"], call["brand"], call["agent_gender"], call["customer_gender"])
    if t and _sim(t[0]["text_roman"], g) < 0.8:
        F.append(("greeting_template", f"turn 0 must be the assigned greeting exactly: \"{g}\""))
    if t and _sim(t[-1]["text_roman"], so) < 0.8:
        F.append(("signoff_template", f"last turn must be the assigned sign-off: \"{so}\""))
    return F


# --------------------------------------------------------------------------- din ban (agent turns)
# Scope: AGENT turns only (loss is on agent text; a customer may say "do din se wait kar raha hoon").
DIN_RE = re.compile(r"\bdin\b", re.I)


def din_checks(call):
    return [("din", f"agent turn {i} uses 'din' ('{t['text_roman']}'); the agent never says 'din' (no 'nice din', "
             "'din achha rahe'); use the assigned sign-off") for i, t in enumerate(call["turns"])
            if t["speaker"] == "agent" and DIN_RE.search(t["text_roman"])]


# --------------------------------------------------------------------------- nukta checks on text_tts
# Lexicon of roman tokens whose sound needs a nukta letter in Devanagari. f/z: English words (+ a few very common
# Hindi-Urdu loans with f/z). ṛ: there are no English words with ṛ, so that list is Hindi words (+ Gurgaon/Udaan).
# Check per turn: #nukta letters of the class in NFD(text_tts) >= #lexicon tokens of the class in text_roman.
NUK = "़"
NUKTA_F = set("""for from flight flights refund refunds refunded phone phones headphones earphones confirm confirmed
confirmation perfect perfectly flat flats family safe safely safety unsafe food office fare fares fee fees free info
information feel feeling fries fried preference prefer preferred full fully after first fast final finally fine fix fixed
definitely specific confuse confused confusion gift gifts uncomfortable comfortable helpful verify verified
professional successfully frustrated frustration affected face traffic form forms file feedback few friday february friend
friends fresh fault failed fail offer offers coffee half staff transfer profile different difference floor field front
forward forgot forget fund funds future follow official lift left soft if off fit fruit fruits phase photo photos pharmacy
swift fan fans finance financial fraud fraudulent fourth fifth often sufficient refer referral
father shift shifted shifting scarf feature features inform informed successful notification notifications fryer
identify specify refresh further effect afternoon frustrating fish
kaafi kafi sirf taraf maaf saaf fikar fikr tareef""".split())
NUKTA_Z = set("""zero zone size sizes realize realise recognize prioritize organize apologize lazy crazy amazing amazon
pizza zip zoom dozen frozen citizen quiz prize magazine horizon blazer
zyada zaroor zaroori zaroorat cheez cheezein cheezon roz rozana bazaar mazaa awaaz aawaaz intezaar intezar zara zindagi
tez hazaar""".split())
NUKTA_R = set("""thoda thodi thode gaadi gaadiyan khada khadi khade bada badi bade padega padegi padenge pada padi pade
padta padti padh padhna padhai chhod chhoda chhodo chhodna chhodkar chhodiye jod jodo joda jodna ladka ladki sadak pakad
pakdo ghadi gadbad udaan badhiya badhia badh badhna chadh chadhna gurgaon""".split())
# user 2026-10-04: 'reference' removed from NUKTA_F (exempt; prompt still shows रेफ़रेंस)
NUKTA_CLASSES = (("f", NUKTA_F, ("फ" + NUK,), "फ़"), ("z", NUKTA_Z, ("ज" + NUK,), "ज़"),
                 ("r", NUKTA_R, ("ड" + NUK, "ढ" + NUK), "ड़/ढ़"))


def _rtoks(s):
    return [x for x in re.split(r"[^a-z'-]+", s.lower()) if x]


def nukta_checks(call):
    F = []
    for i, t in enumerate(call["turns"]):
        rt = [x.strip("'-") for x in _rtoks(t["text_roman"])]
        tts = unicodedata.normalize("NFD", t["text_tts"])
        for name, lex, letters, show in NUKTA_CLASSES:
            need = [w for w in rt if w in lex]
            have = sum(tts.count(L) for L in letters)
            if len(need) > have:
                F.append(("nukta", f"turn {i} text_tts has {have} {show} but text_roman has {len(need)} word(s) that need "
                          f"it: {need[:5]} (write {show} with the nukta dot, e.g. "
                          + {"f": "फ़ॉर, फ़्लाइट, रिफ़ंड, फ़ोन, कन्फ़र्म", "z": "ज़ीरो, साइज़, ज़्यादा, ज़रूर",
                             "r": "थोड़ा, गाड़ी, पड़ेगा, बढ़िया, गुड़गांव"}[name] + ")"))
    return F


# --------------------------------------------------------------------------- read tags
GENERIC = set("""your with from this that have order orders will been there they what please thank minutes minute hello
sure okay about just status delivery booking flight ride plan account payment customer registered email phone number
details detail rs item items amount date time older older new current via paid""".split())


def _info_facts(call):
    """(digit/ID tokens, content tokens) that count as INFORMATION facts for a read turn."""
    info = call["role_prompt"].split("Information:", 1)[-1]
    rec = call.get("record") or {}
    vals = list((rec.get("facts") or {}).values()) if isinstance(rec.get("facts"), dict) else \
        [x.get("value", "") for x in rec.get("facts") or []]
    vals += list(rec.get("distractors") or [])
    toks = lambda s: [x for x in re.split(r"[^a-z0-9]+", s.lower()) if x]  # noqa: E731
    dig = {x for x in toks(info) if re.search(r"\d", x) and x not in ("1", "2")}
    info_set = set(toks(info))
    from validate import STOP  # lazy: validate imports v4
    import hindi_share as hs
    hindi, _ = hs._lexicons()
    cont = {x for v in vals for x in toks(str(v)) if len(x) >= 3 and not re.search(r"\d", x) and x in info_set
            and x not in GENERIC and x not in STOP and x not in hindi}
    return dig, cont


def read_checks(call, check_like):
    """A turn tagged read: spoken by the agent, not also a check_line, contains >= 1 INFORMATION fact (a digit/ID token
    of Information, or a content word of a record fact value), and is not check-line-like (check_like(i) -> bool)."""
    F = []
    dig, cont = _info_facts(call)
    for i, t in enumerate(call["turns"]):
        if "read" not in t["tags"]:
            continue
        if t["speaker"] != "agent":
            F.append(("read_tag", f"turn {i} is a customer turn tagged read (only agent turns stating an INFORMATION fact)"))
            continue
        if "check_line" in t["tags"] or check_like(i):
            F.append(("read_tag", f"read turn {i} contains a check-line; a read turn states the fact directly"))
        tt = set(x for x in re.split(r"[^a-z0-9]+", t["text_roman"].lower()) if x)
        if not (tt & dig or tt & cont):
            F.append(("read_tag", f"turn {i} is tagged read but states no INFORMATION fact: '{t['text_roman']}' (tag it "
                      "read only when it states a fact from INFORMATION, copied exactly; otherwise empty tags)"))
    return F


# --------------------------------------------------------------------------- duration (f5cs rates)
def est_duration(call):
    """Like validate.est_duration but with measured f5cs units/s per voice instead of Kokoro wps."""
    from validate import tts_word_units  # lazy (validate imports v4)
    total = 1.0
    for i, t in enumerate(call["turns"]):
        total += tts_word_units(t["text_tts"]) / F5_UNITS_PER_S[call["voices"][t["speaker"]]]
        if i < len(call["turns"]) - 1:
            total += t.get("pause_after_s", 0.45)
        total -= t.get("overlap_offset_s", 0.0)
    return round(total, 1)


# --------------------------------------------------------------------------- prompt pieces
def lang_block(v3_lang):
    """V4 language block = a modified COPY of the rendered V3 language block (V3 text itself is untouched)."""
    old_len = "Hinglish lines run long: keep customer lines to at most 14 words and the call to at most 16 turns."
    assert old_len in v3_lang, "V3 language block changed; update v4.lang_block"
    out = v3_lang.replace(old_len, "Line lengths and the number of turns for THIS call are given under LENGTH at the end.")
    i = out.index("Sign-off: natural and short")
    out = out[:i] + SIGNOFF_RULE
    return out


SIGNOFF_RULE = ("Sign-off: use the sign-off given for this call at the end (exactly, after the customer says thanks/bye). "
                "It adds no new fact and no promise (nothing about when an order, ride, refund or update will arrive or "
                "happen). The AGENT never uses the word \"din\" (no \"have a nice din\", no \"aapka din achha rahe\").")

FACTS_EXTRA = ("- Answer only what is asked: the agent states the INFORMATION facts the customer's question or the current "
               "flow step needs, and NEVER recites other facts (an older order, the registered email, payment method, "
               "full item list, price) unless the customer asks for them. Restating a fact the customer just asked "
               "about is fine; reading out unasked facts is rejected.\n")

STRUCTURE = """STRUCTURE AND LENGTH
- Turn 1: the agent greeting, exactly the greeting given below (tag "greeting"). Last turn: the agent sign-off given below (tag "signoff"), after the customer says thanks/bye.
- Follow the scenario flow steps in order; expand each step into 1-3 turns. No backchannels ("hmm", "okay" alone) as separate turns.
- 10-18 turns in total; line lengths, turn count and word budget for THIS call are under LENGTH below. Confirmations echo the full value with a SHORT wrapper (e.g. "Done ji, the instruction is added: <value>."). Make the call full and natural by letting the customer explain their situation and ask follow-ups, not by the agent reciting INFORMATION.
- Tags: "greeting", "read" (agent turn that states a fact from INFORMATION; it must contain at least one INFORMATION fact copied exactly - an ID, number, time, date, status, name or item - and never a check-line), "check_line", "confirm_write", "signoff"; other turns have an empty tags list.
- "truncated": false on every turn unless this call has an interruption (see below)."""

NUKTA_RULE = (" NUKTA: write the nukta dot wherever the sound needs it (checked): English f -> फ़ (for -> फ़ॉर, from -> फ़्रॉम, "
              "flight -> फ़्लाइट, refund -> रिफ़ंड, phone -> फ़ोन, confirm -> कन्फ़र्म, perfect -> परफ़ेक्ट, office -> ऑफ़िस, "
              "food -> फ़ूड, flat -> फ़्लैट, Friday -> फ़्राइडे, family -> फ़ैमिली, safe -> सेफ़, info -> इन्फ़ो, reference -> रेफ़रेंस), z -> ज़ "
              "(zero -> ज़ीरो, size -> साइज़, zyada -> ज़्यादा, zaroor -> ज़रूर), and the Hindi flapped r -> ड़ / ढ़ "
              "(thoda -> थोड़ा, gaadi -> गाड़ी, padega -> पड़ेगा, badhiya -> बढ़िया, Gurgaon -> गुड़गांव). Hindi-Urdu words with f/z "
              "too: kaafi -> काफ़ी, taraf -> तरफ़, sirf -> सिर्फ़, maaf -> माफ़, saaf -> साफ़, cheez -> चीज़, roz -> रोज़. "
              "Hindi ph stays फ (phir -> फिर).")

RETRY_FIX = ("rewrite the lines named above (a speaker with too little Hindi -> natural spoken Hinglish with Hindi grammar "
             "words and verbs; a Hindi word in Latin letters in text_tts -> Devanagari; a missing nukta -> add it: फ़ ज़ ड़ ढ़), "
             "keep every line inside this call's LENGTH band (shorten over-long lines, lengthen short ones with natural "
             "phrasing, add or remove turns to fit the word budget), use the assigned greeting and sign-off exactly, no "
             "'din', no unasked INFORMATION facts, and tag read only on agent turns that state an INFORMATION fact. "
             "Keep text_tts in sync with text_roman.")


def patch_prefix(p):
    """Turn the rendered V3-style common prefix into the V4 one: STRUCTURE block replaced, unasked-facts rule added to
    FACTS AND TOOLS, nukta rule appended to the text_tts rules. Asserts every anchor so a V3 text change can't
    silently skip a patch."""
    a, b = p.index("STRUCTURE AND LENGTH\n"), None
    b = p.index("\n\n", a)
    p = p[:a] + STRUCTURE + p[b:]
    anchor = "- WRITE actions (only those listed"
    assert p.count(anchor) == 1
    p = p.replace(anchor, FACTS_EXTRA + anchor)
    anchor = " (This overrides the guidance's script rule: in text_tts English words are ALSO Devanagari.)"
    assert p.count(anchor) == 1
    p = p.replace(anchor, anchor + NUKTA_RULE)
    assert "190-240 words" not in p and "16 turns" not in p and "din achha" not in p.replace("\"aapka din achha rahe\"", "")
    return p


def call_block(call_id, rec, scen_s, ag, cg):
    """Per-call V4 additions: (greeting, extra lines appended after the flow/writes)."""
    agent = rec["agent_name_f"] if ag == "f" else rec["agent_name_m"]
    gk, sk = template_ids(call_id)
    band = length_band(call_id)
    greet = greeting_text(gk, rec["brand"], agent, ag)
    so = signoff_text(sk, rec["brand"], ag, cg)
    extra = [band_prompt(band), f"Sign-off (last turn, exactly): \"{so}\""]
    return greet, extra, {"length_band": band, "greeting_template": gk, "signoff_template": sk}


# --------------------------------------------------------------------------- judge (V4 only)
JUDGE_SCHEMA = {"type": "object", "properties": {"violations": {"type": "array", "items": {
    "type": "object", "properties": {"turn": {"type": "integer"}, "quote": {"type": "string"},
                                     "kind": {"type": "string", "enum": ["unsupported_fact", "dynamic_result",
                                                                         "false_capability", "wrong_fact", "unasked_fact"]},
                                     "reason": {"type": "string"}}, "required": ["turn", "quote", "kind", "reason"]}}},
    "required": ["violations"]}

UNASKED_RULE = ("- unasked_fact: the agent volunteers an INFORMATION fact that the customer did not ask about and the "
                "current scenario step does not need (e.g. reading out an older order, the registered email, the payment "
                "method, the full item list or the price when the customer only asked about the ETA or wants a change). "
                "Not this: a fact that answers the customer's question or the scenario step, repeating the ID/name/value "
                "the customer gave, the agent's confirmation of a requested action.\n")


def judge_prompt(v3_prompt, call):
    """V4 judge = the V3 judge text plus the scenario flow and the unasked_fact rule (V3 text untouched)."""
    anchor = "Do NOT flag:"
    assert anchor in v3_prompt
    flow = call.get("scenario_flow") or []
    head = ("SCENARIO STEPS (what this call is about):\n" + "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(flow)) + "\n\n"
            if flow else "")
    p = v3_prompt.replace("CALL:\n", head + "CALL:\n", 1)
    rep = "repeating INFORMATION or the customer's words"
    assert p.count(rep) == 1
    p = p.replace(rep, "repeating the customer's words or an INFORMATION fact the customer asked about", 1)
    return p.replace(anchor, UNASKED_RULE + anchor, 1)


def judge_keep(call, x):
    """Drop unasked_fact flags on greeting / confirm_write / signoff turns."""
    if x.get("kind") != "unasked_fact":
        return True
    return not set(call["turns"][x["turn"]]["tags"]) & {"greeting", "confirm_write", "signoff", "check_line"}
