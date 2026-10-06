"""Validation before audio (HANDOFF §3). Deterministic checks are pure python; judge_calls() needs an LLM.

Standalone:  python validate.py data/V1/calls.jsonl [--judge]   -> per-check pass rates
"""
import argparse
import difflib
import json
import math
import os
import re

_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
from collections import Counter

import hindi_share as hs
import v4
from common import (CHECK_LINES, CHECK_LINES_EN, CONTROL_MAX_HINDI, ECHO_ARG, GEN, PAIRINGS, REGIMES, SHARE_BANDS,
                    V3_MAX_EN_LINES, V3_MIN_HINDI, V3_TTS_SCRIPT, load_scenarios)

KOKORO = json.load(open(os.path.join(GEN, "kokoro_wps.json")))
WPS = {v: x["wps"] for v, x in KOKORO["per_voice"].items()}
DEV_RE = re.compile(r"[ऀ-ॿ]")
LATIN_TOK_RE = re.compile(r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NUM_RE = re.compile(r"\d+(?:[ :.,/-]\d+)*")
ID_RE = re.compile(r"\b[A-Z]{1,3}\d{2,}[A-Z0-9]*\b|\b\d+[A-Z]\b")
FEM_1P = re.compile(r"\b\w*(?:ti|rahi|gayi|gai|chuki|sakti|payi|saki|leti|deti|karti|thi)\s+(?:hoon|hun|hu|hoo)\b|\b\w+(?:ungi|oongi|ungee)\b")
MASC_1P = re.compile(r"\b\w*(?:ta|raha|gaya|chuka|sakta|paya|saka|leta|deta|karta|tha)\s+(?:hoon|hun|hu|hoo)\b|\b\w+(?:unga|oonga)\b")
# first-person past tense ("main ... ja raha tha" / "main ... gayi thi"); subject 'main' within the same clause
FEM_PAST = re.compile(r"\bmain\b[^.?!,]{0,30}?\b\w*(?:ti|rahi|gayi|gai|chuki|sakti|payi|aayi)\s+thi\b")
MASC_PAST = re.compile(r"\bmain\b[^.?!,]{0,30}?\b\w*(?:ta|raha|gaya|chuka|sakta|paya|aaya)\s+tha\b")
# completed-write claims ("updated", "cancel kar diya", "has been added"); only allowed in confirm_write turns
DONE_RE = re.compile(r"\b(?:i have|i've|we have|has been|have been|is now|are now)\s+(?:now\s+|also\s+|successfully\s+)?"
                     r"(?:added|updated|changed|submitted|paused|raised|registered|cancell?ed)\b"
                     r"|\b(?:is|are|was)\s+(?:also\s+|successfully\s+)?(?:added|updated|changed|submitted)\b"
                     r"|\b(?:update|add|change|cancel|submit|pause|register|raise|note)\s+(?:kar\s+)?(?:diya|di|diye|dia)\b"
                     r"|\b(?:update|add|change|cancel|submit|pause|register|raise)\s+ho\s+(?:gaya|gayi|gaye|gai|chuka|chuki)\b")
# agent announcing a lookup in first person ("taaki main details check kar sakun", "so I can check", "main account check karta hoon")
AGENT_CHECK_RE = re.compile(r"\b(?:main|i|we|hum)\b[^.?!]{0,30}\b(?:check|dekh)(?!-in)\b")
STOP = set("the a an to of and or at dot com in on for with near please se ko ka ki ke pe par mein me hai hain ji haan "
           "is my mera meri mere aapka aapki aapke wala wali wale new naya nayi address number instruction ka".split())
TAGS_FORBID_TRUNC = {"greeting", "check_line", "confirm_write", "signoff"}
NICE_DIN_RE = re.compile(r"\b(?:nice|good|great|achha|accha|acha)\s+din\b|\bhave\s+a\s+\w+\s+din\b", re.I)
# agent promising an outcome/timing (v3review 2026-10-03: 3/8 trial sign-offs said 'aapka order jaldi pahunch jayega',
# which the judge does not check because it skips sign-offs). V3/CONTROL only (V1/V2 validation unchanged).
PROMISE_RE = re.compile(r"\bjaldi\b[^.?!]{0,30}\b(?:pahunch|aa\s+ja|aa\s+jay|mil\s+ja|deliver|ho\s+ja|process|credit|wapas)"
                        r"|\b(?:pahunch|aa|mil)\w*\s+ja(?:yega|yegi|yenge|ega|egi)\b[^.?!]{0,20}\b(?:jaldi|time\s+(?:pe|par|par\s+hi))\b"
                        r"|\b(?:will|'ll|should|is going to|are going to)\b[^.?!]{0,40}\b(?:soon|shortly|on time|quickly|in no time|right away)\b"
                        r"|\b(?:arriv|reach|deliver|refund|credit)\w*\b[^.?!]{0,30}\b(?:soon|shortly|on time|quickly|in no time)\b", re.I)


def tts_mixed_issues(t, ex):
    """V3 mixed-script text_tts: Hindi in Devanagari, English in Latin, same words as text_roman."""
    out = []
    tts = EMAIL_RE.sub(" ", t["text_tts"])
    if not tts.strip():
        return ["text_tts empty"]
    rl = hs.classify_line(t["text_roman"], ex)
    n_hi_roman = sum(1 for _, l in rl if l == "H")
    if n_hi_roman and not DEV_RE.search(tts):
        out.append(f"text_roman has Hindi words but text_tts has no Devanagari (write Hindi words in Devanagari)")
    lat = " ".join(LATIN_TOK_RE.findall(DEV_RE.sub(" ", tts)))
    bad = [w for w, l in hs.classify_line(lat, ex) if l == "H" and not re.search(r"\d", w)]
    if bad:
        out.append(f"text_tts has Hindi words in Latin letters: {bad[:5]} (write them in Devanagari)")
    n_en_roman = sum(1 for _, l in rl if l == "E")
    n_lat_tts = len([w for w in LATIN_TOK_RE.findall(lat) if not re.search(r"\d", w)])
    if n_en_roman >= 3 and n_lat_tts < 0.5 * n_en_roman:
        out.append(f"text_tts writes English words in Devanagari ({n_en_roman} English words in text_roman, {n_lat_tts} "
                   "Latin words in text_tts); keep English words in Latin letters")
    nr, nt = len(words(t["text_roman"])), len(words(t["text_tts"]))
    if abs(nr - nt) > max(3, 0.25 * nr):
        out.append(f"text_tts has {nt} words but text_roman {nr}; they must be the same line")
    return out


def words(s):
    return [w for w in re.split(r"\s+", s.strip()) if re.search(r"[\wऀ-ॿ]", w)]


def toks(s):
    return [t for t in re.split(r"[^a-z0-9@]+", s.lower()) if t]


def digits(s):
    return re.sub(r"\D", "", s)


def tts_word_units(text_tts):
    n = 0
    for w in words(text_tts):
        d = digits(w)
        if d and len(d) == len(re.sub(r"[^\w]", "", w)):
            n += 1 if len(d) == 1 else (2 if len(d) <= 3 else len(d))
        else:
            n += 1
    return n


def est_duration(call):
    """Estimated call length (s): 1.0 lead + speech (tts word units / per-voice Kokoro wps) + gaps - overlaps."""
    total = 1.0
    for i, t in enumerate(call["turns"]):
        v = call["voices"][t["speaker"]]
        total += tts_word_units(t["text_tts"]) / WPS[v]
        if i < len(call["turns"]) - 1:
            total += t.get("pause_after_s", 0.45)
        total -= t.get("overlap_offset_s", 0.0)
    return round(total, 1)


def overlap_frac(value, text):
    """Fraction of the value's content tokens present in text (exact token match; digit tokens too, so
    '502' is not found inside '4502')."""
    vt = [t for t in toks(value) if t not in STOP and len(t) >= 2]
    if not vt:
        return 1.0
    tt = set(toks(text))
    return sum(t in tt for t in vt) / len(vt)


def digit_toks_ok(value, text):
    """Every digit-bearing token of the value (flat/sector/house numbers) appears as a token in text."""
    tt = set(toks(text))
    return all(t in tt for t in toks(value) if any(ch.isdigit() for ch in t))


def echo_ok(tool, argname, value, text):
    if argname == "phone" or (digits(value) and len(digits(value)) >= 8 and argname not in ("address", "location", "instruction")):
        return digits(value) in digits(text)
    if argname in ("order_id", "ride_id", "pnr", "reference_id"):
        return value.replace(" ", "").lower() in text.replace(" ", "").lower()
    if argname == "email":
        local = [t for t in toks(value.split("@")[0]) if len(t) >= 2]
        dom = toks(value.split("@")[1])[:1] if "@" in value else []  # 'gmail' / 'yahoo' must be said too
        return all(t in toks(text) or t in text.lower().replace(" ", "") for t in local + dom)
    return overlap_frac(value, text) >= 0.6 and digit_toks_ok(value, text)


def number_sources(call, upto):
    src = [call["role_prompt"]]
    src += [t["text_roman"] for t in call["turns"][:upto] if t["speaker"] == "customer"]
    nums = set()
    ids = set()
    for s in src:
        for m in NUM_RE.finditer(s):
            nums.add(digits(m.group(0)))
            for part in re.split(r"[ :.,/-]", m.group(0)):
                nums.add(part)
        for m in ID_RE.finditer(s):
            ids.add(m.group(0).upper())
        for m in re.finditer(r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+", s):
            nums.add(digits(m.group(0)))
    return nums, ids


def latin_allow(call):
    allow = set()
    blob = json.dumps(call.get("record", {}), ensure_ascii=False) + " " + call["role_prompt"]
    for m in hs.PLATE_RE.finditer(blob):
        allow.update(re.findall(r"[A-Z]+", m.group(0)))
    for m in re.finditer(r"\b([A-Z]{1,3})[ -]?\d{2,5}\b", blob):  # flight codes like 'SI 302'
        allow.add(m.group(1))
    for m in re.finditer(r"\b[A-Z]{1,3}\b", json.dumps(call.get("record", {}).get("facts", {}), ensure_ascii=False)):
        allow.add(m.group(0))  # short upper-case pieces of record values (plate letters, series codes)
    allow -= {"UPI", "PNR", "ETA", "PM", "AM", "SOS", "OTP", "GB", "KG", "ID", "OK", "EMI", "AC"}
    return allow


def check_call(call, scen, interrupt_flag, twopart_flag):
    """Returns list of (check, message). Empty list = pass. call is the post-processed full call dict."""
    F = []
    turns = call["turns"]
    v = call["variant"]
    caller_reg, agent_reg = REGIMES["V3" if v == "V4" else v]
    ag, cg = call["agent_gender"], call["customer_gender"]

    def fail(c, m):
        F.append((c, m))

    tts_mode = "english" if v == "CONTROL" else ("mixed" if v == "V3" and V3_TTS_SCRIPT == "mixed" else "devanagari")

    tlo, thi = v4.TURNS if v == "V4" else (8, 16)
    if not (tlo <= len(turns) <= thi):
        fail("turn_count", f"{len(turns)} turns; need {tlo}-{thi}")
    if not turns:
        return F
    t0 = turns[0]
    if t0["speaker"] != "agent" or "greeting" not in t0["tags"]:
        fail("greeting_first", "first turn must be the agent greeting tagged greeting")
    elif call["brand"].lower() not in t0["text_roman"].lower() or call["agent_name"].lower() not in t0["text_roman"].lower():
        fail("greeting_first", "greeting must name the brand and the agent")
    tl = turns[-1]
    if tl["speaker"] != "agent" or "signoff" not in tl["tags"]:
        fail("signoff_last", "last turn must be the agent sign-off tagged signoff")

    # per-line checks
    for i, t in enumerate(turns):
        nw = len(words(t["text_roman"]))
        lo, hi = (6, 18) if t["speaker"] == "agent" else (4, 16)
        if v != "V4" and (nw > hi or (nw < lo and not t.get("truncated"))):  # V4: per-band caps in v4.band_checks
            fail("line_words", f"turn {i} ({t['speaker']}) has {nw} words; need {lo}-{hi}")
        if DEV_RE.search(t["text_roman"]):
            fail("roman_no_devanagari", f"turn {i} text_roman contains Devanagari")
        if tts_mode in ("mixed", "english"):
            continue
        if not t["text_tts"].strip() or not DEV_RE.search(t["text_tts"]):
            fail("tts_no_latin", f"turn {i} text_tts empty or not Devanagari")
    rec0 = call.get("record") or {}
    ex0 = hs.build_exclude(rec0, [call["agent_name"], call["brand"]] + rec0.get("customer_name", "").split())
    if tts_mode == "mixed":
        for i, t in enumerate(turns):
            for m in tts_mixed_issues(t, ex0):
                fail("tts_script", f"turn {i} {m}")
    elif tts_mode == "english":
        for i, t in enumerate(turns):
            if not t["text_tts"].strip() or DEV_RE.search(t["text_tts"]):
                fail("tts_script", f"turn {i} text_tts must be English in Latin letters (no Devanagari)")
            elif t["text_tts"].strip() != t["text_roman"].strip() and \
                    abs(len(words(t["text_tts"])) - len(words(t["text_roman"]))) > max(3, 0.25 * len(words(t["text_roman"]))):
                fail("tts_script", f"turn {i} text_tts must be the same line as text_roman")
    allow = latin_allow(call)
    for i, t in enumerate(turns):
        if tts_mode in ("mixed", "english"):
            break
        tts = EMAIL_RE.sub(" ", t["text_tts"])  # emails count as IDs (TTS stage spells them out)
        bad = [w for w in LATIN_TOK_RE.findall(tts) if not re.search(r"\d", w) and w not in allow]
        if bad:
            fail("tts_no_latin", f"turn {i} text_tts has Latin outside IDs: {bad[:5]} (write them in Devanagari)")

    # duration
    d = v4.est_duration(call) if v == "V4" else est_duration(call)
    call["est_duration_s"] = d
    dlo, dhi = v4.DUR if v == "V4" else (60, 110)
    if not (dlo <= d <= dhi):
        fail("duration", f"estimated {d}s; need {dlo}-{dhi} s (total words too {'few' if d < dlo else 'many'})")

    # hindi share (recomputed here with the frozen counter; the stored value is refreshed)
    rec = call.get("record") or {}
    cn = rec.get("customer_name", "").split()
    call["hindi_token_share"] = hs.call_shares(call, hs.build_exclude(rec, [call["agent_name"], call["brand"]] + cn))
    if v in ("V3", "V4", "CONTROL"):
        ex = hs.build_exclude(rec, [call["agent_name"], call["brand"]] + cn)
        for spk in ("agent", "customer"):
            s = call["hindi_token_share"][spk]
            ls = [(hs.share([t["text_roman"]], ex), t["text_roman"]) for t in turns
                  if t["speaker"] == spk and not set(t["tags"]) & {"greeting", "check_line"}]
            if v in ("V3", "V4"):  # floor on FREE lines: the fixed greeting/check-lines/sign-off must not carry the share
                free = [t["text_roman"] for t in turns
                        if t["speaker"] == spk and not set(t["tags"]) & {"greeting", "check_line", "signoff"}]
                s = hs.share(free, ex) if free else 0.0
                en_lines = [x for x in free if hs.share([x], ex) == 0]
                if len(en_lines) > V3_MAX_EN_LINES:
                    fail("hinglish_both", f"{spk} has {len(en_lines)} pure-English lines (max {V3_MAX_EN_LINES}); make them "
                         "natural spoken Hinglish: " + " | ".join(f"'{x}'" for x in en_lines[:3]))
            if v in ("V3", "V4") and s < V3_MIN_HINDI:
                ls = sorted(ls)[:3]
                fail("hinglish_both", f"{spk} Hindi word share {s:.2f} < {V3_MIN_HINDI:.2f}: the {spk} must speak natural "
                     "Hinglish, not English; make these lines Hinglish: " + " | ".join(f"'{x}' ({p:.0%})" for p, x in ls))
            if v == "CONTROL" and s >= CONTROL_MAX_HINDI:
                ls = sorted(ls, reverse=True)[:3]
                fail("english_only", f"{spk} Hindi word share {s:.2f} >= {CONTROL_MAX_HINDI:.2f}: English only; fix: "
                     + " | ".join(f"'{x}' ({p:.0%})" for p, x in ls))
        for i, t in enumerate(turns):
            if t["speaker"] == "agent" and PROMISE_RE.search(t["text_roman"]):
                fail("promise", f"agent turn {i} promises an outcome/timing not in INFORMATION: "
                     f"'{PROMISE_RE.search(t['text_roman']).group(0)}' (say nothing about when it will arrive/happen)")
            if NICE_DIN_RE.search(t["text_roman"]):
                fail("nice_din", f"turn {i} uses '{NICE_DIN_RE.search(t['text_roman']).group(0)}' (half-translated "
                     "sign-off); use a natural sign-off")
    for spk, reg in (() if v in ("V3", "V4", "CONTROL") else (("agent", agent_reg), ("customer", caller_reg))):
        lo, hi = SHARE_BANDS[(reg, spk)]
        s = call["hindi_token_share"][spk]
        if not (lo <= s <= hi):
            ex = hs.build_exclude(rec, [call["agent_name"], call["brand"]] + cn)
            ls = sorted(((hs.share([t["text_roman"]], ex), t["text_roman"]) for t in turns
                         if t["speaker"] == spk and not set(t["tags"]) & {"greeting", "check_line"}), reverse=s > hi)[:3]
            fail("hindi_share", f"{spk} Hindi word share {s:.2f}; need {lo:.2f}-{hi:.2f} ("
                 + ("too much Hindi; make these lines more English: " if s > hi else "too little Hindi; add Hindi words to: ")
                 + " | ".join(f"'{x}' ({v:.0%})" for v, x in ls) + ")")

    # gender
    if v not in ("GATE1", "CONTROL"):
        for spk, g in (("agent", ag), ("customer", cg)):
            txt = " ".join(t["text_roman"].lower() for t in turns if t["speaker"] == spk)
            fem, masc = FEM_1P.findall(txt) + FEM_PAST.findall(txt), MASC_1P.findall(txt) + MASC_PAST.findall(txt)
            right, wrong = (fem, masc) if g == "f" else (masc, fem)
            if wrong:
                fail(f"gender_{spk}", f"{spk} is {'female' if g == 'f' else 'male'} but uses {wrong[:3]}")
            elif not right:
                fail(f"gender_{spk}", f"{spk} has no first-person gendered Hindi verb (e.g. {'kar rahi hoon' if g == 'f' else 'kar raha hoon'})")

    # writes
    writes = call["writes"]
    need = scen["writes"]
    got = [w["tool"] for w in writes]
    if sorted(got) != sorted(need):
        fail("writes_allowed", f"writes {got} but scenario needs exactly {need}")
    confirms = [i for i, t in enumerate(turns) if "confirm_write" in t["tags"]]
    checks = [i for i, t in enumerate(turns) if "check_line" in t["tags"]]
    if len(confirms) != len(writes) or len(checks) != len(writes):
        fail("write_structure", f"{len(writes)} writes but {len(checks)} check_line and {len(confirms)} confirm_write turns")
    lines_ok = CHECK_LINES_EN if v in ("GATE1", "CONTROL") else [c[0] if ag == "f" else c[1] for c in CHECK_LINES]
    for i in checks:
        t = turns[i]
        if t["speaker"] != "agent":
            fail("write_structure", f"check_line turn {i} not spoken by agent")
        best = max(difflib.SequenceMatcher(None, t["text_roman"].lower(), c.lower()).ratio() for c in lines_ok)
        if best < 0.8:
            fail("write_structure", f"check_line turn {i} '{t['text_roman']}' is not one of the 5 phrasings")
        if not (0.8 <= t.get("pause_after_s", 0) <= 1.5):
            fail("write_structure", f"check_line turn {i} pause_after_s not in 0.8-1.5")
        if i + 1 >= len(turns) or "confirm_write" not in turns[i + 1]["tags"] or turns[i + 1]["speaker"] != "agent":
            fail("write_structure", f"check_line turn {i} must be followed directly by the agent confirm_write turn")
        if i == 0 or turns[i - 1]["speaker"] != "customer":
            fail("write_structure", f"check_line turn {i} must come right after the customer's request/value turn")
    # static reads are never preceded by a check-line: no check-line-like agent turn outside the tagged write structure
    rnc = set()  # check-line-like agent turns (used by the V4 read-tag check)
    for i, t in enumerate(turns):
        if t["speaker"] != "agent" or "check_line" in t["tags"]:
            continue
        low = t["text_roman"].lower()
        sim = max(difflib.SequenceMatcher(None, low, c.lower()).ratio() for c in lines_ok)
        if sim >= 0.8 or re.search(r"let me (?:check|look|see)|main (?:abhi |phir |zara )?(?:check|dekh)|check karke bata|check kar(?:ti|ta|ungi|unga) |check kar le(?:ti|ta)|dekh le(?:ti|ta)|\bchecking\b|one moment|just a (?:sec|second|moment)|ek (?:minute|second) rukiye", low) \
                or AGENT_CHECK_RE.search(low):
            rnc.add(i)
            fail("read_no_check", f"agent turn {i} sounds like a check-line but is not a tagged write check-line: '{t['text_roman']}'")
    # a completed-write claim outside confirm_write (extra write with no check-line) fails, unless it only restates
    # the argument of a write already confirmed earlier in the call
    for i, t in enumerate(turns):
        if t["speaker"] != "agent" or "confirm_write" in t["tags"] or not DONE_RE.search(t["text_roman"].lower()):
            continue
        prior = [(w.get("args") or {}).get(ECHO_ARG.get(w["tool"], ""), "") for k, w in enumerate(writes)
                 if k < len(confirms) and confirms[k] < i]
        if not any(val and echo_ok("", "", val, t["text_roman"]) for val in prior):
            fail("untagged_write", f"agent turn {i} claims a completed action outside the check-line + confirm_write "
                 f"structure: '{t['text_roman']}' (only the listed writes may be done, each as check-line then confirm_write)")
    for k, w in enumerate(writes):
        if k >= len(confirms):
            break
        ci = confirms[k]
        w["confirm_turn_idx"] = ci
        arg = ECHO_ARG.get(w["tool"])
        val = (w.get("args") or {}).get(arg, "")
        if not val:
            fail("write_echo", f"write {w['tool']} missing arg {arg}")
            continue
        if not echo_ok(w["tool"], arg, val, turns[ci]["text_roman"]):
            fail("write_echo", f"confirm turn {ci} does not echo {arg}='{val}'")
        # origin of the argument: an earlier customer turn (or record for IDs/plan)
        cands = [(overlap_frac(val, turns[j]["text_roman"]) * digit_toks_ok(val, turns[j]["text_roman"]) if arg not in ("phone",)
                  else float(digits(val) in digits(turns[j]["text_roman"])), j)
                 for j in range(ci) if turns[j]["speaker"] == "customer"]
        best = max(cands) if cands else (0, None)
        w["caller_turn_idx"] = best[1]
        in_record = echo_ok(w["tool"], arg, val, call["role_prompt"])
        if arg in ("address", "location", "instruction", "phone", "email") and best[0] < (0.6 if arg != "phone" else 1):
            fail("write_echo", f"{arg}='{val}' was never spoken by the customer before turn {ci}")
        elif arg not in ("address", "location", "instruction", "phone", "email") and best[0] < 0.6 and not in_record:
            fail("write_echo", f"{arg}='{val}' not in record or an earlier customer turn")

    # two-part argument
    if twopart_flag and writes and confirms:
        c = writes[0].get("caller_turn_idx")
        if c is None or c < 2 or turns[c - 1]["speaker"] != "agent" or turns[c - 2]["speaker"] != "customer" \
                or "check_line" not in turns[min(c + 1, len(turns) - 1)]["tags"]:
            fail("twopart", "first write must be: customer intent, agent asks for value, customer gives value, check-line, confirm")
        elif overlap_frac((writes[0].get("args") or {}).get(ECHO_ARG[writes[0]["tool"]], ""), turns[c - 2]["text_roman"]) >= 0.6 \
                and ECHO_ARG[writes[0]["tool"]] in ("address", "location", "instruction", "phone", "email"):
            fail("twopart", "customer gave the value already in the intent turn; the value must come only after the agent asks")

    # interruption
    tr = [i for i, t in enumerate(turns) if t.get("truncated")]
    if interrupt_flag:
        if len(tr) != 1:
            fail("interruption", f"need exactly 1 truncated agent turn, got {len(tr)}")
        else:
            i = tr[0]
            if turns[i]["speaker"] != "agent" or set(turns[i]["tags"]) & TAGS_FORBID_TRUNC:
                fail("interruption", f"truncated turn {i} must be a plain agent turn (not greeting/check/confirm/signoff)")
            if i + 1 >= len(turns) or turns[i + 1]["speaker"] != "customer":
                fail("interruption", f"truncated turn {i} must be followed by the interrupting customer turn")
            if re.search(r"[.?!।]\s*$", turns[i]["text_roman"]):
                fail("interruption", f"truncated turn {i} should end mid-sentence (no final punctuation)")
    elif tr:
        fail("interruption", f"this call must have no interruption, got truncated turns {tr}")

    # numbers / IDs in agent lines must come from record or earlier caller turns
    for i, t in enumerate(turns):
        if t["speaker"] != "agent":
            continue
        nums, ids = number_sources(call, i)
        for m in NUM_RE.finditer(t["text_roman"]):
            dg = digits(m.group(0))
            parts = [p for p in re.split(r"[ :.,/-]", m.group(0)) if p]
            if dg in ("1", "2") or dg in nums or all(p in nums for p in parts) or any(dg and dg in n for n in nums if len(dg) >= 4):
                continue
            fail("numbers_regex", f"agent turn {i} says '{m.group(0)}' which is not in the record or earlier customer turns")
        for m in ID_RE.finditer(t["text_roman"]):
            if m.group(0).upper() not in ids and digits(m.group(0)) not in nums:
                fail("numbers_regex", f"agent turn {i} says ID '{m.group(0)}' not in record")

    if v == "V4":  # D2 additions: band caps/shape, greeting/sign-off templates, din ban, nukta, read tags
        for c, m in (v4.band_checks(call) + v4.template_checks(call) + v4.din_checks(call) + v4.nukta_checks(call)
                     + v4.read_checks(call, lambda i: i in rnc)):
            fail(c, m)
    return F


# --------------------------------------------------------------------------- LLM judge
JUDGE_SCHEMA = {"type": "object", "properties": {"violations": {"type": "array", "items": {
    "type": "object", "properties": {"turn": {"type": "integer"}, "quote": {"type": "string"},
                                     "kind": {"type": "string", "enum": ["unsupported_fact", "dynamic_result", "false_capability", "wrong_fact"]},
                                     "reason": {"type": "string"}}, "required": ["turn", "quote", "kind", "reason"]}}},
    "required": ["violations"]}


def judge_prompt(call):
    if call.get("variant") == "V4":
        return v4.judge_prompt(_judge_prompt_v3(call), call)
    return _judge_prompt_v3(call)


def _judge_prompt_v3(call):
    lines = "\n".join(f"[{i}] {t['speaker'].upper()}{' (confirms the requested action)' if 'confirm_write' in t['tags'] else ''}: "
                      f"{t['text_roman']}" for i, t in enumerate(call["turns"]))
    return f"""You are a strict fact-checker for a synthetic customer-support call. The AGENT may only state facts that are in the INFORMATION below or that the CUSTOMER said earlier in the call.

INFORMATION (everything the agent knows):
{call['role_prompt']}

CALL:
{lines}

Flag an AGENT turn ONLY if it contains one of these:
- unsupported_fact: a specific fact (number, time, date, amount, name, place, item, status, ETA, policy) that is not in INFORMATION and was not said earlier by the customer.
- wrong_fact: a fact that contradicts INFORMATION (e.g. different ETA, price, seat, address).
- dynamic_result: a result only a live system could give: a refund amount/approval/timeline, processing time, new ETA or fare after a change, live location, seat/slot availability.
- false_capability: claiming to contact/call/message the rider, driver, restaurant, seller or airline staff, to track live, or to investigate/escalate.
Do NOT flag: greetings, empathy, apologies, generic reassurance ("aapka order on the way hai" when INFORMATION says so), asking questions, repeating INFORMATION or the customer's words, an agent turn that confirms the customer's requested change/cancellation/update is done or the refund/cancellation request is submitted (the agent IS able to do these actions; only flag such a turn if it adds an amount, timeline, approval or other outcome), check-lines like "ek minute, main check karti hoon", sign-offs, telling the customer to use an in-app SOS or call 112 only if INFORMATION mentions them.
Return JSON {{"violations": [...]}} with turn index, the exact quote, kind and a short reason. Empty list if the call is clean."""


def judge_calls(llm, calls):
    sch = JUDGE_SCHEMA if not any(c.get("variant") == "V4" for c in calls) else \
        [v4.JUDGE_SCHEMA if c.get("variant") == "V4" else JUDGE_SCHEMA for c in calls]
    outs = llm.chat([judge_prompt(c) for c in calls], schema=sch, temperature=0.0, max_tokens=800)
    res = []
    for c, (txt, fin, _) in zip(calls, outs):
        try:
            vio = json.loads(txt)["violations"]
            vio = [x for x in vio if 0 <= x.get("turn", -1) < len(c["turns"]) and c["turns"][x["turn"]]["speaker"] == "agent"]
            # a confirm_write turn saying the requested action is done is allowed by design; drop the judge's
            # dynamic_result/false_capability flags on it unless the quote carries a number or a refund outcome/timeline
            vio = [x for x in vio if not ("confirm_write" in c["turns"][x["turn"]]["tags"]
                                          and x.get("kind") in ("dynamic_result", "false_capability")
                                          and not re.search(r"\d|amount|days?|din|hours?|ghante|approve|mil jaye|credited|wapas|weeks?|hafte|hafta|"
                                                            r"mahin|months?|within|shortly|soon|sms|message|notification|minutes?|\btak\b",
                                                            x.get("quote", ""), re.I))]
            if c.get("variant") == "V4":
                vio = [x for x in vio if v4.judge_keep(c, x)]
        except Exception as e:  # noqa
            vio = [{"turn": -1, "quote": "", "kind": "judge_error", "reason": f"{fin}: {e}"}]
        res.append(vio)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--scenarios", help="scenario file (default: guidance/scenario_creation.json; V4 calls: _v2.json)")
    a = ap.parse_args()
    from common import assignments
    calls = [json.loads(x) for x in open(a.calls)]
    if a.scenarios or any(c.get("variant") == "V4" for c in calls):
        scen = v4.load_scenarios(a.scenarios or v4.V4_SCENARIOS)
    else:
        scen = load_scenarios()
    interrupt, twopart = assignments(scen)
    cnt = Counter()
    npass = 0
    for c in calls:
        F = check_call(c, scen[c["scenario_id"]], c["call_id"] in interrupt, c["call_id"] in twopart)
        for k in {f[0] for f in F}:
            cnt[k] += 1
        npass += not F
        for f in F:
            print(c["call_id"], f)
    print(f"deterministic pass {npass}/{len(calls)}; failing calls per check: {dict(cnt)}")
    if a.judge:
        from common import LLM
        llm = LLM()
        for c, v in zip(calls, judge_calls(llm, calls)):
            print(c["call_id"], "judge:", v)


if __name__ == "__main__":
    main()
