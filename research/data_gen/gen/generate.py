"""Call generator (Gemma 4 31B, vLLM offline, batched) -> data/<V>/calls.jsonl, gen_log.jsonl, SAMPLES.md

  python generate.py --variant V1                      # all 288 calls
  python generate.py --variant V1 --calls food_05_g1 cab_15_g2 --out /workspace/hinglish/data/trial/V1
  python generate.py --variant GATE1                   # 10 English-only calls (common.GATE1_CALLS)
  python generate.py --job V1:food_01_g1,cab_15_g2 --job V3:... --job GATE1 --with-records   # several in one model load
Each call: up to 3 attempts; attempt k>1 gets the previous failure reasons as feedback.
"""
import argparse
import json
import os
import random
import re

_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
import time
from collections import Counter

import hindi_share as hs
import v4
import validate as V
from common import (CHECK_LINES, CHECK_LINES_EN, DATA, GATE1_CALLS, GUIDE, PAIRINGS, REGIMES, SHARE_BANDS, TOOL_ARGS,
                    V3_TTS_SCRIPT, all_call_ids, append_jsonl, assignments, control_call_ids, extract_json, h01, jdump,
                    load_scenarios)

MAX_ATTEMPTS = 3


def load_examples():
    rows = []
    for ln in open(f"{GUIDE}/r1_vs_r2.md", encoding="utf-8"):
        p = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(p) == 3 and p[0].isdigit():
            rows.append((int(p[0]), p[1].replace("’", "'"), p[2].replace("’", "'")))
    agent_rows = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18}
    ex = {}
    for col, k in (("R1", 1), ("R2", 2)):
        ex[(col, "agent")] = [r[k] for r in rows if r[0] in agent_rows]
        ex[(col, "customer")] = [r[k] for r in rows if r[0] not in agent_rows]
    return ex


EXAMPLES = load_examples()

REGIME_TEXT = {
    "R1": ("REGIME 1 (R1): English grammar and English word order; Hindi nouns, fillers and short fixed phrases dropped in. "
           "If you swapped the Hindi words for English the sentence would still parse as English. "
           "The main verbs are ENGLISH (is, are, have updated, will arrive, want, need, can you, please change). "
           "Hindi appears only as: openers/fillers (haan ji, ji, achha, arre yaar, bas, theek hai, matlab), connectives and adverbs "
           "(aur, bhi, abhi, abhi tak, toh, lekin, phir, pehle hi), short fixed phrases (koi baat nahi, ek minute, bilkul, sahi hai), "
           "possessives (aapka/aapki/mera/meri), everyday nouns (ghar, gaadi, paise, din), and an occasional trailing 'hai'. "
           "NOT R1: whole Hindi clauses with Hindi verbs and postpositions (\"maine aapka number update kar diya hai\", "
           "\"address change karna hai\", \"order cancel kar do\") - those are Regime 2 and make the line too Hindi. "
           "Exceptions: the greeting, the check-lines, and for the CUSTOMER two short first-person clauses that show gender."),
    "R2": ("REGIME 2 (R2): the grammar itself switches to Hindi. Hindi verb-final clauses with postpositions (ko, se, mein, pe, "
           "ke liye, wala/wali/wale) and Hindi verb conjugation (kar do, ho gaya hai, karna hai, jaayega, mil jaayega), alternating "
           "with short clean English clauses inside the same turn. English is used for content nouns (order, address, delivery, "
           "refund, plan, booking) inside Hindi sentences. Keep roughly half the words English: not every clause is Hindi."),
    "EN": "ENGLISH ONLY: plain natural Indian-English customer-support language. No Hindi words at all (no ji, haan, achha, yaar).",
}


R1_DENSITY = {
    "agent": ["Ji, your order FD1234 is out for delivery, aur ETA abhi bhi 15 minutes hai.",
              "Theek hai, I have updated the address, ab delivery Tower B, Flat 402 pe hogi.",
              "Ji bilkul, your driver is Amit, aur car number DL 01 AB 4821 hai.",
              "Achha ji, thank you for calling, have a good day.",
              "Koi baat nahi ji, your new number 98000 12345 is now updated on the account.",
              "Bas itna hi, your plan is Premium Monthly at Rs 499, and renewal 18 October ko hai."],
    "customer": ["Arre yaar, I ordered one hour ago, aur abhi tak koi update nahi.",
                 "Achha, please change the address to my office, main wahan pe hi rahungi. (female caller; male: rahunga)",
                 "Matlab I have been waiting for forty minutes, main bahut pareshan ho raha hoon yaar. (male caller; female: ho rahi hoon)",
                 "Haan ji, my new number is 98000 12345, please isko jaldi update karo.",
                 "Achha theek hai, then please cancel it, mujhe ab nahi chahiye."],
}


AGENT_COUNT = {
    "R1": "Aim for about 27% overall. Practical rule: in a 13-word agent line use only 3-4 Hindi words (e.g. ji, achha, abhi, aur, bhi, aapka, hai) and keep the verb English (is / have updated / will arrive / please tell me). The greeting and check-lines are already very Hindi, so the other agent lines must be lighter. Sign-off in English with at most one Hindi word.",
    "R2": "Aim for about 57% overall. Practical rule: in a 13-word agent line use about 7-8 Hindi words; Hindi verb at the end of most clauses, plus one short English clause or English nouns.",
}
CUST_COUNT = {
    "R1": "Aim for about 43% overall. Practical rule: in a 12-word customer line use about 5 Hindi words (fillers, adverbs, short phrases), with an English main verb; one or two lines may carry a short Hindi first-person clause that shows gender.",
    "R2": "Aim for about 57% overall. Practical rule: in a 12-word customer line use about 6-7 Hindi words, never a fully Hindi line: every line keeps 4+ English words (nouns or a short English clause).",
}


MIX_NOTE = ("IMPORTANT: the customer speaks Regime 2 (Hindi grammar) but the AGENT stays strictly Regime 1: English verbs and "
            "English sentence structure with only a few Hindi words. Do NOT mirror the customer's Hindi; agents who drift into "
            "Hindi verb phrases (bataiye, kar deti hoon, ho gaya hai) are rejected. Aim for about 28% Hindi in agent lines.\n")


def pct(b):
    return f"{int(b[0] * 100)}-{int(b[1] * 100)}%"


# ---------------------------------------------------------------- V3 (novasynth Hinglish guidance) and CONTROL (English)
import novasynth_hinglish as NS  # noqa: E402  verbatim copies of the user's novasynth Hinglish prompts

SIGNOFF_BAN = ("Sign-off: natural and short, in your own words for this call (vary it; do not copy these examples), e.g. "
               "\"Thank you ji, aapka din achha rahe.\", \"Koi baat nahi sir, aur kuch ho toh zaroor call kijiyega.\", "
               "\"Theek hai ma'am, thank you for calling, take care.\" The sign-off adds no new fact and no promise (nothing about "
               "when an order, ride, refund or update will arrive or happen). NEVER \"have a nice din\" / \"have a good din\" "
               "(a half-translated phrase nobody says).")
V3_AGENT_NOTE = ("For THIS call script two of those fillers are NOT allowed: \"Just checking, sir…\" and \"एक सेकंड सर…\" "
                 "(any \"checking\" / \"ek second\" / \"main check/dekh kar ...\" line outside a write check-line is rejected, see FACTS "
                 "AND TOOLS). When the agent needs an ID it just asks: \"Ji sir, aapka order ID bataiye.\" with no reason "
                 "like \"taaki main check/dekh sakun\".")

V3_LANG_T = f"""LANGUAGE: BOTH speakers speak natural, everyday spoken Hinglish in EVERY line
The AGENT and the CUSTOMER both talk the way real people talk on a support call in urban India: colloquial Hindi-English code-mix, Hindi grammar words and verbs mixed with English content words. No line is pure English and no line is formal shuddh Hindi. Write it as it would really be said; do not count words.
Follow the guidance below (copied from our persona prompts). It describes HOW each side talks. {{script_note}}

CUSTOMER guidance ("you (the caller)" = the CUSTOMER; "the bot" / "user" = the AGENT):
<<<
{NS.PERSONA_HINGLISH_CONSTRAINT}
{NS.SPEECH_FORMATTING_HINDI_HINGLISH}
{NS.HINGLISH_CALLER_INSTRUCTIONS}
>>>

AGENT guidance (from our Hinglish telecaller agent prompt; "Riya" there = this call's agent; a MALE agent uses the male forms, e.g. "सर, मैं सुन रहा हूँ"; say sir/ma'am to match the customer):
<<<
{NS.AGENT_LANGUAGE_STYLE}

{NS.AGENT_FILLERS}
>>>
The customer speaks Hinglish too, so the agent stays in Hinglish for the whole call (never switches to pure English). Fillers: at most one per line, not in every line.
{V3_AGENT_NOTE}
Every line is Hinglish, including the customer's dictation lines (e.g. "Haan, naya address hai Flat 502, Tower C, Sector 45, Gurgaon."). Hinglish lines run long: keep customer lines to at most 14 words and the call to at most 16 turns.
{SIGNOFF_BAN}"""

V3_SCRIPT_NOTE = {
    "mixed": "Its script rules apply to text_tts only (see SCRIPTS below); text_roman is always typed Latin.",
    "devanagari": ("Its SCRIPT rules (Devanagari for Hindi, Roman for English) do NOT apply to this JSON; only its speaking "
                   "style does. Here text_roman is the line typed in Latin letters (Hindi romanised) and text_tts is the same "
                   "line fully in Devanagari, English words too (see the text_roman / text_tts rules below)."),
}

V3_TTS_MIXED = (
    "SCRIPTS (two fields per line, the SAME words in the same order):\n"
    "- text_roman: the line typed WhatsApp style in Latin letters only: English words as English, Hindi words romanised "
    "(kya, hai, nahi, theek, achha, kar do, mujhe), never IAST/diacritics, never Devanagari.\n"
    "- text_tts: the same line for a Hindi-English code-switch TTS: every Hindi word in Devanagari, every English word in "
    "Latin letters (exactly as the guidance above shows). Digits stay digits (15, 6:40, 98000 01234), IDs/plates/emails "
    "exactly as in text_roman (FD4821, DL 01 AB 4821, mehta.home21@yahoo.com), names of people, brands, restaurants and "
    "places in Latin letters as in text_roman, Rs 540 stays Rs 540. Never write a Hindi word in Latin letters in text_tts, "
    "never write an English word in Devanagari.\n"
    "  e.g. text_roman \"Theek hai, ek second, main dekh leti hoon.\" -> text_tts \"ठीक है, एक second, मैं देख लेती हूँ.\"\n"
    "  e.g. text_roman \"Achha, aapka order FD1452 out for delivery hai, ETA 12 minutes hai.\" -> text_tts "
    "\"अच्छा, आपका order FD1452 out for delivery है, ETA 12 minutes है.\"")

CONTROL_LANG = """LANGUAGE
Both speakers: ENGLISH ONLY, natural conversational English as in a normal customer-support phone call (clear, friendly, everyday phrasing; contractions are fine). No Hindi words at all (no ji, haan, achha, yaar, theek hai), even though the names and places are Indian.
Sign-off: natural and short, in your own words for this call (vary it), e.g. "Thanks for calling, have a great day." It adds no new fact and no promise (nothing about when an order, ride, refund or update will arrive or happen).""".rstrip()

CONTROL_TTS = ("text_tts = the SAME line in plain English Latin script (identical words to text_roman). Digits stay digits "
               "(15, 6:40, 98000 01234), IDs/plates/emails exactly as written, Rs 540 stays Rs 540; no Devanagari anywhere.")


RETRY_FIX = {
    "V3": ("rewrite the lines named above (a speaker with too little Hindi -> make those lines natural spoken Hinglish with "
           "Hindi grammar words and verbs; a Hindi word in Latin letters in text_tts -> write it in Devanagari), shorten "
           "over-long lines, add or lengthen turns if the call is too short. Keep text_tts in sync with text_roman."),
    "CONTROL": ("rewrite the lines named above (any Hindi word -> plain English), shorten over-long lines, add or lengthen "
                "turns if the call is too short. Keep text_tts identical to text_roman."),
    "V4": v4.RETRY_FIX,
}


def default_ids(variant, scen):
    if variant == "GATE1":
        return GATE1_CALLS
    if variant == "CONTROL":
        return control_call_ids()
    if variant == "V4":  # D2: all g1-g4 calls of the v2 scenario list (no drop list)
        return all_call_ids(scen)
    if variant == "V3":  # drop ids in data/dropped_call_ids.json (air_13_g1); V1/V2 defaults unchanged
        drop = set(json.load(open(f"{DATA}/dropped_call_ids.json"))["ids"])
        return [c for c in all_call_ids(scen) if c not in drop]
    return all_call_ids(scen)


def inputs_for(variant, scen_path=None, rec_path=None, records=True):
    """(scenarios, records) for a variant. V1/V2/V3/CONTROL/GATE1 default to the original files; V4 (D2) defaults to
    guidance/scenario_creation_v2.json + data/V4/records.json and exits if they are missing (no silent fallback)."""
    if variant == "V4":
        scen = v4.load_scenarios(scen_path or v4.V4_SCENARIOS)
        return scen, (v4.load_records(rec_path or v4.V4_RECORDS) if records else None)
    scen = v4.load_scenarios(scen_path) if scen_path else load_scenarios()
    return scen, (json.load(open(rec_path or f"{DATA}/records.json")) if records else None)


def common_prefix(variant):
    caller_reg, agent_reg = REGIMES["V3" if variant == "V4" else variant]  # V4 = V3 language recipe
    ab, cb = SHARE_BANDS[(agent_reg, "agent")], SHARE_BANDS[(caller_reg, "customer")]
    en = variant in ("GATE1", "CONTROL")

    def ex(reg, spk):
        if reg == "EN":
            return ""
        out = "Examples of this shape (" + spk + " lines):\n" + "\n".join(f"  - {x}" for x in EXAMPLES[(reg, spk)])
        if reg == "R1":
            out += ("\nThe examples above are mostly English; your " + spk + " lines need somewhat MORE Hindi, like these lines at the "
                    "target density (" + pct(SHARE_BANDS[(reg, spk)]) + " Hindi):\n" + "\n".join(f"  - {x}" for x in R1_DENSITY[spk]))
        return out

    tts_rules = (
        "text_tts = the SAME line written fully in Devanagari script, English words transliterated as they are pronounced "
        "(thank you -> थैंक यू, order -> ऑर्डर, delivery -> डिलीवरी, UPI -> यूपीआई, PNR -> पीएनआर, ETA -> ईटीए, PM -> पीएम, "
        "OK -> ओके, SOS -> एसओएस, GB -> जीबी, kg -> केजी, Rs 540 -> 540 रुपये, Gmail -> जीमेल, Tower B -> टावर बी, "
        "DLF -> डीएलएफ, NCR -> एनसीआर, Block C -> ब्लॉक सी, H-12 -> एच-12). Digits stay digits (15, 6:40, 98000 01234). "
        "IDs stay exactly as written in Latin letters+digits (FD4821, 18A, 42B), vehicle plates stay Latin (DL 01 AB 4821). "
        "Brand, restaurant, product and person names: Devanagari (QuickBite -> क्विकबाइट, Domino's -> डोमिनोज़). "
        "Emails in Devanagari words (mehta.home21@gmail.com -> मेहता डॉट होम 21 एट जीमेल डॉट कॉम). "
        "NO other Latin letters anywhere in text_tts; never the rupee symbol.")
    if en:
        lang = f"""LANGUAGE
Both speakers: {REGIME_TEXT['EN']}"""
        gender = "GENDER: English has no verb gender; just keep names consistent."
        checks = "\n".join(f"  {k + 1}. \"{c}\"" for k, c in enumerate(CHECK_LINES_EN))
    else:
        lang = f"""LANGUAGE (the most important rule; lines are counted word by word)
AGENT lines: {REGIME_TEXT[agent_reg]}
  Target: {pct(ab)} of the agent's words are Hindi (counted over all agent lines; numbers, IDs and names don't count). {AGENT_COUNT[agent_reg]}
{ex(agent_reg, 'agent')}
CUSTOMER lines: {REGIME_TEXT[caller_reg]}
  Target: {pct(cb)} of the customer's words are Hindi (counted over all customer lines). {CUST_COUNT[caller_reg]} Natural fillers: yaar, haan, achha, matlab, ek minute, arre.
{ex(caller_reg, 'customer')}
{MIX_NOTE if caller_reg != agent_reg else ""}Romanised Hindi spelling in text_roman: typed WhatsApp style (kya, hai, jo, wala, nahi, theek, achha, kar do), never IAST/diacritics, never Devanagari."""
        gender = """GENDER (verbs must match the speaker's gender; this is checked)
- Female speaker first-person verbs: kar rahi hoon, karti hoon, sakti hoon, deti hoon, leti hoon, batati hoon, gayi hoon, chahti hoon, karungi.
- Male speaker first-person verbs: kar raha hoon, karta hoon, sakta hoon, deta hoon, leta hoon, batata hoon, gaya hoon, chahta hoon, karunga.
- The CUSTOMER must use at least two first-person Hindi verbs that show their gender (e.g. "main kab se wait kar rahi hoon", "main office ja raha hoon").
- The AGENT's greeting already shows gender ("kar sakti hoon" / "kar sakta hoon"); keep every other agent first-person verb in the same gender."""
        checks = "\n".join(f"  {k + 1}. female: \"{c[0]}\" | male: \"{c[1]}\"" for k, c in enumerate(CHECK_LINES))
    if variant in ("V3", "V4"):
        if variant == "V4" and V3_TTS_SCRIPT != "devanagari":
            raise SystemExit("V4 uses all-Devanagari text_tts; unset HINGLISH_V3_TTS_SCRIPT")
        lang = V3_LANG_T.replace("{script_note}", V3_SCRIPT_NOTE[V3_TTS_SCRIPT])
        if V3_TTS_SCRIPT == "mixed":
            tts_rules = V3_TTS_MIXED
        else:  # full Devanagari text_tts (overrides the guidance's Latin-for-English script rule)
            tts_rules = ("text_roman: typed WhatsApp style in Latin letters (kya, hai, jo, wala, nahi, theek, achha, kar do), "
                         "never IAST/diacritics, never Devanagari.\n" + tts_rules +
                         " (This overrides the guidance's script rule: in text_tts English words are ALSO Devanagari.)")
        if variant == "V4":
            lang = v4.lang_block(lang)
    elif variant == "CONTROL":
        lang, tts_rules = CONTROL_LANG, CONTROL_TTS
    p = f"""You write ONE realistic phone call between a customer-support AGENT and a CUSTOMER in Delhi/NCR, India, as strict JSON. It is training data for a speech model, so follow every rule exactly.

{lang}

{gender}

FACTS AND TOOLS
- The agent knows ONLY the INFORMATION given below plus what the customer says in the call. Every fact the agent speaks (ID, item, price, time, date, status, ETA, address, name, seat, plan, number) must come from INFORMATION or an earlier customer line, copied exactly. Never invent anything else: no refund amounts, refund timelines, processing times, new ETAs, availability, SMS/email notifications, policies.
- Static facts: the agent answers DIRECTLY from INFORMATION with no check-line. Outside the write structure the agent NEVER says anything like "let me check", "main check karti hoon", "ek minute rukiye", "one moment", not even when asking for or confirming an ID (just ask: "Please aapka order ID bataiye."), and never give a reason like "taaki main details check kar sakun" / "so I can check the details" / "I will check".
- Only a confirm_write turn may say an action is done. No other agent turn says "I have added/updated ...", "request is submitted", "update kar diya" (no extra actions, no repeated "done" after the confirmation).
- The agent never claims to contact the rider/driver/restaurant/seller, track live, investigate or escalate. If the customer asks for something unknown, the agent says politely that it does not have that detail and repeats what is known.
- WRITE actions (only those listed for this call, each exactly once, in the listed order) ALWAYS use this exact structure:
    customer line giving the request + value  ->  agent CHECK-LINE (tag "check_line")  ->  agent CONFIRMATION (tag "confirm_write")
  The check-line and the confirmation are two consecutive AGENT turns. The check-line is copied EXACTLY from the phrasing assigned below (correct gender form). The confirmation says the action is done (or the request is submitted) and repeats the customer's value word for word (full address, full phone number digit by digit, full instruction, email, or the order/ride/booking ID for cancellations/refunds).
- List each write in "writes" with "tool" and "args"; the main argument value must be copied verbatim from the customer's line (address/location/instruction/phone/email) or from INFORMATION (IDs, plan name).
- Write all numbers as digits in text_roman too (15 minutes, Rs 540, 6:40 PM, 98000 01234); phone numbers as 5+5 digits.

STRUCTURE AND LENGTH
- Turn 1: the agent greeting, exactly the greeting given below (tag "greeting"). Last turn: a short agent sign-off (tag "signoff"), 6-12 words, after the customer says thanks/bye.
- Follow the scenario flow steps in order; expand each step into 1-3 turns. No backchannels ("hmm", "okay" alone) as separate turns.
- 14-16 turns in total. Agent lines 10-16 words each (hard limits 6-18). Customer lines 7-14 words each (hard limits 4-16; a long address or number dictation still must stay under 16 words). Confirmations echo the full value with a SHORT wrapper so they stay within 18 words (e.g. "Done ji, the instruction is added: <value>."). The whole call needs 190-240 words (about 80 seconds of speech); short calls are rejected, so make every line full and natural and expand the flow (customer explains their situation, agent restates details).
- Tags: "greeting", "read" (agent states a fact from INFORMATION), "check_line", "confirm_write", "signoff"; other turns have an empty tags list.
- "truncated": false on every turn unless this call has an interruption (see below).

{tts_rules}

CHECK-LINE PHRASINGS (for each write the exact phrasing to use is assigned below):
{checks}

OUTPUT: JSON {{"turns":[{{"speaker":"agent"|"customer","text_roman":"...","text_tts":"...","tags":[...],"truncated":false}}],"writes":[{{"tool":"...","args":{{...}}}}]}}

=== THIS CALL ===
"""
    return v4.patch_prefix(p) if variant == "V4" else p


def call_prompt(call_id, variant, scen, rec, interrupt, twopart):
    sid, g = call_id.rsplit("_", 1)
    s = scen[sid]
    p = PAIRINGS[g]
    ag, cg = p["agent_gender"], p["customer_gender"]
    agent = rec["agent_name_f"] if ag == "f" else rec["agent_name_m"]
    cust = (rec["customer_first_f"] if cg == "f" else rec["customer_first_m"]) + " " + rec["customer_last"]
    en = variant in ("GATE1", "CONTROL")
    v4_extra = []
    if en:
        greet = f"Hello, thank you for calling {rec['brand']}, this is {agent}. How can I help you today?"
    elif variant == "V4":
        greet, v4_extra, _ = v4.call_block(call_id, rec, s, ag, cg)
    else:
        greet = f"Hello, thank you for calling {rec['brand']}, this is {agent}. Main aapki kaise help kar {'sakti' if ag == 'f' else 'sakta'} hoon?"
    lines = [f"Company: {rec['brand']} ({rec['agent_type_plain']}). Scenario: {s['title']}.",
             f"AGENT: {agent}, {'FEMALE' if ag == 'f' else 'MALE'}. CUSTOMER: {cust}, {'FEMALE' if cg == 'f' else 'MALE'}.",
             f"INFORMATION (the agent's only knowledge): {rec['role_prompts'][g].split('Information: ', 1)[1]}",
             f"Customer's situation: {rec['caller_situation']}"]
    if rec["caller_values"]:
        lines.append("Values the CUSTOMER dictates in this call (use these exact values): " +
                     "; ".join(f"{k} = {v}" for k, v in rec["caller_values"].items()))
    lines.append("Scenario flow (follow in order):\n" + "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(s["flow"])))
    lines.append(f"Greeting (turn 1, exactly): \"{greet}\"")
    if len(s["flow"]) <= 6 and len(s["writes"]) <= 1 and variant != "V4":
        lines.append("This flow is short: expand it to at least 14 turns (the customer explains the situation and asks follow-ups, "
                     "the agent restates the known details) so the call reaches 190+ words.")
    if s["writes"]:
        wl = []
        for k, w in enumerate(s["writes"]):
            if en:
                cl = CHECK_LINES_EN[int(h01(call_id, "chk", str(k)) * 5)]
            else:
                c = CHECK_LINES[int(h01(call_id, "chk", str(k)) * 5)]
                cl = c[0] if ag == "f" else c[1]
            wl.append(f"  {k + 1}. {w}(args: {', '.join(TOOL_ARGS[w])}) - check-line: \"{cl}\"")
        lines.append("WRITE actions in this call (each: customer value -> check-line -> confirmation):\n" + "\n".join(wl))
    else:
        lines.append("WRITE actions in this call: NONE (writes must be an empty list; no check-lines, no confirm_write).")
    if twopart and s["writes"]:
        lines.append(f"TWO-PART ARGUMENT for the first write ({s['writes'][0]}): the customer first states the intent WITHOUT the value; "
                     "the agent asks for the value; the customer then gives the value; then the check-line; then the confirmation.")
    if interrupt:
        lines.append("INTERRUPTION: exactly once, the customer cuts the agent off mid-sentence. Pick one agent turn that is a "
                     "statement/explanation (not greeting, check-line, confirmation or sign-off), cut its text_roman and text_tts at a "
                     "word boundary in the middle of the sentence (no final punctuation), set \"truncated\": true on it, and make the "
                     "next turn the customer cutting in (e.g. " + ("\"Yes yes, I know that, but...\"" if variant == "CONTROL" else
                                                               "\"Haan haan, woh pata hai, but...\"") + "). All other turns truncated false.")
    else:
        lines.append("No interruption in this call: every turn has \"truncated\": false.")
    lines += v4_extra
    return "\n".join(lines) + "\n\nReturn only the JSON."


def call_schema(scen_writes):
    argprops = {}
    for w in scen_writes:
        for a in TOOL_ARGS[w]:
            argprops[a] = {"type": "string"}
    turn = {"type": "object", "properties": {
        "speaker": {"type": "string", "enum": ["agent", "customer"]},
        "text_roman": {"type": "string"}, "text_tts": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string", "enum": ["greeting", "read", "check_line", "confirm_write", "signoff"]}},
        "truncated": {"type": "boolean"}}, "required": ["speaker", "text_roman", "text_tts", "tags", "truncated"]}
    wr = {"type": "object", "properties": {"tool": {"type": "string", "enum": scen_writes or ["none"]},
                                           "args": {"type": "object", "properties": argprops}},
          "required": ["tool", "args"]}
    return {"type": "object", "properties": {"turns": {"type": "array", "items": turn, "minItems": 6, "maxItems": 20},
                                             "writes": {"type": "array", "items": wr, "maxItems": max(len(scen_writes), 0)}},
            "required": ["turns", "writes"]}


def postprocess(call_id, variant, scen, rec, raw, interrupt):
    sid, g = call_id.rsplit("_", 1)
    p = PAIRINGS[g]
    turns = []
    for i, t in enumerate(raw["turns"]):
        t = {"speaker": t["speaker"], "text_roman": t["text_roman"].strip(), "text_tts": t["text_tts"].strip(),
             "pause_after_s": 0.0, "truncated": bool(t.get("truncated")), "overlap_offset_s": 0.0,
             "tags": list(dict.fromkeys(t.get("tags") or []))}
        if "check_line" in t["tags"]:
            t["pause_after_s"] = round(0.8 + 0.7 * h01(call_id, "pause", str(i)), 2)
        else:
            t["pause_after_s"] = round(0.3 + 0.3 * h01(call_id, "pause", str(i)), 2)
        turns.append(t)
    for i, t in enumerate(turns):
        if t["truncated"]:  # cut mid-sentence: drop trailing punctuation/ellipsis/dashes
            t["text_roman"] = re.sub(r"[\s.,!?;:\u2026\u2014\u2013-]+$", "", t["text_roman"])
            t["text_tts"] = re.sub(r"[\s.,!?;:\u2026\u2014\u2013\u0964-]+$", "", t["text_tts"])
        if t["truncated"] and i + 1 < len(turns):
            turns[i + 1]["overlap_offset_s"] = round(0.3 + 0.5 * h01(call_id, "overlap"), 2)
            t["pause_after_s"] = 0.0
    if turns:
        turns[-1]["pause_after_s"] = 0.0
    writes = [{"tool": w["tool"], "args": {k: v for k, v in (w.get("args") or {}).items() if v},
               "caller_turn_idx": None, "confirm_turn_idx": None} for w in raw.get("writes", []) if w.get("tool") != "none"]
    agent = rec["agent_name_f"] if p["agent_gender"] == "f" else rec["agent_name_m"]
    cust_first = rec["customer_first_f"] if p["customer_gender"] == "f" else rec["customer_first_m"]
    record = {k: rec[k] for k in ("primary_id", "secondary_id", "phone", "facts", "distractors", "caller_values", "information")}
    record["customer_name"] = f"{cust_first} {rec['customer_last']}"
    call = {"call_id": call_id, "variant": variant, "scenario_id": sid, "agent_type": scen[sid]["agent_type"],
            "brand": rec["brand"], "agent_name": agent, "agent_gender": p["agent_gender"],
            "customer_gender": p["customer_gender"], "voices": dict(p["voices"]), "role_prompt": rec["role_prompts"][g],
            "record": record, "turns": turns, "writes": writes}
    if variant == "V4":  # D2: band + template ids (reproducible from call_id) and the flow (for the V4 judge)
        _, _, meta = v4.call_block(call_id, rec, scen[sid], p["agent_gender"], p["customer_gender"])
        call.update(meta)
        call["scenario_flow"] = list(scen[sid]["flow"])
    ex = hs.build_exclude(record, [agent, rec["brand"], cust_first, rec["customer_last"]])
    call["hindi_token_share"] = hs.call_shares(call, ex)
    return call


def run_job(llm, variant, call_ids, out_dir, scen, records, judge=True, seed_base=0, samples=(3, 2, 2), keep_existing=None):
    """Up to MAX_ATTEMPTS rounds. Round r draws samples[r-1] candidates per pending call (round 1 fresh; later rounds
    repair the least-failing previous candidate with its failure list). A call is accepted with its first candidate
    that passes all deterministic checks and the LLM judge."""
    os.makedirs(out_dir, exist_ok=True)
    interrupt, twopart = assignments(scen)
    prefix = common_prefix(variant)
    pending = {cid: "" for cid in call_ids}
    prev = {}
    done, log = {}, []
    first_fail = Counter()
    first_pass = 0
    sample_stats = {"samples": 0, "det_pass": 0, "full_pass": 0}
    per_attempt = []
    t_start = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        ids = list(pending)
        n_s = samples[min(attempt, len(samples)) - 1]
        reqs, prompts, schemas = [], [], []
        for cid in ids:
            sid = cid.rsplit("_", 1)[0]
            pr = prefix + call_prompt(cid, variant, scen, records[sid], cid in interrupt, cid in twopart)
            if pending[cid]:
                if prev.get(cid):
                    pr += ("\n\nYOUR PREVIOUS VERSION OF THIS CALL:\n" + prev[cid] +
                           "\n\nIt was REJECTED for these reasons:\n" + pending[cid] +
                           "\n\n(turn numbers count from 0 as in the \"turn\" field.) Return the corrected complete JSON (without the "
                           "\"turn\" field). You MUST change the flagged lines - an unchanged copy is rejected again. Fix exactly these problems: "
                           + (RETRY_FIX.get(variant) or
                              "rewrite the lines named above (too much Hindi -> replace Hindi verb phrases with English ones; too "
                              "little Hindi -> add Hindi fillers/adverbs/phrases), shorten over-long lines, add or lengthen turns if "
                              "the call is too short. Keep text_tts in sync with text_roman."))
                else:
                    pr += "\n\nYOUR PREVIOUS VERSION OF THIS CALL WAS REJECTED FOR THESE REASONS - fix all of them:\n" + pending[cid]
            for k in range(n_s):
                reqs.append((cid, k))
                prompts.append(pr)
                schemas.append(call_schema(scen[sid]["writes"]))
        outs = llm.chat(prompts, schema=schemas, temperature=0.8, max_tokens=4000,
                        seeds=[int(h01(cid, variant, str(attempt), str(k), str(seed_base)) * 1e9) for cid, k in reqs])
        cands = []  # (cid, k, call, F, ntok, fin, prevjson)
        raw_rows = []
        for (cid, k), (txt, fin, ntok) in zip(reqs, outs):
            sid = cid.rsplit("_", 1)[0]
            raw_rows.append({"call_id": cid, "attempt": attempt, "sample": k, "finish": fin, "text": txt})
            pj = None
            try:
                if fin == "length":
                    raise ValueError("output truncated (finish_reason=length)")
                raw = extract_json(txt)
                pj = json.dumps({"turns": [{"turn": i, **{kk: t.get(kk) for kk in ("speaker", "text_roman", "text_tts", "tags", "truncated")}}
                                           for i, t in enumerate(raw["turns"])], "writes": raw.get("writes", [])}, ensure_ascii=False)
                call = postprocess(cid, variant, scen, records[sid], raw, cid in interrupt)
                F = V.check_call(call, scen[sid], cid in interrupt, cid in twopart)
            except Exception as e:  # noqa
                call, F = None, [("parse", f"{type(e).__name__}: {str(e)[:200]}")]
            cands.append([cid, k, call, F, ntok, fin, pj])
        append_jsonl(f"{out_dir}/raw_outputs.jsonl", raw_rows)
        sample_stats["samples"] += len(cands) if attempt == 1 else 0
        # LLM judge on deterministic passes
        to_judge = [c for c in cands if not c[3]]
        if attempt == 1:
            sample_stats["det_pass"] += len(to_judge)
        if judge and to_judge:
            verdicts = V.judge_calls(llm, [c[2] for c in to_judge])
            for c, vio in zip(to_judge, verdicts):
                c[2]["judge"] = vio
                for x in vio:
                    c[3].append(("llm_judge", f"turn {x['turn']} {x['kind']}: '{x['quote']}' - {x['reason']}"))
        if attempt == 1:
            sample_stats["full_pass"] += sum(1 for c in cands if not c[3])
        n_ok = 0
        by_cid = {}
        for c in cands:
            by_cid.setdefault(c[0], []).append(c)
        for cid in ids:
            cs = by_cid[cid]
            for c in cs:
                log.append({"call_id": cid, "variant": variant, "attempt": attempt, "sample": c[1], "pass": not c[3],
                            "out_tokens": c[4], "finish": c[5], "failures": [f"{a}: {m}" for a, m in c[3]],
                            "hindi_token_share": c[2]["hindi_token_share"] if c[2] else None,
                            "est_duration_s": c[2].get("est_duration_s") if c[2] else None})
            ok = [c for c in cs if not c[3]]
            if attempt == 1:
                if ok:
                    first_pass += 1
                else:
                    for a in {a for c in cs for a, _ in c[3]}:
                        first_fail[a] += 1
            if ok:
                call = ok[0][2]
                n_ok += 1
                call["gen_attempts"] = attempt
                call["gen_sample"] = ok[0][1]
                done[cid] = call
                pending.pop(cid)
            else:
                best = min(cs, key=lambda c: (c[2] is None, len(c[3])))
                if best[6]:
                    prev[cid] = best[6]
                pending[cid] = "\n".join(f"- {m}" for _, m in best[3][:12])
                if attempt == MAX_ATTEMPTS and best[2] is not None:
                    best[2]["gen_attempts"] = attempt
                    best[2]["validation_failures"] = [f"{a}: {m}" for a, m in best[3]]
                    done.setdefault(cid + "#FAILED", best[2])
        per_attempt.append({"attempt": attempt, "submitted": len(ids), "samples_each": n_s, "passed": n_ok})
        print(f"[{variant}] attempt {attempt} ({n_s} samples/call): {n_ok}/{len(ids)} passed", flush=True)
    wall = time.time() - t_start
    if keep_existing:
        old = {}
        if os.path.exists(f"{out_dir}/calls.jsonl"):
            for ln in open(f"{out_dir}/calls.jsonl", encoding="utf-8"):
                c = json.loads(ln)
                old[c["call_id"]] = c
        old.update({c: done[c] for c in call_ids if c in done})
        done.update(old)
        call_ids = [c for c in keep_existing if c in done or c + "#FAILED" in done]
    good = [done[c] for c in call_ids if c in done]
    failed = [done[c + "#FAILED"] for c in call_ids if c + "#FAILED" in done]
    with open(f"{out_dir}/calls.jsonl", "w", encoding="utf-8") as f:
        for c in good:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/failed_calls.jsonl", "w", encoding="utf-8") as f:
        for c in failed:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/gen_log.jsonl", "a", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_samples(f"{out_dir}/SAMPLES.md", variant, good)
    summ = {"variant": variant, "n_calls": len(call_ids), "final_pass": len(good), "first_attempt_pass": first_pass,
            "first_attempt_samples": sample_stats,
            "first_attempt_fail_by_check": dict(first_fail), "per_attempt": per_attempt, "wall_s": round(wall, 1),
            "calls_per_min_final": round(len(good) / (wall / 60), 2) if wall else None}
    jdump(f"{out_dir}/gen_summary.json", summ)
    print(json.dumps(summ), flush=True)
    return summ


def write_samples(path, variant, calls, n=5):
    rnd = random.Random(1234)
    pick = rnd.sample(calls, min(n, len(calls)))
    out = [f"# SAMPLES — {variant} ({len(pick)} random calls of {len(calls)})\n"]
    for c in pick:
        out.append(f"## {c['call_id']}  (agent {c['agent_name']} {c['agent_gender']} / customer {c['customer_gender']}; "
                   f"voices {c['voices']}; Hindi share {c['hindi_token_share']}; est {c.get('est_duration_s')} s)\n")
        out.append(f"**role_prompt:** {c['role_prompt']}\n")
        out.append("| # | spk | text_roman | text_tts | pause | trunc | overlap | tags |\n|---|---|---|---|---|---|---|---|")
        for i, t in enumerate(c["turns"]):
            out.append(f"| {i} | {t['speaker']} | {t['text_roman']} | {t['text_tts']} | {t['pause_after_s']} | "
                       f"{'Y' if t['truncated'] else ''} | {t['overlap_offset_s'] or ''} | {','.join(t['tags'])} |")
        out.append(f"\n**writes:** `{json.dumps(c['writes'], ensure_ascii=False)}`\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["V1", "V2", "V3", "GATE1", "CONTROL", "V4"])
    ap.add_argument("--calls", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--job", action="append", default=[], help="VARIANT[:id1,id2,...[:outdir]]")
    ap.add_argument("--with-records", action="store_true", help="(re)build data/records.json first in the same model load")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the first prompt and exit (no GPU)")
    ap.add_argument("--samples", default="3,2,2", help="candidates per call in rounds 1,2,3")
    ap.add_argument("--only-missing", action="store_true",
                    help="generate only call_ids not yet in <out>/calls.jsonl (new seeds), merge into it")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--scenarios", help="scenario file (default: guidance/scenario_creation.json; V4: _v2.json)")
    ap.add_argument("--records", help="records file (default: data/records.json; V4: data/V4/records.json)")
    a = ap.parse_args()
    samples = tuple(int(x) for x in a.samples.split(","))
    jobs = []
    if a.variant:
        scen, _ = inputs_for(a.variant, a.scenarios, a.records, records=False)
        ids = a.calls or default_ids(a.variant, scen)
        jobs.append((a.variant, ids, a.out or f"{DATA}/{a.variant}"))
    for j in a.job:
        parts = j.split(":")
        v = parts[0]
        scen, _ = inputs_for(v, a.scenarios, a.records, records=False)
        ids = parts[1].split(",") if len(parts) > 1 and parts[1] else default_ids(v, scen)
        jobs.append((v, ids, parts[2] if len(parts) > 2 else f"{DATA}/{v}"))
    if a.dry:
        v, ids, _ = jobs[0]
        scen, records = inputs_for(v, a.scenarios, a.records)
        interrupt, twopart = assignments(scen)
        cid = ids[0]
        print(common_prefix(v) + call_prompt(cid, v, scen, records[cid.rsplit('_', 1)[0]], cid in interrupt, cid in twopart))
        return
    from common import LLM
    llm = LLM()
    if a.with_records or not os.path.exists(f"{DATA}/records.json"):
        from records import make_records
        make_records(llm)
    allsum = []
    for v, ids, out in jobs:
        scen, records = inputs_for(v, a.scenarios, a.records)
        keep = None
        if a.only_missing:
            have = set()
            if os.path.exists(f"{out}/calls.jsonl"):
                have = {json.loads(x)["call_id"] for x in open(f"{out}/calls.jsonl", encoding="utf-8")}
            keep = list(ids)
            ids = [c for c in ids if c not in have]
            print(f"[{v}] only-missing: {len(ids)} to generate", flush=True)
        allsum.append(run_job(llm, v, ids, out, scen, records, judge=not a.no_judge, seed_base=a.seed_base,
                              samples=samples, keep_existing=keep))
    print(f"[done] llm load {llm.load_s:.0f}s, gen {llm.gen_s:.0f}s, out tokens {llm.tok_out} "
          f"({llm.tok_out / max(llm.gen_s, 1):.0f} tok/s aggregate)", flush=True)


if __name__ == "__main__":
    main()
