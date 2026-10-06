"""Role prompt + voice selection used in training and in the live demo (the PersonaPlex record/pairing format).

Single source (packages/personaplex_lora, D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07): imported by
deploy/common/session.py (worker + Space; the builds copy this file next to session.py), by research/eval_harness
(tcommon.ascii_prompt) and as `personaplex_lora.role_prompt` when pip-installed. Origin: DEP1 common/session.py and the
hinglish repo's tests/tcommon.py ascii_prompt. The role prompt is the training-format prompt stored in a record's
role_prompts[gN], folded to ASCII (the PersonaPlex tokenizer maps unknown characters to id 0 == EPAD).

    rec = json.load(open("examples/food_23.json"))
    prompt = build_role_prompt(rec, "g1")      # "You work for <brand> which is a <agent_type_plain> and your name is
                                               #  <agent name>. Information: Customer: <name>, phone <phone>. <facts>"
    voice = voice_for("g1")                    # "NATF2.pt"

Pairings: g1/g3 = female agent (voice NATF2.pt), g2/g4 = male agent (voice NATM1.pt); g1/g2 use the male customer
name, g3/g4 the female one. The voices are the stock PersonaPlex voice prompts from the base repo's voices.tgz.
"""
import re
import unicodedata

PAIRINGS = ("g1", "g2", "g3", "g4")
AGENT_GENDER = {"g1": "f", "g2": "m", "g3": "f", "g4": "m"}
VOICE = {"f": "NATF2.pt", "m": "NATM1.pt"}

_ASCII_MAP = {"₹": "Rs ", "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "}


def ascii_prompt(s: str) -> str:
    for k, v in _ASCII_MAP.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def build_role_prompt(record: dict, pairing: str = "g1") -> str:
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}")
    return ascii_prompt(record["role_prompts"][pairing])


def voice_for(pairing: str = "g1") -> str:
    return VOICE[AGENT_GENDER[pairing]]
