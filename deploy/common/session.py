"""S2S serverless: fork of DEP1 common/session.py (session config; logic unchanged). D-SINGLE-SOURCE 2026-10-07: the
role-prompt format (ascii_prompt, build_role_prompt, PAIRINGS, AGENT_GENDER, VOICE) lives ONCE in
packages/personaplex_lora/role_prompt.py; it is imported from $S2S_FORMATS, from next to this file (the staged Space
tree copies it there), from /opt/s2s/packages/personaplex_lora (worker image) or <repo>/packages/personaplex_lora.

Only the two record paths changed: they come from env (DESIGN.md 1.2) with defaults relative to this file, so the
same file works in the worker image (/opt/s2s/common), in the Space image and in the repo on runpod2:
    S2S_RECORDS          default <this dir>/data/records_v4.json   (copy of the DEP1 V4 demo records)
    S2S_NEEDLE_RECORDS   default <repo>/packages/needle_router/n1/src/records.json (V3 records, offline eval only)
Pure python, no GPU, importable from every venv (venv-pp, venv-asr, venv-needle) and from the Space:
    sys.path.insert(0, <repo>/common); import session

Role prompt = the exact training-format prompt already stored in records.json (role_prompts[gN]), folded to
ASCII exactly as tests/tcommon.ascii_prompt does for the V3 tests (PersonaPlex tokenizer unk id 0 == EPAD).
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in (os.environ.get("S2S_FORMATS"), str(_HERE), str(_HERE.parent / "packages" / "personaplex_lora"),   # image: /opt/s2s/packages
           str(_HERE.parent.parent / "packages" / "personaplex_lora")):                                       # checkout
    if _d and (Path(_d) / "role_prompt.py").is_file():
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
from role_prompt import AGENT_GENDER, PAIRINGS, VOICE, ascii_prompt, build_role_prompt  # noqa: E402,F401
RECORDS_V4 = Path(os.environ.get("S2S_RECORDS") or _HERE / "data" / "records_v4.json")   # demo records (spec)
NEEDLE_RECORDS_V3 = Path(os.environ.get("S2S_NEEDLE_RECORDS")
                         or _HERE.parent.parent / "packages" / "needle_router" / "n1" / "src" / "records.json")   # V3 records (offline eval)
SUPPORTED_AGENT_TYPES = ("food_delivery_support", "ecommerce_support", "cab_ride_support",
                         "subscription_account_support", "airport_ticket_counter")  # == needle tools.AGENT_TYPES
DEFAULT_SEED = 1001


@lru_cache(maxsize=4)
def load_records(path: str = str(RECORDS_V4)) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def demo_records(path: str = str(RECORDS_V4)) -> dict:
    """record_id (== scenario_id, e.g. 'food_01') -> record, restricted to SUPPORTED_AGENT_TYPES."""
    return {k: v for k, v in load_records(path).items() if v["agent_type"] in SUPPORTED_AGENT_TYPES}


def session_config(record_id: str, pairing: str = "g1", seed: int = DEFAULT_SEED,
                   records_path: str = str(RECORDS_V4)) -> dict:
    """The session config object (INTERFACE.md section 3). Raises KeyError on unknown record_id."""
    rec = load_records(records_path)[record_id]
    g = AGENT_GENDER[pairing]
    return {"type": "session_config", "agent_type": rec["agent_type"], "record_id": record_id,
            "pairing": pairing, "agent_gender": g, "voice": VOICE[g],
            "role_prompt": build_role_prompt(rec, pairing), "seed": int(seed),
            "router_supported": rec["agent_type"] in SUPPORTED_AGENT_TYPES}


if __name__ == "__main__":
    recs = load_records()
    demo = demo_records()
    print(f"records {len(recs)}, demo (router-supported types) {len(demo)}")
    c = session_config("food_01", "g2")
    print(json.dumps(c, indent=1)[:600])
    bad = [(k, g) for k, v in recs.items() for g in PAIRINGS
           if build_role_prompt(v, g) != v["role_prompts"][g]]
    print(f"prompts changed by ascii folding: {len(bad)} of {len(recs) * 4}")
