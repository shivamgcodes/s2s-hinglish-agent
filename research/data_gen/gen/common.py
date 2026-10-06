"""Shared constants/helpers for the Hinglish data generator. Pure python except LLM() (lazy vLLM import)."""
import hashlib
import json
import os
import re
import time

ROOT = os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")  # data root (data/ lives here)
GUIDE = os.environ.get("HINGLISH_GUIDANCE") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guidance")
DATA = f"{ROOT}/data"
GEN = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "/workspace/gemma/models/gemma-4-31B-it")

# --------------------------------------------------------------------------- scenarios
def load_scenarios():
    d = json.load(open(f"{GUIDE}/scenario_creation.json"))
    out = {}
    for a in d["agents"]:
        for s in a["scenarios"]:
            out[s["id"]] = {**s, "agent_type": a["agent_type"], "brand_context": a["brand_context"],
                            "static_read": a["tools"]["static_read"], "static_write_all": a["tools"]["static_write"],
                            "writes": s.get("tools") or []}
    return out


AGENT_PLAIN = {  # "You work for X which is a <plain>"
    "food_delivery_support": "food delivery company",
    "ecommerce_support": "online shopping company",
    "cab_ride_support": "cab booking company",
    "subscription_account_support": "digital subscription service",
    "airport_ticket_counter": "passenger airline",
}
# three fictional brands per agent type, each with one female and one male agent name
BRANDS = {
    "food_delivery_support": [("QuickBite", "Priya", "Rahul"), ("TiffinGo", "Neha", "Arjun"), ("Biteway", "Kavya", "Vikram")],
    "ecommerce_support": [("ShopKart", "Ananya", "Rohit"), ("Bazaarly", "Pooja", "Karan"), ("Kartify", "Sneha", "Aditya")],
    "cab_ride_support": [("RideNow", "Divya", "Manish"), ("CityCab", "Ritu", "Sandeep"), ("Savaari Go", "Meera", "Nikhil")],
    "subscription_account_support": [("StreamBox", "Isha", "Varun"), ("TunePlus", "Simran", "Gaurav"), ("ReadMore", "Tanvi", "Akash")],
    "airport_ticket_counter": [("Indus Airways", "Nidhi", "Abhishek"), ("SkyIndia Airlines", "Shruti", "Deepak"), ("Udaan Air", "Aditi", "Harsh")],
}
ID_PREFIX = {"food_delivery_support": "FD", "ecommerce_support": "EC", "cab_ride_support": "RD",
             "subscription_account_support": "AC", "airport_ticket_counter": "PN"}

CUSTOMER_LAST = ["Mehta", "Sharma", "Verma", "Gupta", "Malhotra", "Kapoor", "Arora", "Bansal", "Chopra", "Saxena",
                 "Agarwal", "Bhatia", "Khanna", "Sethi", "Tandon", "Jain", "Singh", "Yadav", "Chauhan", "Rawat",
                 "Mishra", "Tiwari", "Pandey", "Srivastava", "Joshi", "Nair", "Iyer", "Reddy", "Das", "Bose",
                 "Ahuja", "Grover", "Kohli", "Luthra", "Mittal", "Goel"]
CUSTOMER_FIRST_M = ["Rohan", "Amit", "Saurabh", "Ankit", "Vivek", "Rajat", "Mohit", "Kunal", "Siddharth", "Pankaj",
                    "Tarun", "Ashish", "Lalit", "Naveen", "Yash", "Pranav", "Dhruv", "Ritesh", "Sumit", "Manoj",
                    "Hemant", "Ajay", "Vishal", "Lokesh"]
CUSTOMER_FIRST_F = ["Riya", "Anjali", "Shreya", "Nisha", "Pallavi", "Swati", "Megha", "Komal", "Preeti", "Sakshi",
                    "Jyoti", "Kritika", "Aarti", "Bhavna", "Rashmi", "Payal", "Garima", "Shalini", "Monika", "Ritika",
                    "Deepika", "Sonal", "Tanya", "Neelam"]

# gender pairings (customer, agent) and voices
PAIRINGS = {
    "g1": {"customer_gender": "m", "agent_gender": "f", "voices": {"customer": "hm_psi", "agent": "hf_beta"}},
    "g2": {"customer_gender": "m", "agent_gender": "m", "voices": {"customer": "hm_psi", "agent": "hm_omega"}},
    "g3": {"customer_gender": "f", "agent_gender": "f", "voices": {"customer": "hf_alpha", "agent": "hf_beta"}},
    "g4": {"customer_gender": "f", "agent_gender": "m", "voices": {"customer": "hf_alpha", "agent": "hm_omega"}},
}
REGIMES = {"V1": ("R1", "R1"), "V2": ("R2", "R1"), "V3": ("R2", "R2"), "GATE1": ("EN", "EN"),
           "CONTROL": ("EN", "EN")}  # (caller, agent)
SHARE_BANDS = {("R1", "agent"): (0.25, 0.40), ("R1", "customer"): (0.35, 0.55),
               ("R2", "agent"): (0.45, 0.70), ("R2", "customer"): (0.45, 0.70),
               ("EN", "agent"): (0.0, 0.08), ("EN", "customer"): (0.0, 0.08)}
# V3 (user decision 2026-10-03 ~19:00 IST): natural Hinglish both sides via novasynth guidance; no R1/R2 regime, no
# Hindi-share band. hindi_token_share is still computed/reported; validation only requires a per-speaker floor.
V3_MIN_HINDI = 0.25       # each speaker's Hindi token share over FREE lines (not greeting/check_line/signoff) must be
                          # >= this (v3review 2026-10-03: was 0.15 on all lines, which passed an agent speaking English
                          # on every free line because the fixed Hinglish greeting/check-lines carried the share)
V3_MAX_EN_LINES = 1       # at most this many pure-English free lines per speaker (Hindi share 0)
CONTROL_MAX_HINDI = 0.05  # CONTROL (English, PersonaPlex style, eval-only forgetting set): each speaker < this
# V3 text_tts script for the IndicF5 code-switch TTS: "mixed" (Hindi words Devanagari, English words Latin, numbers/IDs
# as in text_roman) or "devanagari" (everything Devanagari, same text_tts rules as V1).
# USER DECISION 2026-10-03 ~19:15 IST (audio/F5_BACKEND.md): ALL-DEVANAGARI input for the code-switch model -> default.
V3_TTS_SCRIPT = os.environ.get("HINGLISH_V3_TTS_SCRIPT", "devanagari")

# check-line phrasings (female, male); index used in prompts and validation
CHECK_LINES = [
    ("Ek minute rukiye, main check karke batati hoon.", "Ek minute rukiye, main check karke batata hoon."),
    ("Theek hai, ek second, main dekh leti hoon.", "Theek hai, ek second, main dekh leta hoon."),
    ("Haan ji, abhi update kar deti hoon, ek minute.", "Haan ji, abhi update kar deta hoon, ek minute."),
    ("Sure, let me check that for you, bas ek second.", "Sure, let me check that for you, bas ek second."),
    ("Checking, just a sec, ek minute.", "Checking, just a sec, ek minute."),
]
CHECK_LINES_EN = ["Sure, let me check that for you, one second.", "Checking, just a sec, one moment.",
                  "Okay, let me update that for you, one moment.", "One moment please, let me do that for you.",
                  "Alright, give me just a second."]

# canonical write args per tool (the "argument string" that must be echoed is ECHO_ARG)
TOOL_ARGS = {
    "cancel_order": ["order_id"], "change_delivery_address": ["order_id", "address"],
    "add_delivery_instruction": ["order_id", "instruction"], "update_contact_number": ["phone"],
    "request_refund": ["reference_id", "reason"], "cancel_ride": ["ride_id"],
    "change_pickup_location": ["ride_id", "location"], "change_drop_location": ["ride_id", "location"],
    "add_driver_instruction": ["ride_id", "instruction"], "cancel_subscription": ["plan"],
    "pause_subscription": ["plan"], "update_email_address": ["email"],
    "request_ticket_cancellation": ["pnr"],
}
ECHO_ARG = {"cancel_order": "order_id", "change_delivery_address": "address", "add_delivery_instruction": "instruction",
            "update_contact_number": "phone", "request_refund": "reference_id", "cancel_ride": "ride_id",
            "change_pickup_location": "location", "change_drop_location": "location",
            "add_driver_instruction": "instruction", "cancel_subscription": "plan", "pause_subscription": "plan",
            "update_email_address": "email", "request_ticket_cancellation": "pnr"}
# D2 / V4 (2026-10-04): write tools of the two new agent types in guidance/scenario_creation_v2.json. Keys ADDED only
# (V1/V3 tools unchanged). Dictated args (address/phone/reason) are spoken by the customer; card_last4 / transaction_id
# / service / pack / sim_number come from the record (validate.write_echo: record or earlier customer turn).
TOOL_ARGS.update({
    "block_card": ["card_last4"], "raise_transaction_dispute": ["transaction_id", "reason"],
    "request_card_replacement": ["card_last4", "address"],
    "deactivate_service": ["service"], "activate_pack": ["pack"], "block_sim": ["sim_number"],
    "update_alternate_number": ["phone"],
})
ECHO_ARG.update({
    "block_card": "card_last4", "raise_transaction_dispute": "transaction_id", "request_card_replacement": "address",
    "deactivate_service": "service", "activate_pack": "pack", "block_sim": "sim_number", "update_alternate_number": "phone",
})

# GATE1: English-only plumbing set (training scenarios, not held out)
GATE1_CALLS = [f"food_05_{g}" for g in ("g1", "g2", "g3", "g4")] + [f"cab_06_{g}" for g in ("g1", "g2", "g3", "g4")] \
    + ["sub_09_g1", "sub_09_g2"]


def control_call_ids():
    """CONTROL set: the 25 held-out scenarios x g1-g4 (100 calls), same records/call_ids as the V3 held-out calls."""
    h = json.load(open(f"{DATA}/holdout.json"))
    return [f"{sid}_{g}" for sid in h["test_scenarios"] for g in PAIRINGS]


def h01(*parts):  # noqa: E302
    """Deterministic float in [0,1) from strings."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16) / 16 ** 12


def all_call_ids(scen):
    return [f"{sid}_{g}" for sid in scen for g in PAIRINGS]


def assignments(scen):
    """Deterministic per-call interruption (exactly round(30%) of calls) and two-part-argument (round(20%) of
    write calls) flags. Keyed on call_id only (no variant) so V1/V2/V3 share them."""
    ids = all_call_ids(scen)
    n_int = round(0.30 * len(ids))
    interrupt = set(sorted(ids, key=lambda c: h01("interrupt", c))[:n_int])
    # two-part only where the first write's echoed argument is dictated by the caller (not refund/cancel), and not in
    # the scenarios whose flow already has the caller correcting a value mid-dictation
    wids = [c for c in ids if scen[c.rsplit("_", 1)[0]]["writes"]
            and ECHO_ARG[scen[c.rsplit("_", 1)[0]]["writes"][0]] in ("address", "location", "instruction", "phone", "email")
            and c.rsplit("_", 1)[0] not in ("food_08", "ecom_13")]
    n_two = round(0.20 * sum(1 for c in ids if scen[c.rsplit("_", 1)[0]]["writes"]))  # 20% of ALL write calls
    twopart = set(sorted(wids, key=lambda c: h01("twopart", c))[:n_two])
    return interrupt, twopart


# --------------------------------------------------------------------------- JSON helpers
def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i = text.find("{")
    j = text.rfind("}")
    return json.loads(text[i:j + 1])


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- LLM
class LLM:
    """Thin wrapper over vLLM offline chat with JSON-schema structured output."""

    def __init__(self, gpu_util=0.94, max_model_len=8192, kv_fp8=True):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/workspace/hf")
        os.environ.setdefault("VLLM_CACHE_ROOT", f"{ROOT}/.vllm_cache")
        from vllm import LLM as _LLM
        t0 = time.time()
        kw = dict(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_model_len,
                  limit_mm_per_prompt={"image": 0, "audio": 0}, enable_prefix_caching=True)
        if kv_fp8:
            kw["kv_cache_dtype"] = "fp8"
        try:
            self.llm = _LLM(**kw)
        except Exception as e:  # fall back to bf16 KV
            if not kv_fp8:
                raise
            print(f"[llm] fp8 KV failed ({type(e).__name__}: {str(e)[:200]}); retrying bf16 KV", flush=True)
            kw.pop("kv_cache_dtype")
            self.llm = _LLM(**kw)
        self.kv = kw.get("kv_cache_dtype", "auto")
        self.load_s = time.time() - t0
        self.tok_out = 0
        self.gen_s = 0.0
        print(f"[llm] loaded in {self.load_s:.0f}s kv={self.kv}", flush=True)

    def chat(self, prompts, schema=None, temperature=0.8, max_tokens=3500, seeds=None, system=None):
        """prompts: list of user strings. Returns list of (text, finish_reason, n_out_tokens)."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        params = []
        for i, _ in enumerate(prompts):
            kw = dict(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
            if seeds is not None:
                kw["seed"] = int(seeds[i])
            if schema is not None:
                sc = schema[i] if isinstance(schema, list) else schema
                kw["structured_outputs"] = StructuredOutputsParams(json=sc, disable_any_whitespace=True)
            params.append(SamplingParams(**kw))
        convs = []
        for p in prompts:
            msgs = [{"role": "system", "content": system}] if system else []
            convs.append(msgs + [{"role": "user", "content": p}])
        t0 = time.time()
        outs = self.llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
        dt = time.time() - t0
        res = []
        n = 0
        for o in outs:
            oo = o.outputs[0]
            n += len(oo.token_ids)
            res.append((oo.text, oo.finish_reason, len(oo.token_ids)))
        self.tok_out += n
        self.gen_s += dt
        print(f"[llm] batch {len(prompts)} -> {n} out tokens in {dt:.0f}s ({n / max(dt, 1e-6):.0f} tok/s)", flush=True)
        return res
