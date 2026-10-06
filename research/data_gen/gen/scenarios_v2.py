"""D2 / V4: generate ~90 new scenarios with Gemma 4 31B (one vLLM load) and write guidance/scenario_creation_v2.json.

New file; does not touch guidance/scenario_creation.json or any existing code path.
Stages per round (all in one model load):
  1. generation  : per agent type, prompts of <=5 slots; each slot has a FIXED tools[] list (deterministic plan)
  2. shape check : deterministic (keys, ids, flow 4-8 steps, roles, tools == slot tools, banned-phrase regex, title dup)
  3. rule judge  : Gemma, temperature 0, one scenario per prompt (static writes only, one entity, no dynamic, no reversal,
                   every tool performed+confirmed, no write steps for undeclared actions)
  4. dedupe judge: Gemma, temperature 0, candidate vs the type's original titles + already-accepted new titles
                   (+ earlier candidates of the same type in this round)
Rejected slots are regenerated with feedback, up to MAX_ROUNDS.
Run: cd gen && flock ../gpu.lock env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf \
       /workspace/venv-vllm/bin/python scenarios_v2.py
"""
import copy
import difflib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GUIDE, ROOT, extract_json, h01, jdump  # noqa: E402

SRC = f"{GUIDE}/scenario_creation.json"
OUT = f"{GUIDE}/scenario_creation_v2.json"
WORK = f"{ROOT}/data/V4/scen_v2_work"
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 6))
CHUNK = 5

NEW_TYPES = [
    {"agent_type": "bank_card_support",
     "brand_context": "HDFC/ICICI-style bank credit and debit card helpline",
     "tools": {"static_read": ["get_card_status", "get_card_details", "get_recent_transactions", "get_statement_details",
                               "get_reward_points", "get_contact_details"],
               "static_write": ["block_card", "raise_transaction_dispute", "request_card_replacement",
                                "update_contact_number", "update_email_address"]}},
    {"agent_type": "telecom_prepaid_support",
     "brand_context": "Jio/Airtel-style prepaid mobile operator",
     "tools": {"static_read": ["get_plan_details", "get_balance_details", "get_validity_details", "get_recharge_history",
                               "get_active_services", "get_available_packs", "get_sim_details"],
               "static_write": ["deactivate_service", "activate_pack", "block_sim", "update_alternate_number",
                                "request_refund"]}},
]
ID_STEM = {"food_delivery_support": "food", "ecommerce_support": "ecom", "cab_ride_support": "cab",
           "subscription_account_support": "sub", "airport_ticket_counter": "air", "bank_card_support": "bank",
           "telecom_prepaid_support": "tel"}
N_NEW = {"food_delivery_support": 10, "ecommerce_support": 10, "cab_ride_support": 10,
         "subscription_account_support": 10, "airport_ticket_counter": 10, "bank_card_support": 20,
         "telecom_prepaid_support": 20}
# write-count mix per type: (n0, n1, n2, n3)
MIX = {10: (3, 4, 2, 1), 20: (5, 8, 5, 2)}
PAIRS = {
    "food_delivery_support": [["change_delivery_address", "add_delivery_instruction"],
                              ["update_contact_number", "add_delivery_instruction"], ["cancel_order", "request_refund"]],
    "ecommerce_support": [["cancel_order", "request_refund"], ["change_delivery_address", "add_delivery_instruction"],
                          ["update_contact_number", "change_delivery_address"]],
    "cab_ride_support": [["change_pickup_location", "add_driver_instruction"], ["change_drop_location", "update_contact_number"],
                         ["cancel_ride", "request_refund"]],
    "subscription_account_support": [["update_email_address", "update_contact_number"], ["pause_subscription", "update_email_address"],
                                     ["cancel_subscription", "request_refund"]],
    "airport_ticket_counter": [["request_ticket_cancellation", "request_refund"], ["update_email_address", "request_refund"],
                               ["update_contact_number", "update_email_address"]],  # r2: idx1 swapped (phone+email == air_11)
    "bank_card_support": [["block_card", "request_card_replacement"], ["raise_transaction_dispute", "block_card"],
                          ["update_contact_number", "update_email_address"], ["raise_transaction_dispute", "update_email_address"],
                          ["block_card", "update_contact_number"]],
    "telecom_prepaid_support": [["deactivate_service", "request_refund"], ["activate_pack", "update_alternate_number"],
                                ["block_sim", "update_alternate_number"], ["deactivate_service", "activate_pack"],
                                ["request_refund", "update_alternate_number"]],
}
TRIPLES = {
    "food_delivery_support": [["change_delivery_address", "update_contact_number", "add_delivery_instruction"]],
    "ecommerce_support": [["change_delivery_address", "update_contact_number", "add_delivery_instruction"]],
    "cab_ride_support": [["change_drop_location", "add_driver_instruction", "update_contact_number"]],
    "subscription_account_support": [["update_contact_number", "update_email_address", "pause_subscription"]],
    "airport_ticket_counter": [["request_ticket_cancellation", "request_refund", "update_email_address"]],
    "bank_card_support": [["block_card", "raise_transaction_dispute", "request_card_replacement"],
                          ["block_card", "request_card_replacement", "update_contact_number"]],
    "telecom_prepaid_support": [["deactivate_service", "request_refund", "activate_pack"],
                                ["block_sim", "update_alternate_number", "request_refund"]],
}
TOOL_DESC = {
    "cancel_order": "cancel the order (arg: order_id)",
    "change_delivery_address": "change the delivery address to a new address the customer dictates (args: order_id, address)",
    "add_delivery_instruction": "add a short delivery instruction the customer dictates (args: order_id, instruction)",
    "update_contact_number": "update the registered contact phone number to a new number the customer dictates (arg: phone)",
    "request_refund": "submit a refund request for the known order/charge/recharge (args: reference_id, reason); the agent "
                      "only says the request is submitted, never an amount or a timeline",
    "cancel_ride": "cancel the ride (arg: ride_id)",
    "change_pickup_location": "change the pickup point to a location the customer dictates (args: ride_id, location)",
    "change_drop_location": "change the destination to a location the customer dictates (args: ride_id, location)",
    "add_driver_instruction": "add a short instruction for the driver that the customer dictates (args: ride_id, instruction)",
    "cancel_subscription": "cancel the subscription plan (arg: plan)",
    "pause_subscription": "pause the subscription plan (arg: plan)",
    "update_email_address": "update the registered email to a new email the customer dictates (arg: email)",
    "request_ticket_cancellation": "submit cancellation of the ticket for the PNR (arg: pnr)",
    "block_card": "permanently block the card (arg: card_last4, the last 4 digits from the known card details)",
    "raise_transaction_dispute": "register a dispute on ONE transaction listed in the known recent transactions (args: "
                                 "transaction_id, reason); the agent only says the dispute is registered, never an outcome, "
                                 "a credit amount or a timeline",
    "request_card_replacement": "request a replacement card sent to an address the customer dictates (args: card_last4, "
                                "address); never a delivery date",
    "deactivate_service": "deactivate one active value-added service / caller tune listed in the known facts (arg: service_name)",
    "activate_pack": "activate one pack from the packs listed in the known facts, paid from main balance (arg: pack_name)",
    "block_sim": "block the SIM of the customer's mobile number, e.g. phone lost or stolen (arg: mobile_number)",
    "update_alternate_number": "update the alternate contact number to a new number the customer dictates (arg: phone)",
}
TYPE_NOTES = {
    "bank_card_support": "Bank-specific: the agent identifies the card only by its last 4 digits and NEVER asks for or "
                         "hears an OTP, CVV, PIN, expiry or full card number. No fraud-investigation results, no "
                         "chargeback/credit outcomes, no limit increases, no new card delivery dates, no branch visits.",
    "telecom_prepaid_support": "Telecom-specific: the customer calls about their own single prepaid number. No network/"
                               "coverage diagnosis, no porting, no new SIM delivery, no data-speed troubleshooting, no "
                               "recharge done by the agent; balances, validity, plans, packs and services are only the "
                               "known facts.",
    "airport_ticket_counter": "The caller is a 'Passenger' at the airport ticket counter (flow steps start 'Passenger ...').",
}
CONTEXTS = ["caller is in a hurry between meetings", "elderly caller who needs things said slowly",
            "caller is handling this for a family member", "calm, well-organised caller with a precise question",
            "caller is annoyed after a bad experience", "caller is travelling and on a noisy line",
            "first-time user confused by the app", "caller wants only short factual answers",
            "caller is worried about money", "caller is polite but anxious", "caller is a busy parent at home",
            "caller is a student on a tight budget", "caller is calling late at night", "caller is sceptical and double-checks facts"]
BANNED = [
    (r"\bescalat", "escalation"), (r"\binvestigat", "investigation"), (r"\bsupervisor|\bmanager\b", "supervisor/manager"),
    (r"\b(contact|call|message|ping|reach|inform)s? the (driver|rider|restaurant|seller|courier|delivery partner|branch|airline|merchant)",
     "contacting a third party"),
    (r"\b(live|real[- ]time) (track|location)", "live tracking"), (r"\bcall(s)? (the customer |them |him |her )?back\b", "callback"),
    (r"\bnew eta\b", "new ETA"), (r"\brefund (amount|timeline|within|in \d)", "refund amount/timeline"),
    (r"\b(within|in) \d+\s*(working |business )?(days|hours)\b", "timeline"), (r"changes? (his|her|their) mind", "reversal"),
    (r"\b(undo|undoes|revert|reverts|re-?activat|restor)", "reversal"), (r"\bOTP\b|\bCVV\b|\bPIN\b", "OTP/CVV/PIN"),
    (r"full card number", "full card number"), (r"\btransfer(s)? (the call|the customer|them)", "call transfer"),
    (r"\bcorrects?\b|\bcorrection\b", "value correction"), (r"\bwill (receive|get) (an? )?(sms|email|message)", "notification promise"),
]


def load_src():
    return json.load(open(SRC))


def plan_slots(src):
    """Deterministic slot plan: {agent_type: [ {"slot": sid, "tools": [...]} ]} with ids continuing the numbering."""
    existing = {a["agent_type"]: a for a in src["agents"]}
    plan = {}
    for at, n in N_NEW.items():
        n0, n1, n2, n3 = MIX[n]
        sw = existing[at]["tools"]["static_write"] if at in existing else next(t for t in NEW_TYPES if t["agent_type"] == at)["tools"]["static_write"]
        tl = [[] for _ in range(n0)]
        tl += [[sw[i % len(sw)]] for i in range(n1)]
        tl += [PAIRS[at][i % len(PAIRS[at])] for i in range(n2)]
        tl += [TRIPLES[at][i % len(TRIPLES[at])] for i in range(n3)]
        tl = [t for _, t in sorted(enumerate(tl), key=lambda it: h01("v2slot", at, str(it[0])))]
        start = len(existing[at]["scenarios"]) + 1 if at in existing else 1
        plan[at] = [{"slot": f"{ID_STEM[at]}_{start + i:02d}", "tools": list(t)} for i, t in enumerate(tl)]
    return plan


def type_info(src, at):
    for a in src["agents"]:
        if a["agent_type"] == at:
            return a
    return next(t for t in NEW_TYPES if t["agent_type"] == at)


def examples_for(src, at):
    """3 originals as shape/style examples: same type if it exists, else a mix from other types (with writes)."""
    pick = []
    for a in src["agents"]:
        if a["agent_type"] == at:
            sc = a["scenarios"]
            pick = [sc[2], sc[5], sc[0]] if len(sc) > 5 else sc[:3]
    if not pick:
        allsc = {s["id"]: s for a in src["agents"] for s in a["scenarios"]}
        pick = [allsc["sub_07"], allsc["cab_05"], allsc["food_11"]]
    return [{k: s[k] for k in ("id", "title", "flow", "tools") if k in s} for s in pick]


def role_word(at):
    return "Passenger" if at == "airport_ticket_counter" else "Customer"


RULES = """RULES (every scenario must follow all of them):
1. STATIC FACTS ONLY. Everything the agent says comes from facts known before the call (the read list above) or from what
   the caller says. The agent never needs a live lookup or a result it would have to invent: no refund/credit amounts or
   timelines, no new ETAs or delivery dates, no live tracking, no availability, no policies, no fees not in the facts. The
   agent never contacts a third party (driver, rider, restaurant, seller, courier, branch, merchant), never investigates,
   escalates, transfers, or promises a callback/SMS/email. If the caller asks for something unknown, the agent says it does
   not have that detail and restates what is known.
2. STATIC WRITES ONLY, and only the slot's actions. Each action in the slot's tools happens exactly once: a caller step asks
   for it (and gives any value it needs, e.g. the new address/number/email/instruction/reason), then an agent step performs
   it and confirms it, repeating the key value. No other action is performed or promised.
3. ONE PRIMARY ENTITY: the whole call is about exactly one order / ride / subscription / booking / card / SIM number. All
   actions are on that one entity. (Other known facts, like an older order, may exist but are never acted on.)
4. NO REVERSALS OR CORRECTIONS: the caller never changes their mind, never undoes or reverses an action, and never corrects
   a value after giving it. No action cancels or overrides another action in the same call.
5. FLOW: {nmin}-{nmax} steps, each one short plain-English sentence starting with "{role}" or "Agent", in the style of the
   examples (what happens, not the dialogue). Usually the agent first confirms the identifier (order ID, ride ID, PNR,
   account, card ending, number) and states the relevant known facts. Never write concrete values (digits, IDs, amounts,
   phone numbers, pack sizes) or tool names in the flow: the caller's record is generated later, so describe them
   ("provides the new phone number", "the known transaction ID"). Do not end on an unanswered caller question.
6. TITLE: 4-10 words describing the caller's situation. Each scenario must be a clearly DIFFERENT situation (different reason
   for calling, different facts asked about, different caller mood or context) from every existing title listed below and
   from the other slots."""


def gen_prompt(src, at, slots, existing_titles, feedback):
    ti = type_info(src, at)
    sw = ti["tools"]["static_write"]
    wdesc = "\n".join(f"  - {w}: {TOOL_DESC[w]}" for w in sw)
    ex = "\n".join(json.dumps(e, ensure_ascii=False) for e in examples_for(src, at))
    rules = RULES.format(nmin=5, nmax=8, role=role_word(at))
    note = TYPE_NOTES.get(at, "")
    slot_txt = []
    for i, s in enumerate(slots):
        t = json.dumps(s["tools"])
        ctx = CONTEXTS[int(h01("v2ctx", s["slot"], str(len(feedback.get(s["slot"], "")))) * len(CONTEXTS))]
        line = f"  slot {i + 1}: tools = {t}" + ("  (no actions: the call is only about known facts)" if not s["tools"] else "") \
            + f"; caller context idea (optional): {ctx}"
        if feedback.get(s["slot"]):
            line += "\n     previous attempt for this slot was REJECTED: " + feedback[s["slot"]][:400]
        slot_txt.append(line)
    titles = "\n".join(f"  - {t}" for t in existing_titles)
    return f"""You design customer-support call scenarios for training a Hinglish voice agent.
Company type: {at} ({ti['brand_context']}).
Facts the agent knows before the call (static reads, given to the agent as text): {', '.join(ti['tools']['static_read'])}.
Actions the agent can perform (static writes; deterministic, confirmed as done or submitted):
{wdesc}
{note}

{rules}

EXAMPLES of existing scenarios (exact JSON shape; "tools" lists the actions in the order they happen):
{ex}

EXISTING TITLES for this company type (do NOT repeat or paraphrase these situations):
{titles}

Write {len(slots)} NEW scenarios, one per slot, in slot order. Each slot's "tools" must be EXACTLY the list given:
{chr(10).join(slot_txt)}

Return JSON: {{"scenarios": [{{"title": "...", "flow": ["...", "..."], "tools": [...]}}, ...]}}"""


def gen_schema(sw, k):
    return {"type": "object", "properties": {"scenarios": {"type": "array", "minItems": k, "maxItems": k, "items": {
        "type": "object", "properties": {
            "title": {"type": "string"},
            "flow": {"type": "array", "minItems": 4, "maxItems": 8, "items": {"type": "string"}},
            "tools": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": sw}}},
        "required": ["title", "flow", "tools"], "additionalProperties": False}}},
        "required": ["scenarios"], "additionalProperties": False}


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def content_check(at, sc):
    """Added after review of the first pass (all rounds re-checked): no concrete values, no tool names, no dangling ending."""
    errs = []
    flow = sc.get("flow", []) or []
    txt = re.sub(r"\b(last )?4 digits\b", "", " ".join(flow) + " " + (sc.get("title") or ""), flags=re.I)
    m = re.search(r"\d", txt)
    if m:
        errs.append("do not write concrete values (numbers, IDs, amounts, pack sizes) in the title or flow; describe them "
                    "instead ('provides the new phone number', 'the known transaction ID', 'a data pack from the list')")
    m = re.search(r"\b[a-z]+_[a-z_]+\b", " ".join(flow))
    if m:
        errs.append(f"do not write tool names like '{m.group(0)}' in the flow; describe the action in plain words")
    if flow and flow[-1].strip().startswith(role_word(at)) and re.search(r"\basks?\b|\?", flow[-1]):
        errs.append("the flow must not end on an unanswered caller question")
    return errs


def shape_check(at, slot, sc, all_titles):
    errs = content_check(at, sc)
    if set(sc) != {"title", "flow", "tools"}:
        errs.append(f"keys {sorted(sc)}")
    title = sc.get("title", "")
    if not isinstance(title, str) or not (3 <= len(title.split()) <= 12):
        errs.append("title must be 4-10 words")
    if not title.isascii():
        errs.append("non-ASCII title")
    flow = sc.get("flow", [])
    if not (4 <= len(flow) <= 8):
        errs.append(f"flow has {len(flow)} steps (need 4-8)")
    rw = role_word(at)
    for i, st in enumerate(flow):
        if not isinstance(st, str) or not st.strip():
            errs.append(f"step {i + 1} empty")
            continue
        if not st.isascii():
            errs.append(f"step {i + 1} non-ASCII")
        if not re.match(rf"^({rw}|Agent)\b", st.strip()):
            errs.append(f"step {i + 1} must start with '{rw}' or 'Agent'")
    if flow and not re.match(rf"^{rw}\b", flow[0].strip()):
        errs.append(f"first step must be the {rw.lower()}")
    if sum(1 for st in flow if st.strip().startswith("Agent")) < 2:
        errs.append("needs at least 2 agent steps")
    tools = sc.get("tools", [])
    if tools != slot["tools"]:
        errs.append(f"tools {tools} != required {slot['tools']} (exact list and order)")
    if len(set(tools)) != len(tools):
        errs.append("duplicate tool")
    txt = " ".join(flow) + " " + title
    for rx, why in BANNED:
        m = re.search(rx, txt, re.I)
        if m:
            errs.append(f"banned ({why}): '{m.group(0)}'")
    nt = norm_title(title)
    for t in all_titles:
        if norm_title(t) == nt or difflib.SequenceMatcher(None, norm_title(t), nt).ratio() >= 0.85:
            errs.append(f"title too close to existing '{t}'")
            break
    return errs


JUDGE_RULES = """You check one customer-support call scenario (a plan, not a dialogue) against these rules:
R1 STATIC FACTS ONLY: the agent may only state facts known before the call (company read list: {reads}) or repeat what the
   caller said. FAIL if any agent step needs a live lookup or an invented result: refund/credit amount or timeline, new ETA or
   delivery date, live tracking, availability, policy, outcome of an investigation/dispute, contacting a third party,
   escalation, transfer, callback, SMS/email promise. (Saying "it does not have that detail" is fine.)
R2 WRITES: the declared tools are {tools}. FAIL if a declared tool is not performed AND confirmed by an agent step, or if an
   agent step performs/submits/promises any action that is NOT declared (e.g. updates, cancels, blocks, activates something
   not in the list). With tools [] the agent performs no action at all.
R3 ONE ENTITY: FAIL if the call acts on or is mainly about more than one order/ride/subscription/booking/card/SIM.
R4 NO REVERSAL: FAIL if the caller changes their mind, undoes/reverses an action, corrects a value after giving it, or one
   action cancels/overrides another.
R5 COHERENT: FAIL if the steps contradict each other, are not a plausible phone call for a {at} ({ctx}), or the title does
   not match the flow.{extra}
Be strict on real violations but do not invent problems: a step like "Agent confirms the order ID" or "Agent gives the
known status" is fine."""


def judge_prompt(src, at, sc):
    ti = type_info(src, at)
    extra = ""
    if at == "bank_card_support":
        extra = "\nR6 BANK: FAIL if the agent asks for or the caller gives an OTP, CVV, PIN, expiry or full card number."
    head = JUDGE_RULES.format(reads=", ".join(ti["tools"]["static_read"]), tools=json.dumps(sc.get("tools", [])), at=at,
                              ctx=ti["brand_context"], extra=extra)
    body = json.dumps({"title": sc["title"], "flow": sc["flow"], "tools": sc.get("tools", [])}, ensure_ascii=False, indent=1)
    return head + f"\n\nSCENARIO:\n{body}\n\nReturn JSON {{\"violations\": [{{\"rule\": \"R1..R6\", \"step\": <1-based step or 0>, \"why\": \"...\"}}], \"pass\": true|false}}. pass is true only when violations is empty."


JUDGE_SCHEMA = {"type": "object", "properties": {
    "violations": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {
        "rule": {"type": "string", "enum": ["R1", "R2", "R3", "R4", "R5", "R6"]}, "step": {"type": "integer"},
        "why": {"type": "string"}}, "required": ["rule", "step", "why"], "additionalProperties": False}},
    "pass": {"type": "boolean"}}, "required": ["violations", "pass"], "additionalProperties": False}


def dedupe_prompt(at, sc, refs):
    lst = "\n".join(f"  {rid}: {t}  (actions: {', '.join(tl) if tl else 'none'})" for rid, t, tl in refs)
    cand = json.dumps({"title": sc["title"], "flow": sc["flow"], "tools": sc.get("tools", [])}, ensure_ascii=False)
    return f"""You deduplicate customer-support call scenarios for a {at} company.
EXISTING scenarios (id: title (actions)):
{lst}

CANDIDATE:
{cand}

Is the CANDIDATE essentially the same scenario as one of the EXISTING ones, i.e. the same caller situation/reason for calling
AND the same kind of resolution, so that a training call made from it would be a near copy? Different actions, a clearly
different reason for calling, or a different set of facts asked about make it NOT a duplicate. A shared tool alone is not a
duplicate.
Return JSON {{"closest_id": "<id of the most similar existing scenario>", "similarity": <0.0-1.0>, "duplicate": true|false, "reason": "<one sentence>"}}."""


DEDUPE_SCHEMA = {"type": "object", "properties": {
    "closest_id": {"type": "string"}, "similarity": {"type": "number"}, "duplicate": {"type": "boolean"},
    "reason": {"type": "string"}}, "required": ["closest_id", "similarity", "duplicate", "reason"], "additionalProperties": False}


def main():
    os.makedirs(WORK, exist_ok=True)
    src = load_src()
    plan = plan_slots(src)
    jdump(f"{WORK}/slot_plan.json", plan)
    if "--dry" in sys.argv:
        at = sys.argv[sys.argv.index("--dry") + 1] if len(sys.argv) > sys.argv.index("--dry") + 1 else "bank_card_support"
        orig_titles = [s["title"] for a in src["agents"] if a["agent_type"] == at for s in a["scenarios"]]
        print(gen_prompt(src, at, plan[at][:5], orig_titles, {}))
        print("\n=====JUDGE=====\n" + judge_prompt(src, "food_delivery_support", src["agents"][0]["scenarios"][2]))
        return
    state_p = f"{WORK}/state.json"
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"accepted": {}, "feedback": {}, "rejects": [], "rounds": []}
    # re-check accepted scenarios with the current deterministic rules (+ FORCE_REGEN "slot=reason|slot=reason")
    force = dict(x.split("=", 1) for x in os.environ.get("FORCE_REGEN", "").split("|") if "=" in x)
    slot_by_id = {s["slot"]: (at, s) for at, sl in plan.items() for s in sl}
    for sid in list(state["accepted"]):
        at, slot = slot_by_id[sid]
        acc = state["accepted"][sid]
        errs = content_check(at, {"title": acc["title"], "flow": acc["flow"], "tools": acc.get("tools", [])})
        if acc.get("tools", []) != slot["tools"]:
            errs.append(f"slot tools changed to {slot['tools']}")
        if sid in force:
            errs.append(force[sid])
        if errs:
            state["feedback"][sid] = "; ".join(errs)
            state["rejects"].append({"round": len(state["rounds"]), "slot": sid, "stage": "recheck", "title": acc["title"], "why": errs})
            state.setdefault("demoted", []).append({"slot": sid, "old": acc, "why": errs})
            del state["accepted"][sid]
            print(f"[recheck] demoted {sid}: {errs}", flush=True)
    jdump(state_p, state)
    if "--recheck-only" in sys.argv:
        return
    from common import LLM
    llm = LLM(max_model_len=8192)
    orig = {a["agent_type"]: [(s["id"], s["title"], s.get("tools") or []) for s in a["scenarios"]] for a in src["agents"]}
    log = open(f"{WORK}/raw.jsonl", "a", encoding="utf-8")

    # calibration: rule judge over the 72 originals (false-positive rate of the judge)
    if not os.path.exists(f"{WORK}/calib_originals.json"):
        items = [(a["agent_type"], s) for a in src["agents"] for s in a["scenarios"]]
        res = llm.chat([judge_prompt(src, at, s) for at, s in items], schema=JUDGE_SCHEMA, temperature=0.0, max_tokens=800)
        cal = []
        for (at, s), (txt, fin, _) in zip(items, res):
            try:
                j = json.loads(txt)
            except Exception:
                j = {"pass": None, "violations": [], "parse_error": txt[:200]}
            cal.append({"id": s["id"], **j})
        jdump(f"{WORK}/calib_originals.json", cal)
        print(f"[calib] originals judged FAIL: {sum(1 for c in cal if c.get('pass') is False)}/72", flush=True)

    t_start = time.time()
    for rnd in range(len(state["rounds"]), MAX_ROUNDS):
        pending = {at: [s for s in sl if s["slot"] not in state["accepted"]] for at, sl in plan.items()}
        if not any(pending.values()):
            break
        # 1. generation
        jobs = []
        for at, sl in pending.items():
            if not sl:
                continue
            ti = type_info(src, at)
            acc_titles = [state["accepted"][s["slot"]]["title"] for s in plan[at] if s["slot"] in state["accepted"]]
            ex_titles = [t for _, t, _ in orig.get(at, [])] + acc_titles
            for i in range(0, len(sl), CHUNK):
                ch = sl[i:i + CHUNK]
                jobs.append((at, ch, gen_prompt(src, at, ch, ex_titles, state["feedback"]), gen_schema(ti["tools"]["static_write"], len(ch))))
        seeds = [int(h01("v2gen", str(rnd), at, ch[0]["slot"]) * 2 ** 31) for at, ch, _, _ in jobs]
        res = llm.chat([j[2] for j in jobs], schema=[j[3] for j in jobs], temperature=0.8, max_tokens=4000, seeds=seeds)
        cands = []  # (at, slot, scenario)
        for (at, ch, _, _), (txt, fin, _) in zip(jobs, res):
            log.write(json.dumps({"round": rnd, "stage": "gen", "at": at, "slots": [c["slot"] for c in ch], "finish": fin, "text": txt}, ensure_ascii=False) + "\n")
            try:
                arr = json.loads(txt)["scenarios"]
            except Exception:
                try:
                    arr = extract_json(txt)["scenarios"]
                except Exception:
                    arr = []
            for k, slot in enumerate(ch):
                if k < len(arr) and isinstance(arr[k], dict):
                    sc = {kk: arr[k].get(kk) for kk in ("title", "flow", "tools")}
                    sc["title"] = (sc["title"] or "").strip()
                    sc["flow"] = [str(x).strip() for x in (sc["flow"] or [])]
                    cands.append((at, slot, sc))
                else:
                    state["feedback"][slot["slot"]] = "no output for this slot"
                    state["rejects"].append({"round": rnd, "slot": slot["slot"], "stage": "gen", "why": "no output"})
        # 2. shape check (titles: originals of ALL types + all accepted + earlier candidates this round)
        all_titles = [t for v in orig.values() for _, t, _ in v] + [v["title"] for v in state["accepted"].values()]
        ok1 = []
        for at, slot, sc in cands:
            errs = shape_check(at, slot, sc, all_titles)
            if errs:
                state["feedback"][slot["slot"]] = "; ".join(errs)
                state["rejects"].append({"round": rnd, "slot": slot["slot"], "stage": "shape", "title": sc["title"], "why": errs})
            else:
                ok1.append((at, slot, sc))
                all_titles.append(sc["title"])
        # 3. rule judge
        ok2 = []
        if ok1:
            jr = llm.chat([judge_prompt(src, at, sc) for at, _, sc in ok1], schema=JUDGE_SCHEMA, temperature=0.0, max_tokens=800)
            for (at, slot, sc), (txt, fin, _) in zip(ok1, jr):
                try:
                    j = json.loads(txt)
                except Exception:
                    j = {"pass": False, "violations": [{"rule": "R5", "step": 0, "why": "judge parse error"}]}
                log.write(json.dumps({"round": rnd, "stage": "rule_judge", "slot": slot["slot"], "title": sc["title"], "judge": j}, ensure_ascii=False) + "\n")
                if j.get("pass") and not j.get("violations"):
                    ok2.append((at, slot, sc))
                else:
                    why = "; ".join(f"{v['rule']} step {v['step']}: {v['why']}" for v in j.get("violations", [])) or "judge fail"
                    state["feedback"][slot["slot"]] = why
                    state["rejects"].append({"round": rnd, "slot": slot["slot"], "stage": "rule_judge", "title": sc["title"], "why": why})
        # 4. dedupe judge: refs = originals of the type + accepted new of the type + earlier ok2 candidates of the type
        if ok2:
            prompts = []
            for idx, (at, slot, sc) in enumerate(ok2):
                refs = list(orig.get(at, []))
                refs += [(s["slot"], state["accepted"][s["slot"]]["title"], state["accepted"][s["slot"]].get("tools", []))
                         for s in plan[at] if s["slot"] in state["accepted"]]
                refs += [(s2["slot"] + "(new)", sc2["title"], sc2["tools"]) for at2, s2, sc2 in ok2[:idx] if at2 == at]
                prompts.append(dedupe_prompt(at, sc, refs) if refs else None)
            idxs = [i for i, p in enumerate(prompts) if p]
            dr = llm.chat([prompts[i] for i in idxs], schema=DEDUPE_SCHEMA, temperature=0.0, max_tokens=300) if idxs else []
            dmap = dict(zip(idxs, dr))
            dropped_this_round = set()
            for i, (at, slot, sc) in enumerate(ok2):
                d = {"duplicate": False, "similarity": 0.0, "closest_id": "", "reason": "no refs"}
                if i in dmap:
                    try:
                        d = json.loads(dmap[i][0])
                    except Exception:
                        d = {"duplicate": True, "similarity": 1.0, "closest_id": "?", "reason": "dedupe parse error"}
                log.write(json.dumps({"round": rnd, "stage": "dedupe", "slot": slot["slot"], "title": sc["title"], "dedupe": d}, ensure_ascii=False) + "\n")
                cid = str(d.get("closest_id", ""))
                # a "duplicate" of a same-round candidate that itself got dropped is not a duplicate
                if d.get("duplicate") and cid.endswith("(new)") and cid[:-5] in dropped_this_round:
                    d["duplicate"] = False
                if d.get("duplicate"):
                    dropped_this_round.add(slot["slot"])
                    ref_title = next((t for rid, t, _ in (orig.get(at, []) + [(s["slot"], state["accepted"].get(s["slot"], {}).get("title", ""), []) for s in plan[at]]) if rid == cid.replace("(new)", "")), "")
                    why = f"duplicate of {cid} '{ref_title}' ({d.get('reason', '')}); choose a clearly different situation"
                    state["feedback"][slot["slot"]] = why
                    state["rejects"].append({"round": rnd, "slot": slot["slot"], "stage": "dedupe", "title": sc["title"], "why": why,
                                             "similarity": d.get("similarity")})
                else:
                    out = {"id": slot["slot"], "title": sc["title"], "flow": sc["flow"]}
                    if sc["tools"]:
                        out["tools"] = sc["tools"]
                    state["accepted"][slot["slot"]] = {**out, "_dedupe": d, "_round": rnd}
                    state["feedback"].pop(slot["slot"], None)
        n_acc = len(state["accepted"])
        state["rounds"].append({"round": rnd, "gen_prompts": len(jobs), "candidates": len(cands), "shape_ok": len(ok1),
                                "judge_ok": len(ok2), "accepted_total": n_acc, "elapsed_s": round(time.time() - t_start)})
        jdump(state_p, state)
        print(f"[round {rnd}] cands {len(cands)} shape_ok {len(ok1)} judge_ok {len(ok2)} accepted_total {n_acc}/{sum(map(len, plan.values()))}", flush=True)
    log.close()
    build_output(src, plan, state)


def build_output(src, plan, state):
    v2 = copy.deepcopy(src)
    for a in v2["agents"]:
        at = a["agent_type"]
        a["scenarios"] += [{k: v for k, v in state["accepted"][s["slot"]].items() if not k.startswith("_")}
                           for s in plan[at] if s["slot"] in state["accepted"]]
    for t in NEW_TYPES:
        a = copy.deepcopy(t)
        a = {"agent_type": a["agent_type"], "brand_context": a["brand_context"], "distribution": 0.0, "tools": a["tools"],
             "scenarios": [{k: v for k, v in state["accepted"][s["slot"]].items() if not k.startswith("_")}
                           for s in plan[t["agent_type"]] if s["slot"] in state["accepted"]]}
        v2["agents"].append(a)
    tot = sum(len(a["scenarios"]) for a in v2["agents"])
    for a in v2["agents"]:  # distribution = share of scenarios (field is not read by any code)
        a["distribution"] = round(len(a["scenarios"]) / tot, 3)
    jdump(OUT, v2)
    missing = [s["slot"] for sl in plan.values() for s in sl if s["slot"] not in state["accepted"]]
    print(f"[out] {OUT}: {tot} scenarios; missing slots: {missing}", flush=True)


if __name__ == "__main__":
    if "--build-only" in sys.argv:
        src = load_src()
        build_output(src, plan_slots(src), json.load(open(f"{WORK}/state.json")))
    else:
        main()
