"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle_v2/schema/tools_v2.py (path edits in __main__ only:
the spec is $SCENARIO_CREATION_V2 or research/data_gen/guidance/scenario_creation_v2.json; output dir $TOOLS_JSON_OUT).

N2 (Needle v2) tool schemas for all 7 V4 agent types (static_write tools only; read tools are never
given to Needle, which is only invoked after the agent's check-line).

Every model argument is one of two classes (SCHEMA_V2.md has the table and the evidence):
  REF  a pointer into the record. The model copies the exact value it means from the record that is in its
       system text (order/ride/booking/account/transaction/charge ID, card last 4, plan / service / pack name),
       or for numbers already on file a role word ("registered" / "alternate"), never the number itself.
       resolver_v2.resolve(kind, ref, record) maps it to the exact record value, falls back to spoken phrases
       (N1 rules: ID digits, names, ordinals) and returns ASK when it cannot pick exactly one.
  NEW  a value the caller speaks that is not in the record: new phone (10 digits), new email, new address,
       pickup / drop location, instruction, reason. Copied from the number-converted transcript.

Schema rules:
- REF args are OPTIONAL (Needle withholds a call whose required arg has no span, N1 finding); the resolver
  defaults when the record has exactly one candidate.
- NEW args are REQUIRED, except request_card_replacement.address (omitted -> registered address on file).
- No description contains an example VALUE (no phone, email, address, ID): N1 copied 9800011085 from the old
  phone description. Descriptions give the format only.
- Format: Needle flat tool JSON {"name", "description", "parameters": {"type": "object", "properties",
  "required"}}, the same as N1 tools.py.
"""
import json
import os

AGENT_TYPES = ["food_delivery_support", "ecommerce_support", "cab_ride_support",
               "subscription_account_support", "airport_ticket_counter",
               "bank_card_support", "telecom_prepaid_support"]
OLD_TYPES = AGENT_TYPES[:5]   # the 5 N1 types (for the N1-comparable score)

# guidance/scenario_creation_v2.json agents[].tools.static_write (checked by check_against_source()).
STATIC_WRITE = {
    "food_delivery_support": ["cancel_order", "change_delivery_address", "add_delivery_instruction",
                              "update_contact_number", "request_refund"],
    "ecommerce_support": ["cancel_order", "change_delivery_address", "add_delivery_instruction",
                          "update_contact_number", "request_refund"],
    "cab_ride_support": ["cancel_ride", "change_pickup_location", "change_drop_location",
                         "add_driver_instruction", "update_contact_number", "request_refund"],
    "subscription_account_support": ["cancel_subscription", "pause_subscription", "update_contact_number",
                                     "update_email_address", "request_refund"],
    "airport_ticket_counter": ["update_contact_number", "update_email_address",
                               "request_ticket_cancellation", "request_refund"],
    "bank_card_support": ["block_card", "raise_transaction_dispute", "request_card_replacement",
                          "update_contact_number", "update_email_address"],
    "telecom_prepaid_support": ["deactivate_service", "activate_pack", "block_sim", "update_alternate_number",
                                "request_refund"],
}

# ------------------------------------------------------------------------------------ argument kinds
# REF kinds (resolver_v2): order ride booking account transaction charge (ID-shaped [A-Z]{2}dddd in the record),
#   plan, card, service, pack, phone_on_file.
# NEW kinds: phone email address location instruction reason; address_or_registered (NEW, may be omitted).
REF_KINDS = {"order", "ride", "booking", "account", "transaction", "charge", "plan", "card", "service", "pack",
             "phone_on_file"}
NEW_KINDS = {"phone", "email", "address", "location", "instruction", "reason", "address_or_registered"}
ID_KINDS = {"order", "ride", "booking", "account", "transaction", "charge"}

# Reference descriptions. Format: what to copy from the record, and when it may be omitted. No values.
_REF_DESC = {
    "order": "the order the customer means: copy its order ID exactly as written in the record",
    "ride": "the ride the customer means: copy its ride ID exactly as written in the record",
    "booking": "the booking or charge the customer means: copy the PNR, or the charge reference for a duplicate "
               "charge or fee, exactly as written in the record",
    "account": "the account or charge the customer means: copy the account ID or reference exactly as written "
               "in the record",
    "transaction": "the transaction the customer is disputing: copy its transaction ID exactly as written in the "
                   "record (match by merchant, amount or date if the customer did not say the ID)",
    "charge": "the recharge or charge the customer wants refunded: copy its reference ID exactly as written in "
              "the record",
    "plan": "the subscription plan the customer means: copy the plan name exactly as written in the record",
    "card": "the card the customer means: copy the last 4 digits of the card exactly as written in the record",
    "service": "the value added service to stop: copy the service name exactly as written in the record",
    "pack": "the pack to activate: copy the pack name exactly as written in the record",
    "phone_on_file": "which number on file to block: write registered for the customer's registered number or "
                     "alternate for the alternate number on record; never write the digits",
}
_NEW_DESC = {
    "phone": "the new phone number the customer said, written as exactly 10 digits with no spaces",
    "email": "the new email address the customer said, written as name@domain with no spaces",
    "address": "the new address exactly as the customer said it",
    "address_or_registered": "the new address to send the card to, exactly as the customer said it; omit if the "
                             "card should go to the registered address on file",
    "location": None,   # per tool (pickup / drop)
    "instruction": None,  # per tool (delivery person / driver)
    "reason": "why the customer wants this, as the customer said it",
}

# (agent_type or "*", tool) -> tool description, [(model_arg, kind, gold_arg)]
# gold_arg = the key in calls.jsonl writes[].args (the server-side argument).
_T = {
    ("*", "cancel_order"): ("Cancel an order.", [("order_ref", "order", "order_id")]),
    ("*", "change_delivery_address"): ("Change the delivery address of an order to a new address.",
                                       [("order_ref", "order", "order_id"), ("address", "address", "address")]),
    ("*", "add_delivery_instruction"): (
        "Add a note for the delivery person, such as where to leave the parcel or to call on arrival.",
        [("order_ref", "order", "order_id"), ("instruction", "instruction", "instruction")]),
    ("*", "cancel_ride"): ("Cancel a cab ride.", [("ride_ref", "ride", "ride_id")]),
    ("*", "change_pickup_location"): ("Change where the cab picks the customer up.",
                                      [("ride_ref", "ride", "ride_id"), ("location", "location", "location")]),
    ("*", "change_drop_location"): ("Change the destination where the cab drops the customer.",
                                    [("ride_ref", "ride", "ride_id"), ("location", "location", "location")]),
    ("*", "add_driver_instruction"): ("Add a note for the cab driver, such as where to wait or to call on arrival.",
                                      [("ride_ref", "ride", "ride_id"), ("instruction", "instruction", "instruction")]),
    ("*", "cancel_subscription"): ("Cancel a subscription plan so it does not renew.",
                                   [("plan_ref", "plan", "plan")]),
    ("*", "pause_subscription"): ("Pause a subscription plan for some time.", [("plan_ref", "plan", "plan")]),
    ("*", "update_email_address"): ("Change the registered email address to a new one.",
                                    [("email", "email", "email")]),
    ("*", "request_ticket_cancellation"): ("Cancel a flight ticket booking.", [("booking_ref", "booking", "pnr")]),
    ("*", "block_card"): ("Block a lost, stolen or misused card.", [("card_ref", "card", "card_last4")]),
    ("*", "raise_transaction_dispute"): ("Raise a dispute on a card transaction the customer does not accept.",
                                         [("transaction_ref", "transaction", "transaction_id"),
                                          ("reason", "reason", "reason")]),
    ("*", "request_card_replacement"): ("Send the customer a replacement card.",
                                        [("card_ref", "card", "card_last4"),
                                         ("address", "address_or_registered", "address")]),
    ("*", "deactivate_service"): ("Stop a paid value added service on the prepaid number.",
                                  [("service_ref", "service", "service")]),
    ("*", "activate_pack"): ("Activate a data or add-on pack on the prepaid number.",
                             [("pack_ref", "pack", "pack")]),
    ("*", "block_sim"): ("Block the SIM of a lost or stolen phone.", [("number_ref", "phone_on_file", "sim_number")]),
    ("*", "update_alternate_number"): ("Set a new alternate contact number on the prepaid account.",
                                       [("phone", "phone", "phone")]),
    # update_contact_number: no entity reference (writes[] never carries one); only the NEW phone.
    ("food_delivery_support", "update_contact_number"): (
        "Change the phone number the delivery person should call.", [("phone", "phone", "phone")]),
    ("ecommerce_support", "update_contact_number"): (
        "Change the contact phone number for deliveries.", [("phone", "phone", "phone")]),
    ("cab_ride_support", "update_contact_number"): (
        "Change the phone number the driver should call.", [("phone", "phone", "phone")]),
    ("subscription_account_support", "update_contact_number"): (
        "Change the registered phone number on the account.", [("phone", "phone", "phone")]),
    ("airport_ticket_counter", "update_contact_number"): (
        "Change the contact phone number on the flight booking.", [("phone", "phone", "phone")]),
    ("bank_card_support", "update_contact_number"): (
        "Change the registered mobile number for card alerts.", [("phone", "phone", "phone")]),
    # request_refund: reference kind depends on the agent type (what writes[].reference_id points at in V4).
    ("food_delivery_support", "request_refund"): (
        "Raise a refund request for a food order.",
        [("order_ref", "order", "reference_id"), ("reason", "reason", "reason")]),
    ("ecommerce_support", "request_refund"): (
        "Raise a refund request for an order.",
        [("order_ref", "order", "reference_id"), ("reason", "reason", "reason")]),
    ("cab_ride_support", "request_refund"): (
        "Raise a refund request for a ride fare.",
        [("ride_ref", "ride", "reference_id"), ("reason", "reason", "reason")]),
    ("subscription_account_support", "request_refund"): (
        "Raise a refund request for a subscription charge.",
        [("account_ref", "account", "reference_id"), ("reason", "reason", "reason")]),
    ("airport_ticket_counter", "request_refund"): (
        "Raise a refund request for a flight booking or an extra charge on it.",
        [("booking_ref", "booking", "reference_id"), ("reason", "reason", "reason")]),
    ("telecom_prepaid_support", "request_refund"): (
        "Raise a refund request for a recharge or a wrong charge on the prepaid number.",
        [("charge_ref", "charge", "reference_id"), ("reason", "reason", "reason")]),
}

_LOCATION_DESC = {"change_pickup_location": "the new pickup place exactly as the customer said it",
                  "change_drop_location": "the new drop place exactly as the customer said it"}
_INSTR_DESC = {"add_delivery_instruction": "the instruction for the delivery person as the customer said it",
               "add_driver_instruction": "the instruction for the driver as the customer said it"}


def tool_spec(agent_type, tool):
    """(description, [(model_arg, kind, gold_arg), ...]) for one tool of one agent type."""
    if tool not in STATIC_WRITE[agent_type]:
        raise KeyError(f"{tool} is not a static_write tool of {agent_type}")
    return _T.get((agent_type, tool)) or _T[("*", tool)]


def arg_specs(agent_type, tool):
    return tool_spec(agent_type, tool)[1]


def _arg_desc(tool, kind):
    if kind in _REF_DESC:
        return _REF_DESC[kind]
    if kind == "location":
        return _LOCATION_DESC[tool]
    if kind == "instruction":
        return _INSTR_DESC[tool]
    return _NEW_DESC[kind]


def tool_schema(agent_type, tool):
    desc, specs = tool_spec(agent_type, tool)
    props, required = {}, []
    for marg, kind, _ in specs:
        props[marg] = {"type": "string", "description": _arg_desc(tool, kind)}
        if kind in NEW_KINDS and kind != "address_or_registered":
            required.append(marg)
    params = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    return {"name": tool, "description": desc, "parameters": params}


def tools_for(agent_type):
    """Needle flat-JSON tool list for one agent type, in static_write order."""
    return [tool_schema(agent_type, t) for t in STATIC_WRITE[agent_type]]


TOOLS = {a: tools_for(a) for a in AGENT_TYPES}


def check_against_source(path):
    d = json.load(open(path))
    for a in d["agents"]:
        assert STATIC_WRITE[a["agent_type"]] == a["tools"]["static_write"], a["agent_type"]
    assert set(STATIC_WRITE) == {a["agent_type"] for a in d["agents"]}
    return True


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for src in (os.environ.get("SCENARIO_CREATION_V2") or "",
                os.path.join(here, "..", "..", "..", "research", "data_gen", "guidance", "scenario_creation_v2.json"),
                os.path.join(here, "src", "scenario_creation_v2.json")):
        if os.path.exists(src):
            print("static_write lists match", src, check_against_source(src))
    out = os.environ.get("TOOLS_JSON_OUT") or os.path.join(here, "tools_json")
    os.makedirs(out, exist_ok=True)
    for a, ts in TOOLS.items():
        json.dump(ts, open(os.path.join(out, f"{a}.json"), "w"), indent=1, ensure_ascii=False)
        print(a, len(ts), "tools")
