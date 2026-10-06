"""N1 router tool schemas: one Needle tool set per agent type.

Source of the tool lists: guidance/scenario_creation.json, agents[].tools.static_write
(read tools are not exposed: a read/question turn must route to the empty list []).
Names are kept verb_noun exactly as in that file.

Schema rules (handoff N1 section 1 + Needle's "How to design tools for Needle 3"):
- No tool takes an ID. Every tool takes `order_ref` (string) = the entity as the customer
  referred to it; the server turns it into an ID with resolver.resolve().
- `order_ref` is OPTIONAL (not in `required`): most real write turns never name the entity
  ("Mera naya number 98000 12105 hai, update kar dijiye"), and Needle withholds a call whose
  required argument has no span in the request (N0: 38/48 calls withheld). When order_ref is
  absent the resolver defaults to the single active entity.
- The real arguments (address, instruction, phone, location, email, reason) are REQUIRED, so a
  turn that asks for the action but does not give the value yet is withheld, not guessed.
- Every argument carries a per-argument description with its format (the guide's
  "formats in the description" rule).
- Format: Needle's flat tool JSON, {"name", "description", "parameters": {"type": "object",
  "properties": {...}, "required": [...]}} -- the same dict needle.agent.tools.build_schema()
  emits for a decorated function; Needle(tools=[dict, ...]) accepts the dicts as they are.

Mapping from calls.jsonl writes[].args (common.TOOL_ARGS) to the router target:
  order_id / ride_id / reference_id / pnr  -> dropped (the ID comes from the resolver)
  plan (cancel_subscription, pause_subscription) -> order_ref (the plan IS the entity phrase)
  address, instruction, phone, location, email, reason -> kept, same names
"""
import json
import os

AGENT_TYPES = ["food_delivery_support", "ecommerce_support", "cab_ride_support",
               "subscription_account_support", "airport_ticket_counter"]

# static_write lists, copied from guidance/scenario_creation.json (checked by _check_against_source()).
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
}

# order_ref description per agent type. Pattern from the handoff: "the order/ride/booking/plan as
# the customer referred to it, e.g. Domino's, biryani wala, pehla wala, my flight, FD1565";
# the noun and the examples are made agent-specific (generic, not taken from any one record).
ORDER_REF_DESC = {
    "food_delivery_support":
        "the order as the customer referred to it, e.g. Domino's, biryani wala, pehla wala, "
        "my latest order, FD1565; omit if the customer did not name it",
    "ecommerce_support":
        "the order as the customer referred to it, e.g. headphones wala, Samsung phone, purana order, "
        "my order, EC2469; omit if the customer did not name it",
    "cab_ride_support":
        "the ride as the customer referred to it, e.g. airport wali ride, current ride, pichhli ride, "
        "RD3938; omit if the customer did not name it",
    "subscription_account_support":
        "the subscription plan or account as the customer referred to it, e.g. Premium Monthly plan, "
        "mera plan, purana plan, AC5746; omit if the customer did not name it",
    "airport_ticket_counter":
        "the booking as the customer referred to it, e.g. my flight, Mumbai wali flight, return flight, "
        "pehla ticket, JB7215; omit if the customer did not name it",
}

# Per-tool description and real (non-ID) arguments. Argument descriptions give the format.
_PHONE = {"type": "string", "description": "the new phone number, 10 digits, e.g. 9800011085"}
_EMAIL = {"type": "string", "description": "the new email address, written as name@domain.com"}

_TOOL_DEFS = {
    # food + ecommerce
    "cancel_order": ("Cancel an order.", {}),
    "change_delivery_address": (
        "Change the delivery address of an order to a new address.",
        {"address": {"type": "string", "description": "the new address as spoken, e.g. Flat 204, Sector 15, Gurgaon"}}),
    "add_delivery_instruction": (
        "Add a note for the delivery person, e.g. leave with security, call on arrival, do not ring the bell.",
        {"instruction": {"type": "string", "description": "the instruction for the delivery person as spoken"}}),
    # cab
    "cancel_ride": ("Cancel a cab ride.", {}),
    "change_pickup_location": (
        "Change where the cab picks the customer up.",
        {"location": {"type": "string", "description": "the new pickup place as spoken, e.g. IGI Airport Terminal 3, Gate 4"}}),
    "change_drop_location": (
        "Change the destination where the cab drops the customer.",
        {"location": {"type": "string", "description": "the new drop place as spoken, e.g. DLF Cyber City, Phase 2, Gurgaon"}}),
    "add_driver_instruction": (
        "Add a note for the cab driver, e.g. wait outside the barrier, call on arrival.",
        {"instruction": {"type": "string", "description": "the instruction for the driver as spoken"}}),
    # subscription
    "cancel_subscription": ("Cancel a subscription plan so it does not renew.", {}),
    "pause_subscription": ("Pause a subscription plan for some time.", {}),
    "update_email_address": ("Change the registered email address to a new one.", {"email": _EMAIL}),
    # airport
    "request_ticket_cancellation": ("Cancel a flight ticket booking.", {}),
    # shared, description depends on the agent type (see _shared())
    "update_contact_number": (None, {"phone": _PHONE}),
    "request_refund": (None, {"reason": {"type": "string", "description": "why the customer wants the refund, as spoken"}}),
}

_SHARED_DESC = {
    "update_contact_number": {
        "food_delivery_support": "Change the phone number the delivery person should call for an order.",
        "ecommerce_support": "Change the contact phone number for an order's delivery.",
        "cab_ride_support": "Change the phone number the driver should call for a ride.",
        "subscription_account_support": "Change the registered phone number on the account.",
        "airport_ticket_counter": "Change the contact phone number on a flight booking.",
    },
    "request_refund": {
        "food_delivery_support": "Raise a refund request for a food order.",
        "ecommerce_support": "Raise a refund request for an order.",
        "cab_ride_support": "Raise a refund request for a ride fare.",
        "subscription_account_support": "Raise a refund request for a subscription charge.",
        "airport_ticket_counter": "Raise a refund request for a flight booking.",
    },
}

# writes[].args keys that are entity IDs (dropped) and keys that become order_ref.
ID_ARGS = {"order_id", "ride_id", "reference_id", "pnr"}
REF_ARGS = {"plan"}


def tool_schema(name, agent_type):
    desc, real = _TOOL_DEFS[name]
    if desc is None:
        desc = _SHARED_DESC[name][agent_type]
    props = {"order_ref": {"type": "string", "description": ORDER_REF_DESC[agent_type]}}
    props.update({k: dict(v) for k, v in real.items()})
    params = {"type": "object", "properties": props}
    if real:
        params["required"] = list(real)
    return {"name": name, "description": desc, "parameters": params}


def tools_for(agent_type):
    """The tool list (Needle flat JSON dicts) for one agent type, in static_write order."""
    return [tool_schema(n, agent_type) for n in STATIC_WRITE[agent_type]]


TOOLS = {a: tools_for(a) for a in AGENT_TYPES}


def target_arguments(tool, write_args, order_ref=None):
    """Router target arguments for one writes[] entry: drop ID args, plan -> order_ref,
    keep real args; order_ref (phrase extracted from the customer turn) only when given."""
    out = {}
    ref = order_ref
    for k, v in write_args.items():
        if k in ID_ARGS:
            continue
        if k in REF_ARGS:
            ref = ref or v
            continue
        out[k] = v
    if ref:
        out = {"order_ref": ref, **out}
    return out


def _check_against_source(path):
    d = json.load(open(path))
    for a in d["agents"]:
        assert STATIC_WRITE[a["agent_type"]] == a["tools"]["static_write"], a["agent_type"]
    return True


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "src", "scenario_creation.json")
    if os.path.exists(src):
        print("static_write lists match", src, _check_against_source(src))
    os.makedirs(os.path.join(here, "tools_json"), exist_ok=True)
    for a, ts in TOOLS.items():
        p = os.path.join(here, "tools_json", f"{a}.json")
        json.dump(ts, open(p, "w"), indent=1, ensure_ascii=False)
        print(a, len(ts), "tools ->", p)
