"""DEP1 Track 3: stub write tools. Each mutates an in-memory copy of the session record and returns {"ok": true}.

StubTools(record).execute(name, resolved_id, arguments) -> (result, record_diff)
  - The primary entity's fields live in record["facts"] (V3 records) and in the free-text "information"
    (V4 records have no facts dict); the stub writes the new value to facts[<field>] when facts exists,
    and always to record["entity_updates"][resolved_id][<field>], so every write is visible in one place.
  - record_diff = {"path": {"old": .., "new": ..}, ...} for every key the call changed.
  - Every executed call is appended to record["executed_calls"].
Pure python. No real backend is called.
"""
import copy
import time

# tool -> list of (argument -> field) writes, plus constant field writes
_FIELD = {
    "change_delivery_address": {"address": "delivery_address"},
    "add_delivery_instruction": {"instruction": "delivery_instructions"},
    "update_contact_number": {"phone": "phone"},
    "update_email_address": {"email": "email"},
    "change_pickup_location": {"location": "pickup"},
    "change_drop_location": {"location": "drop"},
    "add_driver_instruction": {"instruction": "driver_instructions"},
    "request_refund": {"reason": "refund_reason"},
}
_CONST = {
    "cancel_order": {"status": "cancelled"},
    "cancel_ride": {"status": "cancelled"},
    "cancel_subscription": {"status": "cancelled", "auto_renew": "off"},
    "pause_subscription": {"status": "paused"},
    "request_ticket_cancellation": {"status": "cancellation_requested"},
    "request_refund": {"refund_status": "requested"},
}
KNOWN = set(_FIELD) | set(_CONST)


class StubTools:
    def __init__(self, record):
        self.record = copy.deepcopy(record)
        self.record.setdefault("entity_updates", {})
        self.record.setdefault("executed_calls", [])

    def execute(self, name, resolved_id, arguments):
        if name not in KNOWN:
            raise ValueError(f"unknown tool {name!r}")
        rec = self.record
        diff = {}
        writes = dict(_CONST.get(name, {}))
        for a, field in _FIELD.get(name, {}).items():
            if a in arguments:
                writes[field] = arguments[a]
        upd = rec["entity_updates"].setdefault(resolved_id, {})
        for field, val in writes.items():
            diff[f"entity_updates.{resolved_id}.{field}"] = {"old": upd.get(field), "new": val}
            upd[field] = val
            facts = rec.get("facts")
            if isinstance(facts, dict) and resolved_id == rec.get("primary_id") and field in facts:
                diff[f"facts.{field}"] = {"old": facts[field], "new": val}
                facts[field] = val
        if name == "update_contact_number" and "phone" in arguments and resolved_id == rec.get("primary_id"):
            diff["phone"] = {"old": rec.get("phone"), "new": arguments["phone"]}
            rec["phone"] = arguments["phone"]
        rec["executed_calls"].append({"name": name, "resolved_id": resolved_id, "arguments": dict(arguments),
                                      "t_wall": time.time()})
        return {"ok": True}, diff
