"""N1-comparable scoring: N1 tuned_full and Needle v2 predictions mapped to the same server shape (writes[]) and
scored against the same gold, per check-line.

    gold_for(call, cl_entry, record)            -> [{"tool", "args"}]   (targets.gold_canonical per write)
    n1_to_server(n1_calls, record, agent_type)  -> [{"tool", "args", "ask"}]   N1 router.route() output
    v2_to_server(v2_calls)                      -> [{"tool", "args", "ask"}]   router_v2.route() output
    score(gold, pred, agent_type)               -> {"tool_match", "args_match", "correct", "per_arg", "group"}

Server shape = the arg names of calls.jsonl writes[] (order_id, ride_id, pnr, reference_id, plan, card_last4,
transaction_id, service, pack, sim_number, phone, email, address, location, instruction, reason).

N1 -> server: N1 emits order_ref (+ real args). Its resolved_id (N1 resolver) fills the tool's ID arg
  (order_id / ride_id / pnr / reference_id). For cancel_subscription / pause_subscription the gold is a plan
  NAME, which the N1 resolver cannot produce (it returns the account ID); N1's order_ref is therefore resolved
  with resolver_v2.resolve("plan", ...), the same resolver v2 uses (favours neither). update_contact_number:
  N1's optional order_ref is ignored (gold has no ID arg). N1 phone -> 10 digits like v2.
  N1 only knows the 5 old agent types: bank/telecom rows are v2-only.
v2 -> server: router_v2 server_args (REF args resolved by resolver_v2, phone 10 digits, email normalised).

Per-arg comparison (N1 evaluate.compare_arg rules, so numbers stay comparable with REPORT.md):
  ID / card / plan / service / pack / sim_number   equality after resolver canonicalisation (exact string)
  phone        10-digit equality
  address, instruction     token F1 >= 0.6 (N1 rule)
  location, reason, email  N1 rule (normalised equality; email after at/dot normalisation); token F1 reported
A call with ask=True counts as not correct (the server would not execute it) and is reported separately.
correct = same tool names in the same order AND every gold arg matches AND no extra non-empty arg AND not ask.
Negative rows do not exist in this scheme: Needle only runs after a check-line and every V4 check-line maps to
exactly one write (targets.checkline_writes).
"""
import os
import re
import sys
from pathlib import Path

# monorepo shim: router_v2/resolver_v2/targets/tools_v2/n1path live in packages/needle_router/v2 ($NEEDLE_V2_DIR);
# N1 evaluate.py (compare_arg, token_f1) lives in research/needle/n1 ($N1_EVAL_DIR). Was: all in the same dir.
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(_REPO / "packages" / "needle_router" / "v2")))
sys.path.insert(0, os.environ.get("N1_EVAL_DIR", str(_REPO / "research" / "needle" / "n1")))
import n1path  # noqa: E402,F401
import evaluate as n1eval  # noqa: E402  # N1 evaluate.py (compare_arg, token_f1)
import resolver_v2 as rv
import targets
import tools_v2

N1_ID_ARG = {"cancel_order": "order_id", "change_delivery_address": "order_id", "add_delivery_instruction":
             "order_id", "cancel_ride": "ride_id", "change_pickup_location": "ride_id", "change_drop_location":
             "ride_id", "add_driver_instruction": "ride_id", "request_ticket_cancellation": "pnr",
             "request_refund": "reference_id"}
REF_GOLD_ARGS = {"order_id", "ride_id", "pnr", "reference_id", "plan", "card_last4", "transaction_id", "service",
                 "pack", "sim_number"}


def gold_for(call, cl_entry, record):
    return [targets.gold_canonical(w, record, call["agent_type"]) for w in cl_entry["writes"]]


def n1_to_server(n1_calls, record, agent_type):
    out = []
    for c in n1_calls or []:
        name, args = c.get("name"), dict(c.get("arguments") or {})
        sa, ask = {}, False
        if name in ("cancel_subscription", "pause_subscription"):
            r = rv.resolve("plan", c.get("order_ref") or "", record, agent_type)
            sa["plan"], ask = r["value"], r["ask"]
        elif name in N1_ID_ARG:
            sa[N1_ID_ARG[name]] = c.get("resolved_id")
            ask = c.get("resolved_id") is None
        for k, v in args.items():
            if k == "order_ref":
                continue
            sa[k] = targets.phone10(v) if k == "phone" else v
        out.append({"tool": name, "args": sa, "ask": ask})
    return out


def v2_to_server(v2_calls):
    return [{"tool": c["name"], "args": dict(c.get("server_args") or {}), "ask": bool(c.get("ask"))}
            for c in v2_calls or []]


def compare_arg(key, gold, pred):
    if key in REF_GOLD_ARGS:
        ok = pred is not None and str(pred) == str(gold)
        return ok, {}
    if key == "phone":
        ok = pred is not None and targets.phone10(pred) == targets.phone10(gold) and len(targets.phone10(gold)) == 10
        return ok, {}
    ok, det = n1eval.compare_arg(key, gold, pred)
    if "token_f1" not in det and pred is not None:
        det["token_f1"] = round(n1eval.token_f1(gold, pred), 4)
    return ok, det


def score(gold, pred, agent_type=None):
    gn, pn = [g["tool"] for g in gold], [p["tool"] for p in pred]
    s = {"gold_tools": gn, "pred_tools": pn, "tool_match": gn == pn, "args_match": None, "ask": False,
         "per_arg": [], "correct": False}
    if s["tool_match"]:
        all_ok = True
        for g, p in zip(gold, pred):
            res = {}
            for k in sorted(set(g["args"]) | {k for k, v in p["args"].items() if v not in (None, "")}):
                ok, det = compare_arg(k, g["args"].get(k), p["args"].get(k))
                res[k] = {"ok": ok, "gold": g["args"].get(k), "pred": p["args"].get(k), **det}
                all_ok &= ok
            s["per_arg"].append(res)
            s["ask"] |= bool(p.get("ask"))
        s["args_match"] = all_ok
        s["correct"] = all_ok and not s["ask"]
    if not s["correct"]:
        if not pn:
            s["group"] = "missed call"
        elif not s["tool_match"]:
            s["group"] = "wrong tool" if not set(pn) & set(gn) else "partial / extra call"
        elif not s["args_match"]:
            bad = sorted({k for r in s["per_arg"] for k, v in r.items() if not v["ok"]})
            s["group"] = "wrong argument: " + ",".join(bad)
        else:
            s["group"] = "ask (resolver/format)"
    return s


def is_old_type(agent_type):
    return agent_type in tools_v2.OLD_TYPES
