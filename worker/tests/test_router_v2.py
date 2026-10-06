"""S2S serverless worker: D-ROUTER-V2 CPU test of router/router_core.py with the Needle v2 router (and the N1 rollback).

  <venv-needle>/bin/python worker/tests/test_router_v2.py --v2-weights <needle_v2 tuned_full.cact> \
        --n1-weights <N1 tuned_full.cact> [--out results.json]

Real Needle (CPU, ctypes lib) + the real worker code (router_core, needle_v2/schema, needle_v2/numconv, the N1 subset in
worker/needle, stub_tools) + the V4 demo records (common/data/records_v4.json). ASR is a stub that returns the given
transcript as text_raw (and text), so each check is one handle_trigger() call on a non-silent pcm window.

Checks:
  K1  the live bug (2026-10-06, session s-20261006T084506-47103a): food_01, "mujhe order cancel karna hai. order cancel
      karna hai. I want to cancel the order" -> v2: cancel_order EXECUTED on FD4124 (status cancelled), nothing dropped
  K2  same input with the N1 router (S2S_ROUTER=n1 path, unchanged) -> cancel_order UNBOUND (reproduces the bug)
  R*  rows Needle v2 scored correct in its eval (needle_v2/eval/runs/v2_test_*.json): phone / email / address updates
      and a cancel by spoken ID are EXECUTED with the expected server args (phone/email have no record reference:
      the old is_unbound guard would have dropped them)
  A1  model-driven ask (eval row ecom_07_g1__cl0:opus, phone heard as "SS397071564" = 9 digits) -> needs_clarification
      event, nothing executed
  A2  deterministic ask: resolve_calls on a 9-digit phone / a malformed email / an unknown order ID -> ask, and the
      needs_clarification event path through handle_trigger (Needle output injected)
  P   the stage protocol: trigger -> asr -> needle -> resolved -> executed | needs_clarification
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.dirname(HERE)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   <- {str(detail)[:600]}"), flush=True)
    return cond


KARNA = "mujhe order cancel karna hai. order cancel karna hai. I want to cancel the order"
# (id, agent record, transcript as Trelis gave it, expected tool, expected server args subset) - v2 eval rows marked
# correct (opus = Trelis on opus-coded audio; exact = the scripted text)
ROWS = [
    ("R1 food_07 change_delivery_address (opus)", "food_07",
     "hello mujhe mera delivery address change karna hai galti se purana office address select ho gaya hai order id fd "
     "9206 hai main abhi office ja raha hoon toh please jaldi kar do naye address hai house 15 sector 31 gurgaon",
     "change_delivery_address", {"order_id": "FD9206", "address": "house 15 sector 31 gurgaon"}),
    ("R2 ecom_07 update_contact_number (opus, no record ref)", "ecom_07",
     "delivery ke time ghar pe nahi rahunga haan wahi chahiye kyunki main office ja raha hoon aur ghar pe koi aur "
     "receive kar lega ji aap ye wala number note kar lijiye 6397071564 ye mere brother ka number hai",
     "update_contact_number", {"phone": "6397071564"}),
    ("R3 air_11 update_contact_number (opus, no record ref)", "air_11",
     "mera PNR TM3847 hai main flight alerts miss nahi karna chahta haan woh sab theek hai bas mujhe phone number change "
     "karna hai mera naya number 9650018553 hai is pe saare updates bhej dena",
     "update_contact_number", {"phone": "9650018553"}),
    ("R4 air_11 update_email_address (exact, no record ref)", "air_11",
     "Mera naya number 9650018553 hai, ispe saare updates bhej dena. Thank you. Ab mera email address bhi change kar "
     "dijiye please. Mera naya email banerjee42@yahoo.co.in hai, isko system mein add kar lo.",
     "update_email_address", {"email": "banerjee42@yahoo.co.in"}),
    ("R5 ecom_03 cancel_order by spoken ID (opus)", "ecom_03",
     "arre yaar maine wrong variant select kar liya main toh 256 GB wala chahta tha haan wahi karna hai main abhi is "
     "order ko cancel karwana chahta hoon order ID EEC6168 hai please isko jaldi cancel kar dijiye kaafi problem ho "
     "jayegi",
     "cancel_order", {"order_id": "EC6168"}),
]
ASK_ROW = ("A1 ecom_07 phone heard as 9 digits (opus) -> ask", "ecom_07",
           "3 par delivery ke time main ghar pe nahi rahunga haan wahi problem hai main office ja raha hoon toh please "
           "mere brother ka number add kar dijiye theek hai note kijiye mere brother ka number SS397071564 is pe call "
           "kar lena", "update_contact_number")


def pcm_window(seconds=3.0, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)   # RMS 0.07 > SILENCE_RMS


TR = {"phrase": "ek minute", "rule": "test", "score": 1.0, "segment": "ek minute rukiye", "frame": 100,
      "model_segments": ["test"]}


def run(core, transcript):
    evs = []
    core.emit_fn = evs.append
    core.asr_fn = lambda pcm: {"text": transcript, "text_raw": transcript, "backend": "stub"}
    t = time.perf_counter()
    summ = core.handle_trigger(dict(TR), pcm_window())
    return summ, evs, round((time.perf_counter() - t) * 1000, 1)


def stages(evs):
    return [e["stage"] for e in evs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-weights", required=True)
    ap.add_argument("--n1-weights", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    os.environ["S2S_NEEDLE_V2_WEIGHTS"] = a.v2_weights
    os.environ["S2S_NEEDLE_WEIGHTS"] = a.n1_weights
    os.environ.pop("S2S_ROUTER", None)              # default must be v2
    os.environ.pop("N1_DIR", None)
    sys.path.insert(0, os.path.join(WORKER, "router"))
    sys.path.insert(0, WORKER)
    import paths
    paths.ensure_common_on_path()
    import router_core as rc
    import session as sessmod
    recs = sessmod.load_records()
    out = {"v2_weights": a.v2_weights, "n1_weights": a.n1_weights, "cases": []}

    check("default router is v2 (S2S_ROUTER unset)", rc.ROUTER == "v2", rc.ROUTER)
    core = rc.RouterCore("food_delivery_support", recs["food_01"], None)
    import n1path
    import numconv  # noqa: F401
    check("v2: N1 code = the worker's own subset (N1_DIR)", os.path.realpath(n1path.N1_DIR) ==
          os.path.realpath(str(paths.NEEDLE)), n1path.N1_DIR)
    check("v2: router module = worker/needle_v2/schema/router_v2.py", os.path.realpath(core.router.__file__) ==
          os.path.realpath(os.path.join(WORKER, "needle_v2", "schema", "router_v2.py")), core.router.__file__)
    check("v2: weights = S2S_NEEDLE_V2_WEIGHTS", core.weights == a.v2_weights, core.weights)
    check("v2: the 5 demo agent types are supported", all(t in core.router.tools_v2.AGENT_TYPES for t in
                                                          sessmod.SUPPORTED_AGENT_TYPES), sessmod.SUPPORTED_AGENT_TYPES)

    # ---- K1: the live bug, v2
    summ, evs, ms = run(core, KARNA)
    nd = next((e for e in evs if e["stage"] == "needle"), {})
    ex = [e for e in evs if e["stage"] == "executed"]
    out["cases"].append({"id": "K1", "router": "v2", "transcript": KARNA, "stages": stages(evs),
                         "function_calls": nd.get("payload", {}).get("function_calls"),
                         "calls": summ.get("calls"), "executed": [e["payload"] for e in ex], "ms": ms})
    print("   K1 needle:", json.dumps(nd.get("payload", {}).get("function_calls")), "| resolved:",
          [(c["name"], c["order_ref"], c["resolved_id"], c["resolver_rule"], c["ask"]) for c in summ.get("calls", [])])
    check("K1 v2: stages trigger asr needle resolved executed", stages(evs)[:4] == ["trigger", "asr", "needle", "resolved"]
          and "executed" in stages(evs) and "error" not in stages(evs), stages(evs))
    check("K1 v2: cancel_order EXECUTED on FD4124 (not dropped)", any(e["payload"]["name"] == "cancel_order" and
          e["payload"]["resolved_id"] == "FD4124" for e in ex), [e["payload"] for e in ex])
    check("K1 v2: no unbound / needs_clarification", not ({"unbound", "needs_clarification"} & set(stages(evs))),
          stages(evs))
    check("K1 v2: stub record FD4124 status cancelled",
          core.tools.record["entity_updates"].get("FD4124", {}).get("status") == "cancelled",
          core.tools.record["entity_updates"])

    # ---- K2: same input, N1 (rollback path, unchanged) -> the bug
    n1 = rc.RouterCore("food_delivery_support", recs["food_01"], None, router_kind="n1")
    check("n1: router = worker/needle/router.py, N1 weights", os.path.basename(n1.router.__file__) == "router.py"
          and n1.weights == a.n1_weights, (n1.router.__file__, n1.weights))
    summ1, evs1, ms1 = run(n1, KARNA)
    nd1 = next((e for e in evs1 if e["stage"] == "needle"), {})
    ub = [e for e in evs1 if e["stage"] == "unbound"]
    out["cases"].append({"id": "K2", "router": "n1", "transcript": KARNA, "stages": stages(evs1),
                         "function_calls": nd1.get("payload", {}).get("function_calls"),
                         "unbound": [e["payload"] for e in ub], "ms": ms1})
    print("   K2 n1 needle:", json.dumps(nd1.get("payload", {}).get("function_calls")), "| unbound:",
          [(e["payload"]["order_ref"], e["payload"]["resolver_rule"]) for e in ub])
    check("K2 n1: reproduces the live bug (cancel_order unbound, nothing executed)",
          any(e["payload"]["name"] == "cancel_order" for e in ub) and "executed" not in stages(evs1), stages(evs1))

    # ---- R*: eval rows v2 scored correct
    for name, rid, text, tool, want in ROWS:
        rec = recs[rid]
        c = rc.RouterCore(rec["agent_type"], rec, None)
        s, e, ms = run(c, text)
        exr = [x["payload"] for x in e if x["stage"] == "executed" and x["payload"]["name"] == tool]
        out["cases"].append({"id": name, "record": rid, "stages": stages(e), "function_calls":
                             next((x["payload"]["function_calls"] for x in e if x["stage"] == "needle"), None),
                             "executed": [x["payload"] for x in e if x["stage"] == "executed"],
                             "asks": [x["payload"] for x in e if x["stage"] == "needs_clarification"], "ms": ms})
        ok = any(all(p["arguments"].get(k) == v for k, v in want.items()) for p in exr)
        check(f"{name}: {tool} EXECUTED with {want}", ok and "unbound" not in stages(e) and "error" not in stages(e),
              (stages(e), exr, [x["payload"] for x in e if x["stage"] in ("needle", "needs_clarification")]))
        if tool == "update_contact_number" and ok:
            check(f"{name}: stub record phone updated (entity = primary_id {rec['primary_id']})",
                  c.tools.record.get("phone") == want["phone"], c.tools.record.get("phone"))

    # ---- A1: model-driven ask
    name, rid, text, tool = ASK_ROW
    rec = recs[rid]
    c = rc.RouterCore(rec["agent_type"], rec, None)
    s, e, ms = run(c, text)
    asks = [x["payload"] for x in e if x["stage"] == "needs_clarification"]
    out["cases"].append({"id": name, "record": rid, "stages": stages(e), "asks": asks, "ms": ms})
    check(f"{name}: needs_clarification event, {tool} not executed",
          any(p["name"] == tool and any("phone_not_10_digits" in r for r in p["ask_reasons"]) for p in asks)
          and not any(x["stage"] == "executed" and x["payload"]["name"] == tool for x in e), (stages(e), asks))

    # ---- A2: deterministic ask paths (resolver_v2 + router_v2 NEW-arg checks) and the event path
    rv2 = core.router
    rec = recs["food_01"]
    cases = [("phone 9 digits", [{"name": "update_contact_number", "arguments": {"phone": "987654321"}}],
              "phone_not_10_digits"),
             ("email malformed", None, None),
             ("unknown order ID", [{"name": "cancel_order", "arguments": {"order_ref": "FD9999"}}], "unknown_id"),
             ("empty address", [{"name": "change_delivery_address", "arguments": {"address": ""}}], "address_missing")]
    for label, fc, why in cases:
        if fc is None:
            continue
        r = rv2.resolve_calls(fc, rec, "food_delivery_support")
        check(f"A2 resolve_calls {label} -> ask ({why})", r and r[0]["ask"] and any(why in x for x in r[0]["ask_reasons"]),
              r)
    r = rv2.resolve_calls([{"name": "update_email_address", "arguments": {"email": "pandey at gmail"}}],
                          recs["sub_01"] if "sub_01" in recs else rec, "subscription_account_support")
    check("A2 resolve_calls email malformed -> ask", r and r[0]["ask"], r)
    r = rv2.resolve_calls([{"name": "cancel_order", "arguments": {"order_ref": "Karna"}}], rec, "food_delivery_support")
    check("A2 resolve_calls cancel_order order_ref 'Karna' (the N1 output) -> FD4124, no ask (rule "
          f"{r[0]['resolver_rule'] if r else '?'})", r and not r[0]["ask"] and r[0]["resolved_id"] == "FD4124", r)
    r = rv2.resolve_calls([{"name": "cancel_order", "arguments": {}}], rec, "food_delivery_support")
    check("A2 resolve_calls cancel_order without a reference -> active order FD4124", r and not r[0]["ask"] and
          r[0]["resolved_id"] == "FD4124", r)
    c = rc.RouterCore("food_delivery_support", rec, None)
    c.router = type("R", (), {"route_raw": staticmethod(lambda *a_, **k: {"function_calls": [
        {"name": "update_contact_number", "arguments": {"phone": "987654321"}}]}),
        "resolve_calls": staticmethod(rv2.resolve_calls)})
    s, e, ms = run(c, "mera naya number 98765 4321 hai")
    check("A2 event path: injected 9-digit phone -> needs_clarification, nothing executed, record phone unchanged",
          stages(e) == ["trigger", "asr", "needle", "resolved", "needs_clarification"]
          and c.tools.record.get("phone") == rec.get("phone"), (stages(e), [x["payload"] for x in e][-1:]))
    p = e[-1]["payload"] if e else {}
    check("A2 needs_clarification payload has name, ask_reasons, server_args, reason",
          all(k in p for k in ("name", "ask_reasons", "server_args", "reason")), p)
    json.dumps(e)   # every event must be JSON-serialisable (they are POSTed to /internal/action)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    out["passed"], out["total"] = len(RESULTS) - n_fail, len(RESULTS)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
