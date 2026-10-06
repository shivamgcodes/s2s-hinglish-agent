"""Unit tests for tools_v2 / resolver_v2 / targets / router_v2.resolve_calls / score_map on the V4 records.

Run: python3 test_resolver_v2.py [V4_DIR]      (or pytest test_resolver_v2.py; V4_DIR env var)
V4_DIR default: /root/n2/V4 (runpod2), else /workspace/hinglish/data/V4 (pod1), else ./src.
CPU only, a few seconds. Writes nothing.
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # score_map.py
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", os.path.join(HERE, "..", "..", "..", "..", "packages", "needle_router", "v2")))  # monorepo shim
import resolver_v2 as rv  # noqa: E402
import router_v2  # noqa: E402
import score_map  # noqa: E402
import targets  # noqa: E402
import tools_v2  # noqa: E402


def _v4_dir():
    for d in [os.environ.get("V4_DIR"), (sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else None),
              os.environ.get("N2_ROOT", "/root/n2") + "/V4", os.environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/data/V4", os.path.join(HERE, "src")]:
        if d and os.path.exists(os.path.join(d, "records.json")):
            return d
    raise FileNotFoundError("V4 records.json not found")


V4 = _v4_dir()
R = json.load(open(os.path.join(V4, "records.json"), encoding="utf-8"))
C = [json.loads(l) for l in open(os.path.join(V4, "calls.jsonl"), encoding="utf-8") if l.strip()]


def res(kind, ref, sid):
    return rv.resolve(kind, ref, R[sid], R[sid]["agent_type"])


def val(kind, ref, sid):
    return res(kind, ref, sid)["value"]


def asks(kind, ref, sid):
    return res(kind, ref, sid)["ask"]


# ---------------------------------------------------------------------------------------------- schemas
def test_schemas():
    for a in tools_v2.AGENT_TYPES:
        ts = tools_v2.tools_for(a)
        assert [t["name"] for t in ts] == tools_v2.STATIC_WRITE[a]
        for t in ts:
            blob = json.dumps(t)
            assert not re.search(r"\d{4}", blob), (a, t["name"], "digits in description")
            assert not re.search(r"\w@\w+\.", blob) and not re.search(r"\b[A-Z]{2}\d", blob), (a, t["name"])
            for p in t["parameters"].get("required", []):
                assert p in t["parameters"]["properties"]
            assert all(not k.endswith("_id") for k in t["parameters"]["properties"]), "no raw ID args"


def test_every_write_arg_is_classified():
    for x in C:
        for w in x["writes"]:
            specs = tools_v2.arg_specs(x["agent_type"], w["tool"])
            assert {g for _, _, g in specs} == set(w["args"]), (x["call_id"], w)


# ---------------------------------------------------------------------------------------------- gold coverage
def test_gold_refs_resolve_from_record():
    """Every gold REF value of every V4 write is a candidate (or resolves to one) and the canonical target
    resolves back exactly."""
    n = Counter()
    for x in C:
        rec, at = R[x["scenario_id"]], x["agent_type"]
        for w in x["writes"]:
            for marg, kind, garg in tools_v2.arg_specs(at, w["tool"]):
                if kind not in tools_v2.REF_KINDS:
                    continue
                canon = targets.canonical_ref(kind, w["args"][garg], rec, at)
                assert canon in [c["value"] for c in rv.candidates(kind, rec)], (x["call_id"], kind, canon)
                assert rv.resolve(kind, canon, rec, at)["value"] == canon
                n[kind] += 1
    assert sum(n.values()) == 469, n


def test_roundtrip_all_checklines():
    """gold writes -> canonical target -> router_v2.resolve_calls -> server shape == gold, for all check-lines."""
    n = 0
    for x in C:
        rec, at = R[x["scenario_id"]], x["agent_type"]
        for cl in targets.checkline_writes(x):
            assert len(cl["writes"]) == 1, (x["call_id"], cl["cl_idx"])
            tgt = [targets.target_call(w, rec, at) for w in cl["writes"]]
            pred = score_map.v2_to_server(router_v2.resolve_calls(tgt, rec, at))
            s = score_map.score(score_map.gold_for(x, cl, rec), pred, at)
            assert s["correct"], (x["call_id"], cl["cl_idx"], tgt, s)
            n += 1
    assert n == 647, n


def test_phone_targets_are_10_digits():
    for x in C:
        for w in x["writes"]:
            if "phone" in w["args"]:
                t = targets.target_call(w, R[x["scenario_id"]], x["agent_type"])
                assert re.fullmatch(r"\d{10}", t["arguments"]["phone"]), t


def test_block_sim_never_writes_digits():
    for x in C:
        for w in x["writes"]:
            if w["tool"] == "block_sim":
                t = targets.target_call(w, R[x["scenario_id"]], x["agent_type"])
                assert t["arguments"] == {"number_ref": "registered"}, t


# ---------------------------------------------------------------------------------------------- spoken refs
def test_order_refs():
    s = "food_01"   # FD4124 Burger Singh (active), older FD3076 Sagar Ratna
    assert val("order", "FD4124", s) == "FD4124"
    assert val("order", "F D four one two four", s) == "FD4124"
    assert val("order", "4124", s) == "FD4124"
    assert val("order", "Burger Singh wala", s) == "FD4124"
    assert val("order", "Sagar Ratna wala", s) == "FD3076"
    assert val("order", "purana order", s) == "FD3076"
    assert val("order", "", s) == "FD4124"
    assert res("order", "FD9999", s)["rule"] == "unknown_id" and asks("order", "FD9999", s)


def test_transaction_refs():
    s = "bank_08"   # TX6742 Coursera Rs 4,127 ; TX3004 Zomato Rs 843
    assert val("transaction", "TX6742", s) == "TX6742"
    assert val("transaction", "Coursera wala", s) == "TX6742"
    assert val("transaction", "Zomato", s) == "TX3004"
    assert val("transaction", "4127 wala charge", s) == "TX6742"
    assert asks("transaction", "", s)                                  # two transactions, nothing said
    assert asks("transaction", "Indian Oil", "bank_20")               # two identical Indian Oil charges
    assert val("transaction", "TX6180", "bank_20") == "TX6180"
    assert val("transaction", "", "bank_19") == "TX6416"              # single transaction on record


def test_card_refs():
    assert val("card", "8923", "bank_01") == "8923"
    assert val("card", "", "bank_01") == "8923"
    assert val("card", "mera credit card", "bank_01") == "8923"
    assert val("card", "card ending eight nine two three", "bank_01") == "8923"
    assert asks("card", "1234", "bank_01")


def test_service_pack_refs():
    s = "tel_03"   # services Daily Horoscope, Cricket Alerts ; packs DataBoost 10, DataBoost 25
    assert set(c["value"] for c in rv.candidates("service", R[s])) == {"Daily Horoscope", "Cricket Alerts"}
    assert set(c["value"] for c in rv.candidates("pack", R[s])) == {"DataBoost 10", "DataBoost 25"}
    assert val("service", "Daily Horoscope", s) == "Daily Horoscope"
    assert val("service", "horoscope wali service", s) == "Daily Horoscope"
    assert val("service", "cricket alerts band karo", s) == "Cricket Alerts"
    assert asks("service", "", s)
    assert val("pack", "DataBoost 25", s) == "DataBoost 25"
    assert val("pack", "data boost 25", s) == "DataBoost 25"
    assert val("pack", "25 wala pack", s) == "DataBoost 25"
    assert asks("pack", "DataBoost", s)                                # number decides; not said -> ASK
    assert asks("pack", "DataBoost 50", s)
    assert val("pack", "Data Booster 10GB", "tel_11") == "Data Booster 10GB"
    assert val("service", "", "tel_06") == "Cricket Alert Pack"         # single service on record


def test_plan_refs():
    assert val("plan", "CineMax Ultra", "sub_04") == "StreamBox CineMax Ultra"
    assert val("plan", "", "sub_04") == "StreamBox CineMax Ultra"
    assert val("plan", "purana plan", "sub_04") == "Basic Starter"
    assert val("plan", "Basic Starter", "sub_04") == "Basic Starter"


def test_phone_on_file_refs():
    s = "tel_07"   # registered 76780 47746, alternate 99580 87417
    assert val("phone_on_file", "registered", s) == "76780 47746"
    assert val("phone_on_file", "", s) == "76780 47746"
    assert val("phone_on_file", "mera number", s) == "76780 47746"
    assert val("phone_on_file", "alternate", s) == "99580 87417"
    assert val("phone_on_file", "87417", s) == "99580 87417"
    assert asks("phone_on_file", "9876543210", s)
    assert asks("phone_on_file", "alternate", "tel_03")              # no alternate number on record


def test_booking_refs():
    s = "air_18"   # PNR LZ5607 ; duplicate charge WS5041
    assert val("booking", "", s) == "LZ5607"
    assert val("booking", "WS5041", s) == "WS5041"
    assert val("booking", "duplicate charge", s) == "WS5041"
    assert val("booking", "L Z five six zero seven", s) == "LZ5607"


def test_replacement_address():
    rec = R["bank_01"]
    assert rv.resolve_address("Flat 606, Purva Riviera, Marathahalli, Bangalore", rec)["value"].startswith("Flat 606")
    assert rv.resolve_address("", rec)["value"] == "Flat 204, Prestige Shantiniketan, Whitefield, Bangalore"
    assert rv.resolve_address("same address", rec)["rule"] == "registered_address"


# ---------------------------------------------------------------------------------------------- router glue
def test_router_new_value_checks():
    rec, at = R["bank_02"], "bank_card_support"
    out = router_v2.resolve_calls([{"name": "update_contact_number", "arguments": {"phone": "91490 44112"}}], rec, at)
    assert out[0]["server_args"] == {"phone": "9149044112"} and not out[0]["ask"]
    out = router_v2.resolve_calls([{"name": "update_contact_number", "arguments": {"phone": "91490 4411"}}], rec, at)
    assert out[0]["ask"]
    out = router_v2.resolve_calls([{"name": "update_email_address",
                                    "arguments": {"email": "online dot prasad3 at the rate yahoo dot co dot in"}}],
                                  rec, at)
    assert out[0]["server_args"]["email"] == "online.prasad3@yahoo.co.in" and not out[0]["ask"], out
    out = router_v2.resolve_calls([{"name": "update_contact_number", "arguments": {}}], rec, at)
    assert out[0]["ask"]
    out = router_v2.resolve_calls([{"name": "cancel_order", "arguments": {}}], rec, at)
    assert out[0]["ask"] and out[0]["ask_reasons"] == ["unknown_tool"]


def test_n1_mapping():
    rec, at = R["food_01"], "food_delivery_support"
    n1_out = [{"name": "change_delivery_address", "arguments": {"order_ref": "my order", "address": "Flat 9"},
               "order_ref": "my order", "resolved_id": "FD4124", "resolver_rule": "default_active_unmatched"}]
    srv = score_map.n1_to_server(n1_out, rec, at)
    assert srv == [{"tool": "change_delivery_address", "args": {"order_id": "FD4124", "address": "Flat 9"},
                    "ask": False}]
    rec, at = R["sub_04"], "subscription_account_support"
    srv = score_map.n1_to_server([{"name": "pause_subscription", "arguments": {}, "order_ref": None,
                                   "resolved_id": "AC0000"}], rec, at)
    assert srv[0]["args"] == {"plan": "StreamBox CineMax Ultra"}


def test_review_unknown_id_and_context_split():
    """REVIEW-schema-1/2 (2026-10-05): unknown IDs in any spoken form ASK; ID contexts do not bleed."""
    for ref in ["7138", "1496", "F D seven one three eight", "FD 7138", "F D 7 1 3 8",
                "\u090f\u092b\u0921\u0940 \u0938\u093e\u0924 \u090f\u0915 \u0924\u0940\u0928 \u0906\u0920"]:
        r = res("order", ref, "food_01")
        assert r["ask"] and r["rule"] == "unknown_id", (ref, r)
    assert val("order", "Sagar Ratna wala", "food_01") == "FD3076"
    assert val("order", "purana order", "food_01") == "FD3076"
    assert val("order", "haan ji", "food_01") == "FD4124"           # non-ID phrase: N1 default kept
    assert val("order", "FD3076", "food_01") == "FD3076"
    assert val("charge", "10 September", "tel_16") == "TL5911"
    assert val("charge", "7 December wala", "tel_16") == "TL7413"
    assert val("transaction", "Amazon", "bank_03") == "TX1791"
    assert val("transaction", "Amazon", "bank_07") == "TX8032"
    assert val("transaction", "Zomato", "bank_07") == "TX3938"
    assert val("transaction", "Rs 2,841", "bank_03") == "TX1791"     # amount-only ref still resolves
    assert val("transaction", "618", "bank_03") == "TX3836"
    assert asks("transaction", "TX9999", "bank_03")


def main():
    fails = 0
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for k, f in tests:
        try:
            f()
            print("PASS", k)
        except AssertionError as e:
            fails += 1
            print("FAIL", k, repr(e)[:600])
    print(f"{len(tests) - fails}/{len(tests)} passed  (V4 dir {V4}, numconv {'loaded' if rv._numconv else 'MISSING'})")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
