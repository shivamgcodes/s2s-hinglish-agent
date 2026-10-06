"""Resolver tests built from real data/records.json entries (copy in src/records.json).

Every expected ID below was typed by hand from the record (primary_id / secondary_id /
distractors), not copied from resolver output. Columns:
  (record key, lang, order_ref phrase, expected id, what the case exercises)
Phrases marked [real] are copied from a customer turn in V1/V3 calls.jsonl.
Run:  python3 test_resolver.py   (writes test_resolver_results.json next to it)
"""
import collections
import json
import os
import sys

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))  # build_data, resolver, router, tools, src/records.json
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (data, results/); was this script's own dir
sys.path.insert(0, N1_PKG)
from resolver import resolve, N0_RECORD  # noqa: E402

CASES = [
    # ---------------- food_delivery_support
    ("food_01", "hi", "Behrouz wala order", "FD1000", "restaurant name -> primary"),
    ("food_01", "hi", "momos wala", "FD3500", "item word of the older order (plural)"),
    ("food_01", "en", "my Wow Momo order", "FD3500", "restaurant name -> older order"),
    ("food_01", "hi", "purana order", "FD3500", "purana -> past entity"),
    ("food_01", "hi", "galouti kebab wala", "FD1000", "item name -> primary"),
    ("food_03", "en", "the biryani order", "FD3726", "food_03: biryani is the OLDER order here"),
    ("food_03", "hi", "momo wala order", "FD1226", "food_03: momo is the current order"),
    ("food_04", "hi", "Haldiram wala", "FD3839", "restaurant (apostrophe-less) -> older"),
    ("food_06", "hi", "F D one five six five", "FD1565", "spoken ID, English digits"),
    ("food_06", "hi", "FD ek paanch chhe paanch", "FD1565", "spoken ID, Hindi digits"),
    ("food_06", "en", "order 4065", "FD4065", "digit block of the older ID"),
    ("food_09", "hi", "mera order FD1904", "FD1904", "[real] explicit ID in Hinglish turn"),
    ("food_08", "hi", "purana address", "FD1791", "purana modifies address -> not ordinal, default active"),
    ("food_06", "hi", "", "FD1565", "no phrase -> single active entity"),
    ("food_06", "en", "pehla wala", "FD4065", "pehla -> chronologically first (older)"),
    ("food_06", "en", "my latest order", "FD1565", "latest -> active"),
    ("food_07", "hi", "pichhla order", "FD4178", "pichhla -> past"),
    # ---------------- ecommerce_support
    ("ecom_01", "en", "the Sony headphones", "EC2469", "brand/product -> primary"),
    ("ecom_01", "hi", "Samsung phone wala order", "EC4969", "brand -> older order"),
    ("ecom_03", "hi", "boAt headphones wala", "EC5195", "ecom_03: headphones are the OLDER order"),
    ("ecom_03", "en", "the Galaxy M35", "EC2695", "product -> primary"),
    ("ecom_03", "hi", "EC2695", "EC2695", "[real] 'please EC2695 cancel kar dijiye'"),
    ("ecom_04", "hi", "yeh toh mera purana ghar hai", "EC2808", "[real] purana ghar -> not ordinal"),
    ("ecom_09", "en", "the power bank order", "EC5873", "item -> older"),
    ("ecom_11", "en", "e c three five nine nine", "EC3599", "spoken letters + digits"),
    ("ecom_02", "en", "my order", "EC2582", "only text-less secondary exists -> default active"),
    ("ecom_13", "en", "previous order", "EC6325", "previous -> past"),
    # ---------------- cab_ride_support
    ("cab_01", "hi", "airport wali ride", "RD3938", "drop place -> current ride"),
    ("cab_01", "hi", "Noida wali ride", "RD6438", "place of past ride"),
    ("cab_03", "en", "my current ride", "RD4164", "current -> active"),
    ("cab_03", "hi", "pichhli ride", "RD6664", "pichhli -> past"),
    ("cab_03", "en", "R D four one six four", "RD4164", "spoken ID"),
    ("cab_07", "hi", "", "RD4616", "no phrase -> active ride"),
    ("cab_11", "en", "the Dwarka ride", "RD7568", "place of past ride"),
    ("cab_15", "hi", "Oberoi wali ride", "RD5520", "drop hotel -> current ride"),
    ("cab_15", "en", "ride 8020", "RD8020", "digit block -> past ride"),
    # ---------------- subscription_account_support
    ("sub_07", "hi", "Family Annual plan", "AC6424", "[real] plan name -> account"),
    ("sub_01", "en", "the Basic Quarterly plan", "AC8246", "previous plan name (ID-less distractor) -> old ref"),
    ("sub_02", "hi", "Premium Annual plan", "AC5859", "[real]-style plan -> account"),
    ("sub_02", "en", "the old Basic Monthly plan", "AC8359", "old plan name -> old reference"),
    ("sub_05", "hi", "mera account AC6198", "AC6198", "[real] 'My account ID is AC6198'"),
    ("sub_11", "en", "invoice 9376", "AC9376", "digit block -> past invoice"),
    ("sub_09", "hi", "purana number", "AC6650", "purana number -> not ordinal, default"),
    ("sub_12", "hi", "mera plan", "AC6989", "generic phrase -> default active"),
    # ---------------- airport_ticket_counter
    ("air_01", "en", "my return flight", "PB9715", "return -> return PNR"),
    ("air_01", "hi", "wapsi wali flight", "PB9715", "wapsi -> return PNR"),
    ("air_01", "hi", "Mumbai wali flight", "JB7215", "route place -> onward booking"),
    ("air_01", "en", "the first ticket", "JB7215", "first -> onward (return is later)"),
    ("air_01", "en", "second booking", "PB9715", "second -> return"),
    ("air_15", "en", "Reference is HQ2297", "HQ2297", "[real] V1 air_15_g1 refund against the reference"),
    ("air_15", "hi", "Jaipur wala trip", "HQ2297", "place of past reference"),
    ("air_15", "en", "my booking ID is MD8797", "MD8797", "[real] explicit primary ID"),
    ("air_09", "en", "HR1819", "HR8119", "record misread_id -> primary"),
    ("air_16", "hi", "N J two four one zero", "NJ2410", "spoken return PNR"),
    ("air_17", "hi", "", "FJ9023", "no phrase -> active booking"),
    ("air_10", "en", "the London flight", "QH8232", "route -> onward"),
    # ---------------- batch 2: written AFTER batch 1 passed 64/64, resolver not changed for them
    ("food_01", "hi", "beharoz biryani", "FD1000", "b2: misspelt restaurant (ASR-like)"),
    ("food_01", "hi", "ef dee teen paanch zero zero", "FD3500", "b2: spoken letters + mixed Hindi/English digits"),
    ("food_02", "en", "FD 3613", "FD3613", "b2: spaced ID"),
    ("food_05", "hi", "do momos wala", "FD3952", "b2: 'do' (two) before item"),
    ("cab_02", "hi", "Connaught Place wali ride", "RD4051", "b2: drop place -> current"),
    ("cab_06", "hi", "airport se Noida wali", "RD7003", "b2: past ride places"),
    ("cab_14", "hi", "CP wali ride", "RD7907", "b2: abbreviation CP = Connaught Place (past ride)"),
    ("ecom_06", "hi", "mouse wala order", "EC3034", "b2: product word"),
    ("ecom_10", "hi", "boAt wala", "EC5986", "b2: brand of older order"),
    ("sub_08", "en", "Basic Monthly", "AC9037", "b2: previous plan name"),
    ("air_13", "en", "return flight", "SH2071", "b2: 'Reference ... for return flight'"),
    ("air_14", "hi", "Z X 2184", "ZX2184", "b2: spaced letters + digits"),
]

N0_CASES = [  # needle-exp0 tests.jsonl labels; two ACTIVE orders, A1234 placed earlier
    ("en", "Domino's", "B5678", "act-en-01"),
    ("en", "the biryani order", "A1234", "act-en-02"),
    ("en", "my first order", "A1234", "act-en-06"),
    ("en", "my latest order", "B5678", "act-en-12"),
    ("hi", "pehla wala order", "A1234", "act-hi-03"),
    ("hi", "dusre wale order", "B5678", "act-hi-09"),
    ("hi", "order A1234", "A1234", "act-hi-06"),
    ("hi", "", None, "no phrase, two active orders -> ambiguous (None)"),
]


def main():
    path = os.path.join(N1_PKG, "src", "records.json")        # packages copy (md5 = pod's)
    if not os.path.exists(path):
        path = os.path.join(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish"), "data", "records.json")  # pod
    records = json.load(open(path))
    rows = []
    for key, lang, ref, want, note in CASES:
        rec = records[key]
        got = resolve(ref, rec, rec["agent_type"])
        rows.append({"set": "records", "record": key, "agent_type": rec["agent_type"], "lang": lang,
                     "order_ref": ref, "expected": want, "got": got["id"], "rule": got["rule"],
                     "ok": got["id"] == want, "note": note})
    for lang, ref, want, note in N0_CASES:
        got = resolve(ref, N0_RECORD, "food_delivery_support")
        rows.append({"set": "n0", "record": "N0", "agent_type": "food_delivery_support(N0)", "lang": lang,
                     "order_ref": ref, "expected": want, "got": got["id"], "rule": got["rule"],
                     "ok": got["id"] == want, "note": note})

    def table(keyf, title):
        c = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            c[keyf(r)][0] += r["ok"]
            c[keyf(r)][1] += 1
        print(f"\n{title}")
        for k in sorted(c):
            print(f"  {k:<32} {c[k][0]}/{c[k][1]}")

    n_ok = sum(r["ok"] for r in rows)
    print(f"resolver accuracy: {n_ok}/{len(rows)}")
    table(lambda r: r["agent_type"], "by agent type")
    table(lambda r: r["lang"], "by language")
    table(lambda r: r["rule"].split(":")[0], "by rule that fired")
    table(lambda r: "non-primary expected" if r["set"] == "records" and r["expected"] != records[r["record"]]["primary_id"]
          else "primary/other expected", "by target")
    fails = [r for r in rows if not r["ok"]]
    print(f"\nfailures ({len(fails)}):")
    for r in fails:
        print(f"  {r['record']} [{r['lang']}] {r['order_ref']!r}: expected {r['expected']} got {r['got']} "
              f"(rule {r['rule']}) -- {r['note']}")
    json.dump(rows, open(os.path.join(HERE, "test_resolver_results.json"), "w"), indent=1, ensure_ascii=False)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
