"""CPU unit tests for records_v2 (D2/V4): det pools, deterministic checks, uniqueness pass. Run: python test_records_v2.py [scen.json]"""
import json
import sys
from collections import Counter

import records_v2 as RV

scen_path = sys.argv[1] if len(sys.argv) > 1 else "../guidance/scenario_creation.json"
sc = RV.load_scenarios_v2(scen_path)
det = RV.det_fields_v2(sc)
assert det == RV.det_fields_v2(sc), "det_fields_v2 not deterministic"

# pools: every pooled value offered to <= 3 records, names too
for k in ("customer_last", "customer_first_m", "customer_first_f"):
    c = Counter(d[k] for d in det.values())
    assert c.most_common(1)[0][1] <= 3, (k, c.most_common(1))
c = Counter(v for d in det.values() for v in set(d["dates"].values()))
c2 = Counter(" ".join(v.split()[-2:]) if not v.split()[-1].isdigit() or len(v.split()[-1]) < 4 else v for d in det.values() for v in d["dates"].values())
dm = Counter()
for d in det.values():
    for v in {RV.DATE_RE.search(x).group(1) + " " + RV.DATE_RE.search(x).group(2) for x in d["dates"].values()}:
        dm[v] += 1
assert dm.most_common(1)[0][1] <= 3, dm.most_common(3)
c = Counter(v for d in det.values() for k, v in d["assigned"].items() if k not in RV.CATEGORICAL_ASSIGNED)
assert c.most_common(1)[0][1] <= 3, c.most_common(3)
ids = [d["primary_id"] for d in det.values()] + [d["secondary_id"] for d in det.values()]
assert len(ids) == len(set(ids))
ph = [d[k] for d in det.values() for k in ("phone", "new_phone", "other_phone")]
assert len(ph) == len(set(ph))
em = [d[k] for d in det.values() for k in ("old_email", "new_email")]
assert len(em) == len(set(em))
agent_names = {n for v in RV.V2_BRANDS.values() for b in v for n in b[1:]}
assert not any(d["customer_first_m"] in agent_names or d["customer_first_f"] in agent_names for d in det.values())
# date order by construction
import datetime as dt
for sid, d in det.items():
    call = dt.date.fromisoformat(d["call_date"])
    assert RV.CALL_START <= call < RV.CALL_START + dt.timedelta(RV.CALL_DAYS)
print(f"pools OK over {len(det)} scenarios; date max reuse {dm.most_common(1)}; assigned max {c.most_common(1)}")

# deterministic checks
sid = "food_08" if "food_08" in sc else next(s for s in sc if sc[s]["agent_type"] == "food_delivery_support")
f, s = det[sid], sc[sid]
A = f["assigned"]
info = (f"Order {f['primary_id']} from {A['restaurant']}: Masala Dosa x2, Rs 386, status out for delivery, ETA {A['eta_minutes']}, "
        f"delivery to {A['current_delivery_address']}, paid by UPI. Older order {f['secondary_id']} from "
        f"{A['older_order_restaurant']} on {f['dates']['older_order_date']}, Rs 212.")
cvs = [{"key": "new_address", "value": A["new_delivery_address"]}] if "new_delivery_address" in A else []
if "new_delivery_address_wrong" in A:
    cvs.append({"key": "new_address_wrong", "value": A["new_delivery_address_wrong"]})
good = {"facts": [{"key": "order_id", "value": f["primary_id"]}, {"key": "restaurant", "value": A["restaurant"]},
                  {"key": "items", "value": "Masala Dosa x2"}, {"key": "status", "value": "out for delivery"}],
        "distractors": ["x"], "information": info, "caller_values": cvs, "caller_situation": "x"}
errs, nt = RV.check_record_v2(sid, s, f, good)
assert not errs, errs
print("good record passes; tokens", nt)
bad = json.loads(json.dumps(good))
bad["information"] += (" Saved address Sector 15, Gurgaon, last order 12 Oct, delivered 3 March, email foo@yahoo.com, "
                       f"phone 98000 01234, {f['customer_first_f']} likes it. Also {RV.ADDRESSES['Mumbai'][0] if RV.ADDRESSES['Mumbai'][0] not in A.values() else RV.ADDRESSES['Mumbai'][1]}.")
if bad["caller_values"]:
    bad["caller_values"][0]["value"] = "Flat 2, Somewhere Else, Mumbai"
errs, _ = RV.check_record_v2(sid, s, f, bad)
for e in errs:
    print("   bad ->", e)
need = ["not one of the fixed dates", "month 'Oct'", "email foo@yahoo.com", "phone 98000 01234", "customer's name",
        "is not this record's address", "over-used default"] + (["must be exactly one of the FIXED"] if bad["caller_values"] else [])
for n in need:
    assert any(n in e for e in errs), ("missing check", n)
long = json.loads(json.dumps(good))
long["information"] = info + " " + " ".join(["Extra fact number"] * 40)
assert any("too long" in e for e in RV.check_record_v2(sid, s, f, long)[0])
print("bad record flags all expected checks")

# value extraction + uniqueness
rec = RV._record(sid, s, f, good, nt, [])
vi = RV.value_items(rec)
print(sorted(vi))
assert ("amount", "rs 386") in vi and ("restaurant", RV.norm(A["restaurant"])) in vi
cab = next(x for x in sc if sc[x]["agent_type"] == "cab_ride_support")
r2 = RV._record(cab, sc[cab], det[cab], {"facts": [{"key": "flight_status", "value": "On time"}, {"key": "seat_preference", "value": "window"}],
                                          "distractors": [], "caller_values": [], "caller_situation": "",
                                          "information": f"Ride X, {det[cab]['assigned']['plate']}, flight IU 402, seat 14F window, Gate 22B, boarding 7:25 PM, Rs 1,349, card ending 4412"}, 1, [])
vi2 = RV.value_items(r2)
print(sorted(vi2))
assert ("flight", "iu 402") in vi2 and ("seat", "14f") in vi2 and not any(v == "on time" or v == "window" for _, v in vi2)
assert not any(t == "flight" and v.split()[0] in ("dl", "hr", "up", "mh", "ka") for t, v in vi2), "plate read as flight"
allr = {k: dict(rec, scenario_id=k) for k in "abcde"}
v, _ = RV.reuse_violations(allr, list(allr))
assert set(v) == {"d", "e"}, v
st = RV.diversity_stats(allr)
assert st["max_reuse_per_value_type"]["restaurant"]["max"] == 5
print("uniqueness pass OK:", {k: len(x) for k, x in v.items()})
print("ALL TESTS PASSED")

# existing-record registration: dropping one scenario's det and re-deriving must keep all others identical
recs = {sid: {**d, "agent_type": sc[sid]["agent_type"]} for sid, d in det.items()}
drop = list(sc)[len(sc) // 2]
ex = {k: v for k, v in recs.items() if k != drop}
det2 = RV.det_fields_v2(sc, existing=ex)
assert all(det2[k] == {k2: det[k][k2] for k2 in RV.DET_KEYS if k2 in det[k]} for k in ex), "existing changed"
assert det2[drop]["primary_id"] not in {v["primary_id"] for v in ex.values()} | {v["secondary_id"] for v in ex.values()}
c = Counter(v for d in det2.values() for k, v in d["assigned"].items() if k not in RV.CATEGORICAL_ASSIGNED)
assert c.most_common(1)[0][1] <= 3, c.most_common(3)
print("existing-record registration OK (re-derived", drop, ")")
