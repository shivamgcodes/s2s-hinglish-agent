"""CPU tests for the D2 repair additions in records_v2 (scenario overrides, categorical/year checks, judge date facts,
repair det redraw). Run with venv-vllm: python test_records_v2_repair.py"""
import json
from collections import Counter

import records_v2 as RV

sc = RV.load_scenarios_v2(RV.SCEN_V2)
recs = json.load(open(f"{RV.V4_DIR}/run1_records.json" if __import__("os").path.exists(f"{RV.V4_DIR}/run1_records.json")
                      else f"{RV.V4_DIR}/records.json"))
# 1. first-run det fields unchanged when overrides=False (V4 run-1 reproducible)
d0 = RV.det_fields_v2(sc)
for sid, r in recs.items():
    assert all(d0[sid][k] == r[k] for k in RV.DET_KEYS if k in r), sid
print("det overrides=False reproduces run-1 det fields for", len(recs))
# 2. overrides
o = {s: RV.scenario_overrides(sc[s]) for s in sc}
assert o["bank_07"].get("city") == "Mumbai", o["bank_07"]
assert o["cab_03"].get("pickup_kind") == "airport" and o["cab_04"].get("pickup_kind") == "airport"
assert o["cab_01"].get("drop_kind") == "airport"
assert o["cab_26"].get("pickup_kind") == "hospital"
assert o["food_23"].get("office_address") and o["sub_21"].get("new_signup")
assert o["sub_10"].get("cycle_not") == "monthly", o["sub_10"]
assert "2026" in RV.fmt_today(recs["sub_14"]) and "[2027-02-28]" in RV.date_facts_text(recs["sub_14"])
p = RV.judge_prompt_repair("sub_14", sc["sub_14"], recs["sub_14"], recs["sub_14"])
assert "current year is 2026" in p
p0 = RV.judge_prompt("sub_14", sc["sub_14"], recs["sub_14"], recs["sub_14"])
assert "FIXED dates (computed" not in p0  # run-1 judge prompt unchanged
print("override flags:", Counter(k for v in o.values() for k in v))
# 3. redraw with overrides: keep everything except the redraw set; caps hold
redo, keep_det, reasons = RV.plan_repair(sc, recs)
redraw = [s for s in redo if s not in keep_det]
existing = {k: v for k, v in recs.items() if k not in redraw}
d1 = RV.det_fields_v2(sc, existing=existing, overrides=True)
for sid in existing:
    assert all(d1[sid][k] == recs[sid][k] for k in RV.DET_KEYS if k in recs[sid]), sid
for sid in redraw:
    f = d1[sid]
    tmp = {**recs[sid], **f}
    assert not RV.override_conflicts(sc[sid], tmp), (sid, RV.override_conflicts(sc[sid], tmp), f["assigned"])
    print(" redraw", sid, f["city"], f["dates"], f["assigned"])
ids = [d["primary_id"] for d in d1.values()] + [d["secondary_id"] for d in d1.values()]
assert len(ids) == len(set(ids))
ph = [d[k] for d in d1.values() for k in ("phone", "new_phone", "other_phone")]
assert len(ph) == len(set(ph))
c = Counter(v for d in d1.values() for k, v in d["assigned"].items() if k not in RV.CATEGORICAL_ASSIGNED)
assert c.most_common(1)[0][1] <= 3, c.most_common(3)
assert d1 == RV.det_fields_v2(sc, existing=existing, overrides=True), "not deterministic"
# 4. categorical + year checks
r = recs["sub_10"]
assert any("monthly" in e for e in RV.categorical_errors(r, r["facts"], r["information"]))
f = dict(recs["sub_07"])
rr = {"information": f["information"], "facts": [{"key": k, "value": v} for k, v in f["facts"].items()],
      "caller_values": [{"key": k, "value": v} for k, v in f["caller_values"].items()], "distractors": [], "caller_situation": ""}
errs, _ = RV.check_record_v2("sub_07", sc["sub_07"], f, rr)
assert any("wrong year" in e for e in errs), errs
assert RV.fut_year_ok("21 January", {"21 January 2027"}) and not RV.fut_year_ok("21 January", {"21 January"})
# 5. judge date text: tel_02 states 84 days
t = RV.date_facts_text(recs["tel_02"])
assert "+ 84 days" in t, t
print(t)
print(RV.date_facts_text(recs["sub_14"]))
cp = RV.compress_prompt("tel_18", sc["tel_18"], recs["tel_18"], ({"information": recs["tel_18"]["information"], "facts": []}, 165))
assert "at most" in cp
# 6. reuse check over kept + redo order: kept never violators
clean = [s for s in sc if s in recs and s not in redo]
viol, _ = RV.reuse_violations({k: recs[k] for k in clean}, clean)
assert not viol, viol
print("ALL REPAIR TESTS PASSED; redo", len(redo), "redraw", len(redraw))
