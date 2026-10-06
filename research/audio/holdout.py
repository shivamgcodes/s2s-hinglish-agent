"""Write data/holdout.json (same hold-out for V1/V2/V3). Stdlib only.

TEST scenarios = ids ending _03/_07/_11 + last two scenarios (file order) of each agent_type -> 25.
Test calls = one per held-out scenario, gN rotated g1,g2,g3,g4,g1,... over sorted scenario ids.
Train = all calls (g1-g4) of non-held-out scenarios.
"""
import json
import sys
from pathlib import Path

SC = Path(__import__("os").environ.get("HINGLISH_GUIDANCE") or Path(__file__).resolve().parents[1] / "data_gen" / "guidance") / "scenario_creation.json"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/data/holdout.json")

d = json.loads(SC.read_text(encoding="utf-8"))
test, by_type, all_ids = set(), {}, []
for ag in d["agents"]:
    ids = [s["id"] for s in ag["scenarios"]]
    by_type[ag["agent_type"]] = ids
    all_ids += ids
    test |= {i for i in ids if i.endswith(("_03", "_07", "_11"))}
    test |= set(ids[-2:])
test = sorted(test)
test_calls = [f"{s}_g{i % 4 + 1}" for i, s in enumerate(test)]
train_sc = sorted(set(all_ids) - set(test))
doc = {"rule": "ids ending _03/_07/_11 + last two per agent_type; test call gN rotated over sorted ids",
       "test_scenarios": test, "test_calls": test_calls, "train_scenarios": train_sc,
       "n_test_scenarios": len(test), "n_train_scenarios": len(train_sc),
       "n_train_calls_per_variant": 4 * len(train_sc)}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, indent=1))
print(f"{len(test)} test scenarios, {len(train_sc)} train scenarios -> {OUT}")
print(test_calls)
