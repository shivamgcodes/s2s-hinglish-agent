"""D2 / V4 hold-out over the v2 scenario list -> data/V4/holdout.json (stdlib only, deterministic).

  python3 holdout_v2.py [--scenarios guidance/scenario_creation_v2.json] [--out data/V4/holdout.json] [--force]

TEST (30 scenarios). Same selection rule as audio/holdout.py, applied to the v2 list:
  candidates = ids ending _03/_07/_11 + the last two scenarios (file order) of each agent_type.
  Guard: a candidate that was a TRAINING scenario of the old split (the 72 v1 scenarios minus data/holdout.json
  test_scenarios = the 47 V3_A trained on) is never put in test (D2 §8 tests V3_A on this set); it is dropped and listed.
  Trim to 30 if there are more: drop one candidate per agent_type, types in order of h01("hv2_trim_type", type), the
  dropped one being the type's candidate with the largest h01("hv2_trim", id); repeat the type cycle if still > 30.
  Extend to 30 if there are fewer: add non-candidate, non-old-train scenarios round-robin over types (same type order),
  within a type by smallest h01("hv2_extend", id).
VAL (15 scenarios, disjoint from test): round-robin over agent types in order of h01("hv2_val_type", type), within a
  type the remaining (non-test) scenario with the smallest h01("hv2_val", id), until 15.
TRAIN = all other scenarios.
test_calls = one call per test scenario, gN rotated g1,g2,g3,g4,g1,... over the sorted test ids (as before).
val_calls = all g1-g4 calls of the val scenarios (never a test call: val and test scenarios are disjoint).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

H = Path(__import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import h01  # noqa: E402

N_TEST, N_VAL = 30, 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(H / "guidance/scenario_creation_v2.json"))
    ap.add_argument("--old-scenarios", default=str(H / "guidance/scenario_creation.json"))
    ap.add_argument("--old-holdout", default=str(H / "data/holdout.json"))
    ap.add_argument("--out", default=str(H / "data/V4/holdout.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different --out")
    a = ap.parse_args()
    if Path(a.scenarios).resolve() == Path(a.old_scenarios).resolve() and a.out == str(H / "data/V4/holdout.json"):
        raise SystemExit("refusing to write data/V4/holdout.json from the v1 scenario file (pass --out for a dry run)")

    d = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))
    by_type = {ag["agent_type"]: [s["id"] for s in ag["scenarios"]] for ag in d["agents"]}
    all_ids = [i for ids in by_type.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)), "duplicate scenario ids"
    type_of = {i: t for t, ids in by_type.items() for i in ids}

    old = json.loads(Path(a.old_scenarios).read_text(encoding="utf-8"))
    old_ids = {s["id"] for ag in old["agents"] for s in ag["scenarios"]}
    old_test = set(json.loads(Path(a.old_holdout).read_text())["test_scenarios"])
    old_train = old_ids - old_test

    cand = set()
    for t, ids in by_type.items():
        cand |= {i for i in ids if i.endswith(("_03", "_07", "_11"))}
        cand |= set(ids[-2:])
    excluded_old_train = sorted(cand & old_train)
    test = set(cand) - old_train

    types_trim = sorted(by_type, key=lambda t: h01("hv2_trim_type", t))
    trimmed, extended = [], []
    k = 0
    while len(test) > N_TEST:
        t = types_trim[k % len(types_trim)]
        k += 1
        mine = [i for i in test if type_of[i] == t]
        if len(mine) <= 1:
            continue
        drop = max(mine, key=lambda i: h01("hv2_trim", i))
        test.discard(drop)
        trimmed.append(drop)
    k = 0
    while len(test) < N_TEST:
        t = types_trim[k % len(types_trim)]
        k += 1
        pool = [i for i in by_type[t] if i not in test and i not in cand and i not in old_train]
        if not pool:
            if k > 10 * len(types_trim):
                raise SystemExit("cannot extend test to 30")
            continue
        add = min(pool, key=lambda i: h01("hv2_extend", i))
        test.add(add)
        extended.append(add)
    test = sorted(test)

    types_val = sorted(by_type, key=lambda t: h01("hv2_val_type", t))
    val, k = [], 0
    while len(val) < N_VAL:
        t = types_val[k % len(types_val)]
        k += 1
        pool = [i for i in by_type[t] if i not in test and i not in val]
        if pool:
            val.append(min(pool, key=lambda i: h01("hv2_val", i)))
        elif k > 10 * len(types_val):
            raise SystemExit("cannot fill val")
    val = sorted(val)
    train = sorted(set(all_ids) - set(test) - set(val))
    test_calls = [f"{s}_g{i % 4 + 1}" for i, s in enumerate(test)]
    val_calls = [f"{s}_{g}" for s in val for g in ("g1", "g2", "g3", "g4")]
    assert not set(val) & set(test) and not set(test) & old_train and not set(val_calls) & set(test_calls)

    doc = {"rule": ("TEST: ids ending _03/_07/_11 + last two per agent_type over the v2 list, minus old-split training "
                    "scenarios, trimmed/extended to 30 by hash (see gen/holdout_v2.py docstring); test call gN rotated "
                    "over sorted ids. VAL: 15 scenarios disjoint from test, round-robin over types by hash. "
                    "TRAIN: the rest."),
           "source": a.scenarios, "source_md5": hashlib.md5(Path(a.scenarios).read_bytes()).hexdigest(),
           "test_scenarios": test, "test_calls": test_calls,
           "val_scenarios": val, "val_calls": val_calls,
           "train_scenarios": train,
           "n_test_scenarios": len(test), "n_val_scenarios": len(val), "n_train_scenarios": len(train),
           "n_train_calls_per_variant": 4 * len(train),
           "selection": {"candidates": sorted(cand), "excluded_old_train": excluded_old_train,
                         "trimmed": trimmed, "extended": extended,
                         "per_type_test": {t: sum(type_of[i] == t for i in test) for t in by_type},
                         "per_type_val": {t: sum(type_of[i] == t for i in val) for t in by_type},
                         "old_test_kept_in_test": sorted(set(test) & old_test),
                         "old_test_in_train": sorted(set(train) & old_test)}}
    out = Path(a.out)
    txt = json.dumps(doc, indent=1)
    if out.exists() and out.read_text() != txt and not a.force:
        raise SystemExit(f"{out} exists and differs; pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(f"{len(test)} test / {len(val)} val / {len(train)} train scenarios -> {out}")
    print(json.dumps(doc["selection"]))


if __name__ == "__main__":
    main()
