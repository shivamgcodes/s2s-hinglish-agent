"""Build common/data/scripts_v4.json: the expected conversation ("script") for every demo record x pairing, shown in
the client's "Script: what to say" panel (DECISIONS D-SCRIPT-PANEL, 2026-10-06).

  python3 ops/build_scripts.py --calls <V4 calls.jsonl> --holdout <V4 holdout.json> [--out common/data/scripts_v4.json]

Sources (pod1, read only): /workspace/hinglish/data/V4/calls.jsonl (md5 7f540269...) and holdout.json (md5 0a4d5483...).
A demo record is a record of common/data/records_v4.json whose agent_type is one of session.SUPPORTED_AGENT_TYPES
(record_id == scenario_id); its call for pairing gN is call_id <record_id>_gN. A few calls were dropped at generation
time; those pairings are absent from the output and the Space answers "no script for this pairing".

Per call only what the panel shows is kept: speaker, text_roman, tags per turn, the expected tool writes, and the
holdout split:
  test           the call is in holdout.test_calls (the scored test set; never trained on)
  test_scenario  another pairing of a test scenario (never trained on, not in the scored set)
  val            a validation scenario (never trained on)
  train          a training scenario: the model was fine-tuned on this exact dialogue
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
os.environ.setdefault("S2S_RECORDS", str(ROOT / "common" / "data" / "records_v4.json"))
import session as sessmod  # noqa: E402


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", default=str(ROOT / "common" / "data" / "scripts_v4.json"))
    a = ap.parse_args()
    h = json.loads(Path(a.holdout).read_text())
    test_calls, val_calls = set(h["test_calls"]), set(h["val_calls"])
    test_scn, val_scn, train_scn = set(h["test_scenarios"]), set(h["val_scenarios"]), set(h["train_scenarios"])
    demo = sessmod.demo_records()
    scripts, missing = {}, []
    calls = {}
    for line in open(a.calls, encoding="utf-8"):
        c = json.loads(line)
        calls[c["call_id"]] = c
    for rid in sorted(demo):
        for g in sessmod.PAIRINGS:
            cid = f"{rid}_{g}"
            c = calls.get(cid)
            if c is None:
                missing.append(cid)
                continue
            # the script must belong to the same role prompt the worker builds for this record + pairing
            if sessmod.ascii_prompt(c["role_prompt"]) != sessmod.session_config(rid, g)["role_prompt"]:
                raise SystemExit(f"{cid}: role prompt differs from session.session_config({rid}, {g})")
            if cid in test_calls:
                split = "test"
            elif rid in test_scn:
                split = "test_scenario"
            elif cid in val_calls or rid in val_scn:
                split = "val"
            elif rid in train_scn:
                split = "train"
            else:
                raise SystemExit(f"{cid}: not in any holdout split")
            scripts[cid] = {
                "record_id": rid, "pairing": g, "split": split,
                "agent_name": c["agent_name"], "agent_gender": c["agent_gender"],
                "customer_gender": c["customer_gender"],
                "turns": [{"speaker": t["speaker"], "text": t["text_roman"], "tags": list(t.get("tags") or [])}
                          for t in c["turns"]],
                "writes": [{"tool": w["tool"], "args": w["args"]} for w in c.get("writes") or []],
            }
    out = {
        "comment": "Expected conversations for the demo records (V4 synthetic calls, text_roman). Built by "
                   "ops/build_scripts.py; served by the Space at GET /api/script/{record_id}?pairing=gN.",
        "source": {"calls_md5": md5(a.calls), "holdout_md5": md5(a.holdout)},
        "splits": {
            "test": "held-out test call (never trained on)",
            "test_scenario": "held-out test scenario, another pairing (never trained on)",
            "val": "validation scenario (never trained on)",
            "train": "training scenario (the model was fine-tuned on this exact dialogue)",
        },
        "missing": missing,
        "scripts": scripts,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"{a.out}: {len(scripts)} scripts, {Path(a.out).stat().st_size / 1024:.0f} KiB, missing {missing}, "
          f"splits {dict(Counter(s['split'] for s in scripts.values()))}")


if __name__ == "__main__":
    main()
