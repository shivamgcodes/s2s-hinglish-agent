"""D2 section 7 re-rank summary: reads tests/out/rr4_<run>_s<step>/V4/*_s1001.nat.json (judge.py --naturalness),
val/control losses from /workspace/runs/<run>/eval_metrics.csv; prints the tests/V4_RERANK.md body and, with
--write-final, writes /workspace/runs/<run>/FINAL_CKPT (pick = highest NATURAL rate; tie -> lower val total_pooled).
  python3 v4_rerank.py [--write-final] > V4_RERANK.md
"""
import csv
import json
import sys
from pathlib import Path

H = __import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish")
RUNS_ROOT = __import__("os").environ.get("RUNS_ROOT", "/workspace/runs")
T = Path(H) / "tests"
RUNS = {"V4_A": [400, 350, 300], "V4_A2": [600, 500, 450]}
SUB = ("air_11_g2,air_26_g3,bank_07_g1,bank_20_g4,cab_03_g1,cab_26_g4,ecom_11_g3,ecom_23_g1,food_03_g2,food_07_g3,"
       "sub_11_g4,tel_03_g2").split(",")
LABS = ["NATURAL", "STIFF", "PEPPERED", "NONSENSE"]
BANDS = ["standard", "long", "mixed"]
calls = {c["call_id"]: c for c in (json.loads(l) for l in open(f"{H}/data/V4/calls.jsonl"))}


def evals(run):
    out = {}
    for r in csv.DictReader(open(f"{RUNS_ROOT}/{run}/eval_metrics.csv")):
        out.setdefault(int(r["step"]), {})[r["split"]] = float(r["total_pooled"])
    return out


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


rows, per_call, missing = {}, {}, []
for run, steps in RUNS.items():
    ev = evals(run)
    for st in steps:
        tag = f"rr4_{run}_s{st}"
        agg = {k: 0 for k in LABS + ["n", "trunc", "hc", "calls"]}
        band = {}
        for cid in SUB:
            f = T / "out" / tag / "V4" / f"{cid}_s1001.nat.json"
            if not f.exists():
                missing.append(str(f))
                continue
            d = json.loads(f.read_text())
            if "error" in d:
                missing.append(str(f) + " (error)")
                continue
            agg["calls"] += 1
            bb = band.setdefault(calls[cid]["length_band"], {"n": 0, "NATURAL": 0, "NONSENSE": 0})
            agg["n"] += d["n"]
            agg["trunc"] += d.get("trunc", 0)
            agg["hc"] += d.get("hindi_content", 0)
            bb["n"] += d["n"]
            for k in LABS:
                agg[k] += d.get("counts", {}).get(k, 0)
            bb["NATURAL"] += d.get("counts", {}).get("NATURAL", 0)
            bb["NONSENSE"] += d.get("counts", {}).get("NONSENSE", 0)
            per_call.setdefault(cid, {})[tag] = (d.get("counts", {}).get("NATURAL", 0), d["n"])
        agg["band"] = band
        agg["val"] = ev[st]["val"]
        agg["control"] = ev[st].get("control", float("nan"))
        agg["rate"] = agg["NATURAL"] / agg["n"] if agg["n"] else -1.0
        rows[(run, st)] = agg

picks = {run: sorted(steps, key=lambda s: (-rows[(run, s)]["rate"], rows[(run, s)]["val"]))[0]
         for run, steps in RUNS.items()}

L = ["# V4 re-rank (D2 spec section 7)", "",
     "Top-3 checkpoints by val total_pooled per run, re-ranked by the Gemma 4 31B naturalness judge "
     "(tests/judge.py --naturalness, v3_hinglish_review rubric: NATURAL / STIFF / PEPPERED / NONSENSE per utterance). "
     "Final pick per run = highest NATURAL rate; tie -> lower val loss.", "",
     "- Test subset (12 of the 30 V4 test calls, seed 1001, no gate0): " + ", ".join(SUB),
     "- Balance: 7 agent types (air, bank, cab, ecom, food 2 each; sub, tel 1 each); g1-g4 3 each (6 female / 6 male "
     "agents); length band 6 long / 5 standard / 1 mixed (air_26_g3 is the only mixed test call); writes per call "
     "0:2, 1:4, 2:5, 3:1.",
     "- Segmentation: tcommon.segments() (new utterance after >= 15 PAD/EPAD frames or after . ? !), deterministic; "
     "the same segmenter score.py uses.",
     "- Judge: Gemma 4 31B, temperature 0, one prompt per call (agent and customer gender, brand), JSON schema with "
     "one label per utterance; outputs tests/out/rr4_<run>_s<step>/V4/<call>_s1001.nat.json.", ""]
if missing:
    L += ["**Missing / errored judge files:** " + ", ".join(missing), ""]
L += ["## Results", "",
      "| Run | Step | Val total_pooled | Control total_pooled | Calls | Utterances | NATURAL | STIFF | PEPPERED "
      "| NONSENSE (trunc) | Hindi carries content | Pick |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (run, st), a in rows.items():
    L.append(f"| {run} | {st} | {a['val']:.5f} | {a['control']:.5f} | {a['calls']} | {a['n']} | "
             f"{pct(a['NATURAL'], a['n'])} | {a['STIFF']} | {a['PEPPERED']} | {a['NONSENSE']} ({a['trunc']}) | "
             f"{pct(a['hc'], a['n'])} | {'**FINAL**' if picks[run] == st else ''} |")
L += ["", "## NATURAL / NONSENSE by length band", "",
      "| Run | Step | " + " | ".join(f"{b} NATURAL | {b} NONSENSE" for b in BANDS) + " |",
      "|---|---|" + "---|---|" * len(BANDS)]
for (run, st), a in rows.items():
    cells = []
    for b in BANDS:
        bb = a["band"].get(b, {"n": 0, "NATURAL": 0, "NONSENSE": 0})
        cells += [pct(bb["NATURAL"], bb["n"]), pct(bb["NONSENSE"], bb["n"])]
    L.append(f"| {run} | {st} | " + " | ".join(cells) + " |")
tags = [f"rr4_{r}_s{s}" for r, ss in RUNS.items() for s in ss]
L += ["", "## NATURAL per call (natural / utterances)", "", "| Call | band | " + " | ".join(tags) + " |",
      "|---|---|" + "---|" * len(tags)]
for cid in SUB:
    L.append(f"| {cid} | {calls[cid]['length_band']} | " + " | ".join(
        "{}/{}".format(*per_call.get(cid, {}).get(t, ("-", "-"))) for t in tags) + " |")
print("\n".join(L))
if "--write-final" in sys.argv:
    if missing:
        sys.exit("refusing --write-final: missing/errored judge files")
    for run, st in picks.items():
        a = rows[(run, st)]
        Path(f"{RUNS_ROOT}/{run}/FINAL_CKPT").write_text(
            f"step={st}\npath={RUNS_ROOT}/{run}/checkpoints/checkpoint_{st:06d}/consolidated\n"
            f"criterion=highest Gemma naturalness NATURAL rate over the top-3 val checkpoints {RUNS[run]} "
            f"(tie -> lower val total_pooled); D2 spec section 7; tests/V4_RERANK.md\n"
            f"natural={a['NATURAL']}/{a['n']} ({100 * a['rate']:.1f}%)\nnonsense={a['NONSENSE']}/{a['n']}\n"
            f"val_total_pooled={a['val']:.5f}\ncontrol_total_pooled={a['control']:.5f}\n"
            f"test_subset=12 V4 test calls, seed 1001: {','.join(SUB)}\n"
            f"judge_outputs={H}/tests/out/rr4_{run}_s{st}/V4/*_s1001.nat.json\n")
        print(f"wrote {RUNS_ROOT}/{run}/FINAL_CKPT step={st}", file=sys.stderr)
