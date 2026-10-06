"""(V3/CONTROL copy of make_stats.py: id count from generate.default_ids, F5 timing/tail stats, CONTROL eval-only note)
data/<V>/STATS.md from calls.jsonl, gen_log.jsonl, gen_rounds.json, manifests, qc/qc_stats.jsonl, qc_summary.json.
  python3 /workspace/hinglish/gen/make_stats.py V1"""
import json
import statistics as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from pathlib import Path

V = sys.argv[1]
D = Path(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")) / "data"
R = D / V
calls = [json.loads(l) for l in open(R / "calls.jsonl", encoding="utf-8")]
log = [json.loads(l) for l in open(R / "gen_log.jsonl", encoding="utf-8")]
rounds = json.load(open(R / "gen_rounds.json"))
dropped = json.load(open(D / "dropped_call_ids.json"))
ho = json.load(open(D / "holdout.json"))
qcs = [json.loads(l) for l in open(R / "qc/qc_stats.jsonl")]
qsum = json.load(open(R / "qc/qc_summary.json"))
import generate
from common import load_scenarios
# requested ids = round-0 submission (default_ids re-reads dropped_call_ids.json, which shrinks after V3 drops are appended)
NID = rounds[0]["submitted"] if rounds else len(generate.default_ids(V, load_scenarios()))
missing = json.load(open(R / "gen_missing.json")) if (R / "gen_missing.json").exists() else []
# per-chunk TTS facts from work/ (best-effort: plan.json chunk count, timing.jsonl tail re-renders, asr try results)
W = R / "work"
tim = [json.loads(l) for l in open(W / "timing.jsonl")] if (W / "timing.jsonl").exists() else []


def man(p):
    return [json.loads(l) for l in open(R / p)]


train, test = man("train/train.jsonl"), man("test/test.jsonl")
held = man("heldout/heldout_all.jsonl")

# attempts: global attempt index = 3*round + attempt of the passing sample
pass_at = {}
for r in rounds:
    for row in log[r["log_lines"][0]:r["log_lines"][1]]:
        if row["pass"] and row["call_id"] not in pass_at:
            pass_at[row["call_id"]] = (r["round"], 3 * r["round"] + row["attempt"])
hist = Counter(v[1] for v in pass_at.values())
cands = Counter(row["call_id"] for row in log)
fails = Counter()
for row in log:
    for f in row.get("failures") or []:
        fails[str(f).split(":")[0]] += 1
first = rounds[0]["summary"]["first_attempt_fail_by_check"]

ag = [c["hindi_token_share"]["agent"] for c in calls]
cu = [c["hindi_token_share"]["customer"] for c in calls]
ms = lambda x: f"{st.mean(x):.3f} ± {st.pstdev(x):.3f} (min {min(x):.3f}, max {max(x):.3f})"
mins = lambda m: sum(x["duration"] for x in m) / 60
over = [q["call_id"] for q in qcs if q["over_100s"]]
durs = [q["duration"] for q in qcs]
test_sc = set(ho["test_scenarios"])
leak = [x["path"] for x in train if Path(x["path"]).stem.rsplit("_", 1)[0] in test_sc]

o = [f"# STATS — {V}\n",
     f"## Generation",
     f"- call_ids requested: {NID} (generate.default_ids({V!r})); generated (in calls.jsonl): {len(calls)}; not generated: {len(missing)} {missing}; dropped_call_ids.json now: {len(dropped['ids'])} {dropped['ids']}",
     f"- rounds (one vLLM load, gen/gen_loop.py): " + "; ".join(
         f"r{r['round']} seed {r['seed_base']}: {r['submitted']} submitted -> total {r['passed']} ({r['wall_min']} min)" for r in rounds),
     f"- candidates drawn: {len(log)} total; mean {len(log) / NID:.2f} per call_id",
     f"- attempts histogram (global attempt of the accepted candidate = 3*round + attempt; 1-3 = round 0):",
     "  " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)),
     f"- failure reasons, all {len(log)} candidates (a candidate can fail several checks): "
     + ", ".join(f"{k} {v}" for k, v in fails.most_common()),
     f"- failure reasons, round 0 attempt 1 (per call): " + ", ".join(f"{k} {v}" for k, v in sorted(first.items(), key=lambda x: -x[1])),
     f"- dropped: " + "; ".join(f"{d['call_id']}: {d['reason']}" for d in dropped["dropped"] if d.get("dropped_by") == V),
     "",
     "## Audio",
     f"- TTS backend: {'f5cs (IndicF5 code-switch, Tharshan/indicf5_hindi-english_code_switch)' if V == 'V3' else 'kokoro_en (Kokoro English)' if V == 'CONTROL' else 'see plan.json'}",
     *([f"- CONTROL is EVAL-ONLY (forgetting measure). Eval manifest = {R}/heldout/heldout_all.jsonl (all {len(held)} calls). {R}/train/train.jsonl must never reach the trainer."] if V == "CONTROL" else []),
     f"- stereo calls: {len(qcs)}; total {sum(durs) / 60:.1f} min ({sum(durs) / 3600:.2f} h); per call {min(durs):.1f}-{max(durs):.1f} s, mean {st.mean(durs):.1f} s",
     f"- train manifest ({len(train)} calls, non-held-out scenarios): {mins(train):.1f} min -> {R}/train/train.jsonl",
     f"- test manifest ({len(test)} calls, holdout.json test_calls): {mins(test):.1f} min -> {R}/test/test.jsonl",
     f"- all held-out-scenario calls: {len(held)} ({mins(held):.1f} min) -> {R}/heldout/heldout_all.jsonl",
     f"- train paths in a held-out scenario: {len(leak)}",
     f"- agent speech {qsum.get('agent_hours')} h of {qsum.get('hours')} h",
     "",
     "## Hindi share (frozen counter, per call pooled per speaker)",
     f"- agent: {ms(ag)}",
     f"- customer: {ms(cu)}",
     "",
     "## TTS / alignment QC",
     f"- TTS chunks: {qsum['chunks']}; CER-flagged (best of 3 kept): {qsum['cer_flagged_chunks']}; mean chunk CER {qsum['mean_cer']}",
     f"- TTS stages (chunks rendered per try; try1/2 = re-renders of CER-rejected chunks): " + "; ".join(f"{t['stage']} {t['chunks']}" for t in tim if t["stage"].startswith("tts")),
     f"- F5 tail guard: re-rendered {sum(t.get('tail_rerender', 0) for t in tim)}, fixed {sum(t.get('tail_fixed', 0) for t in tim)}; s/chunk " + ", ".join(str(t.get('s_per_chunk')) for t in tim if t["stage"].startswith("tts")),
     f"- alignment fallback utterances: {qsum['align_fallback_utts']}; low-confidence align utterances: {qsum['align_low_utts']}",
     f"- interruptions realised: {qsum['interruptions']} (mode {qsum.get('interrupt_mode')})",
     f"- calls > 100 s: {len(over)} {over}",
     f"- failed/missing inputs at assembly: {qsum.get('failed_missing_inputs')}",
     ]
(Path(sys.argv[2]) if len(sys.argv) > 2 else R / "STATS.md").write_text("\n".join(o) + "\n", encoding="utf-8")
print("\n".join(o))
