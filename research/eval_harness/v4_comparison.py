#!/usr/bin/env python3
"""D2 spec section 9: write tests/V4_COMPARISON.md from existing outputs only (CPU, no models).

Inputs (read-only): tests/out/{base_V4,V3_A200_V4,V4_A,V4_A2}/v4_scores.json (tests/v4_eval.py),
/workspace/runs/{V4_A,V4_A2}/{eval_metrics.csv,BEST_CKPT,FINAL_CKPT}, data/V4/STATS.md (quoted by hand in the text),
tests/V4_RERANK.md (numbers quoted). Output: tests/V4_COMPARISON.md.
"""
import csv, json, os, datetime

H = __import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish")
RUNS = __import__("os").environ.get("RUNS_ROOT", "/workspace/runs")
TAGS = [("base_V4", "base"), ("V3_A200_V4", "V3_A@200"), ("V4_A", "V4_A@350"), ("V4_A2", "V4_A2@600")]
BANDS = ["all", "standard", "long", "mixed"]

S = {t: json.load(open(f"{H}/tests/out/{t}/v4_scores.json")) for t, _ in TAGS}

# (label, seed-key or None, pooled-key or None, unit of the pooled n)
METRICS = [
    ("**Judge: task score** (COMPLETE 1 / PARTIAL 0.5 / FAILED 0)", "tc_score", None, "calls"),
    ("Judge: COMPLETE calls", "tc_complete", None, "calls"),
    ("Judge: FAILED calls", "tc_failed", None, "calls"),
    ("**Judge: facts CONSISTENT**", "fc_consistent", None, "calls"),
    ("Judge: facts CONTRADICTS", "fc_contradicts", None, "calls"),
    ("Judge: wrong facts per call", "fc_n_wrong", None, "calls"),
    ("Judge: write confirmed with the right value", "cj_confirm_value_rate", "cj_confirm_value_pooled", "writes"),
    ("**Echo** (validate.echo_ok on the confirm slot)", "echo_rate", "echo_pooled", "writes"),
    ("Check-line before write (score.py m3)", "m3_check_rate", "m3_check_pooled", "writes"),
    ("**NATURAL lines**", "nat_natural", "nat_natural_pooled", "lines"),
    ("- NATURAL and Hinglish (en=false)", None, "nat_natural_hi_pooled", "lines"),
    ("- NATURAL and pure English (en=true)", None, "nat_natural_en_pooled", "lines"),
    ("STIFF lines", "nat_stiff", "nat_stiff_pooled", "lines"),
    ("PEPPERED lines", "nat_peppered", "nat_peppered_pooled", "lines"),
    ("NONSENSE lines", "nat_nonsense", "nat_nonsense_pooled", "lines"),
    ("- NONSENSE that is a cut-off fragment (trunc)", None, "nat_trunc_pooled", "lines"),
    ("Lines where Hindi carries content", None, "nat_hindi_content_pooled", "lines"),
    ("**Trelis CER m1** (roman vs roman, trimmed)", "cer_m1", "cer_m1_pooled", "ref chars"),
    ("Trelis WER m1", "wer_m1", "wer_m1_pooled", "ref words"),
    ("Model Hindi share (score.py m1)", "m1_model_hindi_share", None, "calls"),
    ("Read rate (m4)", "m4_read_rate", None, "calls"),
    ("Value rate (m4)", "m4_value_rate", None, "calls"),
    ("Invented-fact flag (m4)", "m4_invented", None, "calls"),
    ("Greeting (m5)", "m5_greeting", None, "calls"),
    ("Degenerate (m5)", "m5_degenerate", None, "calls"),
    ("Judge r2 (Hinglish register)", "m2_judge_r2", None, "calls"),
    ("Words per call", "n_words", None, "calls"),
]


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def cell(b, seed_key, pooled_key):
    parts = []
    if seed_key and seed_key in b and isinstance(b[seed_key], dict) and b[seed_key].get("mean") is not None:
        v = b[seed_key]
        nd = 1 if seed_key == "n_words" else 3
        s = fmt(v["mean"], nd)
        if v.get("n_seeds", 1) > 1:
            s += f" ± {fmt(v['std'], nd)}"
        parts.append(s)
    if pooled_key and pooled_key in b and isinstance(b[pooled_key], dict) and b[pooled_key].get("den"):
        p = b[pooled_key]
        if parts:
            parts.append(f"(pooled {fmt(p['rate'])}, n={p['den']})")
        else:
            parts.append(f"{fmt(p['rate'])} (n={p['den']})")
    return " ".join(parts) if parts else "-"


def table(section_key, band, tags):
    hdr = ["metric"]
    for t, name in tags:
        b = S[t][section_key].get(band, {})
        k = len(b.get("seeds", []))
        nc = b.get('n_calls', 0)
        hdr.append(f"{name} ({nc} call{'s' if nc != 1 else ''} × {k} seed{'s' if k != 1 else ''})")
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for label, sk, pk, unit in METRICS:
        row = [f"{label} [n: {unit}]"]
        for t, _ in tags:
            row.append(cell(S[t][section_key].get(band, {}), sk, pk))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def band_delta(section_key, tags):
    keys = [("task score", "tc_score", None), ("facts CONSISTENT", "fc_consistent", None),
            ("judge confirm+value", None, "cj_confirm_value_pooled"), ("echo", None, "echo_pooled"),
            ("NATURAL", None, "nat_natural_pooled"), ("NATURAL Hinglish", None, "nat_natural_hi_pooled"),
            ("NONSENSE", None, "nat_nonsense_pooled"), ("Trelis CER m1", None, "cer_m1_pooled")]
    hdr = ["model"] + [f"{k} standard / long" for k, _, _ in keys]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for t, name in tags:
        row = [name]
        for _, sk, pk in keys:
            vals = []
            for band in ("standard", "long"):
                b = S[t][section_key][band]
                vals.append(fmt(b[sk]["mean"], 2) if sk else fmt(b[pk]["rate"], 2))
            row.append(" / ".join(vals))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def curve(run):
    rows = list(csv.DictReader(open(f"{RUNS}/{run}/eval_metrics.csv")))
    val = {int(r["step"]): r for r in rows if r["split"] == "val"}
    ctl = {int(r["step"]): r for r in rows if r["split"] == "control"}
    best = min(val, key=lambda s: float(val[s]["total_pooled"]))
    cmin = min(ctl, key=lambda s: float(ctl[s]["total_pooled"]))
    final = int(open(f"{RUNS}/{run}/FINAL_CKPT").read().split("step=")[1].split()[0])
    end = max(val)
    steps = sorted({0, 50, 100, cmin, best, final, end} | ({400} if run == "V4_A" else set()))
    out = ["| step | val total_pooled | val text | val cb1 | val cb2-8 | control total_pooled | control text | control cb2-8 | note |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in steps:
        v, c = val.get(s), ctl.get(s)
        note = []
        if s == best: note.append("val best (BEST_CKPT)")
        if s == final: note.append("FINAL_CKPT (re-rank pick)")
        if s == cmin: note.append("control min")
        if s == end: note.append("end of run")
        f = lambda r, k: fmt(float(r[k]), 4) if r else "-"
        out.append(f"| {s} | {f(v,'total_pooled')} | {f(v,'text_pooled')} | {f(v,'cb1_pooled')} | {f(v,'cb2_8_pooled')} | "
                   f"{f(c,'total_pooled')} | {f(c,'text_pooled')} | {f(c,'cb2_8_pooled')} | {', '.join(note)} |")
    return "\n".join(out), best, final, end


four = TAGS
three = TAGS[:3]
for t, _ in TAGS:
    assert S[t]["bands_s1001"]["all"]["n_calls"] == 30, t
assert S["V4_A2"]["complete_seeds"] == [1001], S["V4_A2"]["complete_seeds"]
cA, bA, fA, eA = curve("V4_A")
cA2, bA2, fA2, eA2 = curve("V4_A2")
h = lambda t: S[t]["bands_s1001"]["all"]
excl = S["V4_A2"]["incomplete_seed_runs_excluded"]

md = f"""# V4 comparison: base / V3_A / V4_A / V4_A2 (D2 spec section 9)

Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by tests/v4_comparison.py from tests/out/<tag>/v4_scores.json (tests/v4_eval.py) and /workspace/runs/<run>/eval_metrics.csv. Facts only; decisions and definitions are in NOTES.md ('D2 / V4 ...' entries) and tests/V4_EVAL.md, tests/V4_RERANK.md, data/V4/STATS.md.

Models (adapter = checkpoints/checkpoint_000NNN/consolidated, merged at load, 253 LoRA pairs, scaling 2.0):
- **base**: PersonaPlex, no adapter (tag base_V4).
- **V3_A@200**: old V3 adapter /workspace/runs/V3_A step 200, run on the NEW V4 test inputs (tag V3_A200_V4).
- **V4_A@350**: lr 1.5e-5, 1200 steps; FINAL_CKPT step {fA} (re-rank pick; val-best is step {bA}).
- **V4_A2@600**: lr 7.5e-6, 1500 steps; FINAL_CKPT step {fA2} (= its val-best).

Test set: the 30 V4 holdout test calls (one per test scenario; 13 standard / 16 long / 1 mixed), PersonaPlex seeds 1001-1003. **V4_A2 has seed 1001 only** (user, 2026-10-05 ~10:30 IST: single seeds); its {sum(excl.values())} leftover seed-1002/1003 runs ({excl}) are excluded. So the like-for-like 4-way comparison is **section B on seed 1001**; section C adds 3 seeds for the three tags that have them.

How to read the cells: `mean ± std` = per-seed mean over the band's calls, then mean and population std over seeds (std shown only when there are 3 seeds; one seed has no std). `(pooled r, n=N)` = rate pooled over every run of the band with its denominator N in the unit given in the row ([n: calls / writes / lines / ref chars / ref words]); for one seed and per-call metrics n = the calls in the column header. Rows with only a pooled value have no per-seed statistic in v4_scores.json.

## A. Headline (seed 1001, all 30 calls)

| | base | V3_A@200 | V4_A@350 | V4_A2@600 |
|---|---|---|---|---|
| task score | {fmt(h('base_V4')['tc_score']['mean'],2)} | {fmt(h('V3_A200_V4')['tc_score']['mean'],2)} | {fmt(h('V4_A')['tc_score']['mean'],2)} | {fmt(h('V4_A2')['tc_score']['mean'],2)} |
| COMPLETE calls | {h('base_V4')['tc_counts'].get('COMPLETE',0)}/30 | {h('V3_A200_V4')['tc_counts'].get('COMPLETE',0)}/30 | {h('V4_A')['tc_counts'].get('COMPLETE',0)}/30 | {h('V4_A2')['tc_counts'].get('COMPLETE',0)}/30 |
| facts CONSISTENT calls | {h('base_V4')['fc_counts'].get('CONSISTENT',0)}/30 | {h('V3_A200_V4')['fc_counts'].get('CONSISTENT',0)}/30 | {h('V4_A')['fc_counts'].get('CONSISTENT',0)}/30 | {h('V4_A2')['fc_counts'].get('CONSISTENT',0)}/30 |
| echo (writes) | {round(h('base_V4')['echo_pooled']['rate']*h('base_V4')['echo_pooled']['den'])}/{h('base_V4')['echo_pooled']['den']} | {round(h('V3_A200_V4')['echo_pooled']['rate']*h('V3_A200_V4')['echo_pooled']['den'])}/{h('V3_A200_V4')['echo_pooled']['den']} | {round(h('V4_A')['echo_pooled']['rate']*h('V4_A')['echo_pooled']['den'])}/{h('V4_A')['echo_pooled']['den']} | {round(h('V4_A2')['echo_pooled']['rate']*h('V4_A2')['echo_pooled']['den'])}/{h('V4_A2')['echo_pooled']['den']} |
| NATURAL lines | {fmt(h('base_V4')['nat_natural_pooled']['rate'],2)} | {fmt(h('V3_A200_V4')['nat_natural_pooled']['rate'],2)} | {fmt(h('V4_A')['nat_natural_pooled']['rate'],2)} | {fmt(h('V4_A2')['nat_natural_pooled']['rate'],2)} |
| NATURAL and Hinglish | {fmt(h('base_V4')['nat_natural_hi_pooled']['rate'],2)} | {fmt(h('V3_A200_V4')['nat_natural_hi_pooled']['rate'],2)} | {fmt(h('V4_A')['nat_natural_hi_pooled']['rate'],2)} | {fmt(h('V4_A2')['nat_natural_hi_pooled']['rate'],2)} |
| NONSENSE lines | {fmt(h('base_V4')['nat_nonsense_pooled']['rate'],2)} | {fmt(h('V3_A200_V4')['nat_nonsense_pooled']['rate'],2)} | {fmt(h('V4_A')['nat_nonsense_pooled']['rate'],2)} | {fmt(h('V4_A2')['nat_nonsense_pooled']['rate'],2)} |
| Trelis CER m1 | {fmt(h('base_V4')['cer_m1_pooled']['rate'])} | {fmt(h('V3_A200_V4')['cer_m1_pooled']['rate'])} | {fmt(h('V4_A')['cer_m1_pooled']['rate'])} | {fmt(h('V4_A2')['cer_m1_pooled']['rate'])} |

Seed-to-seed std of the task score on the three 3-seed tags: base ±{fmt(S['base_V4']['bands']['all']['tc_score']['std'],3)}, V3_A ±{fmt(S['V3_A200_V4']['bands']['all']['tc_score']['std'],3)}, V4_A ±{fmt(S['V4_A']['bands']['all']['tc_score']['std'],3)}. The V4_A vs V4_A2 differences on seed 1001 (task 0.70 vs 0.72, CONSISTENT 16 vs 19 of 30, echo 17 vs 11 of 34) are single-seed, on 30 calls / 34 writes.

### Length band: standard (13 calls) / long (16 calls), seed 1001

{band_delta('bands_s1001', four)}

Long calls have lower fact consistency than standard calls for base and both V4 adapters (V3_A@200 is the exception on seed 1001); NATURAL rate and CER differ little by band. The mixed band is 1 call (air_26_g3) and is not interpretable.

## B. All metrics × length band, seed 1001, all four models

"""
for band in BANDS:
    n = S["V4_A"]["bands_s1001"][band]["n_calls"]
    md += f"### B: band {band} ({n} call{'s' if n != 1 else ''})\n\n{table('bands_s1001', band, four)}\n\n"

md += "## C. All metrics × length band, 3 seeds (1001-1003), base / V3_A@200 / V4_A@350\n\nV4_A2 is absent here (seed 1001 only).\n\n"
for band in BANDS:
    n = S["V4_A"]["bands"][band]["n_calls"]
    md += f"### C: band {band} ({n} call{'s' if n != 1 else ''})\n\n{table('bands', band, three)}\n\n"
md += "### C: standard / long, 3 seeds (seed means; pooled rates)\n\n" + band_delta("bands", three) + "\n\n"

md += f"""## D. Data (data/V4/STATS.md, md5 a2278a74)

- Scenarios: 162 (guidance/scenario_creation_v2.json; 72 old + new, 7 agent types incl. bank and telecom). Records: data/V4/records.json, regenerated with seeded pools and a 3-record reuse cap.
- Calls: **645 of 648** (4 gender pairings per scenario); missing food_18_g1, sub_12_g2, sub_12_g3 (none in test or val). First text run 570/648 accepted; a top-up regenerated the 78 drops and added 75.
- Length band (hash-assigned): standard 265, long 261, mixed 119 calls. Audio 878.2 min (14.64 h), 67.4-116.0 s per call, mean 81.7 s; standard 77.9 s, long 86.7 s, mixed 79.1 s mean.
- Splits: train 465 calls / 117 scenarios / 633.8 min (188 / 188 / 89 standard / long / mixed); val 60 calls / 15 scenarios / 81.6 min; test 30 calls / 30 scenarios / 40.8 min. No scenario overlaps between train, val and test.
- TTS: IndicF5 code-switch (f5cs), the four V3 voices; long and mixed calls use 24-word / 140-Devanagari chunks. Trelis CER flags (> 0.10, best of 3 tries kept): 142/9174 chunks = 1.55% (standard 1.64%, long 1.27%, mixed 1.88%). Number-word mismatches (reported only): 419/2528 chunks with numbers = 16.6%. MMS alignment fallbacks 0. Chunks still ending loud after the tail guard 0.40%.
- Hindi share (agent / customer): standard 0.544 / 0.603, long 0.576 / 0.630, mixed 0.567 / 0.610.
- Trainer window 140 s (longest train/val call 116.0 s, so every call is one window).

## E. Training curves (runs/V4_A, runs/V4_A2: loss.png, eval_metrics.csv)

Common config: rank 64, scaling 2.0, batch 8, duration_sec 140, keep_and_shift on (text stream shifted; new runs only), eval every 50 steps on val (60 calls) and control (English CONTROL set, 100 windows). Peak VRAM 37.2 GB, 5.6 s/step. Losses are not comparable with V3_A (keep_and_shift, different val set).

### V4_A (lr 1.5e-5, 1200 steps, 126 GPU-min)

{cA}

Val bottoms at step {bA} and rises to the end. Control total goes below step 0 early, then rises above step 0 from step 650; control cb2-8 (acoustic) rises from step 0 for the whole run, so the early control gain is text and cb1 only. The early-stop condition was met at step 800; the run was not stopped (NOTES 'D2 / V4 training queue').

### V4_A2 (lr 7.5e-6, 1500 steps, 158 GPU-min)

{cA2}

Val bottoms at step {bA2}; control total at the end (1.2792) stays just below step 0 (1.3114), but control cb2-8 also rises from step 0 (2.19 to 2.90). V4_A2 forgets English less than V4_A at the end of training (control total_pooled 1.2792 vs 1.4169); at the val-best steps the control cost is similar (1.1336 vs 1.1493).

## F. Re-rank (spec section 7; tests/V4_RERANK.md)

Top-3 val checkpoints per run, PersonaPlex on a 12-call subset (seed 1001), Gemma 4 31B naturalness judge; pick = highest NATURAL rate, tie -> lower val loss.

| run | step | val total_pooled | NATURAL | NONSENSE | pick |
|---|---|---|---|---|---|
| V4_A | 400 | 1.01392 | 99/182 (54.4%) | 32 (17.6%) | |
| V4_A | 350 | 1.01520 | 108/193 (56.0%) | 53 (27.5%) | **FINAL** |
| V4_A | 300 | 1.01933 | 108/197 (54.8%) | 37 (18.8%) | |
| V4_A2 | 600 | 1.03260 | 107/181 (59.1%) | 36 (19.9%) | **FINAL** |
| V4_A2 | 500 | 1.03305 | 105/199 (52.8%) | 39 (19.6%) | |
| V4_A2 | 450 | 1.03355 | 89/184 (48.4%) | 50 (27.2%) | |

Judge check against the hand review (V3_A@200, same 12 calls): judge 54.8% NATURAL / 27.1% NONSENSE vs hand 51% / 30%; 13/16 lines agree on air_03_g1.

## G. Caveats

1. **V4_A2 is single-seed** (seed 1001). Single-seed differences between V4_A and V4_A2 are within the seed-to-seed spread seen on the other tags (task score std up to ±0.06).
2. **V4_A step 350 vs 400 is open for the user.** The re-rank pick (350) is a tie within noise (1.6 pp spread vs ~3.6 pp SE) and has the most NONSENSE of the three; step 400 (val-best, fewest NONSENSE) was not run on the full test set.
3. **Mixed band = 1 test call**; its cells are not interpretable. Long vs standard is 16 vs 13 calls.
4. **Base's NATURAL rate is fluent pure English** (the rubric labels real English formulas NATURAL, en=true); compare the "NATURAL and Hinglish" row.
5. **Judges are Gemma 4 31B**, temperature 0; on the calibration set it was ~4 pp more lenient on NATURAL than the hand review. Task / fact judgements are per call, without human check.
6. **Echo is strict** (validate.echo_ok string match on the confirm slot); a garbled or translated value fails echo while the call judge may still accept it ("judge confirm+value" row).
7. **CER method**: Trelis Whisper-Hinglish, roman vs roman (romanised via the V4 data's own spellings, fixed-table fallback), silence-trimmed at -50 dBFS (0.2 s pad, gaps -> 0.3 s); the reference is the model's own text stream, so CER measures speech-vs-text agreement, not correctness.
8. **Training losses** are not comparable to V3_A; V4_A control cb2-8 got worse from step 0.
9. **Data**: 3 calls missing (none in test/val); 16.6% of number-bearing chunks have a number-word mismatch (not re-rendered; open question). data/V4/SAMPLES.md and gen_report.json describe the first 570-call text run; STATS.md describes the final 645 calls. The top-up regenerated the dropped calls with greeting/sign-off turns exempt from the per-band line caps and 'reference' exempt from the nukta check (data/V4/topup_go.sh header: user, 2026-10-04 ~20:00 IST).
10. V3_A@200 was trained on V3 data and is tested here on V4 inputs (new scenarios, new records, longer calls).
"""

out = f"{H}/tests/V4_COMPARISON.md"
open(out, "w").write(md)
print(out, len(md.splitlines()), "lines")
