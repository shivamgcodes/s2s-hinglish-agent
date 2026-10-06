#!/bin/bash
# D2 section 9 deliverables bundle (CPU only; reads existing files, writes only under /workspace/deliverables/<ts>)
set -euo pipefail
H=${HINGLISH_ROOT:-/workspace/hinglish}; R=${RUNS_ROOT:-/workspace/runs}
TS=$(date -u +%Y%m%dT%H%M%SZ)
B=/workspace/deliverables/$TS
mkdir -p $B/{reports,runs,qc,samples,scores,wavs,adapters}
cp $H/tests/V4_COMPARISON.md $H/tests/V4_EVAL.md $H/tests/V4_RERANK.md $H/tests/V1_COMPARISON.md $B/reports/
cp $H/data/V4/STATS.md $B/reports/V4_STATS.md
cp $H/NOTES.md $B/
cp $H/D2_v4_dataset_retrain.md $B/reports/
for v in V1 V3 CONTROL V4 V4_topup; do [ -f $H/data/$v/SAMPLES.md ] && cp $H/data/$v/SAMPLES.md $B/samples/SAMPLES_$v.md; done
cp $H/data/V4/{gen_report.json,topup_result.json,dropped_call_ids.json} $B/samples/ 2>/dev/null || true
for r in $(ls $R); do
  for f in loss.png metrics.csv eval_metrics.csv BEST_CKPT FINAL_CKPT args.yaml; do
    [ -f $R/$r/$f ] && { mkdir -p $B/runs/$r; cp $R/$r/$f $B/runs/$r/; }
  done
done
cp -r $H/data/V4/qc $B/qc/V4
for v in V1 V3 CONTROL; do [ -d $H/data/$v/qc ] && { mkdir -p $B/qc/$v; cp $H/data/$v/qc/*.png $B/qc/$v/ 2>/dev/null || true; }; done
for t in base_V4 V3_A200_V4 V4_A V4_A2; do
  mkdir -p $B/scores/$t; cp $H/tests/out/$t/v4_scores.{json,csv} $H/tests/out/$t/v4_runs.csv $B/scores/$t/
done
CALLS="air_11_g2 air_26_g3 bank_07_g1 cab_03_g1 cab_26_g4 ecom_23_g1 food_03_g2 food_07_g3 sub_11_g4 tel_03_g2"
for t in base_V4 V3_A200_V4 V4_A V4_A2; do
  mkdir -p $B/wavs/$t
  for c in $CALLS; do
    for e in _stereo.wav .nat.json .call.json .trelis.json; do cp $H/tests/out/$t/V4/${c}_s1001$e $B/wavs/$t/; done
  done
done
mkdir -p $B/wavs/inputs; for c in $CALLS; do cp $H/tests/inputs/V4/$c.wav $H/tests/inputs/V4/$c.meta.json $B/wavs/inputs/; done
cp -r $R/V4_A/checkpoints/checkpoint_000350/consolidated $B/adapters/V4_A_step350_FINAL
cp -r $R/V4_A/checkpoints/checkpoint_000400/consolidated $B/adapters/V4_A_step400_valbest_ALTERNATIVE
cp -r $R/V4_A2/checkpoints/checkpoint_000600/consolidated $B/adapters/V4_A2_step600_FINAL
cat > $B/README.md <<EOR
# D2 / V4 deliverables ($TS)

Bundle built by /workspace/hinglish/bundle_v4.sh from existing files only (no GPU). Main report: reports/V4_COMPARISON.md.

- reports/: V4_COMPARISON.md (spec section 9: base / V3_A@200 / V4_A@350 / V4_A2@600 x metrics x length band), V4_EVAL.md (section 8), V4_RERANK.md (section 7), V4_STATS.md (data/V4/STATS.md), V1_COMPARISON.md (earlier), D2_v4_dataset_retrain.md (the spec).
- NOTES.md: all decisions (see 'D2 / V4 ...' headings).
- runs/<run>/: loss.png, metrics.csv, eval_metrics.csv, BEST_CKPT, FINAL_CKPT, args.yaml for every run in /workspace/runs that has them.
- qc/V4: data/V4/qc (alignment PNGs, Mimi round-trips, qc_stats.jsonl, qc_summary.json); qc/V1, V3, CONTROL: PNGs only.
- samples/: SAMPLES.md per variant. SAMPLES_V4.md and gen_report.json describe the first 570-call V4 text run; SAMPLES_V4_topup.md the top-up; final counts are in reports/V4_STATS.md.
- scores/<tag>/: v4_scores.{json,csv}, v4_runs.csv (tests/v4_eval.py).
- wavs/<tag>/: 10 representative V4 test calls, seed 1001, stereo PersonaPlex output (*_s1001_stereo.wav) + naturalness (.nat.json), call judge (.call.json) and Trelis ASR (.trelis.json). Same 10 calls for every tag (base_V4, V3_A200_V4, V4_A = step 350, V4_A2 = step 600): $CALLS. All 7 agent types; 4 long / 5 standard / 1 mixed; g1 3, g2 3, g3 2, g4 2. wavs/inputs/: the matching test input wavs + meta.
- adapters/: consolidated LoRA (lora.safetensors + config.json), merged at load by tests/driver.py (rank 64, scaling 2.0):
  - V4_A_step350_FINAL: /workspace/runs/V4_A/checkpoints/checkpoint_000350 (re-rank pick, used in all V4_A tests)
  - V4_A_step400_valbest_ALTERNATIVE: checkpoint_000400 (val-best; the 350-vs-400 choice is open for the user; not run on the full test set)
  - V4_A2_step600_FINAL: /workspace/runs/V4_A2/checkpoints/checkpoint_000600 (re-rank pick = val-best)
  - Note: runs/<run>/lora.safetensors at the run root is the END-of-run adapter (step 1200 / 1500), not a pick; it is not included.
  - V3_A@200 (compared, not new): /workspace/runs/V3_A/checkpoints/checkpoint_000200/consolidated, not included.
- No REPORT.md or SUMMARY.md exists on the pod; none was created for this bundle.
- MANIFEST.md5: md5 of every file.
EOR
(cd $B && find . -type f ! -name MANIFEST.md5 | sort | xargs md5sum > MANIFEST.md5)
du -sh $B
tar -C /workspace/deliverables -czf /workspace/deliverables/$TS.tar.gz $TS
ls -la /workspace/deliverables/$TS.tar.gz
md5sum /workspace/deliverables/$TS.tar.gz
echo BUNDLE_DIR=$B
echo BUNDLE_TGZ=/workspace/deliverables/$TS.tar.gz
