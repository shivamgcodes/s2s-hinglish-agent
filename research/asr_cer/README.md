# research/asr_cer: ASR and CER study scripts

This folder holds the scripts behind the reported CER/WER numbers for Hinglish speech: how model output and TTS audio were scored against text, and how Trelis Whisper-Hinglish compares with stock Whisper large-v3.

It contains code and the study write-ups only. Transcripts, references, maps, metrics JSON/CSV and audio are left out.

## What each part produced

| dir | scripts | what it produced (reported in) |
|---|---|---|
| `cer_study/` | `compare.py` | `comparison.csv`: the 10 (reference, Whisper large-v3 hypothesis) pairs of base_V1/V1_A/V1_B/V1_C model outputs, scored three ways: score.py raw/fold, Method 1 and Method 2. Also prints the Spearman correlations. Reported in `cer_study/COMPARISON.md`. |
| `cer_study/w1_romanise/<pair>/` | `score_w1.py` (one per pair) | **Method 1**: hand-romanise the Whisper text, then plain Levenshtein CER/WER with S/D/I (`metrics.json`, `verify.json`) |
| `cer_study/w2_charmap/<pair>/` | `align.py`, `build*_map.py`, `word_align.py`, `fix.py`, `finalize.py`, `verify*.py`, `write_verify.py`, `tok.py`/`norm.py` (they vary by pair) | **Method 2**: cross-script character and word alignment by sound (`char_map.jsonl`, `word_map.jsonl`, `char_metrics.json`, `word_metrics.json`, verify) |
| `cer_study/trelis/` | `research/eval_harness/trelis_tx.py` (one copy; deduplicated 2026-10-07) | Trelis/whisper-hinglish-preview transcripts of the same 10 wavs (`trelis_raw.json`, `inputs/<pair>.whisper.txt`). Settings are in `TRANSCRIBE.md`. Same md5 as pod1 `hinglish/tests/trelis_tx.py`, which `needle/n1/asr_turns.py` copies its logic from. |
| `cer_study/trelis/m1/<pair>/`, `m2/<pair>/` | `score.py`; `align.py`, `build_*map.py`, `metrics.py`, `verify*.py`, `finalize.py`, `normalize.py` | Method 1 and Method 2 re-run on the Trelis transcripts |
| `cer_study/trelis/` | `compare_trelis.py` | `comparison_trelis.csv`: Trelis vs large-v3 for every method, with deltas. Reported in `trelis/COMPARISON_TRELIS.md`. It contains a verbatim copy of score.py's `raw_norm`/`fold`/`cer`. |
| `cer_study/` | `cer_trelis_workflow.js` | the workflow script that ran the Trelis re-run with agents. Its `const D` path is now a placeholder. |
| `trelis_calib/` | `calib.py`, `ana.py` | Calibration of Trelis + Method 1 CER against the stored large-v3 CER on 1,059 V3/V4 chunks (`tmp_trelis/calib.jsonl`): the p50/p90/p95/p99 values, Spearman 0.27/0.07, and rejects by threshold. These set the V4 audio-QC threshold change from 0.35 (large-v3) to 0.10 (Trelis m1). Reported in pod1 `NOTES.md` (D2/V4 audio QC: Trelis ASR) and `tmp_trelis/notes_add.md`. |
| `trelis_calib/` | `v3chk.py` | check that the per-record `cer_max` retry rule gives the same result as the old fixed 0.35 rule on V3. The NOTES entry above reports "4439 comparisons, 0 differences". |
| `tts_checks/kokoro-check/` | `asr.py`, `asr_long_tail.py`, `analyze.py`, `make_report.py` | Kokoro Hindi TTS check: faster-whisper large-v3 transcript-back, rough CER, report (`kokoro-check/REPORT.md`) |
| `tts_checks/indicf5-check/scripts/` | `asr.py`, `analyze.py`, `marks.py` | IndicF5 TTS check, with the same ASR settings and the same CER (analyze imports `../../kokoro-check/analyze.py`, so the sibling layout is kept). Report: `indicf5-check/REPORT.md` |
| `tts_checks/review_f5_edge/` | `score.py` | F5 `edge_s` A/B (24 e2e texts x 3 seeds, large-v3): loud ends, mean CER, syl/s and duration. Reported in `research/audio/F5_BACKEND.md` (the edge_s table). |
| `tts_checks/tts_ab_f5/` | `asr_all.py` | large-v3 transcripts (`asr.json`) of the F5/Kokoro TTS A/B clips. The script that scored them was not found among the pod1 `tts_ab_f5/scripts`. |

The 10 study pairs are model outputs of synthetic calls. Their references and transcripts are not here. The HF dataset `shivamgupta/hinglish-s2s-synthetic-calls` has the call data.

**Embedded text:** 55 of the per-pair Method 2 scripts quote Devanagari/Roman fragments of the synthetic transcripts inside their hand-judgement tables (for example `fix.py`'s `SET`/`PATCH`). These are part of the code of record and were kept.

## Related code elsewhere (not copied here)

| file | where | role |
|---|---|---|
| `score.py` (raw_norm, fold, cer) | `research/eval_harness/score.py` (from `hinglish/tests/score.py`) | the score.py raw and fold CER columns |
| `trelis_m1.py`, `asr.py`, `synth.py`, `tts_norm.py`, `tts_backends.py` | `research/audio/` | Trelis + Method 1 audio QC (`calib.py`, `v3chk.py` and `review_f5_edge/score.py` import these from `$AUDIO_DIR`, default `research/audio`) |
| `trelis_pass.py`, `v4_eval.py` | `research/eval_harness/` | Trelis pass in the PersonaPlex V4 eval |
| `asr_span_probe.py` | `research/needle/n1/experiments/` | N1 decision A2: does base Needle suppress a call when a required arg is spoken as number words |
| `asr_turns.py` | `research/needle/n1/` | Trelis ASR of the V1/V3 customer turns (N1 rendering b) |
| `asr_smoke.py`, `opus_rt.py` | `research/needle/n2/setup/` | Trelis on a 30 s window, clean vs Opus round trip |
| `asr_windows.py`, `review_trelis.py` | `research/needle/n2/data/tools/`, `n2/windows/windows_tools/` | Trelis on the N2 router windows |

## Paths

| env var | default (original) | used by |
|---|---|---|
| `HINGLISH_ROOT` | `/workspace/hinglish` | `trelis_calib/*`, `../eval_harness/trelis_tx.py` (wavs `tests/out/<model>/V1/`), `tts_ab_f5/asr_all.py` |
| `AUDIO_DIR` | `research/audio` (was `/workspace/hinglish/audio`, and `tmp_trelis/audio_new` for `v3chk.py`, whose synth.py has the same md5) | `calib.py`, `v3chk.py`, `review_f5_edge/score.py` |

**Hard-coded paths that remain:**

| file | path | kind |
|---|---|---|
| `tts_checks/kokoro-check/asr.py`, `indicf5-check/scripts/asr.py` | `/workspace/venv-asr/bin/python`, `/root/hf-asr`, `/workspace/hf` (an HF cache dir, not a token) | docstrings (run lines) |
| `tts_checks/kokoro-check/make_report.py` | `/workspace/kokoro-check/`, `/workspace/venv-tts`, `/workspace/venv-asr`, `/root/hf-asr` | text written into the report |
| `tts_checks/tts_ab_f5/asr_all.py` | `/workspace/hinglish/tts_ab_f5/asr.json` | docstring |
| `cer_study/trelis/TRANSCRIBE.md` | `/workspace/hinglish/tests/out/...` | doc |
| `cer_study/w*/`, `trelis/m*/` per-pair scripts | none. They read and write relative to their own dir (`os.path.dirname(__file__)`), where the excluded reference, whisper and map files lived. | |
| `tts_checks/kokoro-check/*`, `indicf5-check/scripts/*`, `cer_study/compare.py`, `trelis/compare_trelis.py` | none. They read and write relative to their own (or the parent) dir: `out/`, `transcripts.json`, `metrics.csv`, `args.json`, `comparison*.csv`. | |

## Excluded, and why

- **Data**: `cer_study/inputs/*.txt`, `pairs.json`, `args.json` (the score.py CER input to compare.py), `comparison*.csv`, `trelis_raw.json`, every per-pair `reference.txt`/`whisper*.txt`/`romanisation_notes.txt`/`*_map.*`/`*metrics.json`/`verify.json`/`auto_align.json`; kokoro and indicf5 `transcripts.json`, `metrics.csv`, `summary.json`, `ratings*.csv`, `listen*.html`, `out/` wavs; `tmp_trelis/calib.jsonl`; `tts_ab_f5` jsons and wavs; `review_f5/edge/*.wav` and `index.json`.
- **Invalid run**: `cer_study/trelis_INVALID_no_transcription/` (the name says it is invalid, because it had no transcription).
- **Other agents' folders**: the audio QC modules (`research/audio`), `tests/score.py`, `trelis_pass.py`, `v4_eval.py`, `gen/make_stats_v4.py` (`research/eval_harness`, `research/data_gen`).
- **Not CER/ASR studies**:
  - TTS rendering only: `indicf5-check/scripts/{synth.py, install.sh, run_all.sh}`, `kokoro-check/{asr_install.sh, make_listen2.py}`, `tts_ab/render_ab.py`, `tts_ab_f5/scripts/*render*.py`, `common.py`, `run_ft.sh`, `review_f5/edge/render.py`.
  - Data QC: `review_v1data/*`.
  - Records ASR, computes no CER: `gate0/transcribe.py`.
  - Outside this brief: `exp0b`/`needle-exp0` tools, which contain no CER/ASR comparison.
  - No reported number: `tests/vad_scratch/vad_compare.py` (VAD, not ASR).
  - Earlier copies of the same scripts: the `writeup/src_snapshot*` duplicates.
