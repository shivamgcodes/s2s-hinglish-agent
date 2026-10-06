# research/: the code that made the data, the models and the numbers

This folder is for AI folks. It holds every script of record behind the synthetic dataset, the PersonaPlex V4 Hinglish
LoRA, the Needle N1 and v2 routers and their evaluations. It holds code only:
- **data** is in the HF dataset [`shivamgupta/hinglish-s2s-synthetic-calls`](https://huggingface.co/datasets/shivamgupta/hinglish-s2s-synthetic-calls);
- **weights** are in [`shivamgupta/personaplex-hinglish-v4-lora`](https://huggingface.co/shivamgupta/personaplex-hinglish-v4-lora)
  and [`shivamgupta/needle-hinglish-router-v2`](https://huggingface.co/shivamgupta/needle-hinglish-router-v2)
  (N1 is in `n1/`, the sweep in `checkpoints/`).

The scripts ran on the dev boxes: pod1 `/workspace/hinglish` for PersonaPlex, runpod2 `/root/n2` for N2, plus the laptop.
- Data and work directories come from env vars. Their defaults are the original paths, so a run on the original box
  behaves as before.
- Code is found relative to the script.
- Each subfolder's README has a **Paths** table, and each has a `PROVENANCE.tsv` with source path, md5 and every
  edit made.

Code shared with the live demo, such as the router, the LoRA merge, the role-prompt format and `hindi_share`, is
**not** here. It lives once in [`../packages/`](../packages/README.md), and these scripts import it from there
(D-SINGLE-SOURCE).

## Pipeline map: PersonaPlex speech model (V1 → V3 → V4)

| # | stage | scripts | outputs (dataset folder or run dir) |
|---|---|---|---|
| 1 | Scenario specs | `data_gen/guidance/scenario_creation*.json`, `data_gen/gen/scenarios_v2.py` | 72 scenarios (V1), v2 specs (V4) |
| 2 | Records (customers, orders, facts, role prompts g1-g4) | `data_gen/gen/records.py` (V1/V3), `records_v2.py` (V4); split `holdout_v2.py` / `audio/holdout.py` | `data/<V>/records.json`, `holdout.json` |
| 3 | Dialogue generation (Gemma 4 31B, vLLM, offline) | `data_gen/gen/generate.py`, `gen_loop.py`, `v4.py`, `worker.py` | `data/<V>/calls.jsonl`, `gen_log.jsonl` |
| 4 | Validation (deterministic + LLM judge) | `data_gen/gen/validate.py`, `rescore_trial.py`, `v4_post.py`; stats `make_stats*.py` | `STATS.md`, rejects |
| 5 | TTS + Trelis/Whisper QC (reject loop on CER) | `audio/run_variant.sh` → `synth.py`, `tts_backends.py`, `f5cs.py` (IndicF5 code-switch), `tts_norm.py`, `asr.py`, `trelis_m1.py` | per-turn chunks, `work/plan.json` |
| 6 | Word alignment + stereo assembly | `audio/align.py` (MMS_FA), `assemble.py`, `qc.py` | `stereo/*.wav` + alignment JSON, `train/`/`heldout/` manifests |
| 7 | LoRA training on patched moshi-finetune | `personaplex_train/UPSTREAM.md` (kyutai-labs/moshi-finetune @ 2acc879 + `trainer/personaplex_prefix.patch`), `trainer/configs/{A,B,C}.yaml`, `make_config.py`, `train_run.sh`, `preflight.py`, `queue_v4_train.sh`; voice prompt codes `voice_codes/*.py` | `$RUNS_ROOT/<run>/checkpoints/` |
| 8 | Held-out loss, checkpoint pick | `personaplex_train/trainer/eval_heldout.py`, `plot_val.py`, `pick_best.py` | `BEST_CKPT` |
| 9 | Test harness: inputs → PersonaPlex runs → ASR → judge → score | `eval_harness/make_inputs.py`, `run_tests.py` → `driver.py` (LoRA merged by `packages/personaplex_lora`), `asr_pass.py` / `trelis_pass.py`, `judge.py`, `score.py`, `v4_eval.py`; queues `personaplex_train/queue_v4_eval*.sh` | `tests/out/...`, `V4_EVAL.md` |
| 10 | Re-rank + comparison (V4_A2 step 600 shipped) | `eval_harness/v4_rerank.py`, `v4_comparison.py`; VAD gating study `vad_gate.py` | `FINAL_CKPT`, `V4_RERANK.md`, `V4_COMPARISON.md`, `VAD_GATING.md` |
| 11 | ASR / CER studies (scoring method, Trelis vs large-v3, TTS checks) | `asr_cer/` | `COMPARISON*.md`, calibration numbers |
| 12 | Offline inference with the released LoRA | `inference/personaplex_lora/run_offline.py` (the HF card's example; uses the pip package `personaplex_lora`) | wav + text |

## Pipeline map: Needle router (N1 → v2)

| # | stage | scripts | outputs |
|---|---|---|---|
| N1.1 | Examples from V1/V3 calls (+ ASR turns, romanised) | `packages/needle_router/n1/build_data.py`, `needle/n1/asr_turns.py`, `assemble.py`, `validate_datasets.py` | `train/val/test_*.jsonl` |
| N1.2 | Fine-tune sweep (needle3 + LoRA) | `needle/n1/finetune/sweep/{build,run_sweep}.sh`, `needle/common/{timed,tstamp,val_loss}.py` | `tuned_full.cact` (= HF `n1/`) |
| N1.3 | Eval / audit / report | `needle/n1/evaluate.py`, `audit.py`, `router_audit.py`, `neg_analysis.py`, `make_report.py`, `experiments/` | `REPORT.md` |
| N2.1 | Call windows from V4 audio | `needle/n2/windows/windows_tools/extract_windows.py`, review tools | windows + meta |
| N2.2 | Trelis ASR of the windows → numconv → rows | `needle/n2/data/tools/asr_windows.py`, `packages/needle_router/numconv`, `build_rows.py`, `stats_rows.py` | `data/rows` |
| N2.3 | Fine-tune sweep (A/B/R sets × epochs, + reasoning variants) | `needle/n2/finetune/run_sweep*.sh`, `run_r15.sh`, `add_reasoning.py`, `post_sweep.sh`, `collect.py` | `checkpoints/<run>/` (HF), R_e15 shipped |
| N2.4 | Engine eval, analysis, latency | `needle/n2/finetune/eval_engine.py`, `needle/n2/eval/*.py`, `eval/run_all.sh`, `eval/lat/run_lat.sh`; numconv eval `needle/n2/numconv/eval_numconv.py` | `engine_val_*.json`, `selection.json` |
| N2.5 | Inference example | `inference/needle_router/example.py` (the HF card's quickstart; uses the pip package `needle_router`) | resolved calls |

## Not in this repo (and why)

| what | why | how to supply it |
|---|---|---|
| `gen/novasynth_hinglish.py` + `gen/novasynth_src/` (NovaSynth prompt texts used by V3/V4 generation) | verbatim copies of a third-party (employer) codebase's prompts, licence unclear | provide a `novasynth_hinglish.py` that defines the constants listed in `data_gen/README.md`. Without it, `generate.py` (and every script that imports it) does not run. |
| Google English word lists (`google-20k-en.txt`, `google-10000-en.txt`) | no clear licence | download from github.com/first20hours/google-10000-english and set `HINDI_SHARE_LEXICON=<dir>`. The validators, `score.py` and `v4_eval.py` need it. |
| upstream moshi-finetune tree, PersonaPlex tree | upstream code | `personaplex_train/UPSTREAM.md` (patch + commit). PersonaPlex is unmodified. |
| data (`*.jsonl`, wavs, records dumps, eval outputs), weights, `voices_f5/` reference audio | data / weights | the HF dataset and model repos above |
| backups (`*.bak*`, `bak_*`), logs, `out/`, `inputs/`, `review*/`, one-off probes | not code of record | n/a |

## Smoke test

`bash research/smoke_test.sh` parses every `.py` and `.sh` and runs the pure-python unit tests listed in
`unit_tests.txt`: the VAD gate and numconv. `deploy/run_all_cpu_tests.sh` runs it as the `research_smoke` suite.
Everything else needs the dev boxes' data and GPUs.
