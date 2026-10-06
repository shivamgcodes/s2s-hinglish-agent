# research/data_gen: synthetic Hinglish call text

This folder holds the code that wrote the text for the synthetic customer-support calls (variants V1, V3, V4 and CONTROL) used to fine-tune PersonaPlex. Gemma 4 31B runs offline through vLLM and writes the calls; deterministic checks and an LLM judge then validate them. The code was copied from pod1 `/workspace/hinglish/{gen,guidance}`. Per-file sources and md5s are in `PROVENANCE.tsv`.

## Stages and scripts

| Stage | Script(s) |
|---|---|
| Scenario specs (72 V1 scenarios; V4 v2 specs) | `guidance/scenario_creation.json`, `gen/scenarios_v2.py` -> `guidance/scenario_creation_v2.json` |
| Caller records (one per scenario, shared across g1-g4) | `gen/records.py` (V1/V3), `gen/records_v2.py` (V4) |
| Test / val holdout split | `gen/holdout_v2.py` (V4); the V1 split is `research/audio/holdout.py` |
| Call generation | `gen/generate.py` (one batch), `gen/gen_loop.py` (retry rounds), `gen/worker.py` (job-queue tuning worker that reads `gen/queue/*.json`) |
| V4 extras (length bands, templates, nukta checks) | `gen/v4.py`; post-run report `gen/v4_post.py` |
| Validation | `gen/validate.py` (deterministic checks plus `--judge`), `gen/rescore_trial.py` |
| Hindi-token share | `packages/hinglish_text/hindi_share.py` (shared module, not copied here) |
| Speaking-rate tables | `gen/measure_kokoro_wps.py` -> `gen/kokoro_wps.json`; `gen/f5_wps.json` |
| Stats | `gen/make_stats.py` (V1), `gen/make_stats_v3.py`, `gen/make_stats_v4.py` |
| Self-tests | `gen/test_records_v2.py`, `gen/test_records_v2_repair.py` |
| Deliverables bundle (D2/V4) | `bundle_v4.sh` |

Guidance provenance: `guidance/scenario_creation*.json` and `r1_vs_r2.md` are synthetic scenario specs and examples written in this project. They contain only fictional brands, names and numbers.

## Inputs and outputs

Generated text goes to `$HINGLISH_ROOT/data/<variant>/`: `calls.jsonl`, `gen_log.jsonl`, `SAMPLES.md`, `records.json`, `holdout.json`, `STATS.md` and similar files. The published copies are in the HF dataset `shivamgupta/hinglish-s2s-synthetic-calls`, one dataset folder per variant.

## Required files that are not shipped

- **`gen/novasynth_hinglish.py`** is excluded because it is a verbatim copy of third-party NovaSynth prompts with an unclear licence. `generate.py` imports it at top level (`import novasynth_hinglish as NS`), so any script that imports `generate` fails without it. That covers V1, V3, V4 and CONTROL generation, plus `gen_loop.py`, `worker.py`, `v4_post.py` and `rescore_trial.py`. To run them, provide a `gen/novasynth_hinglish.py` that defines these string constants: `HINGLISH_CALLER_INSTRUCTIONS`, `SPEECH_FORMATTING_HINDI_HINGLISH`, `PERSONA_HINGLISH_CONSTRAINT`, `AGENT_FILLERS`, `AGENT_LANGUAGE_STYLE`. These are the only `NS.<NAME>` uses, all in `generate.py`. `validate.py`, `v4.py` and `records*.py` do not import it.
- **Google word lists** `google-10000-en.txt` and `google-20k-en.txt` are excluded because their licence is unclear. Source: github.com/first20hours/google-10000-english. `hindi_share` reads them from `$HINDI_SHARE_LEXICON`. Only `google-20k-en.txt` is required at runtime. If it is missing, `hindi_share._lexicons()` raises FileNotFoundError, which breaks `validate.py`, `generate.py`, the V4 checks in `v4.py`, and `score.py`/`v4_eval.py` in the eval harness. `google-10000-en.txt` is used only by the `hindi_share` selftest. The project-made `english_extra.txt` and `hindi_extra.txt` are in `packages/hinglish_text/lexicon/`.
- The PersonaPlex SentencePiece tokenizer (from the HF cache) is used by `records.py` `n_tokens`, and therefore by `test_records_v2.py`.

## Paths

| Name | Default | Used by |
|---|---|---|
| `HINGLISH_ROOT` | `/workspace/hinglish` | data root: `common.ROOT/DATA`, make_stats*, v4_post, holdout_v2, rescore_trial, bundle_v4.sh |
| `HINGLISH_GUIDANCE` | `research/data_gen/guidance` (next to `gen/`) | `common.GUIDE`. On pod1 this was `$ROOT/guidance`. `scenarios_v2.py` writes here. |
| `HINGLISH_TEXT_PKG` | `packages/hinglish_text` | sys.path shim for `import hindi_share` (generate, validate, v4, worker, v4_post, rescore_trial) |
| `GEMMA_MODEL_PATH` | `/workspace/gemma/models/gemma-4-31B-it` | `common.MODEL_PATH` |
| `RUNS_ROOT` | `/workspace/runs` | bundle_v4.sh |
| hard-coded | `/workspace/hf/hub/models--nvidia--personaplex-7b-v1/snapshots/*/tokenizer_spm_32k_3.model` | records.py `SPM` |
| hard-coded | `HF_HOME` setdefault `/workspace/hf` | common.LLM |
| hard-coded | `/workspace/deliverables/<ts>`, `/workspace/venv-*` in docstrings | bundle_v4.sh, run lines |

## Excluded

`gen/novasynth_hinglish.py` and `gen/novasynth_src/` (unclear licence; contains a personal path). `gen/lexicon/` (Google lists: unclear licence; extras live in packages). `gen/hindi_share.py` (shared module in packages). `gen/queue/` (empty runtime dir). `gen/bak_*`, `*.bak*`, `logs/`, `review_tmp/`, `__pycache__/`, `selftest_r1_vs_r2.txt` (a test output).

## Smoke test (laptop)

`py_compile` passes for every file. `import validate, v4, common` succeeds and resolves `hindi_share` from packages; it loads 72 scenarios. `test_records_v2.py` needs the PersonaPlex tokenizer and `test_records_v2_repair.py` needs `data/V4`, so neither runs without data.
