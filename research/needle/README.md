# research/needle: Needle router training and eval code (N1 and N2)

This folder holds the code that built, trained and evaluated the two Needle 3 Hinglish action routers. It contains code only: no data, weights or tokens.

- **N1** is the first router: 5 agent types, the N1 schema, and the `order_ref` resolver.
- **N2** is the router v2: 7 agent types, schema v2, numconv and resolver_v2.

The runtime modules are not copied here. They live once, in `packages/`, and every script below puts that package directory on `sys.path`:

| package dir | modules | imported by (in this folder) |
|---|---|---|
| `packages/needle_router/n1` (`$N1_DIR`) | `build_data`, `resolver`, `router`, `tools`, `romanise`, `src/records.json`, `tools_json/` | n1: `assemble`, `audit` (+ `build_data`, `tools`), `evaluate`, `router_audit`, `test_resolver`, `build_report` (tools_json only), `experiments/{asr_span_probe, req_vs_opt, real_turn_stress}`; n2: `data/tools/build_rows` (via n1path), `eval/eval_n1` (`router`, via n1path) |
| `packages/needle_router/v2` (`$NEEDLE_V2_DIR`) | `router_v2`, `resolver_v2`, `tools_v2`, `targets`, `n1path` | n2: `data/tools/{build_rows, stats_rows}`, `finetune/{add_reasoning, diag_think_v2, eval_engine}`, `eval/{analyse, eval_n1}`, `schema/{score_map, test_resolver_v2}` |
| `packages/needle_router/numconv` (`$NUMCONV_DIR`) | `numconv` | n2: `numconv/{eval_numconv, test_numconv}`, `windows/windows_tools/review_trelis`; also `router_v2` itself |
| `packages/hinglish_text` | `hindi_share` | indirectly, through the n1 package (`romanise`/`build_data`) |

`n2/schema/score_map.py` is research code and is not in the package. It imports N1 `evaluate.py` (`compare_arg`, `token_f1`) from `research/needle/n1` (`$N1_EVAL_DIR`). Every importer of score_map puts `research/needle/n2/schema` on `sys.path`.

## Layout

```
common/      timed.py tstamp.py val_loss.py         shared fine-tune helpers (N1 and N2 copies were md5-identical)
n1/          assemble evaluate audit asr_turns build_report final_stats make_report neg_analysis
             router_audit validate_datasets test_resolver   (+ experiments/, finetune/, results/*.sh, docs/)
n2/          windows/windows_tools/  data/tools/  numconv/  schema/  finetune/ (+ smoke/)  eval/ (+ lat/)  setup/
```

## Rule for paths

- **Code** is found relative to the script, in this repo. That covers packages/, `common/`, sibling scripts and `env_*.sh`.
- **Data and work dirs** come from env vars. Each defaults to the original value, so a run on the original pod behaves as before.

| env var | default (original) | meaning |
|---|---|---|
| `N1_WORKDIR` | `/workspace/hinglish/needle` | N1 working dir: `*.jsonl`, `src/` (holdout, calls), `results/`, `examples_raw.jsonl`, `REPORT.md`, `vendor/needle_tokenizer.model`. Before, this was the script's own dir. |
| `N1_ROOT` | `/root/needle` (pod2 launchers, `diag_think.py`); `/workspace/hinglish/needle` (`env_pod.sh`) | N1 work dir that the shell launchers use. The `laptop_latency*.sh` scripts require it (they have no default). |
| `N2_ROOT` | `/root/n2` | runpod2 N2 work dir: `windows/`, `V4/`, `data/rows`, `data/asr`, `finetune/` (sweep, evals), `eval/runs`, `needle_n1/` |
| `HINGLISH_ROOT` | `/workspace/hinglish` | pod1 project root (`data/`, `needle_v2/windows`) |
| `N1_DIR`, `NEEDLE_V2_DIR`, `NUMCONV_DIR`, `N1_EVAL_DIR` | `packages/needle_router/{n1,v2,numconv}`, `research/needle/n1` | where the modules are |
| `NEEDLE_COMMON` | `research/needle/common` | timed/tstamp/val_loss. These were in `$F` = `<root>/finetune`. |
| `NEEDLE_VENV` | `/root/venv-needle` | needle venv (pod2 and runpod2) |
| `GPU_LOCK` | `/root/gpu.lock` | flock file |
| `NEEDLE_SITE_PACKAGES` | `/mnt/drive2/venv/needle-exp0/lib/python3.11/site-packages` | installed cactus-needle that `validate_datasets.py` uses |
| `NEEDLE_EXP0_ENV` | (required) | `needle-exp0/env.sh`, which defines `$NEEDLE_PY`; used by `laptop_latency*.sh` |
| `V4_DIR` | `$N2_ROOT/V4`, then `$HINGLISH_ROOT/data/V4` | V4 records and calls for `test_resolver_v2.py` |

**Hard-coded paths that remain** (docstrings, comments, or values the env vars above don't cover):

| file | path | kind |
|---|---|---|
| `n1/asr_turns.py` | `/workspace/hinglish/tests/trelis_tx.py`, `/workspace/venv-tts/bin/python`, `/workspace/hinglish/gpu.lock` | comments (run line) |
| `n1/finetune/env_pod.sh` | `PATH=/workspace/venv-tts/bin` | value |
| `n1/evaluate.py` | `../../needle-exp0/env.sh` | docstring |
| `n2/data/tools/{asr_windows,build_rows,stats_rows,write_stats_md}.py` | `/root/n2/...`, `/root/venv-needle/bin/python` | docstrings |
| `n2/data/tools/write_stats_md.py` | `/root/n2/windows/meta.jsonl` | text written into DATA_STATS.md |
| `n2/finetune/{eval_engine,diag_think_v2}.py`, `n2/eval/eval_n1.py` | `/root/n2/...` | usage docstrings |
| `n2/schema/test_resolver_v2.py` | `/root/n2/V4`, `/workspace/hinglish/data/V4` | docstring |
| `n2/numconv/eval_numconv.py` | `/workspace/hinglish/data/V4/work/chunks` | docstring. Its data paths (`data/chunks.jsonl`, `holdout_calls.json`, `eval_results.json`) stay relative to the script dir, as in the original. |
| `n2/windows/windows_tools/extract_windows.py` | `/workspace/hinglish/...` | usage docstring |
| `n2/windows/windows_tools/review_windows.py` | `/root/n2/windows` | comment |
| `n1/docs/*.md`, `n2/eval/EVAL.md`, `n2/schema/SCHEMA_V2.md` | pod paths, `/mnt/drive2/...` | docs, kept verbatim |

## N1 stages

| # | stage | script | inputs | outputs |
|---|---|---|---|---|
| 1 | records and examples | `packages/needle_router/n1/build_data.py` (+ `tools.py`, `resolver.py`) | `src/records.json`, V1/V3 calls, holdout | `examples_raw.jsonl`, `DATA_STATS.md`, `tests_n0_relabelled.jsonl` |
| 2 | ASR rendering (b) | `n1/asr_turns.py` (Trelis Whisper-Hinglish, GPU); then `packages/.../romanise.py` adds `asr_roman` | per-turn wavs `data/<V>/work/utt/` | `asr_turns_raw.jsonl` -> `asr_turns.jsonl` |
| 3 | assemble | `n1/assemble.py`, then `n1/final_stats.py` and `n1/validate_datasets.py` | `examples_raw.jsonl`, `asr_turns.jsonl`, `tests_n0_relabelled.jsonl` | `train/val/test_heldout_a/test_heldout_b/test_n0.jsonl` + `*_meta.jsonl`, `assemble_summary.json`, `validate_datasets.json` |
| 4 | finetune sweep | `n1/finetune/sweep/run_sweep.sh` (epochs 3/6/10; `common/val_loss.py` on `val.jsonl`), then `sweep/build.sh` (`tuned_l8.cact`, `tuned_full.cact`); environment `finetune/env_pod2.sh` (pod2 GPU) or `env_pod.sh` (pod1 CPU) | `train.jsonl`, `val.jsonl`, `needle3.safetensors` | `adapter_e*.safetensors`, `val_losses.json`, `.cact` |
| 4b | think-block diagnostic | `n1/finetune/diag_think.py` | adapter, `test_heldout_a` | `diag_think.json` |
| 5 | evaluate | `n1/evaluate.py run/score/summary`, launched by `results/run_evals.sh` and `results/run_all_seq.sh` (pod2) and by `results/laptop_latency*.sh` (laptop CPU latency) | test files, weights | `results/<tag>_<test>{_raw.jsonl,.json}`, `<tag>_summary.md` |
| 6 | audit and report | `n1/audit.py`, `n1/router_audit.py`, `n1/neg_analysis.py`, `n1/test_resolver.py`, `n1/make_report.py`, `n1/build_report.py` | `results/*.json` | `results/audit.md`, `router_audit.json`, `neg_analysis.md`, `report_tables.md`, `REPORT.md`, `REPORT_APPENDIX_failures.md` |
| x | decision probes | `n1/experiments/asr_span_probe.py` (A2: digits vs number words), `req_vs_opt.py` (D3), `real_turn_stress.py` (resolver stress), `audit_baseline.py` | as stated in each docstring | `experiments/*.json` |

The released weights are in the `n1/` subfolder of the HF model repo `shivamgupta/needle-hinglish-router-v2`. The data comes from the HF dataset `shivamgupta/hinglish-s2s-synthetic-calls`.

## N2 stages

| # | stage | script | inputs | outputs |
|---|---|---|---|---|
| 1 | router windows | `n2/windows/windows_tools/extract_windows.py` (30 s customer channel ending at each check line), then `arg_coverage.py`; adversarial reviews `review_{windows,audio_all,leak,mapping,trelis}.py` | V4 `calls.jsonl` + stereo wavs | `windows/` (`meta.jsonl`, `clean_16k`/`opus_16k` wavs, `summary.json`) |
| 2 | Trelis ASR of windows | `n2/data/tools/asr_windows.py` (same recipe as `setup/asr_smoke.py`; Opus degradation via `setup/opus_rt.py`) | windows | `data/asr/{clean,opus}.jsonl`, `check_*.json` |
| 3 | numconv | `packages/needle_router/numconv/numconv.py`; tested by `n2/numconv/test_numconv.py` and `eval_numconv.py` | mined V4 chunk transcripts | `eval_results.json` (excluded) |
| 4 | build rows | `n2/data/tools/build_rows.py` (router_v2.prepare_transcript = numconv + romanise; N1 comparison rows via n1path), then `stats_rows.py` and `write_stats_md.py` | windows meta, V4 calls, ASR | `data/rows/{train,train_grounded,val*,test_*,n1_test_*}.jsonl` + meta, `build_stats.json`, `stats.json`, `DATA_STATS.md` |
| 5 | finetune sweep | `n2/finetune/run_sweep.sh` (sets A = train, B = train_grounded; epochs 3/6/10; `val_loss.py` on val/val_clean/val_opus), then `post_sweep.sh` (build `.cact` per adapter, `eval_engine.py` on val, `diag_think_v2.py`) | rows, `needle3.safetensors` | `sweep/adapter_*.safetensors`, `full_*.cact`, `evals/engine_val_*.json` |
| 5r | reasoning variants | `n2/finetune/add_reasoning.py` -> `rows_r/`, then `run_sweep_r.sh` (R_e3/6/10) and `run_r15.sh` (R_e15, the released `tuned_full.cact`) | rows | R adapters, `.cact`, evals |
| 5s | selection and analysis | `n2/finetune/collect.py` (selection table), `harm.py`, `soft.py`, `onfile.py`; `smoke/one.py`, `setup/needle_infer_smoke.py` | `evals/`, `sweep/` | `evals/selection.json`, stdout tables |
| 6 | engine eval | `n2/eval/run_all.sh`: `finetune/eval_engine.py` (v2, test_clean/opus/exact) and `eval/eval_n1.py` (N1 on n1_test_*, with and without numconv) | test rows, weights | `eval/runs/*.json` |
| 7 | analyse and latency | `n2/eval/analyse.py` (-> `eval.json`, the tables in `EVAL.md`); `n2/eval/lat/run_lat.sh` (idle CPU latency) | `runs/` | `eval.json`, `lat/*.json` |
| t | unit tests | `n2/schema/test_resolver_v2.py` (needs V4 `records.json` + `calls.jsonl`), `n2/numconv/test_numconv.py` | | |

## Tests (CPU, seconds; results from 2026-10-07 on the laptop, system python3, with packages/ present)

| command | needs | result |
|---|---|---|
| `python3 research/needle/n2/numconv/test_numconv.py` | nothing | 91/91 passed, rc 0 |
| `N1_WORKDIR=<writable dir> python3 research/needle/n1/test_resolver.py` | a writable `$N1_WORKDIR` (it writes `test_resolver_results.json` there; records come from the package) | 75/76, rc 1. The one failure, `cab_14 'CP wali ride'`, is the same known failure as the recorded `test_resolver_output.txt`. Without `N1_WORKDIR` it prints the results, then fails writing to `/workspace/hinglish/needle`. |
| `V4_DIR=<.../V4> python3 research/needle/n2/schema/test_resolver_v2.py` | V4 `records.json` + `calls.jsonl` (data) | 17/17 passed, rc 0, run read-only against a local copy of the HF dataset's `V4/` |


The released checkpoints are in the HF model repo `shivamgupta/needle-hinglish-router-v2`: `tuned_full.cact` = R_e15, and `checkpoints/` = the sweep. The schema description is in `n2/schema/SCHEMA_V2.md` and the eval write-up in `n2/eval/EVAL.md`, both copied verbatim.

## Dedupes

- `common/timed.py`, `tstamp.py` and `val_loss.py`: the N1 (`needle/finetune/`) and N2 (`needle_v2/finetune/`) copies were md5-identical, so one copy is kept. The launchers call them through `$C` (`$NEEDLE_COMMON`).
- `diag_think.py` (N1) and `diag_think_v2.py` (N2) differ, so both are kept. v2 is positives-only (every v2 row is a positive), reads v2 rows, and imports `router_v2`/`score_map`.
- `needle_v2/tools/{asr_smoke,needle_infer_smoke,opus_rt}.py` are md5-identical to `needle_v2/setup/`. Only `n2/setup/` is kept. `asr_smoke.py` now imports `opus_rt` from its own dir; before, it used `/root/n2/tools`.
- `needle_v2/env_needle.sh` is identical to `needle_v2/setup/env_needle.sh`. Only `n2/setup/env_needle.sh` is kept, and the N2 launchers source it from the repo.
- pod1 `/workspace/hinglish/needle_v2/windows_tools/{arg_coverage,extract_windows}.py` are md5-identical to the laptop `windows/windows_tools/` copies. Every N1 and N2 `.py`/`.sh` on pod1 matched the laptop, so the laptop copies were used.

## Excluded

- **Package modules** (single source in `packages/`): `needle/{build_data,resolver,romanise,router,tools}.py`, `romanise_lexicon.json`, `tools_json/`, `src/records.json`, `vendor/hindi_share.py` and its lexicon, `needle_v2/schema/{router_v2,resolver_v2,tools_v2,targets,n1path}.py`, `needle_v2/numconv/numconv.py`.
- **Data and weights**: every `*.jsonl`/`*.json` (train/val/test, meta, results, `eval_results*.json`, `assemble_summary.json`, `validate_datasets.json`), `src/*` data, `vendor/needle_tokenizer.model`, `*.cact`, `*.safetensors`, `windows/` and `data/` contents, `eval/runs`, `eval/lat` outputs, `finetune/{evals,sweep,rows_r}`.
- **Reports and results**: `REPORT.md`, `DECISIONS.md`, `DATA_STATS.md`, `results/*.md`, `test_resolver_output.txt` (results, not code). `REPORT_APPENDIX_failures.md` is 1.1 MB, over the 300 KB limit.
- **Excluded by the brief**: `needle_v2/review_schema/`, `review_tmp/`, `review_setup/`, `*.bak`, `numconv_pre_review.py.bak`, `MANIFEST_*.md5`.
