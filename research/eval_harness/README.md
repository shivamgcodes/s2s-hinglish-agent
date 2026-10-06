# research/eval_harness: PersonaPlex test harness (V1, V3, V4 and VAD gating)

This harness runs base PersonaPlex or a LoRA-merged PersonaPlex against held-out synthetic calls. The model hears only the customer channel. The harness scores the model's text stream and audio for Hindi use, verb forms, tool-check lines and naturalness, using an ASR pass and a Gemma judge. The code was copied from pod1 `/workspace/hinglish/tests`. Sources and md5s are in `PROVENANCE.tsv`. The `*.md` files are the reports of record: `V1_COMPARISON`, `V4_EVAL`, `V4_RERANK`, `V4_COMPARISON`, `VAD_GATING`.

## Stages and scripts

| Stage | Script |
|---|---|
| Container check | `setup_container.sh` |
| Test inputs (customer-only wav, 2 s lead, plus meta) | `make_inputs.py` |
| PersonaPlex runs (sharded) | `run_tests.py` -> `driver.py` (`--adapter` merges a LoRA; `--customer-mode vad` uses `vad_gate.py`); `run_tests.sh`, `listen_go.sh` |
| ASR | `asr_pass.py` (faster-whisper), `trelis_pass.py` / `trelis_tx.py` (+ `trelis_args.json`) |
| Judge | `judge.py` (Gemma via vLLM; call judge and `--naturalness`) |
| Scoring | `score.py` (V1/V3 metrics 1-6), `v4_eval.py` (V4, split by length band) |
| Reports | `v4_rerank.py` -> `V4_RERANK.md` (+ `FINAL_CKPT`), `v4_comparison.py` -> `V4_COMPARISON.md` |
| Helpers / self-tests | `tcommon.py`, `fake_data.py`, `test_vad_gate.py` |

Queue scripts that chain these stages are in `research/personaplex_train/queue_*.sh`.

## Inputs and outputs

Inputs are `$HINGLISH_ROOT/data/<variant>/` (calls, holdout, stereo wavs) from the HF dataset `shivamgupta/hinglish-s2s-synthetic-calls`, and adapters under `$RUNS_ROOT/<run>/checkpoints/...`. Outputs go to `$HINGLISH_ROOT/tests/{inputs,out}/` (data, not shipped).

## Shared modules (from `packages/`)

- `tcommon.ascii_prompt` and `_ASCII_MAP` are re-exported from `packages/personaplex_lora/role_prompt.py` (env `PP_LORA_PKG`). The output is identical to the pod1 body for every code point. The map literal differs only in a no-op entry: pod1 maps `' '` to `' '`, and the package maps NBSP to `' '`. NFKD folds NBSP to a space anyway.
- `hindi_share` needs `google-20k-en.txt` in `$HINDI_SHARE_LEXICON`. The file is not in the repo; its source is github.com/first20hours/google-10000-english. Without it, `score.py` (the m1 metrics) and `v4_eval.py` raise FileNotFoundError.
- `driver.py` merge hook: `packages/personaplex_lora/trainer/merge_lora.py` (env `PP_MERGE_LORA_HOOK`). If that file is missing, `driver.py` silently falls back to its own generic merge, which does not recompute voice-prompt embeddings.
- `score.py`, `judge.py` and `v4_eval.py` import `hindi_share` from `packages/hinglish_text` (env `HINGLISH_TEXT_PKG`) and `common` from `research/data_gen/gen` (env `HINGLISH_GEN_DIR`). `v4_eval.py` and `trelis_pass.py` import from `research/audio` (env `HINGLISH_AUDIO_DIR`).

## Paths

| Name | Default | Used by |
|---|---|---|
| `HINGLISH_ROOT` | `/workspace/hinglish` | `tcommon.ROOT` (data/, tests/out, tests/inputs), `v4_comparison.py`, `v4_rerank.py` |
| `RUNS_ROOT` | `/workspace/runs` | `v4_comparison.py`, `v4_rerank.py` |
| hard-coded | `/workspace/personaplex-gate0` (Gate-0 experiment specs), `/workspace/venv-pp/bin/python`, cwd `/workspace/personaplex` | `run_tests.py` |
| hard-coded | `/workspace/hinglish/tests/out/...` | `trelis_tx.py` |
| hard-coded | `/workspace/hinglish/audio/testdata`, `.../tests/selftest/data` | `fake_data.py` |
| hard-coded | `T=/workspace/hinglish/tests`, `HF_HOME=/workspace/hf`, venvs | `run_tests.sh`, `listen_go.sh`, `setup_container.sh` |

## Excluded

`out/`, `inputs/`, `review/`, `selftest/`, `selftest_*.log`, `vad_scratch/`, `trelis_tx.log`, `*.bak*` (`*.bak_vad`, `*.bak_d2*`, `*.bak_full`), `__pycache__/`.

## Smoke test (laptop)

`py_compile` and `bash -n` pass for every file. `python3 test_vad_gate.py` reports ALL PASS (4 checks). `import tcommon` resolves `ascii_prompt` from packages.
