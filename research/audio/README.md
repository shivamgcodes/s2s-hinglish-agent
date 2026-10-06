# research/audio: TTS, alignment and assembly of the stereo training calls

This folder turns a generated `calls.jsonl` into the training audio. It synthesises each turn (Kokoro for V1 and CONTROL, IndicF5 code-switch for V3 and V4), rejects bad chunks with ASR CER, aligns words with MMS_FA and assembles stereo wavs: agent on ch0, customer on ch1. Each wav comes with a moshi-finetune alignment JSON, manifests and QC. The code was copied from pod1 `/workspace/hinglish/audio`. Sources and md5s are in `PROVENANCE.tsv`.

## Stages and scripts

`run_variant.sh CALLS.jsonl ROOT [ASR_SHARDS]` drives the whole pipeline and is resumable.

| Stage | Script |
|---|---|
| Plan (text normalisation, chunking, backend fixed in `ROOT/work/plan.json`) | `synth.py plan` (uses `tts_norm.py`, `tts_backends.py`) |
| TTS | `synth.py tts` (Kokoro), `f5cs.py` (IndicF5 code-switch, venv-f5; design notes in `F5_BACKEND.md`) |
| Chunk ASR QC / reject loop | `asr.py` (faster-whisper; Trelis Whisper-Hinglish scored by `trelis_m1.py`) |
| Build chunk timeline | `synth.py build` |
| Word alignment | `align.py` (torchaudio MMS_FA, venv-pp) |
| Stereo wav + alignment JSON + manifests | `assemble.py` |
| QC plots / Mimi round-trip | `qc.py plots`, `qc.py mimi` |
| V1 hold-out split | `holdout.py` |
| One-off test scripts | `run_e2e*.sh`, `run_scripttest*.sh`, `run_test_cpu.sh`, `f5_scripttest.py` |

## Inputs and outputs

The input is `$HINGLISH_ROOT/data/<variant>/calls.jsonl`, produced by `research/data_gen`. Outputs go to `ROOT/` (`stereo/*.wav` plus `<wav>.json` alignments, `train/` and `heldout/` manifests, `qc/`, `work/`). The published audio is in the HF dataset `shivamgupta/hinglish-s2s-synthetic-calls`, in that dataset's per-variant audio folders.

## Paths

| Name | Default | Used by |
|---|---|---|
| `AUDIO_DIR` | `/workspace/hinglish/audio` | `run_variant.sh` code dir (`A=`) |
| `HINGLISH_ROOT` | `/workspace/hinglish` | `assemble.py --holdout` default, `holdout.py` output default |
| `HINGLISH_GUIDANCE` | `research/data_gen/guidance` | `holdout.py` scenario spec |
| `HF_HOME` | `/workspace/hf` | `f5cs.py` hub dir (env) |
| hard-coded | `/workspace/venv-{tts,f5,asr,pp}/bin/python`, `/workspace/personaplex` (cwd for align/qc mimi), `/workspace/hinglish/gpu.lock`, `TORCH_HOME=/workspace/torch` | `run_variant.sh`, `asr.py TRELIS_PY` |
| hard-coded | `A=/workspace/hinglish/audio` | `run_e2e*.sh`, `run_scripttest*.sh`, `run_test_cpu.sh` (one-off tests) |
| hard-coded, not shipped | `/workspace/hinglish/tts_ab_f5/{scripts,sents.json}` (TTS A/B study) | `f5_scripttest.py` |

## Excluded

`voices_f5/` (reference voice audio), `pkgs/` (vendored wheels), `testdata/`, every `f5_*/` output dir, `*.bak*`, `__pycache__/`.

## Smoke test

`py_compile` and `bash -n` pass for every file. Nothing here can run without models or data.
