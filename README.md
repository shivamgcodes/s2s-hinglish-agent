# Hinglish full-duplex voice agent (`s2s-hinglish-agent`)

A speech-to-speech customer-support agent that speaks Hinglish. It is full duplex: it listens and speaks at the same
time.
- **PersonaPlex 7B** with the **V4 Hinglish LoRA** listens and speaks.
- **Trelis Whisper-Hinglish** transcribes the caller.
- The **Needle v2** router turns what was said into tool actions, such as changing an address or cancelling an order.

This repo holds all the code: data generation, training, tests and inference (`research/`), and the live demo
(`deploy/`). It holds no data, weights or secrets. Data is in the HF dataset, weights are in the HF model repos.

> **Private until release.** This repo replaces `shivamgcodes/s2s-worker` and `shivamgcodes/s2s-space`
> (D-MONOREPO, 2026-10-06). It has had its current layout since D-MONOREPO-LAYOUT (2026-10-07).

## Start here: two paths

| you want to... | go to | start with |
|---|---|---|
| **understand, reproduce or extend the models** (data generation, TTS, LoRA training, the Needle router, evaluation) | [`research/`](research/README.md) | `research/README.md`: a stage-by-stage map, from scenarios to the trained checkpoints and their scores |
| **run or deploy the demo** (RunPod Serverless worker, Hugging Face Space, local single-GPU run) | [`deploy/`](deploy/README.md) | `deploy/README.md`; for a local GPU, `deploy/run_local.sh setup && deploy/run_local.sh run` |

## Folder map

| folder | what |
|---|---|
| `packages/` | **Shared code, exactly one copy.** Both `research/` and `deploy/` import it, and two of its folders are pip-installable from GitHub (the HF model cards use them). `needle_router/` holds Needle v2 + N1: router, resolver, tool schemas, numconv, romanise. `personaplex_lora/` holds the LoRA merge, the voice codes and `role_prompt.py` (record → role prompt, pairing → voice). `hinglish_text/hindi_share.py` holds the Hindi-share counter used by the data builders. `selftest.py` is the CPU self-test. |
| `research/` | Every script that produced the data, the models and the test numbers (code only). `data_gen/` covers scenarios, records, Gemma dialogue generation and the validators. `audio/` covers the TTS backends, Trelis QC, alignment and assembly. `personaplex_train/` has the moshi-finetune patch, configs, run/queue scripts, preflight and checkpoint pick. `eval_harness/` has inputs → driver → ASR → judge → score, the V4 eval/compare/re-rank and the VAD gate. `needle/` has N1 + N2 data building, fine-tune sweeps and engine eval. `asr_cer/` has the ASR/CER studies. `inference/` has offline LoRA inference and the Needle example. |
| `deploy/` | The live demo. `worker/` is the GPU worker image (PersonaPlex + LoRA server, Trelis ASR, Needle router). `space/` is the HF Space backend. `client/` is the web page. `common/` holds the session token, session config and demo records. `ops/` has the build/deploy scripts. `tests/` holds the CPU + live tests. Also here: `run_local.sh` and `run_all_cpu_tests.sh`. |
| `docs/` | Decision log (`DECISIONS.md`), go-live runbook (`GO_LIVE.md`), frozen design (`DESIGN.md`), older notes. |
| `.github/workflows/` | `build-worker.yml` builds the worker image from `deploy/` + `packages/` only. It runs on manual dispatch or a `v*` tag, so research changes never trigger it. |

**Single-source rule (D-SINGLE-SOURCE).** No code is copied between folders:
- Code used by more than one place lives in `packages/` and is imported from there. The worker image COPYs
  `packages/` and the Space build stages `role_prompt.py` next to `session.py`.
- The Hugging Face model repos hold **no code** (D-LEAN-HF): only weights, configs, examples and the card. Their
  quickstarts `pip install` `packages/needle_router` / `packages/personaplex_lora` from this repo and point to
  `research/inference/`.
- Staging copies (`deploy/space/common/`, `deploy/space/static/`) are build outputs. They are not committed.

## Links

| | |
|---|---|
| Live demo (HF Space) | https://shivamgupta-hinglish-agent.hf.space |
| Speech model (LoRA) | [`shivamgupta/personaplex-hinglish-v4-lora`](https://huggingface.co/shivamgupta/personaplex-hinglish-v4-lora) |
| Router model (Needle v2; N1 in `n1/`; sweep in `checkpoints/`) | [`shivamgupta/needle-hinglish-router-v2`](https://huggingface.co/shivamgupta/needle-hinglish-router-v2) |
| Synthetic dataset (dialogues, audio, Needle rows, eval outputs) | [`shivamgupta/hinglish-s2s-synthetic-calls`](https://huggingface.co/datasets/shivamgupta/hinglish-s2s-synthetic-calls) |
| Base models | [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) (gated), [`Cactus-Compute/needle3`](https://huggingface.co/Cactus-Compute/needle3), [`Trelis/whisper-hinglish-preview`](https://huggingface.co/Trelis/whisper-hinglish-preview) |

## Use the models from Python

```bash
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router"       # router (CPU)
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora"    # LoRA merge + role prompts
```
Full examples: `research/inference/needle_router/example.py` and `research/inference/personaplex_lora/run_offline.py`.

## Tests

```bash
bash deploy/run_all_cpu_tests.sh     # every CPU suite: deploy tests + packages/selftest.py + research/smoke_test.sh
python3 packages/selftest.py         # just the shared packages (stdlib only)
bash research/smoke_test.sh          # just research/: parses everything + the pure-python unit tests
```

## Credits

The models, datasets, voices and code this project builds on, with links and licences, are listed in [ATTRIBUTION.md](ATTRIBUTION.md).

## License

There is no top-level license file yet. One will be added before release. Some parts carry their own licenses:
- `deploy/client/LICENSE` (MIT, from the upstream PersonaPlex / moshi web client).
- `deploy/worker/vendor/moshi/LICENSE.moshi` and `LICENSE.audiocraft`.
- `research/personaplex_train/` ships a patch against kyutai-labs/moshi-finetune; see its `UPSTREAM.md`.

The models have their own licenses on their Hugging Face pages. PersonaPlex is under the NVIDIA Open Model License.
Some third-party inputs are deliberately **not** in this repo:
- the NovaSynth prompt texts used by V3/V4 generation;
- the Google English word lists used by `hindi_share`.

`research/README.md` says how to supply them.
