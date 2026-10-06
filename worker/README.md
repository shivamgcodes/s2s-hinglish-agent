# S2S serverless GPU worker (`worker/`)

The GPU side of the Hinglish full-duplex agent demo, for a RunPod Serverless **load-balancing** endpoint created
with RunPod's "Deploy from a Docker image". GitHub Actions builds the image (all weights baked in) and pushes it to a
**private** Docker Hub repo. One container holds:
- PersonaPlex 7B + the **V4_A2 step-600 LoRA** (`worker/server/worker_server.py`);
- Trelis Whisper-Hinglish ASR (`worker/router/asr_service.py`);
- the Needle v2 router (`worker/router/router_service.py`; N1 kept for rollback).

**To run the demo on your own GPU, see "Run it locally" in the [top-level README](../README.md)** (`./run_local.sh` at the
repo root, no RunPod). The page itself is `space/` (backend) + `client/` (web source) in this repo.

In the hosted demo the browser never talks to this worker. The Hugging Face Space (`space/`, pushed by `ops/push_space.sh`)
relays the call and holds the RunPod key. Design and history: `docs/DESIGN.md`, `docs/DECISIONS.md`; the deploy steps
are in `docs/GO_LIVE.md`.

**The repo is code only; the image is not.** GitHub rejects files over 100 MB, so the weights come in at build time
(DECISIONS D-BAKE, 2026-10-06). `.github/workflows/build-worker.yml` (Actions → build-worker-image → Run workflow, or push a
`v*` tag) builds from the repo root with `-f worker/Dockerfile` (BuildKit applies `worker/Dockerfile.dockerignore`, a
whitelist) using BuildKit and pushes `docker.io/<DOCKERHUB_USER>/s2s-worker:<sha12>` and `:latest`:
- PersonaPlex 7B (gated, pinned revision): only the 4 files the engine loads, into `/opt/pp` (`S2S_PP_DIR`).
- The V4 adapter (370 MB), the Needle `.cact` (63 MB) and `libneedle.so`, from the private HF repo
  `shivamgupta/s2s-v4-assets`, by `worker/fetch_assets.py` itself (md5-checked against
  `worker/assets_manifest.json`) into `/opt/s2s/assets`.
- Trelis Whisper-Hinglish (public, pinned revision), the Python envs and the cactus-needle lib.

The workflow runs in two phases (the runner disk cannot hold the 16 GB PersonaPlex layer twice): BuildKit builds and
pushes everything except PersonaPlex as `:base-<sha12>` (HF token = BuildKit **secret** `--secret id=hf_token`, never a
build arg); then the 4 PersonaPlex files are downloaded on the runner (token only in that step's env, sent only to
huggingface.co), sha256-checked against the HF LFS ids, and streamed onto the base as one layer with `crane append`.
The token is never in a layer. A plain local `docker build` (`BAKE_PP=1`, the default) bakes everything in one go. At runtime the
worker is fully offline (`HF_HUB_OFFLINE=1`): no HF token, no model cache, no network volume. `fetch_assets.py` and
`resolve_models.py` still run at start and find everything in place (the volume / HF-repo / model-cache fallbacks
remain for a custom setup).

Actions secrets: `HF_TOKEN` (read access to `nvidia/personaplex-7b-v1` and `shivamgupta/s2s-v4-assets`),
`DOCKERHUB_USER`, `DOCKERHUB_TOKEN`. The workflow refuses to push unless the Docker Hub repo exists and is private.

## Behaviour carried over from the live pod1 demo (2026-10-06)
Each item matches the demo exactly and has its own DECISIONS entry there:
- **Input AGC** (D-AGC): speech is brought to −21 dBFS. Gain is capped at +24 dB, with a 3 dB/frame slew and a soft
  limiter. Env: `S2S_INPUT_AGC`, `S2S_AGC_TARGET_DB`, `S2S_AGC_MAX_GAIN_DB`, `S2S_AGC_GATE_DB`,
  `S2S_AGC_NOISE_GAIN_DB`.
- **Noise gate** (D-GATE, essential): raw frames below −55 dBFS become exact zeros, with a 4-frame hangover. Without
  the gate, the model stays silent on a real microphone. Env: `S2S_INPUT_GATE`, `S2S_GATE_OPEN_DB`,
  `S2S_GATE_HANGOVER_FRAMES`.
- **Turn filler** (D-FILL2, D-TOGGLE): when the model has been quiet for 400 ms after an utterance that is not a
  check-line, its input gets a 1 s ticker at −34 dB RMS.
  - Env defaults: `S2S_TURN_FILL` (ticker), `S2S_TURN_FILL_S`, `S2S_TURN_FILL_QUIET_MS`, `S2S_TURN_FILL_DB`.
  - Optional JSON override: `S2S_TURN_FILL_JSON`.
  - Per call: the `turn_fill=ticker|off` websocket query parameter, which the Space sets from its toggle.
  - Mid-call: the in-band control frame `0x08` + `{"type":"turn_fill","mode":"ticker"|"off"}`. The GPU thread
    applies it at the next frame.
- **Session recording** is off by default. `S2S_RECORD_SESSIONS=1` writes to `S2S_SESSIONS_DIR` (default
  `$S2S_LOGS/sessions`).

## Endpoint environment (console → endpoint → Environment Variables)
| var | value | note |
|---|---|---|
| `S2S_SESSION_SECRET` | ≥ 32 chars | **secret**; the same value as the Space secret |
| `S2S_AUDIENCE` | e.g. `hinglish-lb-1` | the same value as the Space variable |
| `S2S_MODE` | `lb` | |
| `PORT`, `PORT_HEALTH` | `80`, `80` | set both explicitly (runpod/docs#853) |
| `CALL_MAX_S` | `300` | optional; keep it below 330 (the LB processing cap) |
| `S2S_TURN_FILL` | `ticker` | optional; the Space's toggle overrides it per call |
| `S2S_RECORD_SESSIONS` | `0` | optional |

No `HF_TOKEN` and no **Model** field: the weights are in the image. The endpoint needs a Docker Hub **registry
credential** (RunPod → Settings → Container Registry Auth) to pull the private image.

## Layout
| path | what |
|---|---|
| `worker/Dockerfile`, `worker/Dockerfile.dockerignore` | image (BuildKit; weights baked); built from the repo root (`docker build -f worker/Dockerfile .`) |
| `.github/workflows/build-worker.yml` | GitHub Actions: build + push to private Docker Hub |
| `worker/entrypoint.sh`, `worker/stack.sh` | lb: `stack.sh` = worker_server → (engine loaded) ASR → router; one dies, all die |
| `worker/fetch_assets.py`, `worker/assets_manifest.json` | the V4 adapter + Needle weights, fetched at build time (md5-checked) |
| `worker/resolve_models.py`, `worker/paths.py` | PersonaPlex location; every path and port from env |
| `worker/server/` | `worker_server.py` (public `/ping` `/status` `/session/*` `/api/chat`), `core.py` (demo fixes), `engine.py` |
| `worker/router/`, `worker/needle/`, `worker/needle_v2/`, `worker/hinglish/`, `worker/vendor/moshi/` | router + ASR, Needle N1 runtime subset, Needle v2 router code, LoRA merge, moshi |
| `run_local.sh` (repo root) | run everything on one local GPU (D-LOCAL): `setup` (client build + venvs + downloads), `run` (worker + page) |
| `worker/local/` | dev-box env files (`pod1.env`, `runpod2.env`) + `local/run_local.sh mock` or `gpu` (stack outside Docker) used by the CPU tests |
| `worker/tests/` | CPU tests: `test_worker_mock.py`, `test_fixes.py`, `check_dockerfile.py` (excluded from the image) |
| `common/` | session token + records (V4, 162 records; the demo uses the 5 Needle agent types) |

## Router
The router is **Needle v2** (2026-10-06, DECISIONS D-ROUTER-V2): `worker/needle_v2/schema/router_v2.py` (+ the sibling
`worker/needle_v2/numconv/`, path-edit copies of `hinglish/needle_v2`), weights `needle_v2/tuned_full.cact` from the
assets repo (baked at `/opt/s2s/assets/needle_v2/tuned_full.cact`). It gets the raw Trelis text, executes
`server_args`, and a call it cannot fill safely (9-digit phone, unknown order ID, ...) is not executed: it is sent as
the action stage `needs_clarification`. A call without a record reference (e.g. "cancel the order") goes to the
record's active order. Demo agent types: the 5 of `common/session.py` (v2 also knows bank/telecom; the stub tools and
the demo record list do not).

Rollback to Needle N1 without a rebuild: set the endpoint env `S2S_ROUTER=n1` (N1 weights stay in the image).

## Rebuilding
Run the workflow again (or push a `v*` tag). The new `:<sha12>` tag must then be set on the RunPod endpoint (or keep
the endpoint on `:latest` and start new workers). Build args are not passed by the workflow, so edit the `ARG`
defaults in `worker/Dockerfile` (for example `TRELIS_BF16=1` saves about 3 GB per pull; untested).
