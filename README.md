# Hinglish full-duplex voice agent (`s2s-hinglish-agent`)

This repo holds a speech-to-speech customer-support agent that speaks Hinglish. It is a full-duplex model, so it can
listen and speak at the same time:
- **PersonaPlex 7B** with the **V4 Hinglish LoRA** speaks and listens;
- **Trelis Whisper-Hinglish** transcribes the caller;
- the **Needle v2** router turns what was said into tool actions (change the address, cancel the order, ...).

The repo holds all of it: the GPU worker, the web page and its backend, and the scripts that build and deploy them.

> **Private until release.** This repo replaces the two earlier repos `shivamgcodes/s2s-worker` (GPU worker) and
> `shivamgcodes/s2s-space` (the Space contents) (D-MONOREPO, 2026-10-06). The model cards on Hugging Face link here.

## What runs where

```
browser ──https/wss──> Hugging Face Space  (space/ backend + client/ page, CPU; holds the RunPod key)
                              │ wss + Bearer
                              v
                       RunPod Serverless load-balancing endpoint  (worker/ image, one GPU per call, scales to 0)

local:   browser ──> run_local.sh: space/ backend (S2S_MODE=local) ──> worker/ stack on your own GPU
```

| folder | what | runs on |
|---|---|---|
| `worker/` | GPU worker: PersonaPlex + LoRA server, Trelis ASR, Needle router, supervisor (`stack.sh`), `Dockerfile` | RunPod Serverless (image), or your GPU (`run_local.sh`) |
| `space/` | aiohttp backend: serves the page, session API, wakes the endpoint, relays the call websocket | Hugging Face Docker Space (CPU), or locally |
| `client/` | web page source (React + TypeScript + Vite; a small fork of the PersonaPlex/moshi client) | built to `client/dist`, served by `space/` |
| `common/` | shared by worker and space: HMAC session token, session config, demo records + call scripts | both |
| `ops/` | build, stage and deploy scripts + ops notes | your machine / a build box |
| `tests/`, `worker/tests/`, `space/tests/` | CPU test suites (no GPU, no network), GPU/live probes | any Linux box |
| `docs/` | design, decision log, go-live runbook (internal history) | |
| `.github/workflows/build-worker.yml` | GitHub Actions: builds the worker image and pushes it to private Docker Hub | GitHub runners |
| `run_local.sh` | the whole demo on one local NVIDIA GPU, no RunPod, no Space | your GPU box |
| `run_all_cpu_tests.sh` | every CPU suite in one go | any Linux box with the venvs |

Models (private Hugging Face repos): [`shivamgupta/personaplex-hinglish-v4-lora`](https://huggingface.co/shivamgupta/personaplex-hinglish-v4-lora)
and [`shivamgupta/needle-hinglish-router-v2`](https://huggingface.co/shivamgupta/needle-hinglish-router-v2). The base
model is [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) (gated).

## Run it locally (one NVIDIA GPU, no RunPod, no Space)

`run_local.sh` runs the whole demo on your own machine:
- it starts the worker stack (PersonaPlex 7B + V4 LoRA, Trelis ASR, Needle v2 router) as plain processes in uv venvs;
- it starts the page backend from `space/` in `S2S_MODE=local`, which talks to the worker directly.

There is no RunPod and no API key. The web page is the same one the hosted demo uses.

**Requirements**
- Linux x86_64 with one NVIDIA GPU. 24 GB works but is tight (the stack measured 23.4 of 24.5 GB on a 4090); 32 GB or
  more is comfortable. Tested on an RTX 5090 (32 GB).
- An NVIDIA driver for CUDA 12.8 or newer. The script picks the torch wheels itself: `cu130` if `nvidia-smi` reports
  CUDA ≥ 13.0, else `cu128` (torch 2.11.0). Set `TORCH_INDEX=cu128|cu130` to override. Tested: cu128, driver 570.
- About 30 GB of disk (a few GB more during setup): venvs ~7 GB, PersonaPlex 16.4 GB, Trelis 5.8 GB, LoRA + router
  0.45 GB. The Needle native lib (~1 MB) goes to `~/.cache/cactus-needle`.
- System packages: `sudo apt install libopus0 ffmpeg gcc curl`. `uv` is used if present, otherwise the script installs
  it into the run directory.
- **Node.js ≥ 18 with npm.** `setup` builds the web page once (`client/` → `client/dist`). Or point `S2S_STATIC` at
  an already built client.
- A Hugging Face account and read token. PersonaPlex is gated: first open
  https://huggingface.co/nvidia/personaplex-7b-v1 and accept the NVIDIA Open Model License.

**Steps**
```bash
git clone https://github.com/shivamgcodes/s2s-hinglish-agent
cd s2s-hinglish-agent
export HF_TOKEN=hf_...                     # your read token; used by setup only, never stored
./run_local.sh setup                       # once: web client build + venvs + downloads (~30 GB)
./run_local.sh run                         # worker + page; Ctrl-C stops both
```
1. Open **http://localhost:7860**.
2. Pick an agent type, a record and a pairing, then press **Connect**. Use headphones.
3. Wait for the models to load. The first start loads and compiles for about 2-5 min; the page shows "warming up" and
   the terminal prints `READY` when the worker is ready.
4. Speak as the customer. The "Script: what to say" panel, to the right of the live text, shows the expected
   conversation for that record.

`./run_local.sh check` lists what is in place or missing.

**What setup downloads**
- `nvidia/personaplex-7b-v1`, pinned revision `fdaf4090`, 4 files only. Needs your token.
- `Trelis/whisper-hinglish-preview`, pinned revision, public.
- The V4 LoRA from `shivamgupta/personaplex-hinglish-v4-lora` (`config.json`, `lora.safetensors`).
- The Needle v2 router weights from `shivamgupta/needle-hinglish-router-v2` (`tuned_full.cact`).
- The Needle native library (the `cactus-needle` package fetches it).

Do not use the Docker Hub image that the RunPod endpoint runs. It is private and contains the gated base weights.
`worker/Dockerfile` also expects a private assets repo, so for a local run use `run_local.sh`.

**Settings (env vars, all optional)**

| var | default | meaning |
|---|---|---|
| `S2S_LOCAL_HOME` | `./.local-run` | venvs, models, logs (`logs/stack.log`, `logs/space.log`, per-process logs) |
| `S2S_PP_DIR` / `S2S_ASR_DIR` | download | use an existing PersonaPlex / Trelis snapshot dir instead of downloading |
| `S2S_LORA_REPO` (`S2S_LORA_SUBDIR`) / `S2S_ROUTER_REPO` (`S2S_ROUTER_FILE`) | the two repos above | other HF sources for the LoRA / router weights |
| `S2S_ADAPTER` / `S2S_NEEDLE_V2_WEIGHTS` | download | local LoRA dir (`config.json` + `lora.safetensors`) / local `.cact` file |
| `WORKER_PORT` | 8000 | worker HTTP + websocket port. It binds all interfaces, and every call needs a session token signed with the local secret. The worker also uses 8999, 8996 and 8995 on localhost |
| `SPACE_PORT` | 7860 | web page port |
| `HOST` | 127.0.0.1 | address the page binds to; `0.0.0.0` exposes it on the LAN (see the microphone note) |
| `S2S_SPACE_DIR` / `S2S_STATIC` | `./space` / `./client/dist` | page backend dir / built web client |
| `S2S_TURN_FILL_DEFAULT` | ticker | initial turn filler (the page's top-right toggle changes it) |
| `S2S_PASSCODE` | empty | ask for a passcode on Connect |
| `CALL_MAX_S` | 300 | call length cap (the model's context ends after about 3.5 min anyway) |

**Troubleshooting**
- **No microphone prompt, or Connect does nothing, on a remote GPU box.** Browsers allow the microphone only on https
  or on `localhost`. Forward the port and open the page as localhost:
  `ssh -L 7860:localhost:7860 user@gpu-box`, then open http://localhost:7860. Don't open `http://<ip>:7860`.
- **The agent never answers you.** The worker has an input noise gate (D-GATE):
  - microphone frames below −55 dBFS become exact silence;
  - the input AGC lifts quieter speech to −21 dBFS (up to +24 dB).

  A very quiet or far-away microphone can stay under the gate, so speak closer or raise the input gain. Use
  headphones: the agent's voice leaking into the microphone confuses the model.
- **The ticker.** When the agent pauses after an utterance, the worker feeds a soft 1 s ticker into the *model's*
  input so that it hands the turn back.
  - You never hear it.
  - The round "ticker (model only)" indicator shows when it fires.
  - The top-right **Turn filler: Ticker | Off** toggle switches it, also during a call. Try Off if the agent cuts in
    or rambles.
- **`setup` fails on PersonaPlex with 401/403.** Accept the license on the model page with the same account as the
  token.
- **`setup` says npm is missing.** Install Node.js 18 or newer, or build `client/` elsewhere and set `S2S_STATIC`.
- **`torch sees no GPU`.** The driver is older than the chosen wheels. Re-run with `TORCH_INDEX=cu128`, or update the
  driver.
- **CUDA out of memory, or the worker dies while loading.** Something else is using the GPU. The stack needs ~23.4 GB
  free (`nvidia-smi`).
- **The page says "no worker became ready".** See `.local-run/logs/stack.log`. The page backend waits up to 20 min.
- **Port already in use.** Set `WORKER_PORT` / `SPACE_PORT`, or stop the other process.
- **Only 5 agent types are offered** (food delivery, e-commerce, cab, subscription, airport): these are the ones the
  router supports. The records, customers and orders are synthetic, and actions run against stub tools on an
  in-memory copy of the record.

## Build the web client

```bash
cd client && npm ci && npm run build       # -> client/dist (Node 20.12 used so far; .nvmrc)
```
`client/.env.production` sets `VITE_QUEUE_API_PATH` (a same-origin path, not a secret). `client/dist` is not
committed. The Space push and `run_local.sh setup` build it.

## Deploy (hosted demo)

**1. Worker image (GitHub Actions → private Docker Hub).**
- Actions → **build-worker-image** → Run workflow (or push a `v*` tag).
- Result: `docker.io/<DOCKERHUB_USER>/s2s-worker:<sha12>` and `:latest`.
- The workflow builds from the repo root with `worker/Dockerfile`. `worker/Dockerfile.dockerignore` is a whitelist,
  so only `worker/` and three `common/` files enter the image.
- It runs in two phases:
  1. BuildKit builds `:base-<sha12>` with everything except PersonaPlex.
  2. The 4 PersonaPlex files are sha256-checked and appended as one reproducible layer with `crane`.
- Actions secrets:
  - `HF_TOKEN`: read access to `nvidia/personaplex-7b-v1` and the private `shivamgupta/s2s-v4-assets`. It is used
    only as a BuildKit secret and in one download step.
  - `DOCKERHUB_USER`, `DOCKERHUB_TOKEN`.
- The run refuses to push unless the Docker Hub repo exists and is private.
- Details: `worker/README.md`.

**2. RunPod endpoint.**
- Set the new `:<sha12>` tag on the endpoint's template: Serverless → endpoint → Manage → Edit → container image.
- The endpoint env (`S2S_SESSION_SECRET`, `S2S_AUDIENCE`, `S2S_MODE=lb`, `PORT=80`, `PORT_HEALTH=80`) is in
  `worker/README.md`.
- No HF token or model cache is needed: the weights are in the image. The endpoint needs a Docker Hub registry
  credential.

**3. Hugging Face Space.**
- `ops/push_space.sh` assembles the Space tree from this checkout and uploads it with `huggingface_hub`:
  - `space/` minus tests;
  - `common/` (token, session, records, scripts);
  - `client/dist` → `static/`.
- It runs `ops/stage_space.sh` for the assembly and `ops/scan_secrets.sh` before uploading.
- It is a dry run unless you pass `--yes`.

```bash
PATH=/path/to/node/bin:$PATH bash ops/push_space.sh --build-client --out /tmp/space-tree   # dry run: build + stage + scan
HF_TOKEN_WRITE=... bash ops/push_space.sh --yes                                            # upload (token never printed)
```
The Space's secrets (`RUNPOD_API_KEY`, `S2S_SESSION_SECRET`) and variables (`RUNPOD_ENDPOINT_ID`,
`S2S_AUDIENCE`, limits) live only in the Space settings: `space/README.md` lists them. Never put them in this repo.

The full runbook (ordering, checks, costs, quick fixes) is `docs/GO_LIVE.md`. Why things are the way they are:
`docs/DECISIONS.md`.

## Tests

```bash
bash run_all_cpu_tests.sh       # 10 CPU suites (no GPU, no network); logs in .test-out/
```
The defaults are the dev box's venv paths. Point them at yours with `PP_PY`, `SPACE_PY`, `NEEDLE_PY` and
`S2S_LOCAL_ENV` (see the script header). A suite whose prerequisite is missing is reported as SKIP. GPU and live
probes are in `tests/` (`gpu_smoke.sh`, `gpu_call.py`, `latency_probe.py`, ...); each documents itself.

## License

There is no top-level license file yet; one will be added before release. Parts carry their own licenses:
- `client/LICENSE` (MIT, from the upstream PersonaPlex / moshi web client this page is forked from);
- `worker/vendor/moshi/LICENSE.moshi` and `LICENSE.audiocraft`.

The models have their own licenses on their Hugging Face pages. PersonaPlex is under the NVIDIA Open Model License.
