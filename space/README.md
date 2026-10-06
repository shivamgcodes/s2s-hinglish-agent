---
title: Hinglish Full-Duplex Agent (demo)
emoji: 📞
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Hinglish support-agent voice demo on RunPod Serverless
---

# Hinglish full-duplex agent: Space front

This Space serves the web client and relays the call to a GPU worker on RunPod Serverless. The GPU worker
(PersonaPlex 7B + V4 Hinglish LoRA + Trelis ASR + Needle router) scales to zero; opening the page and pressing
Connect wakes it. The first call after a quiet period waits through a cold start (about 1-2 min, up to 8 min on a
fresh host).

Open the direct URL `https://<owner>-<space>.hf.space/`, not the `huggingface.co/spaces/...` page: the iframe
there can block the microphone (DESIGN R8). Use headphones.

Design: `DESIGN.md` sections 2 and 5 in the source folder (`hinglish/deploy_serverless` on the laptop); deploy
steps: its `GO_LIVE.md`. The GPU worker is the `s2s-worker` repo.

**Turn filler toggle** (top right, "Turn filler: Ticker | Off"; 2026-10-06, same as the pod1 demo): when the agent
pauses after a normal utterance, the worker feeds the *model's* input a 1 s soft ticker so it yields the turn; you never
hear it. `GET/POST /api/filler?mode=ticker|off`. The setting is one value per Space process (initially
`S2S_TURN_FILL_DEFAULT`): it applies at once to a live call (in-band control frame on the relay) and to later calls.
The browser playback buffer is the loosened one from the demo (320 ms start, drops only above ~1.9 s queued).

**Script panel** (D-SCRIPT-PANEL, 2026-10-06): "Script: what to say", to the right of the live text, shows the expected
conversation for the selected record + pairing. It is the V4 synthetic call for that record (`common/data/scripts_v4.json`,
built by `ops/build_scripts.py` in the source folder). Customer lines ("you say") are what you speak; agent lines are
dimmed and show what the model is expected to say. `check` / `write` / `read` mark the router trigger line, the
agent's confirmation of a change, and a fact read from the record. The panel says whether the call was in the training
set or held out (val/test). A rough heuristic highlights the next customer line as the model's text advances. Three
pairings have no call (`food_18` g1, `sub_12` g2/g3): the panel says "no script for this pairing".
`GET /api/script/{record_id}?pairing=gN`.

## Run it locally (your own GPU, no RunPod)

Clone this repo **next to** [`s2s-worker`](https://github.com/shivamgcodes/s2s-worker) and follow its README section
"Run it locally". In short:
```bash
git clone https://github.com/shivamgcodes/s2s-worker && git clone https://github.com/shivamgcodes/s2s-space
cd s2s-worker && export HF_TOKEN=hf_...      # accept https://huggingface.co/nvidia/personaplex-7b-v1 first
./run_local.sh setup && ./run_local.sh run   # then open http://localhost:7860
```
`run_local.sh` starts the worker stack and this backend with `S2S_MODE=local`. In that mode the backend skips RunPod
entirely. It claims the worker at `S2S_WORKER_URL` (default `http://127.0.0.1:8000`) with the same session token and
websocket relay as in production, but without a RunPod key, Bearer header, worker pin or keepalive. Both sides get the
same generated `S2S_SESSION_SECRET` (audience `local`). The per-IP / per-day limits default to effectively off.
To run this backend by hand against a worker you started yourself:
`S2S_MODE=local S2S_WORKER_URL=http://127.0.0.1:8000 S2S_SESSION_SECRET=<the worker's> python -m app.main`.
On a remote GPU box, open the page through `ssh -L 7860:localhost:7860 …` and http://localhost:7860: browsers block
the microphone on plain-http non-localhost pages.

## How it works

```
browser ──https/wss (same origin)──> this Space ──Bearer RUNPOD_API_KEY──> RunPod LB endpoint ──> GPU worker
```

- `POST /api/session` checks the passcode and limits, then wakes the endpoint and claims one worker.
  - LB mode polls `GET /status`, then sends `POST /session/claim` with a session token.
  - Queue mode sends `/run`, then polls `/status/{job}` for the worker's IP and port.
- The browser polls `GET /api/session/{sid}` and shows "warming up" until the state is `ready`.
- `wss://<space>/api/chat?sid=…` is relayed frame for frame to the worker. The RunPod key and the HMAC session token
  never reach the browser.

## Where this repo comes from

`ops/make_repos.sh` in the source folder assembles it: `space/` without `tests/`, plus `common/` (staged by
`space/stage_common.sh`) and `static/` (the client built from `client/` source with `npm ci && npm run build`).
This repo already contains the built client, so the Space build needs no Node.

## Settings (Space -> Settings -> Variables and secrets)

| name | kind | default | meaning |
|---|---|---|---|
| `RUNPOD_API_KEY` | **secret** | — | RunPod API key (Bearer for the LB proxy and the queue API) |
| `S2S_SESSION_SECRET` | **secret** | — | HMAC key, ≥ 32 chars, the SAME value as the worker's env |
| `S2S_PASSCODE` | **secret** | empty | if set, Connect asks for it (cost brake, R7) |
| `RUNPOD_ENDPOINT_ID` | variable | — | the endpoint id |
| `S2S_MODE` | variable | `lb` | `lb` or `queue` (must match the worker); `local` = no RunPod, see "Run it locally" |
| `S2S_WORKER_URL` | variable | `http://127.0.0.1:8000` | local mode only: the worker's port |
| `S2S_AUDIENCE` | variable | endpoint id | token `aud`; must equal the worker's `S2S_AUDIENCE` |
| `MAX_CONCURRENT_CALLS` | variable | 1 | set equal to the endpoint's max workers |
| `RATE_PER_IP_PER_HOUR` | variable | 6 | session creations per client IP per hour |
| `MAX_CALLS_PER_DAY` | variable | 100 | all IPs, per UTC day |
| `CALL_MAX_S` | variable | 300 | the Space force-closes at this + 15 s (the worker's own cap wins) |
| `S2S_TOKEN_TTL_S` | variable | 120 | token lifetime |
| `WAKE_POLL_S` / `WAKE_TIMEOUT_S` / `WAKE_HTTP_TIMEOUT_S` | variable | 3 / 600 / 130 | wake polling |
| `CLAIM_TTL_S` | variable | 90 | a `ready` session that is not connected within this is released |
| `ABANDON_S` | variable | 120 | a `waking`/`busy` session whose browser stopped polling for this long is cancelled (end reason `abandoned`); 0 = off (REVIEW-space-3). Mobile browsers pause timers in the background, so a phone user who switches apps during a long cold start may need to reconnect; raise it or set 0 if that matters |
| `UPSTREAM_OPEN_TIMEOUT_S` | variable | 60 | upstream websocket open timeout |
| `S2S_TOKEN_IN_QUERY` | variable | 0 | LB only: 1 = also send the session token as `?token=` (claim, websocket upgrade, release), for when the RunPod proxy drops the `X-S2S-Token` header (CRIT-3, O-W1) |
| `KEEPALIVE_S` / `KEEPALIVE_PATH` / `KEEPALIVE_TIMEOUT_S` | variable | 20 / `/status` / 10 | LB keepalive during a call; `KEEPALIVE_S=0` turns it off (hold test #3) |
| `QUEUE_EXEC_TIMEOUT_MS` / `QUEUE_TTL_MS` / `QUEUE_POLL_S` | variable | 900000 / 1200000 / 2 | queue mode job policy and polling |
| `RUNPOD_LB_URL` / `RUNPOD_API_URL` | variable | `https://{id}.api.runpod.ai` / `https://api.runpod.ai/v2/{id}` | overrides (tests point them at localhost) |
| `S2S_ALLOW_FREE_PROMPT` | variable | 0 | 1 shows the stock free-prompt session (the worker must allow it too) |
| `S2S_SELFTEST` | variable | 0 | 1 enables `GET /api/selftest/outbound?host=&port=&passcode=` (needs `S2S_PASSCODE`) |
| `S2S_BUILD` | variable | empty | free text shown in `/api/config` |
| `S2S_TURN_FILL_DEFAULT` | variable | `ticker` | initial turn-filler mode (`ticker` or `off`); the page toggle changes it |
| `LOG_LEVEL` | variable | INFO | |

**Visibility.** Make the Space public or protected, never private. A private Space returns 404 to visitors, and
also to their websockets (R1).

## Launchers

- (a) **Docker SDK** (this README's header). This needs a paid HF plan (PRO) for a personal account ([HF-OV]).
- (b) **Gradio SDK** (`app.py`), for a free account's ZeroGPU Space. Change the header to
  `sdk: gradio`, `sdk_version: <current>`, `app_file: app.py`, and drop `app_port`. The shim runs the same aiohttp app on
  :7860 without any Gradio UI. This is untested (R3).
- (c) **Any container host**: `docker build -t s2s-space space/ && docker run -p 7860:7860 --env-file … s2s-space`.
  Put TLS in front, because browsers need https for the microphone.

## Routes

`/` (client) · `/api/config` · `/api/records[/{id}]` · `/api/script/{id}?pairing=` · `/api/samples` · `/samples/*` · `POST /api/session` ·
`GET|DELETE /api/session/{sid}` · ws `/api/chat?sid=…` · `GET|POST /api/filler` · `/metrics?sid=` · `/api/diag?sid=` (only for an active call,
at most 1 per 10 s) · ws `/ws-echo` (test #1) · `/healthz`.

## Local test (no GPU, no RunPod)

The tests are not part of this repo; they live in the source folder (`space/tests/`, `tests/`). Run them on pod1
in `/root/dep2` with `bash tests/run_all_cpu_pod1.sh`.
