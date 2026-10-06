# S2S serverless: start here

> **2026-10-06 (D-MONOREPO): one repo.** Everything below is now in the private GitHub monorepo
> `shivamgcodes/s2s-hinglish-agent`. It replaces `s2s-worker` and `s2s-space`, and its README is the overview.
> - `bash ops/export_monorepo.sh <checkout>` exports this folder into it;
> - `ops/push_space.sh` stages and uploads the Space from it;
> - `run_local.sh` and `run_all_cpu_tests.sh` are at the root.
>
> In the monorepo, this file is `docs/SERVERLESS_NOTES.md`.

> **2026-10-06 update (D-BAKE): the weights are now baked into the image.** GitHub Actions builds
> `docker.io/shivamgupta579/s2s-worker:<sha12>` (private Docker Hub) and RunPod uses "Deploy from a Docker image" with a
> registry credential: no RunPod model cache, no Model field, no network volume, no HF token on the endpoint. Where
> this README still describes the model cache / GitHub-release build, **GO_LIVE.md and DECISIONS D-BAKE win.**

> **2026-10-06: read `GO_LIVE.md` first.**
> - The live pod1 demo's fixes are ported: AGC, the noise gate, the ticker turn filler + its toggle, and the loose
>   playback buffer.
> - The adapter is now **V4_A2 step 600**, fetched at worker start from a private HF repo.
> - The worker is built by RunPod from the private GitHub repo `shivamgcodes/s2s-worker`. The Space contents are in
>   `shivamgcodes/s2s-space`.
> - The registry-image route below (BUILD.md, build_and_push.sh, prepare_build_context.sh, baked V3 assets) is
>   superseded. DECISIONS.md "D-GOLIVE-PREP" has the details.

This is the serverless version of the DEP1 Hinglish full-duplex agent demo (PersonaPlex 7B + V3 LoRA + Mimi, Trelis
Whisper-Hinglish ASR, Needle router). DEP1 runs on one always-on pod. This version:
- serves the web client from a CPU front on a **Hugging Face Space** (`space/`, `client/`);
- runs the GPU part on a **RunPod Serverless load-balancing endpoint** (`worker/`). It scales to zero, and each worker
  takes one call;
- keeps a **queue-endpoint fallback** that uses the same image (`S2S_MODE=queue`).

**Status (2026-10-05): built, not deployed.**
- **Verified on CPU:** every test passes (list in "CPU tests" below).
- **Verified on the GPU:** the real worker ran once, outside Docker, on runpod2's RTX 3090:
  - `/ping` turned 200 after 39.6 s;
  - a 75 s call had step p95 69 ms (`results/integration.md`).
- **Never done:** no `docker build` has run, no image was pushed, and no endpoint or Space exists.
- **Dry runs only:** `build_and_push.sh` and `create_endpoint.sh` were dry-run by the earlier ops stages, with fake
  `curl`/`docker`/`git`/`hf`/`npm` first on PATH, which made 0 external calls. The critic stage could not re-run them,
  because a permission block stopped it. They have never run for real, so read their printed plan before you add
  `--push` or `--yes`.

Two corrections to the original brief (DESIGN section 0):
- **The Space is not free.** A Docker (or Gradio) Space needs a paid HF plan (PRO for a personal account). Only Static
  Spaces are free, and a Static Space cannot hold the RunPod key or relay audio. cpu-basic hardware then costs nothing
  per hour, but it sleeps after 48 h without visitors and wakes on the next visit.
- **The browser never talks to the worker.** RunPod documents only `Authorization: Bearer <API key>` for LB HTTP and
  websockets. A browser cannot set that header on a websocket, and the key must never reach the browser. So the Space
  backend relays the websocket in **both** modes and adds the key server-side. The client still opens a same-origin
  `/api/chat`, so the client fork is tiny.

## Architecture

```
                       HF Space (CPU basic, Docker SDK, :7860)                RunPod Serverless
                       space/app: aiohttp, no Gradio                          one worker = one GPU = one call
 ┌─────────┐ https    ┌───────────────────────────────────────┐
 │ Browser │────────> │ static client (client/dist)           │
 │ (stock  │ POST     │ /api/session: passcode, rate, caps,   │
 │ Persona │ /api/    │   mint HMAC token (S2S_SESSION_SECRET)│
 │ Plex TS │ session  │ wake task: poll until a worker is up  │
 │ client) │          │ relay: /api/chat <-> upstream ws      │
 │         │ wss      │ keepalive (LB): GET /status every 20 s│
 │         │<───────> │   pinned to the claimed worker        │
 └─────────┘ same     └──────────────┬────────────────────────┘
             origin                  │
   LB mode (primary) ────────────────┤ https/wss + Authorization: Bearer RUNPOD_API_KEY
                                     │ + X-S2S-Token (or ?token=) + X-Runpod-Worker-Id: strict <id>
                                     v
                       https://<ENDPOINT_ID>.api.runpod.ai  (RunPod LB proxy, TLS)
                                     │ http/ws to $PORT=80
                                     v
                     ┌──────────────────────────────────────────────────────────────┐
                     │ worker image (worker/Dockerfile, ~14 GB)                     │
                     │  entrypoint.sh -> stack.sh (one dies => all die)             │
                     │   worker_server.py :80  /ping /status /session/claim         │
                     │      /session/release /api/chat(ws)   verifies the token,    │
                     │      PersonaPlex 7B + V3 LoRA merged + 2xMimi  refuses a 2nd │
                     │      127.0.0.1:8999 internal API (inject/mute, /metrics)     │
                     │   asr_service.py    127.0.0.1:8996  Trelis Whisper-Hinglish  │
                     │   router_service.py 127.0.0.1:8995  Needle .cact on CPU      │
                     │   rp_handler.py     (queue mode only, runpod SDK)            │
                     │ PersonaPlex weights: RunPod model cache (or network volume)  │
                     │ baked: Trelis, V3 LoRA (370 MB), Needle .cact + lib, code    │
                     └──────────────────────────────────────────────────────────────┘
                                     ^
   Queue mode (fallback) ────────────┤ 1) POST api.runpod.ai/v2/<ID>/run (Bearer) wakes a worker
                                     │ 2) handler publishes {public_ip, tcp_port} via progress_update;
                                     │    the Space polls /status/<job>
                                     │ 3) Space relays to ws://<public_ip>:<RUNPOD_TCP_PORT_8765>
                                     │    (plaintext, X-S2S-Token is the only auth)
```

### LB mode (primary, `S2S_MODE=lb`)
1. The browser loads the Space and sends `POST /api/session`. The Space checks the passcode, the per-IP rate, the
   daily cap and `MAX_CONCURRENT_CALLS`, and returns `{sid, state: "waking"}`.
2. The wake task polls `GET https://<ID>.api.runpod.ai/status` every 3 s, for up to 600 s. The proxy routes only to
   workers whose `/ping` is 200 (204 means initializing).
3. On the first ready answer, the Space mints a token and sends `POST /session/claim`. The worker answers 200 with
   its `worker_id`, or 409 `busy` if that worker is in a call; on 409 the Space retries every 3 s. State: `ready`.
4. The browser opens `wss://<space>/api/chat?sid=…`. The relay mints a fresh single-use token and opens the upstream
   websocket with the Bearer key, the token and strict affinity to the claimed worker (open timeout 60 s).
5. The worker checks the token (HMAC, expiry, mode, audience, sid, config, replay) and then runs the DEP1 handler
   unchanged: prompt phase about 9 s, then 0x00 handshake, then 0x01/0x02/0x07 frames.
6. During the call the Space sends a pinned `GET /status` every 20 s, so the worker is not scaled down if the
   open websocket does not count as work (R2).
7. The call ends on Disconnect, context full (about 221 s), the 300 s cap (`CALL_MAX_S`, under the 330 s LB cap), or
   an error. The worker frees itself for the next claim, and idles out after the endpoint idle timeout (120 s).
   If the worker vanishes mid-call, the browser gets a `session_end` with reason `worker_lost`.

### Queue mode (fallback, `S2S_MODE=queue` on both sides, separate endpoint, same image)
1. The Space sends `POST /v2/<ID>/run` with the session config. That job wakes a worker.
2. In the worker, `entrypoint.sh` starts the stack in the background and then `rp_handler.py`. The handler waits for
   `/ping` 200, claims the sid internally, reads `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_8765`, and publishes them
   with `progress_update`.
3. The Space polls `/v2/<ID>/status/<job>` every 2 s, then relays to `ws://ip:port/api/chat` with the token.
   No keepalive is needed: a running job keeps the worker alive.
4. When the session ends, the handler returns the session summary as the job output. If the browser leaves before
   connecting, the Space cancels the job.
5. Caveats: the RunPod-to-Space leg is plaintext (R6), the Space must be able to reach a random non-443 port (R17),
   and the endpoint needs "Expose TCP ports: 8765" and an execution timeout of 900 s.

### Security model
- Secrets live only in HF Space *Secrets* and RunPod endpoint env: `RUNPOD_API_KEY` (Space only),
  `S2S_SESSION_SECRET` (both sides, same value), `S2S_PASSCODE` (Space). No secret is in the image, the code or any repo.
- The HF token for the gated `nvidia/personaplex-7b-v1` goes only into the endpoint's **Model** field (model cache).
  The image never sees it.
- The worker accepts only tokens minted by our Space (HMAC-SHA256 with the shared secret, 120 s TTL, single use on the
  upgrade), and it refuses a second concurrent call with 409.
- `S2S_SESSION_SECRET`, `S2S_MODE` and `S2S_AUDIENCE` must be **equal** on both sides, or every call gets 401.

## Step-by-step deploy (details: `ops/BUILD.md`, `ops/ENDPOINT_SETTINGS.md`, `ops/space_push.md`, `ops/ENV.md`)

Every ops script prints its plan by default and acts only with an explicit flag (`--push`, `--yes`).

**1. Stage the build context (runpod2).**
```bash
cd /root/deploy_serverless
bash ops/build_client.sh --copy-only && bash space/stage_common.sh   # stage the Space first, so the bundle has it
bash ops/prepare_build_context.sh --bundle /root/s2s_build_context.tar
```
- The baked assets (V3 LoRA, Needle `.cact`, the Needle native library, `records_v4.json`) are in runpod2
  `worker/build/` and are **not** in the laptop or pod1 code copies.
- If runpod2 is gone, use the parked copy on pod1:
  `runpod:/workspace/hinglish/deploy_serverless_assets/s2s_build_assets.tar` (466 MB, md5
  `f620575d6357e9a20fd743213c803c0b`). See `ops/BUILD.md`, "If runpod2 is gone".

**2. Build and push the image** from a docker host with 60 GB or more of disk. The pods have no docker daemon.
```bash
scp runpod2:/root/s2s_build_context.tar . && mkdir s2s && tar -xf s2s_build_context.tar -C s2s && cd s2s
docker login                                   # use a PRIVATE registry repo
IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh          # read the plan
IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh --push   # about 14 GB, 20-40 min
```

**3. Create the RunPod endpoint** (LB). Use the script or the console; every setting is in `ops/ENDPOINT_SETTINGS.md`.
```bash
export S2S_SESSION_SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))")   # keep it for the Space
export S2S_AUDIENCE=hinglish-lb-1 IMAGE=docker.io/<you>/s2s-worker TAG=v1
bash ops/create_endpoint.sh --mode lb --max-workers 1            # read the request
read -s RUNPOD_API_KEY && export RUNPOD_API_KEY
bash ops/create_endpoint.sh --mode lb --max-workers 1 --yes      # prints ENDPOINT_ID
```
Then, in the console:
- **Model** field: `nvidia/personaplex-7b-v1` plus a read-only, fine-grained HF token for that repo. There is no API
  for this step. If an LB endpoint has no Model field, follow `ops/NETWORK_VOLUME.md`.
- Check: endpoint type Load Balancer; GPU priority A6000/A40 (48 GB), then 4090 PRO (24 GB), then L40/L40S. **Not**
  the 24 GB L4/A5000/3090 pool.
- Check: CUDA 13.0 and newer; active workers 0; max workers 1-2; request count scaler 1; idle timeout 120 s;
  FlashBoot on; container disk 40 GB; expose HTTP port 80.
- Env: `S2S_MODE=lb`, `S2S_SESSION_SECRET`, `S2S_AUDIENCE=hinglish-lb-1`, **`PORT=80`, `PORT_HEALTH=80`**,
  `CALL_MAX_S=300`, `CLAIM_TTL_S=90`.
- **Do the spend controls the same day** (ENDPOINT_SETTINGS section 4).

**4. Create the HF Space** (`ops/space_push.md`).
- New Space, SDK **Docker**, hardware CPU basic, visibility **Public** (or Protected). **Never Private**: a private
  Space returns 404 to visitors and their websockets.
- Before the first push, set:

  | kind | name | value |
  |---|---|---|
  | Secret | `RUNPOD_API_KEY` | RunPod API key (a dedicated one) |
  | Secret | `S2S_SESSION_SECRET` | same as the endpoint |
  | Secret | `S2S_PASSCODE` | strongly recommended: it is the main cost brake |
  | Variable | `RUNPOD_ENDPOINT_ID` | from step 3 |
  | Variable | `S2S_MODE` | `lb` |
  | Variable | `S2S_AUDIENCE` | same as the endpoint |
  | Variable | `MAX_CONCURRENT_CALLS` | = the endpoint's max workers |
  | Variable (optional) | `S2S_TOKEN_IN_QUERY` | `1` only if the RunPod proxy drops `X-S2S-Token` (O-W1) |

- Push with `hf upload "$HF_SPACE" . . --repo-type space --exclude "tests/*"` from the staged Space folder.
  Plain git fails on the binary files unless LFS tracks them first.
- Open the **direct** URL `https://<you>-<space>.hf.space/`, not the huggingface.co/spaces page (the microphone is
  blocked inside that iframe, R8).

**5. Queue fallback (only if LB fails the hold test).** Create a **separate** endpoint with
`create_endpoint.sh --mode queue`: Queue type, queue-delay scaler 1 s, execution timeout 900 s, expose TCP 8765, env
`S2S_MODE=queue PORT=8765 S2S_AUDIENCE=hinglish-q-1 S2S_QUEUE_EXEC_TIMEOUT_S=900 S2S_QUEUE_LOAD_TIMEOUT_S=420`.
Then set the Space to `S2S_MODE=queue` with the new id and audience.

## Test plan for your later session (`tests/TEST_PLAN.md`)

Run from the laptop (in India), in this order. Scripts are dry runs until you give them `--space URL` or `--direct …`.

| # | test | pass | if it fails |
|---|---|---|---|
| 0 | CPU self-tests (below) | all pass | fix code before deploying |
| 1 | `space_ws_echo.py $SPACE`: wss echo through the HF proxy (R1) | `ws_echo: OK`, 20 s hold kept | 404: check visibility, use the direct URL; still 404 → launcher (c) |
| 1b | queue mode only: outbound non-443 port from the Space (R17), with `S2S_SELFTEST=1` | 200 | queue mode cannot work from this Space |
| 2 | `latency_probe.py --space $SPACE` (leg 1 only) | record the RTTs | far Space region (R10) |
| 3 | `cold_start_timer.py --space $SPACE --runs 1` (cold host), then 3b (cached host, after 4 min idle), 3c (warm) | `ready` in ≤ 600 s; expect 1-2 min on a cached host | `PersonaPlex not found` → Model step or NETWORK_VOLUME.md; "token did not reach the worker" → `S2S_TOKEN_IN_QUERY=1`; repeated 502 → PORT |
| 4 | GPU check, from `/status` during step 5 | `step_ms_p95` < 80 ms | remove that GPU pool (R4) |
| 5 | `latency_probe.py --space $SPACE --call --talk-s 30`, with the DEP1 baseline if it is up | pipeline lag within about 150 ms of DEP1 | pick a data center near the Space |
| 6 | **hold matrix**: `ws_hold_test.py --direct lb` with `--keepalive-s 0` (A), default (B), `--keepalive-path /ping --keepalive-s 30` (C) | each ends `context_full` or `time_limit`, no drop | all fail → queue mode (7) or active workers = 1 for demos |
| 7 | queue fallback: repeat 3 and 6 in queue mode | as above | active workers = 1 for demos |
| 8 | clean-up: active workers 0, max workers 0 when done, `S2S_SELFTEST=0` | nothing billed at idle | — |

The short version: steps 1, 3, 6 and 5, in that order. Watch two logs: HF Space → Logs, and RunPod endpoint →
Workers → a worker → Logs. The worker's stdout carries `[stack]`, `[server]`, `[asr]` and `[router]` lines, plus one
`SESSION_SUMMARY` line per call.

## Measured numbers (Integrate stage, 2026-10-05, runpod2 RTX 3090, real worker outside Docker)

| metric | serverless worker | DEP1, same call |
|---|---|---|
| `/ping` 204 (port bound before the load) | T0 + 0.6 s | — |
| engine load (PersonaPlex + V3 LoRA merge + Mimi) | about 20 s (`load_s` 19.0) | — |
| Trelis ASR load | 17.4 s | — |
| `/ping` 200 (router ready too) | **T0 + 39.6 s** | — |
| Space session `ready` after `POST /api/session` | 39.4 s | — |
| prompt phase (ws open → handshake) | 8.6 s | 8.05-8.86 s |
| frame step p50 / p95 / p99 (75 s call) | 58.5 / **69.05** / 74.3 ms | p95 66.66 ms |
| compute RTF | 0.748 | 0.712 |
| VRAM peak | **23,129 MiB** of 24,576 | 23,129 MiB |
| receive gaps p95 / max | 86.6 / 234 ms, none over 500 ms | — |
| router trigger → asr → needle | asr 3.2-4.0 s, needle 0.5-0.6 s | — |

Caveats:
- runpod2 has 1 TB RAM, so the 16 GB of weights were probably in the page cache. The 39.6 s does **not** include an
  image pull, a model-cache read or a cold disk. DESIGN's estimate on RunPod: about 1-2 min with the image on the
  host, 4-8 min the first time on a host.
- 24 GB is tight (23.1 GB used). The 48 GB pools are priority 1 for headroom.
- The router triggers ended `unbound` because the test recording was made for the V3 records and the worker uses V4.
  That is resolver behaviour, not an integration fault.

## Known risks (full list: DESIGN section 10, R1-R17; open questions O-W1..3 in DECISIONS.md)

| id | risk | how we find out | fallback |
|---|---|---|---|
| R1 | websockets through the HF Space proxy return 404 | test 1 | direct `*.hf.space` URL; launcher (c) on another host |
| R2 | the LB does not count an open websocket as work, so the worker scales down mid-call | test 6 run A | the pinned keepalive (default); active workers = 1 for demos |
| R16 | pinned keepalives queue behind the open websocket (or start a second billed worker) | test 6 keepalive codes | active workers = 1 for demos |
| O-W1 | the LB proxy drops the `X-S2S-Token` header | test 3 | `S2S_TOKEN_IN_QUERY=1` (no code change) |
| CRIT-1 / R12 | no Model (cache) field on LB endpoints, or a different mount layout | endpoint page; test 3 | `ops/NETWORK_VOLUME.md`; `S2S_PP_DIR` |
| R3 | a Docker Space needs HF PRO | Space creation | PRO; launcher (b) ZeroGPU shim (untested); launcher (c) |
| R4 | a slow GPU misses the 80 ms frame budget | test 4 | drop that pool |
| R5 | too few CUDA 13 hosts | long "no workers" | cu128 build option (untested) |
| R6 / R17 | queue mode: plaintext leg; Space may not reach non-443 ports | test 1b | LB mode |
| R7 | a public Space starts paid workers for anyone | — | passcode, rate, daily cap, max workers |
| R9 | idle endpoints are throttled to max workers 0 after 7 days | "no worker became ready" | raise max workers in the console |
| R15 | a 14 GB image pulls slowly on fresh hosts | test 3 | pre-merged weights (`worker/tools/premerge.py`, DESIGN 6.4), `TRELIS_BF16=1` |
| O-W2 | a worker whose model load failed may keep billing | endpoint billing | max workers 0, fix, retry |
| O-W3 | a busy worker's 409 may not make RunPod add a worker | two calls at once | `MAX_CONCURRENT_CALLS` = max workers |

Never tested here: `docker build`, the cu128 and `TRELIS_BF16` build options, `tools/premerge.py`, the data-center
ids and the console's "4090 PRO" = `ADA_24` label, two calls at once in queue mode, and the Gradio shim.

## Costs (`ops/COSTS.md`; check the console, prices change)

- Billed per second from worker start until it stops: load + call + idle timeout. $0 at idle with active workers 0.
- A6000/A40 flex: $0.00034/s = $1.22/h. 4090 PRO: $1.12/h. L40/L40S: $1.91/h.
- **One call from zero** (about 90 s load + 240 s call + 120 s idle) ≈ **$0.15** on A6000; about $0.26 the first
  time on a host (image pull). An extra back-to-back call on a warm worker ≈ $0.085.
- **Caps:** `MAX_CALLS_PER_DAY` 100 → at most about $15/day; `RATE_PER_IP_PER_HOUR` 6 → one IP at most about
  $0.90/h; max workers N → at most N × $1.22/h.
- Always-on (active workers 1) costs about $29/day on A6000. Use it only for a demo window (about $2.45 for 2 h).
- The HF Space needs a paid plan (PRO) for a Docker Space; cpu-basic hardware itself is free.
- Without `S2S_PASSCODE`, the daily cap is the only limit. Set a passcode, and do the spend controls on day 1.

## What to read, in order

| # | file | what for |
|---|---|---|
| 1 | `ops/BUILD.md` | the deploy steps and the short command path (LB mode) |
| 2 | `ops/ENV.md` | every variable and secret, on both sides |
| 3 | `ops/ENDPOINT_SETTINGS.md` | every RunPod console setting, and the spend controls (section 4: do them on day 1) |
| 4 | `ops/NETWORK_VOLUME.md` | only if the LB endpoint has no **Model** (cache) field, or the worker says `PersonaPlex not found` |
| 5 | `ops/space_push.md` | create and push the HF Space. A Docker Space needs HF PRO |
| 6 | `tests/TEST_PLAN.md` | the real tests, in order, with pass criteria and what to do on failure |
| — | `ops/COSTS.md` | about $0.15 per call from zero on an A6000, $0 at idle |
| — | `DESIGN.md` (frozen + Amendments), `DECISIONS.md` | why things are the way they are; the risks R1-R17 are in DESIGN section 10 |
| — | `worker/README.md`, `space/README.md`, `client/README_SERVERLESS.md` | per-component detail |

## Day-of-deploy checklist

1. **Build context.**
   - The assets (V3 LoRA, Needle `.cact`, the Needle native library, `records_v4.json`) are not in the laptop copy.
   - On runpod2 they are in `worker/build/`.
   - If runpod2 is gone, use the parked copy (`ops/BUILD.md`, "If runpod2 is gone").
2. **Image.**
   - Build and push it to a **private** registry repo from a docker host with 60 GB or more of disk (`ops/build_and_push.sh`).
   - The image is about 14 GB.
3. **Endpoint.**
   - Create it with `ops/create_endpoint.sh --mode lb` or the console.
   - Then the console **Model** step: `nvidia/personaplex-7b-v1` plus a read-only HF token.
   - Set PORT=80 and PORT_HEALTH=80, idle timeout 120 s, max workers 1-2, CUDA 13.0+.
   - **Set the spend controls the same day.**
4. **Space.**
   - Set the Secrets: `RUNPOD_API_KEY`, `S2S_SESSION_SECRET`, `S2S_PASSCODE`.
   - Set the Variables: `RUNPOD_ENDPOINT_ID`, `S2S_MODE=lb`, `S2S_AUDIENCE`, `MAX_CONCURRENT_CALLS`.
   - Then push.
   - `S2S_SESSION_SECRET`, `S2S_MODE` and `S2S_AUDIENCE` must be the **same** on the endpoint and the Space.
5. **Test.**
   - Run TEST_PLAN steps 1, 3, 6 and 5, in that order.
   - Watch the logs in two places: the HF Space → Logs, and the RunPod endpoint → Workers → a worker → Logs.
   - The worker's stdout carries `[stack]`, `[server]`, `[asr]` and `[router]` lines, plus one `SESSION_SUMMARY` per call.
6. **Clean-up.**
   - Set active workers to 0. Set max workers to 0 when you are done.
   - Turn off `S2S_SELFTEST`.

## Quick fixes for the failures we expect first

| symptom | likely cause | fix |
|---|---|---|
| ws echo (test #1) gives 404 | the HF proxy (R1) | Use the direct `*.hf.space` URL and Public visibility. If it still fails, use launcher (c) (space_push.md 6) |
| session `failed`: "token did not reach the worker … S2S_TOKEN_IN_QUERY", or `relay_error` 401 `missing` | the RunPod proxy drops the `X-S2S-Token` header (O-W1) | Set the Space variable `S2S_TOKEN_IN_QUERY=1` (CRIT-3) |
| session `failed`: "worker rejected the session token" | the secret, audience or mode differ between the two sides | Make all three equal on both sides (ENV.md 1) |
| "RunPod rejected the API key" | `RUNPOD_API_KEY` | Check the Space secret |
| wake times out; endpoint shows `/ping` 500, log says `PersonaPlex not found` | the model cache is missing, or not offered for LB | Redo the console Model step, or follow ops/NETWORK_VOLUME.md |
| repeated 502 for about 8 min | PORT mismatch | Set PORT=80 and PORT_HEALTH=80 (ENDPOINT_SETTINGS 5) |
| "no worker became ready" after a quiet week | the idle throttle took max workers to 0 (R9) | Raise max workers in the console |
| frame step p95 ≥ 80 ms in `/status` | a slow GPU type (R4) | Remove that pool from the endpoint |

## CPU tests (runpod2; no GPU, no network)

```bash
cd /root/deploy_serverless; P=/root/deploy/venv-pp/bin/python
python3 tests/test_token.py
$P tests/test_fake_runpod.py --base-port 51000
$P tests/test_mock_e2e.py --base 52000
$P tests/test_integration.py --base 45000 --phases lb,queue
$P tests/test_crit_token_query.py --base 56000
$P worker/tests/test_worker_mock.py --port-base 58200
.venv-space/bin/python space/tests/test_space_flows.py
$P space/tests/xcheck_repo_fakes.py
python3 worker/tests/check_dockerfile.py
```

Run them one at a time: two suites on the same port base collide. The GPU smoke test (`tests/gpu_smoke.sh`) stops
the live DEP1 demo while it runs. Read its header first, and run it with `setsid nohup`, never inside tmux (INTEG-1).

## File map

```
README.md                 this file
DESIGN.md                 frozen design (read-only) + Amendments S1/S2/W1/W2/C1/R-ops1/R-space
DECISIONS.md              append-only decision log (D-INTEG, D-CRIT, ...)
common/
  s2s_token.py            HMAC session token (stdlib only; used by the Space, the worker and the tests)
  session.py              DEP1 session config fork (records from S2S_RECORDS)
  data/records_v4.json    V4 demo records (staged by prepare_build_context.sh)
  test_vectors.json       fixed token test vectors
worker/
  Dockerfile (+ .dockerignore)   GPU image: CUDA 13.0 base, three venvs, baked Trelis/LoRA/Needle
  entrypoint.sh           S2S_MODE=lb: stack in foreground; queue: stack in background + rp_handler.py
  stack.sh                server -> asr -> router supervisor (fork of DEP1 gpu_stack.sh)
  resolve_models.py       finds PersonaPlex: explicit dir / model cache / network volume / download (debug)
  paths.py                env-driven path layout (no hard-coded /root paths)
  rp_handler.py           queue-mode handler (progress_update with public IP/port)
  server/                 worker_server.py (DEP1 server.py fork + /ping, claim, token, 409), engine, core, ring, mock
  router/                 asr_service.py, router_service.py, router_core.py, trigger.py, stub_tools.py
  hinglish/  needle/      LoRA merge code + voice codes; Needle runtime subset
  requirements-*.txt      pinned from the runpod2 venvs
  build/                  baked assets, NOT synced (see "Copies" below)
  local/run_local.sh      run the stack outside docker (mock | gpu); runpod2.env
  tools/premerge.py       optional pre-merged weights for faster cold start (untested)
  tests/                  test_worker_mock.py, check_dockerfile.py
space/
  app/main.py sessions.py upstream.py   aiohttp backend: client, session API, wake, claim, relay, keepalive
  Dockerfile  README.md   Docker-SDK Space (app_port 7860)
  app.py                  Gradio-SDK shim (launcher b, untested)
  stage_common.sh         copies common/ into the Space folder
  static/                 built client (from client/dist)
  tests/                  test_space_flows.py, xcheck_repo_fakes.py
client/                   minimal fork of the DEP1 PersonaPlex client (README_SERVERLESS.md)
ops/
  BUILD.md ENV.md ENDPOINT_SETTINGS.md NETWORK_VOLUME.md space_push.md COSTS.md
  prepare_build_context.sh build_client.sh build_and_push.sh create_endpoint.sh
tests/
  TEST_PLAN.md            the real tests for later
  fake_runpod.py          local stand-in for the LB proxy and the queue API
  test_token.py test_fake_runpod.py test_mock_e2e.py test_integration.py test_crit_token_query.py   CPU suites
  gpu_smoke.sh gpu_call.py   the GPU smoke (stops DEP1 while it runs)
  space_ws_echo.py cold_start_timer.py latency_probe.py ws_hold_test.py   real tests (dry run by default)
  rt_common.py stub_worker.py
results/
  integration.md          Integrate-stage write-up; gpu_smoke/ logs, integ_cpu*.json
```

## Copies

- **runpod2** `/root/deploy_serverless`: the working copy, with `worker/build/` assets and the venvs. Ephemeral disk.
- **pod1** `/workspace/hinglish/deploy_serverless` and the **laptop** `/home/shivam/Desktop/S2S/hinglish/deploy_serverless`:
  code, docs and results only. Not copied: `.venv-space/`, `.venvs/`, `node_modules/`, `__pycache__/`, `worker/build/`
  and any file over 50 MB.
- **pod1** `/workspace/hinglish/deploy_serverless_assets/s2s_build_assets.tar`: the baked assets (466 MB).
