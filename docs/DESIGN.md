# S2S serverless: DESIGN (frozen after the architect stage)

Status: written by the architect on 2026-10-04. **FROZEN.** Later stages must not edit the sections below. They may only append
`## Amendment N (date, who, why)` sections at the end, and an amendment wins where it disagrees with an earlier section.
Decisions log: `/root/deploy_serverless/DECISIONS.md` (append-only, dated). DEP1 (the live pod demo) is the source:
`/root/deploy` on runpod2, and its contract is `/root/deploy/INTERFACE.md` with Amendment 1. Nothing under `/root/deploy`
is modified; every file we need is copied or forked into `/root/deploy_serverless`.

This stage is BUILD ONLY. No endpoint, pod, image push or Space is created. The user tests later (section 9).

Notation:
- "LB" = RunPod Serverless load-balancing endpoint.
- "queue" = RunPod queue-based endpoint.
- "Space" = the Hugging Face Space front.
- "worker" = one RunPod serverless worker container.
- "relay" = the Space backend's websocket bridge.
- frame = 80 ms Mimi frame, as in DEP1.

Platform sources (fetched 2026-10-04; every platform claim below cites one of these):

| tag | URL |
|---|---|
| [LB-OV] | https://docs.runpod.io/serverless/load-balancing/overview |
| [LB-BUILD] | https://docs.runpod.io/serverless/load-balancing/build-a-worker |
| [LB-AFF] | https://docs.runpod.io/serverless/load-balancing/worker-affinity |
| [EP-CFG] | https://docs.runpod.io/serverless/endpoints/endpoint-configurations |
| [CACHE] | https://docs.runpod.io/serverless/endpoints/model-caching |
| [HF-MODELS] | https://docs.runpod.io/serverless/development/huggingface-models (cached-model path helper) |
| [PRICE] | https://docs.runpod.io/serverless/pricing |
| [ENVV] | https://docs.runpod.io/serverless/development/environment-variables |
| [SEND] | https://docs.runpod.io/serverless/endpoints/send-requests |
| [VOLCACHE] | https://docs.runpod.io/serverless/development/volume-cache |
| [EX-LBWS] | https://github.com/runpod-workers/worker-lb-websocket (README.md, app.py, test_scaling.py) |
| [EX-QWS] | https://github.com/runpod-workers/worker-websocket (README.md, rp_handler.py) |
| [HF-DOCKER] | https://huggingface.co/docs/hub/spaces-sdks-docker |
| [HF-OV] | https://huggingface.co/docs/hub/spaces-overview |
| [HF-CFG] | https://huggingface.co/docs/hub/spaces-config-reference |
| [HF-GPUS] | https://huggingface.co/docs/hub/spaces-gpus (sleep time) |
| [HF-ZERO] | https://huggingface.co/docs/hub/spaces-zerogpu |
| [HF-WS404] | https://discuss.huggingface.co/t/fastapi-websocket-returns-http-404-on-spaces/159865 |

---------------------------------------------------------------------------------------------------------------------
## 0. Premise corrections (read first)

The research turned up four facts that change the plan we were given. Each one is decided here and listed again in section 10.

1. **A Docker or Gradio Space is not free.**
   - [HF-OV] says: "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts ... Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU".
   - cpu-basic hardware has no hourly cost once you have the plan, but it sleeps after 48 h without visitors and wakes on the next visit ([HF-GPUS] "Set a custom sleep time"). So it is not "always on", only "wakes on visit".
   - **Decision.** `space/` is one plain aiohttp app (`space/app/main.py`) with no Gradio or HF dependency. It ships with three launchers:
     - (a) a Docker-SDK Space (primary; needs PRO);
     - (b) a Gradio-SDK `app.py` shim that runs the same aiohttp app on :7860 (for a ZeroGPU Space on a free account; untested platform behaviour, risk R3);
     - (c) the same Dockerfile on any container host (fallback front).
   - A Static Space cannot work as the front: its variables are exposed to client JavaScript ([HF-OV] "For Static Spaces, both are available through client-side JavaScript"), so it cannot keep the RunPod key and it has no backend to relay through.
2. **Each endpoint can cache only one model.**
   - [CACHE] "Current limitations": "Each endpoint is currently limited to one cached model at a time."
   - **Decision.**
     - Cache `nvidia/personaplex-7b-v1`: 16 GB, gated, needs an HF token in the endpoint's model setting.
     - Bake `Trelis/whisper-hinglish-preview` (public, 5.8 GB fp32 on disk) into the image.
     - Bake the V3 LoRA adapter (370 MB), the Needle `.cact` (63 MB), the Needle native library cache (35 MB) and all code into the image.
   - Option B (section 6.4) is a single private HF repo that holds everything and is cached as the one model.
3. **Idle endpoints are throttled.**
   - After 3 days with no requests, max workers drops to 2. After 7 days it drops to 0, and stays there until it is raised by hand ([EP-CFG] "Idle endpoint scale-down").
   - This is an ops item (ops/ENDPOINT_SETTINGS.md) and risk R9.
4. **The browser can never reach the worker directly, in either mode. A relay is mandatory.**
   - The docs show only `Authorization: Bearer <RunPod API key>` for LB HTTP and WS ([LB-BUILD] "Connect from a client"; [EX-LBWS] "Authentication uses the same Authorization: Bearer header"). No query-parameter or cookie alternative is documented.
   - Browsers cannot set headers on `new WebSocket()`, and the key must never reach the browser.
   - In queue mode the worker speaks plain `ws://IP:PORT` with no TLS ([EX-QWS] client.py). An https page cannot open that (mixed content).
   - **Decision.** The browser talks only to the Space, in both modes. The Space backend opens the upstream websocket server-to-server and adds the RunPod key (LB) or connects to the public IP and port (queue). The client fork stays tiny, because its websocket URL is still same-origin `/api/chat`.

---------------------------------------------------------------------------------------------------------------------
## 1. Components and repo layout (`/root/deploy_serverless`)

```
DESIGN.md  DECISIONS.md  README.md (Integrate stage)
common/
  s2s_token.py            HMAC session token (section 4); stdlib only; importable from every venv and from the Space
  session.py              fork of DEP1 common/session.py: RECORDS_V4 from env S2S_RECORDS (default common/data/records_v4.json)
  data/records_v4.json    copy of /root/deploy/assets/ws/hinglish/data/V4/records.json (synthetic records, 587 KB)
  test_vectors.json       fixed token test vectors (section 4.5)
worker/
  Dockerfile              GPU image (section 6)
  entrypoint.sh           S2S_MODE=lb -> supervisor in foreground; S2S_MODE=queue -> supervisor in background + rp_handler.py
  stack.sh                fork of DEP1 gpu_stack.sh + router start: server -> asr -> router, one dies => all die (no flock in the image)
  paths.py                ONE env-driven path layout (section 1.2); every forked module imports it instead of hard-coded /root paths
  resolve_models.py       finds PersonaPlex files (cache / volume / explicit dir / download), extracts voices (section 6.3)
  rp_handler.py           queue-mode handler (section 2.2)
  requirements-pp.txt  requirements-asr.txt  requirements-needle.txt   pinned from the runpod2 venvs (CPU jax for needle)
  server/                 fork of DEP1 server/: worker_server.py (fork of server.py), engine.py, core.py, ring.py, mock_server.py, ws_feed.py
  router/                 copies: asr_service.py, router_service.py, router_core.py, trigger.py, stub_tools.py (path edits only)
  hinglish/               trainer/merge_lora.py, infer/lora_merge.py, trainer/voice_codes/NAT{F2,M1}.*  (path edits only)
  needle/                 runtime subset of /root/needle: router.py build_data.py resolver.py tools.py hindi_share.py romanise.py
                          romanise_lexicon.json tools_json/ vendor/ src/records.json  (no training data)
  build/                  NOT in the repo: filled by ops/prepare_build_context.sh (adapter, .cact, needle lib cache)
space/
  README.md               Space card with YAML header (sdk: docker, app_port: 7860)
  Dockerfile              python:3.11-slim, user 1000, aiohttp app on :7860
  requirements.txt        aiohttp, numpy-free (no audio processing in the Space)
  app.py                  Gradio-SDK shim (launcher b): runs app.main:run() on 0.0.0.0:7860, no Gradio UI
  app/main.py             aiohttp app: static client, records/samples, session API, relay, /ws-echo
  app/upstream.py         RunPod access: LB (ping/claim/keepalive/ws) and queue (run/status/cancel), one interface (section 5.3)
  app/sessions.py         session table + state machine + limits (concurrency, per-IP rate, passcode)
  static/                 built client (copied from client/dist by ops/build_client.sh)
  samples/                optional handpicked wavs (empty by default)
client/                   minimal fork of /root/deploy/client (section 5.4); build output client/dist
ops/
  ENV.md                  every env var, both sides (section 4.4 + 5)
  ENDPOINT_SETTINGS.md    what to click in the RunPod console, per mode
  BUILD.md                docker build/push commands, HF Space push commands (for the user to run later)
  COSTS.md                cost notes
  prepare_build_context.sh  copies adapter/.cact/needle-lib/records into worker/build and common/data (runs on runpod2)
  build_client.sh         npm ci && npm run build in client/, copy dist -> space/static
tests/
  fake_runpod.py          local stand-in for the LB proxy (Bearer check, X-Runpod-Worker-Id, /ping gating) and queue API (/run,/status,/cancel)
  test_token.py           vectors + expiry + replay + tamper
  test_mock_e2e.py        CPU: worker_server with MockEngine + fake_runpod + Space backend + headless ws client (both modes)
  hold_test.py            REAL (user, later): 4-min websocket hold through the Space with keepalive; reports drops
  cold_start_timer.py     REAL: time from POST /api/session to ready, broken into phases
  latency_probe.py        REAL (run from India): per-leg RTT browser->Space, Space->worker (via /api/diag), audio round trip
  space_ws_echo.py        REAL: test #1, wss echo through the deployed Space
```

### 1.1 Worker processes (inside one container)

| process | venv in image | bind | from DEP1 |
|---|---|---|---|
| `worker_server.py` | `/opt/venv-pp` (py3.11, torch 2.14.1) | `0.0.0.0:$PORT` http (public: `/ping`, `/status`, `/session/claim`, `/session/release`, `/api/chat` ws); `127.0.0.1:8999` (DEP1 internal API, unchanged) | server.py fork |
| `asr_service.py` | `/opt/venv-asr` (py3.11, same torch, transformers 5.18.0) | `127.0.0.1:8996` | unchanged except paths |
| `router_service.py` | `/opt/venv-needle` (py3.12, jax CPU) | `127.0.0.1:8995` | unchanged except paths |
| `rp_handler.py` (queue mode only) | `/opt/venv-pp` + `runpod` pip | none (talks to 127.0.0.1:8999) | new |

- There is no TLS inside the worker. The RunPod proxy terminates TLS in LB mode; queue mode is plaintext (section 2.2).
- The `/ping` health port is the same `$PORT` (`PORT_HEALTH` unset; [LB-OV] "Environment variables").

### 1.2 Path layout (`worker/paths.py`, every default overridable by env)

| name | env | image default | DEP1 hard-coded original it replaces |
|---|---|---|---|
| app root | `S2S_ROOT` | `/opt/s2s` | `/root/deploy` |
| adapter dir | `S2S_ADAPTER` | `/opt/s2s/assets/v3_adapter` (contains `config.json`, `lora.safetensors`); `''` = base model; `premerged` = skip merge (6.4) | `assets/V3_CKPT` key=value file |
| merge_lora | `S2S_HINGLISH` | `/opt/s2s/hinglish` (`trainer/merge_lora.py`, `infer/lora_merge.py`, `trainer/voice_codes`) | `/workspace/hinglish/...` (engine.MERGE_LORA, merge_lora sys.path, lora_merge.VOICE_CODES_DIR) |
| needle dir | `S2S_NEEDLE` | `/opt/s2s/needle` | `/root/needle` (router_core.NEEDLE_DIR, asr_service sys.path) |
| needle weights | `S2S_NEEDLE_WEIGHTS` | `/opt/s2s/assets/tuned_full.cact` | `/root/needle/finetune/sweep/tuned_full.cact` |
| needle lib cache | `HOME` | `/root` with `/root/.cache/cactus-needle/v3/3.0.2/{libneedle.so,needle3.cact}` baked | `~/.cache/cactus-needle` (fetched at runtime by cactus-needle 3.0.6) |
| records | `S2S_RECORDS` | `/opt/s2s/common/data/records_v4.json` | `/root/deploy/assets/ws/hinglish/data/V4/records.json` |
| PersonaPlex files | `S2S_PP_DIR` | resolved by `resolve_models.py` (6.3) | `HF_HOME=/root/hf` + `hf_hub_download` |
| voices dir | `S2S_VOICES` | resolved by `resolve_models.py` (6.3): `<snapshot>/voices` if present, else extracted to `/tmp/s2s/voices` | `_get_voice_prompt_dir` extracting next to the HF blob |
| Trelis HF home | `S2S_ASR_HF_HOME` | `/opt/hf-asr` (baked; HF_HUB_OFFLINE=1) | `/root/hf` |
| logs | `S2S_LOGS` | `/tmp/s2s/logs` (stdout is the primary log, [EP-CFG] / RunPod Logs tab) | `/root/deploy/logs` |
| static, samples, certs | n/a | not served by the worker (the Space serves the client) | server.py STATIC_DEFAULT, SAMPLES_DIR, CERT_DIR |

Rule for builders: no string `/root/deploy`, `/root/needle`, `/workspace` or `/root/hf` may remain in `worker/` code, except in comments that cite the DEP1 origin. A grep check is part of tests/test_mock_e2e.py.

---------------------------------------------------------------------------------------------------------------------
## 2. Message flows

### 2.1 LB mode (primary)

```
Browser ──https/wss (same origin)──> HF Space backend ──https/wss + Authorization: Bearer RUNPOD_API_KEY──> RunPod LB proxy ──http/ws──> worker :$PORT
                                       (relay, keepalive, token mint)            https://ENDPOINT_ID.api.runpod.ai/<path>
```

Steps (times are budgets, not measurements):

1. The browser loads `https://<space>.hf.space/` and gets the static client. The Space may itself be waking from cpu-basic sleep ([HF-GPUS]).
2. The browser sends `POST /api/session {record_id, pairing, seed, passcode?}`. The Space:
   - checks the passcode, the per-IP rate limit and `MAX_CONCURRENT_CALLS` (section 5.1);
   - creates `sid`;
   - returns `{sid, state:"waking"}`;
   - starts a wake task.
3. The wake task polls `GET https://ENDPOINT_ID.api.runpod.ai/status` with Bearer, every `WAKE_POLL_S`=3 s.
   - It uses `/status`, a normal application route, and deliberately not the health path `/ping`. The [EX-LBWS] `test_scaling.py` test 1 docstring calls `/ping` polling "the reverse-proxy behaviour that does NOT scale workers". The health path may be treated specially by the LB.
   - `/ping` is used only as RunPod's health contract (the LB polls it itself). Test #3 compares the two (section 9).
   - While no worker is ready, the proxy holds a request up to 2 min, then answers "no workers available" ([LB-OV] "Timeouts and limits: Request timeout 2 min (no worker available)"; [LB-BUILD] Troubleshooting). The Space treats that, 502/503/504 and timeouts as "still waking" and keeps polling, with a per-request HTTP timeout of 130 s.
   - The poll stops at `WAKE_TIMEOUT_S`=600 s, which ends in state `failed`.
   - The proxy routes only to workers whose `/ping` is 200 ([LB-OV] "Health checks": 200 healthy, 204 initializing). So a `/status` answer with `"state"` in `idle`/`claimed`/`busy` means a ready worker.
   - The state the browser polls is `waking`, with `elapsed_s` and `detail` (`"no worker yet"`, `"worker initializing"`).
4. On the first ready `/status`, the Space mints a token (section 4) and sends `POST /session/claim` with header `X-S2S-Token`, no affinity header.
   - A worker answers `200 {worker_id, sid, claim_ttl_s}`, or `409 {"error":"busy"}`, which means the proxy routed the claim to a worker in a call.
   - On 409 the Space retries the claim every 3 s until `WAKE_TIMEOUT_S`. The polling keeps traffic up, so the scaler can add a worker up to max_workers ([EP-CFG] "Auto-scaling type: Request count").
   - The Space records `worker_id` from the JSON body and cross-checks the `X-Runpod-Worker-Id` response header ([LB-AFF] "Worker ID on responses"). If they differ, it uses the header, because the header is what the proxy routes on.
   - State becomes `ready`.
5. The browser sees `ready` and opens `wss://<space>.hf.space/api/chat?sid=…&<DEP1 query params>`. The client is unchanged here apart from the `sid` param.
   - The relay mints a fresh token for this sid (the claim token may be close to expiry). It then opens `wss://ENDPOINT_ID.api.runpod.ai/api/chat?<params>` with these headers:
     - `Authorization: Bearer RUNPOD_API_KEY`
     - `X-Runpod-Worker-Id: strict <worker_id>`
     - `X-S2S-Token: <token>`
   - It sets `open_timeout` ≥ 60 s ([LB-BUILD] "you must set the open_timeout parameter ... A value of 60 seconds works for most use cases").
   - Strict affinity routes only to the claimed worker and returns 404 `affinity_worker_gone` if it is gone ([LB-AFF] "Strict affinity").
6. The worker checks the token. The sid must equal the claimed sid, and the call must not be a replay or a second concurrent call. The worker then runs the DEP1 handler unchanged:
   - prompt phase (about 9 s);
   - 0x00 handshake;
   - 0x01, 0x02 and 0x07 frames (DEP1 INTERFACE section 4).
   The relay pumps binary frames both ways without parsing, except that it peeks at 0x07 `metrics` events (5.1).
7. Keepalive. While the call is up, the Space sends `GET /status` every `KEEPALIVE_S`=20 s with `X-Runpod-Worker-Id: strict <worker_id>` and Bearer.
   - The HTTP client timeout is 10 s, and a timeout is non-fatal: it is logged and the next one is sent.
   - Strict affinity "waits up to ~5 minutes" when the worker is at capacity ([LB-AFF]). If the open websocket occupies the worker's slot, pinned requests may hang, so they must never block the relay (R16).
   - Why it is needed: [EX-LBWS] README warns that an open websocket may not count as active work. Its workaround is "GET /ping every 30 s ... or min_workers = 1". We send the same traffic to `/status` instead of the health path (see step 3).
   - Why pinned: an unpinned ping could land on another worker and reset the wrong idle timer. [LB-AFF] says "a worker receiving regular traffic, with or without strict-resume, will never scale down".
   - Why 20 s and not 30 s: the endpoint idle timeout must be set well above it, to 120 s (ops). The default is 5 s ([EP-CFG] "Idle timeout"), and at 5 s even a 20 s ping cannot hold a worker whose websocket is not counted.
   - We use plain `strict`, not `strict-resume`. Resuming a scaled-down worker cannot save a call whose model state is gone. A 404 or a failed keepalive is logged and surfaced as `relay` detail; the websocket itself decides whether the call is dead.
8. End of call. Any of these ends it:
   - the user clicks Disconnect (the browser closes, the relay closes upstream, the worker ends the session `client_closed`);
   - context full (about 221 s, DEP1 A1.4; the worker sends `session_end` `context_full`);
   - the call time cap `CALL_MAX_S`=300 s from upgrade, including the prompt phase (the worker sends `session_end` `time_limit`);
   - an error.

   When the session has ended, the worker frees the claim and can serve the next claim. The Space stops keepalives, and the worker idles out after the idle timeout ([EP-CFG]).

   Time limits:

   | limit | value | source |
   |---|---|---|
   | per-request processing cap | 5.5 min = 330 s | [LB-OV] "Processing timeout 5.5 min (per request)" |
   | worker call cap | `CALL_MAX_S` = 300 s | ours; leaves 30 s margin under the 330 s cap |
   | Space session cap | `CALL_MAX_S` + 15 s | ours; then the relay force-closes |

   We assume the websocket counts as one request under the 330 s cap. That is unverified (R2).
9. Worker scale-down or crash mid-call. The upstream websocket closes without a `session_end`. The relay then:
   - sends the browser a synthetic 0x07 `{"type":"session_end","reason":"worker_lost","frames":null}`;
   - closes the browser socket with code 1011;
   - marks the session `ended` with detail `worker_lost`.
   The client shows the reason; DEP1 Panels already render `session_end.reason`. There is no resume, because the model KV state lives only on that GPU.
   If the worker gets SIGTERM, worker_server sends `session_end` `worker_shutdown` before closing.

### 2.2 Queue mode (fallback; same image with `S2S_MODE=queue`)

```
Browser ──wss──> Space backend ──https Bearer──> api.runpod.ai/v2/ENDPOINT_ID/{run,status,cancel}
                       └──────── ws://RUNPOD_PUBLIC_IP:RUNPOD_TCP_PORT_<port> (plaintext, X-S2S-Token) ──> worker :$PORT
```

1. The browser call is the same `POST /api/session`. The Space sends `POST https://api.runpod.ai/v2/ENDPOINT_ID/run` with:
   - body `{"input":{"sid":…,"record_id":…,"pairing":…,"seed":…}, "policy":{"executionTimeout": QUEUE_EXEC_TIMEOUT_MS, "ttl": …}}`;
   - `executionTimeout` default 600000 ms. The endpoint setting must be at least this ([EP-CFG] "Execution timeout"; per-request override via the job policy, [SEND] "execution policies").

   This wakes a worker ([EX-QWS] README step 1).
2. In the worker, `entrypoint.sh` starts `stack.sh` in the background and then `rp_handler.py` (runpod SDK). The handler:
   - (a) waits for the local stack to be ready (`GET 127.0.0.1:$PORT/ping` == 200), sending `progress_update` `{"state":"loading"}` every 10 s;
   - (b) claims the job's `sid` in process via `POST 127.0.0.1:8999/internal/claim {"sid","record_id","pairing","seed"}`. That route is on the internal listener only and needs no token.
     - The job carries no token. A minted token has a 120 s TTL, and a cold load can take 1-8 min, so it would expire before this step.
     - The job input is trusted anyway: `/run` itself requires the RunPod key.
     - The relay's freshly minted token is still required on the websocket.
   - (c) reads `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_<PORT>` and publishes `progress_update` `{"state":"ready","public_ip","tcp_port","worker_id":RUNPOD_POD_ID}` ([EX-QWS] rp_handler.py, `runpod.serverless.progress_update`);
   - (d) blocks until the session ends, or until the claim expires unused (`CLAIM_TTL_S`), or until `CALL_MAX_S`+30 s;
   - (e) returns the session summary (DEP1 summary JSON) as the job output.

   One job equals one call. Concurrency per worker is 1.
3. The Space polls `GET /v2/ENDPOINT_ID/status/{job_id}` every 2 s. While the job is `IN_PROGRESS`, the progress payload is in the status `output` ([EX-QWS] README steps 3-4). The browser sees `waking` → `ready`.
4. The relay opens `ws://public_ip:tcp_port/api/chat?<params>` with header `X-S2S-Token`. Everything else is as in LB mode.
   - There is no Authorization header (it is not the RunPod proxy) and no keepalive (a running job keeps the worker alive).
   - This leg is plaintext. The token is the only auth, and audio crosses the internet unencrypted between RunPod and the Space (R6).
5. End. The worker ends the session and the handler returns. If the browser leaves before connecting, the Space sends `POST /v2/ENDPOINT_ID/cancel/{job_id}`.
   - The endpoint must expose TCP port `$PORT` ([EP-CFG] "Expose HTTP/TCP ports"; [EX-QWS] "Expose TCP Ports").
   - A queue endpoint does not support websockets through the RunPod proxy ([EX-LBWS] "Queue-based endpoints do not support WebSocket"), which is why this mode uses the public TCP port.

Queue mode is selected by `S2S_MODE=queue` on both sides. The worker image is the same.

### 2.3 Timeouts summary

| timer | value | where | why |
|---|---|---|---|
| wake poll (`GET /status`) interval / timeout | 3 s / 600 s | Space | the cold start budget (section 6.5) plus the image pull on a fresh host |
| per-poll HTTP timeout | 130 s | Space | the LB holds a request up to 2 min with no worker ([LB-OV]) |
| claim TTL | `CLAIM_TTL_S` = 90 s | worker | the browser must open the websocket within 90 s of `ready`, or the worker frees itself |
| upstream ws open_timeout | 60 s | Space relay | [LB-BUILD] |
| keepalive | `GET /status` every 20 s, strict affinity, 10 s client timeout (non-fatal) | Space (LB only) | [EX-LBWS], [LB-AFF] |
| release (LB) | `POST /session/release`, strict affinity, 10 s client timeout (non-fatal) | Space | [LB-AFF] strict waits up to ~5 min at capacity |
| endpoint idle timeout | 120 s | console | must be above the keepalive interval; default 5 s ([EP-CFG]) |
| call cap | `CALL_MAX_S` = 300 s | worker (authoritative) and Space (+15 s) | below the 330 s LB processing cap ([LB-OV]); context ends about 221 s anyway |
| queue execution timeout | 600 s | console + job policy | covers load (up to about 120 s) + claim (90 s) + call (300 s) |
| browser state poll | 1.5 s | client | UI only |

---------------------------------------------------------------------------------------------------------------------
## 3. Worker detail

### 3.1 worker_server.py (fork of DEP1 server.py)

Changes from DEP1 server.py. Everything else, including the chat handler body, the Hub and the internal API, stays byte-for-byte where possible.

- **Bind first, load later.** `serve()` starts the public site on `0.0.0.0:$PORT` (http) and the internal site on `127.0.0.1:8999` BEFORE the engine load. The load runs as a task. DEP1 bound only after `srv.load()`, which would leave `/ping` unanswered during load.
- **Readiness.** `ready = engine loaded AND GET 127.0.0.1:8996/health ok (unless ASR_BACKEND=off) AND GET 127.0.0.1:8995/health ok (unless ROUTER=off)`. Health of ASR and router is polled every 2 s until ready. After that it is polled every 10 s; a failure there is logged and shown in `/status`, but it does NOT flip `/ping`.
- **Public routes on `$PORT`.**

  | route | auth | response |
  |---|---|---|
  | `GET /ping` | none (the LB proxy already requires the RunPod key) | `204` while loading; `200 {"status":"ok","state":"idle"\|"claimed"\|"busy"}` once ready, **also while busy**; `500` if the load failed (unhealthy; RunPod removes it, [LB-OV]) |
  | `GET /status` | none. The Space's wake and keepalive route (2.1 steps 3 and 7). | `{"state","worker_id","gpu","vram_total_mib","load_s","sessions_served","active":{"sid","conv_s","context_frames_left"}\|null,"step_ms_p95_last","asr":bool,"router":bool,"version"}`. No secrets, no prompts. |
  | `POST /session/claim` | `X-S2S-Token` | `200 {"worker_id","sid","claim_ttl_s"}` (an idempotent re-claim of the same sid → 200) \| `401 {"error":"bad_token"\|"expired"\|"replayed"}` \| `409 {"error":"busy"}` \| `503 {"error":"loading"}` |
  | `POST /session/release` | `X-S2S-Token` (same sid) | `200 {"released":bool}` |
  | `GET /api/chat` (ws) | `X-S2S-Token` header, or `?token=` query (for test tools) | DEP1 section 3 query params and section 4 protocol. Refusals happen BEFORE the upgrade, as HTTP errors with a JSON body: `401` bad, expired or replayed token; `409` busy, or no claim for this sid; `503` loading. After the upgrade, a protocol-level refusal closes with code `4401` (auth), `4409` (busy) or `4503` (loading). |
  | `GET /metrics` | `X-S2S-Token` of the active sid | DEP1 section 7 object |

  Dropped from the public listener: `/`, static files, `/api/records*`, `/api/samples`, `/samples/*`. The Space serves these.
- **Single-session guard.** One state machine guarded by the existing `asyncio.Lock`, with states `idle` → `claimed(sid, until)` → `busy(sid)` → `idle`.
  - A claim is only accepted in `idle`, or as a re-claim of the same sid.
  - A websocket is accepted only for the claimed sid.
  - A second socket gets 409 immediately. This replaces the DEP1 "waits on the lock" behaviour.
  - A claim that times out returns the worker to `idle`.
- **Token verification** (section 4).
  - Every token-carrying route (claim, release, metrics, websocket) checks the signature, `exp`, `mode` and `aud`.
  - **Only the websocket upgrade** also checks the replay set: an LRU of sids that already started a websocket, 1024 entries, entries dropped after `exp`+60 s. Claim, release and metrics tokens for the same sid stay valid after the socket has started.
  - Internal route added on `127.0.0.1:8999` (queue mode): `POST /internal/claim {"sid","record_id","pairing","seed"}`. It has the same semantics as `/session/claim`, without a token.
  - The `record_id`, `pairing` and `seed` in the token must equal the websocket query params, otherwise 401 `cfg_mismatch`.
  - With `S2S_ALLOW_FREE_PROMPT=0` (the default), a non-empty `text_prompt` or `voice_prompt` is rejected (400) and the session must carry a `record_id`. This keeps public callers to the demo records.
- **Call cap.** A `CALL_MAX_S` watchdog in `process_loop` sets `end_reason="time_limit"`. The DEP1 `session_end` reasons gain `time_limit` and `worker_shutdown`.
- **SIGTERM.** The handler sets close with `end_reason="worker_shutdown"`, flushes for up to 2 s, then exits.
- **Logging.** Stdout, in the DEP1 format. One JSON line per session summary, with `worker_id`, `gpu`, `sid` (never the token), `end_reason` and step stats. These also go to `S2S_LOGS/server_sessions.jsonl`. At startup it logs the GPU name, driver, CUDA, free VRAM, and a warning if the GPU name matches `S2S_SLOW_GPUS` (default `L4,A4000,A4500,RTX 4000,A2000`).
- **Engine.** DEP1 engine.py, with the constructor taking explicit file paths: `Engine(adapter, device, pp_files={moshi, mimi, tokenizer}, voice_prompt_dir=…)` instead of `hf_hub_download` (section 6.3). Sampling, merge, warmup and the step loop are unchanged.

### 3.2 stack.sh (fork of gpu_stack.sh)

- Exports `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=${S2S_THREADS:-4}` and `DEP1_TORCH_THREADS` (DEP1 A1.3).
- Runs `resolve_models.py` first. It writes `/tmp/s2s/models.env` with `S2S_PP_DIR` and `S2S_VOICES`; on failure it exits non-zero and `/ping` returns 500.
- Starts `worker_server.py` on `$PORT`.
- After `/status` shows the engine loaded, starts `asr_service.py --backend ${ASR_BACKEND:-trelis} --device cuda --port 8996`. The order is kept from DEP1 (the server loads first and sees the full card).
- Then starts `router_service.py --internal http://127.0.0.1:8999 --asr-url http://127.0.0.1:8996 --window ring30 --unbound strict --port 8995`.
- If any child exits, it kills the rest and exits non-zero. In LB mode the container then exits; RunPod replaces the worker.
- No flock: a serverless worker owns its GPU.

### 3.3 rp_handler.py (queue mode)

Section 2.2 gives the steps. It is plain `runpod.serverless.start({"handler": handler})` from [EX-QWS], with sync waits via urllib and no asyncio. The default port for queue mode is `PORT=8765`, matching the example and the TCP port to expose.

### 3.4 Idle and keepalive behaviour (worker side)

- The worker never exits on its own in LB mode. RunPod's idle timeout scales it down. A keepalive is a normal `GET /status`, pinned with strict affinity by the Space.
- In queue mode the worker stays up while the handler runs. After the handler returns, the RunPod idle timeout applies.

---------------------------------------------------------------------------------------------------------------------
## 4. Session token spec (`common/s2s_token.py`)

### 4.1 Format

```
token = "v1." + b64url(payload_json) + "." + b64url(HMAC_SHA256(key, "v1." + b64url(payload_json)))
b64url = urlsafe base64 without '=' padding
payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)   (UTF-8)
key = S2S_SESSION_SECRET as UTF-8 bytes (required, >= 32 chars; generate: python -c "import secrets;print(secrets.token_urlsafe(48))")
```

### 4.2 Payload fields (all required)

| field | type | meaning |
|---|---|---|
| `sid` | str | 22-char urlsafe random (`secrets.token_urlsafe(16)`), unique per call |
| `iat` | int | unix seconds at mint |
| `exp` | int | `iat + S2S_TOKEN_TTL_S` (default 120) |
| `mode` | str | `"lb"` or `"queue"`; the worker rejects a token for the other mode |
| `aud` | str | `S2S_AUDIENCE`, default the endpoint id; the worker rejects a mismatch. This stops a token minted for endpoint A from working on endpoint B with the same secret. |
| `record_id` | str\|null | as DEP1 |
| `pairing` | str\|null | `g1`..`g4` |
| `seed` | int\|null | as DEP1 |
| `max_s` | int | call cap the worker enforces: min(token `max_s`, worker `CALL_MAX_S`) |

### 4.3 API

```python
mint(secret: str, *, sid, mode, aud, record_id, pairing, seed, max_s, ttl_s=120, now=None) -> str
verify(secret: str, token: str, *, mode, aud, now=None, leeway_s=10) -> dict      # raises TokenError(code) with code in
                                                                                   # {"malformed","bad_sig","expired","not_yet","mode","aud"}
new_sid() -> str
```

- `hmac.compare_digest` is used for the signature check.
- Replay protection is NOT in s2s_token.py. The worker keeps the replay set (3.1), so the check is per worker. Under LB strict affinity a sid only ever reaches its claimed worker, so a per-worker set is sufficient.
- The relay mints a fresh token for each upstream action (claim, websocket open, release, metrics) with the same sid.
- A replay of the websocket token is refused by the worker's `busy`/`claimed(sid)` rule and by the replay set once the socket has started.

### 4.4 Env vars carrying the token config

| var | Space | worker | note |
|---|---|---|---|
| `S2S_SESSION_SECRET` | secret, required | env, required | the same value on both sides; never logged |
| `S2S_MODE` | `lb`\|`queue` | `lb`\|`queue` | must match |
| `S2S_AUDIENCE` | default `RUNPOD_ENDPOINT_ID` | required (set it to the endpoint id) | |
| `S2S_TOKEN_TTL_S` | 120 | n/a | |
| `CALL_MAX_S` | 300 | 300 | the worker value wins |

### 4.5 Test vectors (`common/test_vectors.json`, written by the builder of common/, checked by tests/test_token.py)

- Secret `"test-secret-0123456789-abcdefghijklmnop"`.
- Payload `{"aud":"ep-test","exp":1759600120,"iat":1759600000,"max_s":300,"mode":"lb","pairing":"g1","record_id":"food_01","seed":1001,"sid":"AAAAAAAAAAAAAAAAAAAAAA"}`.
- The expected token string is computed once with the reference implementation and frozen in the file.
- Negative vectors: a flipped signature byte (`bad_sig`), `now = exp + 11` (`expired`), `mode = "queue"` (`mode`).

---------------------------------------------------------------------------------------------------------------------
## 5. Space backend and client

### 5.1 Space HTTP API (`space/app/main.py`, aiohttp, `0.0.0.0:7860`)

| route | in | out |
|---|---|---|
| `GET /` + static | | built client (`space/static`) |
| `GET /api/config` | | `{"mode","passcode_required":bool,"max_call_s","max_concurrent":N,"build"}` |
| `GET /api/records`, `GET /api/records/{id}`, `GET /api/samples`, `/samples/*` | | same JSON as DEP1 INTERFACE section 5, computed locally with `common/session.py` (the Space image copies `common/`) |
| `POST /api/session` | `{"record_id","pairing","seed","passcode"?}` | `200 {"sid","state":"waking"}` \| `400` bad config \| `403 {"error":"passcode"}` \| `429 {"error":"capacity"\|"rate_limited","retry_after_s"}` |
| `GET /api/session/{sid}` | | `{"sid","state","detail","elapsed_s","worker_id"?,"end_reason"?}`; `state` ∈ `waking`, `ready`, `in_call`, `ended`, `failed`, `busy` (`busy` = all workers busy: claim kept getting 409, still retrying) |
| `DELETE /api/session/{sid}` | | `200`; releases the claim (LB) or cancels the job (queue) |
| `GET /api/chat?sid=…&<DEP1 params>` (ws) | | relay. Before the upgrade: `404` unknown sid, `409` not `ready`. After upgrade: frames are passed through both ways. The relay sends a synthetic `session_end` on upstream loss (2.1 step 9). The browser's `record_id`/`pairing`/`seed` must equal the values given at `POST /api/session` (they are bound into the token). |
| `GET /metrics?sid=…` | | the last 0x07 `metrics` event the relay saw for that sid, or `{}` |
| `GET /ws-echo` (ws) | | echoes text and binary frames. Diagnostic for test #1 (R1). |
| `GET /api/diag?sid=…` | | `{"space_to_worker_ms": <RTT of one strict-pinned GET /status>, "mode"}`. It works only for a sid in `ready`/`in_call`, and otherwise returns 409. It never sends an unpinned request, so it cannot wake or keep alive a worker on its own. Without that restriction, a public diag route would let anyone keep a billed worker up (R7). Rate-limited to 1 per 10 s per sid. |
| `GET /healthz` | | `200 ok` (Space liveness) |
| `GET /api/selftest/outbound?host=&port=` | passcode | `{"ok":bool,"ms","error"}`; one TCP connect from the Space. Only when `S2S_SELFTEST=1` (R17). |

Session table and limits (`sessions.py`):

- Everything is in memory; a Space restart drops all sessions.
- At most `MAX_CONCURRENT_CALLS` sessions in `waking`/`ready`/`in_call`. Set it equal to the endpoint's max workers.
- Per client IP (the `X-Forwarded-For` first hop): `RATE_PER_IP_PER_HOUR`=6 session creations and `MAX_CALLS_PER_DAY`=100 in total.
- Optional `S2S_PASSCODE`: when set, `POST /api/session` requires it. A public Space can otherwise start paid GPU workers for anyone (R7).
- Sessions are garbage-collected 10 min after `ended`/`failed`.

The relay uses aiohttp `ClientSession.ws_connect` (upstream) and `web.WebSocketResponse` (downstream).
- It uses one task per direction.
- Each message is passed through as is; there is no re-encoding.
- `max_msg_size=0` matches DEP1.
- Upstream heartbeat is off; the LB keepalive is the HTTP ping in 2.1 step 7.
- Downstream it uses aiohttp `heartbeat=20`, to keep the HF proxy from idling the browser socket.

### 5.2 Space packaging

- `space/README.md` header:
  ```yaml
  ---
  title: Hinglish Full-Duplex Agent (demo)
  emoji: 📞
  colorFrom: indigo
  colorTo: gray
  sdk: docker
  app_port: 7860
  pinned: false
  short_description: Talk to a Hinglish support agent (PersonaPlex + LoRA) on RunPod Serverless
  ---
  ```
  `app_port` is from [HF-DOCKER] and [HF-CFG].
- The Dockerfile follows [HF-DOCKER] "Permissions":
  - `useradd -m -u 1000 user`, `USER user`, `WORKDIR $HOME/app`;
  - `pip install -r requirements.txt` (aiohttp; no uvicorn);
  - `CMD ["python","-u","-m","app.main"]`.
- Secrets (`RUNPOD_API_KEY`, `S2S_SESSION_SECRET`, `S2S_PASSCODE`) go in Space **Secrets**, read at runtime from env. Non-secrets (`RUNPOD_ENDPOINT_ID`, `S2S_MODE`, `MAX_CONCURRENT_CALLS`, …) go in Space **Variables** ([HF-OV] "Managing secrets"; [HF-DOCKER] "Secrets ... Runtime").
- Visibility must be **public** or **protected**, never private. A private Space returns 404 to everyone else ([HF-OV] "Space visibility" table). That is the most likely cause of the websocket 404s reported in [HF-WS404] for clients without HF auth. The thread itself is unresolved (one reply blames `ws://` vs `wss://`).
- Websocket-404 avoidance, all required:
  - (1) the client uses `wss://` + `window.location.host` (5.4);
  - (2) the app server is aiohttp, which has native websocket support (the forum's other cause was a server without a websocket library);
  - (3) users open the direct `https://<owner>-<space>.hf.space/` URL, not the `huggingface.co/spaces/...` iframe page. This also avoids iframe microphone-permission issues (R8);
  - (4) test #1 (`tests/space_ws_echo.py`) is the first thing run after the Space is up. If `/ws-echo` gets 404, the front moves to launcher (c) on another host, with no code change.
- `space/app.py` is launcher (b), the Gradio-SDK shim for a free ZeroGPU Space. It imports `app.main` and runs it on `0.0.0.0:7860`. A dummy `@spaces.GPU` function is declared only if `SPACES_ZERO_GPU` is set, in case the platform requires one. This is untested (R3).

### 5.3 Upstream interface (`space/app/upstream.py`). The relay and session code use only this.

```python
class Upstream:                                   # LBUpstream(endpoint_id, api_key, base_url=None) | QueueUpstream(...)
    async def wake_and_claim(self, sess, on_state) -> None   # drives waking/busy/ready, sets sess.worker_id / sess.ws_url
    async def open_ws(self, sess, query: dict) -> aiohttp.ClientWebSocketResponse   # headers per mode (2.1 step 5 / 2.2 step 4)
    async def keepalive_loop(self, sess) -> None             # LB: strict-pinned GET /status every KEEPALIVE_S (10 s timeout, non-fatal); queue: no-op
    async def release(self, sess) -> None                    # LB: POST /session/release (strict); queue: /cancel/{job_id}
    async def diag(self) -> dict
```

- Base URLs come from env: `RUNPOD_LB_URL` (default `https://{RUNPOD_ENDPOINT_ID}.api.runpod.ai`) and `RUNPOD_API_URL` (default `https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}`).
- `tests/fake_runpod.py` sets both to localhost.

### 5.4 Client changes (fork `/root/deploy/client` → `client/`, minimal)

1. `Conversation.tsx` `buildURL`: `newWorkerAddr = window.location.host`. The DEP1 `hostname + ":" + port` gives `host:` on hf.space, where the port is empty. The code also appends `sid`.
2. `SessionSetup.tsx` Connect flow:
   - `POST /api/session`, then poll `GET /api/session/{sid}` every 1.5 s.
   - It shows a "Warming up the GPU… {elapsed}s ({detail})" panel for `waking`/`busy`, an error panel for `failed` or `429`, and a passcode field if `/api/config` says so.
   - Only on `ready` does it proceed to the existing Conversation (websocket) step.
   - A Cancel button sends `DELETE /api/session/{sid}`.
3. `Panels.tsx` latency readout: drop the 1 s `/metrics` HTTP poll and use the 0x07 `metrics` events the socket already carries every 2 s (DEP1 4.1). `fetchMetrics` is kept but pointed at `/metrics?sid=` for the setup page only. This avoids 1 req/s per viewer through the relay.
4. End of call: show `session_end.reason`, including the new `time_limit`, `worker_lost` and `worker_shutdown`. On `Disconnect`, also send `DELETE /api/session/{sid}`.
5. `env.ts`/`.env.local` stay unchanged (`VITE_QUEUE_API_PATH=/api`). The build is `npm ci && npm run build`. The build goes into `space/static` via `ops/build_client.sh` (runpod2 only; never on the laptop).

Nothing else changes: the protocol is byte-identical to DEP1 and the panels are untouched.

---------------------------------------------------------------------------------------------------------------------
## 6. Resources, image, models, cold start

### 6.1 GPU tier (measured on DEP1: PersonaPlex + LoRA + 2×Mimi 19.3 GB; + Trelis bf16 23.1 GB of 24.6 GB; frame p95 about 60 ms on the 3090)

- The per-frame step is memory-bandwidth bound: a 7B bf16 model reads about 14 GB of weights per frame. The 3090 runs at about 936 GB/s and needs about 46 ms of LM step.
- The console's 24 GB pool is "L4, A5000, 3090" ([EP-CFG] GPU table). The **L4 has about 300 GB/s**, which should put the step well over the 80 ms budget. That is an inference from bandwidth, not a measurement (R4). The pool cannot be split in the documented UI.
- **Recommendation, in priority order** (the console allows 3 GPU types, [EP-CFG] "GPU priority"):
  1. 48 GB `A6000, A40` ($0.00034/s ≈ $1.22/h; about 700-770 GB/s; 24 GB of headroom, so ASR and model fit comfortably);
  2. 24 GB `4090 PRO` ($0.00031/s; about 1 TB/s; 24 GB is tight but measured to fit on the 3090);
  3. 48 GB `L40, L40S, 6000 Ada PRO` ($0.00053/s).
  Do NOT select the `L4, A5000, 3090` pool unless a test shows the L4 keeps p95 < 80 ms. Prices are from [EP-CFG].
- With fewer than 5 workers, all workers use the highest-priority GPU available ([EP-CFG]).
- If 24 GB gets tight, `ASR_BACKEND=fw-small` moves ASR to CPU (DEP1 fallback; slower: about 10 s per trigger on 8 threads).

### 6.2 Image

- Base: `nvidia/cuda:13.0.x-base-ubuntu24.04`, the `-base-` variant as in [EX-LBWS] and [LB-BUILD]. The pip torch cu130 wheels carry their own CUDA and cuDNN libraries, so a `cudnn-runtime` base would duplicate about 3 GB.
- Python 3.11 and 3.12 are installed by uv.
- apt packages, from DEP1 `build_envs.sh`: `libopus-dev` (sphn Opus), `ffmpeg`, `curl`, `ca-certificates`. `espeak-ng` is not needed (it was for IndicF5 only). Pinned versions are from the runpod2 venvs (DEP1 `build_envs.sh`): torch 2.14.1+cu130, transformers 5.18.0, huggingface-hub 0.24.7 in venv-pp, cactus-needle 3.0.6, jax 0.10.2.
- **CUDA.** cu130 wheels need hosts with CUDA 13 drivers. The endpoint's CUDA filter must be 13.0+ ([EP-CFG] "CUDA version selection"). That shrinks the host pool (R5).
  - Build arg `TORCH_INDEX=cu130|cu128` switches both torch venvs to cu12.8 wheels for a wider pool. cu128 is untested and must be verified on runpod2 before use.
- **One torch copy.**
  - `venv-pp` and `venv-asr` are built in ONE `RUN` with `UV_LINK_MODE=hardlink` and a shared uv cache, so torch is stored once. Separate layers would duplicate it, about +5 GB.
  - `venv-needle` uses **CPU jax** (`jax[cpu]==0.10.2`, no `jax-cuda12-*`). Needle runs with `JAX_PLATFORMS=cpu` already (router_core), so dropping the CUDA plugins saves several GB.
- **Baked into the image:**

  | item | size |
  |---|---|
  | Trelis snapshot in `/opt/hf-asr` (`hf download Trelis/whisper-hinglish-preview` at build time; public, no token) | 5.8 GB |
  | V3 adapter dir | 370 MB |
  | `tuned_full.cact` | 63 MB |
  | `/root/.cache/cactus-needle/v3/3.0.2` (`libneedle.so` + base `needle3.cact`, copied from runpod2) | 35 MB |
  | needle runtime subset + code + records | under 10 MB |

  Optional: convert Trelis to bf16 safetensors at build time. That makes it 2.9 GB, and it is numerically identical, because asr_service loads it as bf16 anyway.
- **Size estimate.**

  | part | size |
  |---|---|
  | CUDA base + apt (ffmpeg, opus) | about 0.6 GB |
  | torch cu130 + deps, stored once | about 5.5 GB |
  | transformers etc. | about 0.3 GB |
  | needle venv, CPU jax | about 1 GB |
  | Trelis fp32 | 5.8 GB |
  | adapter etc. | 0.5 GB |
  | **total** | **about 13-14 GB** (about 11 GB with Trelis in bf16) |

  Container disk setting ≥ 30 GB.
- No HF token or RunPod key in any layer. `HF_TOKEN` is read only at runtime, by the fallback downloader (6.3).
- PersonaPlex weights are NOT baked. They are gated, and baking them into an image pushed to a registry redistributes them. Option B (6.4) keeps them in a private HF repo.

### 6.3 Model resolution (`worker/resolve_models.py`, runs first in stack.sh)

PersonaPlex files needed: `model.safetensors`, `tokenizer-e351c8d8-checkpoint125.safetensors` (Mimi), `tokenizer_spm_32k_3.model`, `voices.tgz` or `voices/`. The resolver tries these in order:

1. `S2S_PP_DIR` (explicit dir: a network volume, a pre-merged snapshot, or a local path on runpod2 for testing).
2. The RunPod cached model: `/runpod-volume/huggingface-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/<refs/main or first>/`. This is the [HF-MODELS] `resolve_snapshot_path` helper ([CACHE] "Where cached models are stored").
3. A network-volume HF cache: `/runpod-volume/hf/hub/models--nvidia--personaplex-7b-v1/...` (layout of [VOLCACHE]). Do not attach a network volume together with a cached model: both use `/runpod-volume` ([CACHE] note).
4. If `S2S_ALLOW_DOWNLOAD=1` and `HF_TOKEN` is set: `hf_hub_download` into `/tmp/s2s/hf`. This is slow (16 GB per cold start, billed), logged as a WARNING, and meant only for debugging.

If none of these works, the resolver exits 2 and `/ping` returns 500.

Voices:
- If `<dir>/voices/NATF2.pt` exists, it is used.
- Otherwise `voices.tgz` is extracted to `/tmp/s2s/voices`. The cache mount should be treated as read-only; DEP1's `_get_voice_prompt_dir` writes next to the blob and would fail there.

Outputs:
- `S2S_PP_DIR` and `S2S_VOICES` are written to `/tmp/s2s/models.env`.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set for every process ([HF-MODELS] "Force offline mode").

### 6.4 Options for faster cold start (documented; not the default)

- **A. Pre-merged weights.**
  - Merge the V3 LoRA into `model.safetensors` offline, once, on runpod2 with `merge_lora` + `save_file`.
  - Upload it with the Mimi, tokenizer and voices files to a **private** HF repo `<user>/personaplex-7b-v3merged`, and cache THAT repo as the endpoint model (the HF token goes in the endpoint model setting).
  - The worker runs with `S2S_ADAPTER=premerged`.
  - This saves the merge step (seconds) and a 370 MB read. The NVIDIA license must be checked before re-hosting merged weights; the repo must stay private.
- **B. Everything in one private repo.**
  - Option A plus the Trelis snapshot and the `.cact` in subfolders, cached as the single model.
  - The image drops to about 10 GB, so image pull dominates less. The resolver gets `S2S_ASR_DIR` and `S2S_NEEDLE_WEIGHTS` pointing into the snapshot.
- **C. Network volume.**
  - Put `/runpod-volume/hf` (PersonaPlex + Trelis) on a network volume in one data center ([VOLCACHE]).
  - Slower loads than the cache ([CACHE] note), and it pins the endpoint to that DC ([EP-CFG] "Network volumes").
  - Use it only if model caching misbehaves.
- **FlashBoot** stays enabled; it is the default ([EP-CFG] "FlashBoot").

### 6.5 Cold-start budget (estimates; the user measures with `tests/cold_start_timer.py`)

| phase | estimate | note |
|---|---|---|
| Space wake (cpu-basic slept > 48 h) | 10-60 s | [HF-GPUS]; rare |
| RunPod schedules a worker on a cached-model host | 5-30 s | no billing for the model download if the host lacks the cache ([CACHE] "How it works") |
| image pull (about 13-14 GB) on a host without the image | 2-6 min | first time per host; FlashBoot and host image cache make repeats fast |
| resolve + voices extract | under 5 s | |
| PersonaPlex load + LoRA merge + warmup (DEP1 measured about 35-60 s for the full start.sh, "model load about 1-2 min") | 40-90 s | the disk read of 16 GB dominates |
| Trelis load | 10-20 s | runs after the server, as DEP1 |
| router (Needle) load | under 5 s | |
| **cold start, image cached** | **about 1-2 min** | |
| **first ever on a host** | **about 4-8 min** | covered by `WAKE_TIMEOUT_S`=600 |
| prompt phase per call | about 9-10 s | DEP1 measured 9.1-9.6 s |

`min_workers` (active workers) = 0 by default, so the scale-to-zero holds. For a scheduled demo, set active workers to 1 for that window: no cold start, but billed 24/7 while set ([EP-CFG] "Active workers"; [PRICE] "Worker types").

### 6.6 Cost notes (ops/COSTS.md)

- Billing is per second, from worker start until it fully stops, and includes start time and idle time ([PRICE] "Compute cost breakdown").
- One call on 48 GB A6000 ($0.00034/s) is billed for about 90 s cold start + about 240 s call + 120 s idle timeout, about 450 s ≈ **$0.15**. Back-to-back calls on a warm worker cost about $0.08-0.10 each.
- Container disk costs about $0.10/GB/month ([PRICE]).
- Account spend limit is $80/h by default ([PRICE] "Account limits").
- `MAX_CONCURRENT_CALLS`, `RATE_PER_IP_PER_HOUR`, `MAX_CALLS_PER_DAY` and `S2S_PASSCODE` on the Space are the cost brakes.

---------------------------------------------------------------------------------------------------------------------
## 7. Interfaces for parallel builders

| builder | owns | depends on (frozen here) |
|---|---|---|
| **W (worker)** | `worker/**`, `common/session.py` fork | section 3, the token API 4.3, the paths in 1.2, the model resolution in 6.3 |
| **S (Space + client)** | `space/**`, `client/**` | section 5, the token API 4.3, the worker routes and status codes in 3.1, the upstream flows in 2.1/2.2 |
| **C (common, ops, tests)** | `common/s2s_token.py`, `common/test_vectors.json`, `ops/**`, `tests/**` (incl. `fake_runpod.py`) | sections 2, 4, 6; the route tables 3.1/5.1 |

Contracts between them, exactly:

- **Token.** `common/s2s_token.py` as in 4.3. C writes it first; W and S import it via `sys.path.insert(0, <repo>/common)`, the same pattern as DEP1 session.py.
- **Worker HTTP/WS.** The route table in 3.1, including status codes, JSON bodies, close codes 4401/4409/4503, and the header names `X-S2S-Token` and `X-Runpod-Worker-Id`.
- **Worker id.** The `/session/claim` JSON `worker_id` = env `RUNPOD_POD_ID` if set, else `socket.gethostname()`. The Space prefers the proxy's response header when present (2.1 step 4).
- **Queue progress payload.** `{"state":"loading"|"ready"|"ended","public_ip":str,"tcp_port":int,"worker_id":str,"sid":str}`. The handler output is `{"summary":{...DEP1 session summary...},"end_reason":str}`.
- **fake_runpod.py** (C) behaviour that W and S test against:
  - LB proxy on `127.0.0.1:18080`:
    - requires `Authorization: Bearer test-key`;
    - forwards to one or more local workers;
    - answers "no workers available" (HTTP 503 here, after a short hold) while no worker returns 200 on `/ping`;
    - sets `X-Runpod-Worker-Id` on responses;
    - honours `strict <id>` (404 `affinity_worker_gone` if that worker is down);
    - proxies websockets.
  - Queue API on `127.0.0.1:18081`: `/run` starts `rp_handler.handler` in a thread with `RUNPOD_PUBLIC_IP=127.0.0.1` and `RUNPOD_TCP_PORT_<port>`; `/status` returns its progress; `/cancel` stops it.
  - Not emulated: the real 2-min hold, scaling and billing.
- **Mock engine.** `worker_server.py --mock` uses DEP1 `mock_server.MockEngine` (CPU replay of a V3 call). `ASR_BACKEND=off ROUTER=off` (or the DEP1 router against the mock) gives a GPU-free worker for all local tests.
- **Ports in tests.** Worker `$PORT`=18000 (+1 per extra worker), internal 18999 (`--internal-port`). Space 17860.

---------------------------------------------------------------------------------------------------------------------
## 8. Verification in this build (no docker, no deploy)

- **Static review of both Dockerfiles** against section 6.2, plus the path-grep rule (1.2).
- **CPU, any time.**
  - `tests/test_token.py`.
  - `tests/test_mock_e2e.py`, in both modes: mock worker + fake_runpod + Space + a scripted websocket client. It checks:
    - the state sequence `waking`→`ready`→`in_call`→`ended`;
    - a second call → 409 / `busy`;
    - tampered and expired tokens → 401;
    - the time cap → `time_limit`;
    - killing the worker mid-call → `worker_lost`;
    - `/ws-echo`.
- **GPU smoke (Integrate stage only).**
  1. `bash /root/deploy/stop.sh`.
  2. Under `flock /root/gpu.lock`, run `worker/entrypoint.sh` outside docker on runpod2, with `S2S_PP_DIR` = the runpod2 snapshot, `S2S_ROOT` = the repo, and the venvs = the DEP1 venvs (or new ones built from requirements-*.txt).
  3. Drive one call with `server/ws_feed.py` through the Space backend + fake_runpod.
  4. `bash /root/deploy/start.sh`, and confirm it is healthy.

---------------------------------------------------------------------------------------------------------------------
## 9. The user's later real tests (scripts in tests/, how-to in ops/BUILD.md)

1. `space_ws_echo.py https://<space>.hf.space`: wss echo through the HF proxy (R1). Run this first.
   - The same script also calls `GET /api/selftest/outbound?host=…&port=…`. Through that route the Space opens one TCP connection to a public non-443 test port and reports the result, which tells whether queue mode can work from the Space (R17).
   - The route is passcode-gated and must not be left enabled: `S2S_SELFTEST=1` turns it on.
2. `cold_start_timer.py`: `POST /api/session` → `ready`, with the time spent in each state, three runs:
   - a cold host;
   - scale-from-zero on a cached host;
   - warm (inside the idle timeout).
3. `hold_test.py --minutes 4`: holds one call through the Space for 4 min (silence + periodic tones). It records:
   - drops, and whether the end was `context_full` or `time_limit` vs `worker_lost`;
   - keepalive status codes.
   Run it three times:
   - with `KEEPALIVE_S=0`, to learn whether RunPod counts the open websocket as work (R2);
   - with the default (`/status` keepalive);
   - with `KEEPALIVE_PATH=/ping`, to learn whether health-path traffic counts as work.
4. `latency_probe.py` (from India). It measures:
   - browser→Space TCP+TLS and websocket echo RTT;
   - Space→RunPod `/api/diag`;
   - the full-path audio round trip, from a click sent to its first model audio, against the DEP1 tunnel baseline.
   Then pick RunPod data centers near the Space region ([EP-CFG] "Data centers").
5. GPU check: `/status` `gpu` and `step_ms_p95_last` on whichever GPU the endpoint got. p95 must be below 80 ms.

---------------------------------------------------------------------------------------------------------------------
## 10. Open risks (for the user's later testing)

| id | risk | how it shows | mitigation / test |
|---|---|---|---|
| R1 | Websockets through the HF Space proxy return 404 ([HF-WS404], unresolved upstream) | client never connects | test #1; public or protected visibility; direct `*.hf.space` URL; fallback front = the same container on another host |
| R2 | The RunPod LB does not count an open websocket as work, or applies the 330 s cap differently | the call drops mid-way (`worker_lost`) | strict-pinned keepalive every 20 s + idle timeout 120 s; call cap 300 s; hold test #3; last resort: active workers = 1 during demos |
| R3 | Paid plan needed for a Docker/Gradio Space ([HF-OV]); the ZeroGPU shim is untested | the Space cannot be created on a free account | PRO, or launcher (b) on ZeroGPU, or launcher (c) elsewhere |
| R4 | A slow GPU (L4-class bandwidth) misses the 80 ms frame budget | `step_ms_p95` > 80, audio backlog, choppy speech | GPU priority in 6.1; `/status` check; startup warning |
| R5 | Too few CUDA-13 hosts (torch cu130) | long waits for a worker | CUDA filter 13.0+; cu128 build arg (verify on runpod2 first) |
| R6 | Queue mode: plaintext audio between RunPod and the Space; public TCP port open | | token required on every route; use LB mode in production |
| R7 | A public Space starts paid GPU workers for anyone (also via any Space route that sends unpinned requests to the endpoint) | bill | passcode, per-IP rate, daily cap, `MAX_CONCURRENT_CALLS` = max workers; `/api/diag` only for an active sid and pinned; the selftest route is off by default |
| R8 | Microphone blocked inside the huggingface.co iframe | no audio permission prompt | link the direct `*.hf.space` URL |
| R9 | The endpoint is idle-throttled after 3 or 7 days ([EP-CFG]) | the wake never succeeds | ops note: raise max workers again in the console; Space shows `failed` with detail |
| R10 | The extra relay hop (India → Space region → RunPod DC) adds RTT, and Opus pages are relayed as is | higher turn latency than the DEP1 tunnel | latency probe; co-locate DC with the Space |
| R11 | LB routes a claim to a busy worker (no "busy" signal to the proxy, since `/ping` stays 200 to protect live calls) | `busy` state for some seconds | claim retry loop; `MAX_CONCURRENT_CALLS` = max workers; request-count scaler |
| R12 | The cached-model mount layout or `refs/main` differs from the docs | `/ping` 500, "PersonaPlex not found" in the logs | the resolver's fallback order and clear log; `S2S_PP_DIR` override |
| R13 | Serverless vCPU count is unknown (DEP1 needed a 4-thread cap; Needle uses about 12 threads for 0.3 s) | step p50 near 100 ms during triggers | thread caps kept; check `/status` during a trigger |
| R14 | The `X-Runpod-Worker-Id` header is missing on the websocket upgrade or claim response | the keepalive is unpinned | body `worker_id` fallback (RUNPOD_POD_ID); if that does not match the proxy's ids, keepalive without affinity and rely on active workers |
| R16 | Strict-affinity requests (keepalive, release, diag) queue behind the open websocket if it occupies the worker's capacity slot ([LB-AFF]: strict "waits up to ~5 minutes") | keepalives time out | 10 s client timeout, non-fatal; hold test shows whether keepalive codes are 200 or timeouts; if they always time out, use active workers = 1 for demos |
| R17 | HF Spaces may restrict outbound connections to non-standard ports (unconfirmed). Queue mode needs `ws://IP:<random RUNPOD_TCP_PORT>` | queue-mode relay cannot connect | outbound self-test in test #1; if it fails, queue mode needs launcher (c) on another host |
| R15 | An image of about 13-14 GB is slow to pull on fresh hosts | cold starts of several minutes | option B (6.4); bf16 Trelis |

---------------------------------------------------------------------------------------------------------------------
## Amendment S1 (2026-10-04, Space builder S, small additions found while building space/ and client/)

These are additions only; nothing in a frozen contract changes meaning. The number is "S1" so that builders working in parallel cannot pick the same N.

1. **Synthetic `session_end` from the relay.** The relay adds two fields: `"detail"` (text for people) and `"source":"space"`.
   - **New reason `relay_error`.** The relay sends it when the browser socket has already been upgraded but opening the upstream websocket fails. The failure can be an HTTP refusal (401/409/503), a timeout longer than `UPSTREAM_OPEN_TIMEOUT_S`, or a connect error. The relay then closes with code 1011. A browser websocket never sees HTTP status codes, so without this event the UI could not tell the user why.
   - **`time_limit` from the Space.** The relay also sends `time_limit` when the Space cap (`CALL_MAX_S` + 15 s) fires and the worker never sent `session_end`.
   - The client maps every reason to text in `client/src/dep1/types.ts`, `END_REASON_TEXT`.
2. **Additions to `/api/config` and the Space env.**
   - `/api/config` also returns `allow_free_prompt`, `claim_ttl_s` and `ok`. `ok` is false when required env is missing.
   - New Space env `S2S_ALLOW_FREE_PROMPT`, default 0. It has the same name and meaning as the worker's. At 0 the client hides the stock free-prompt panel, and `POST /api/session` requires a `record_id`.
   - `POST /api/session` returns `503 {"error":"space_misconfigured"}` while required env is missing: RUNPOD_API_KEY, the endpoint id, or a secret of at least 32 characters.
3. **Additions to `GET /api/session/{sid}`.**
   - `state_elapsed_s`.
   - `timeline`, as `[[state, t_since_created_s, detail], ...]`, for tests/cold_start_timer.py.
   - `keepalive` during a call, as `{"sent","last","codes"}`, for the hold test.
   - Two Space-side `end_reason` values for sessions that never reached a call:
     - `cancelled`: the session was deleted with DELETE.
     - `claim_expired`: the session stayed `ready` longer than `CLAIM_TTL_S`. The Space then releases the claim or cancels the job.
4. **`Upstream.diag(sess)` takes the session.** DESIGN 5.3 wrote `diag(self)`, but diag must be pinned to that session's worker.
5. **The relay normalises the upstream query.**
   - It drops `sid` and `token` from the upstream query.
   - It rewrites `record_id`, `pairing` and `seed` to the canonical values bound into the token. It checks the browser's values first, and a mismatch gets 400 `cfg_mismatch` before the upgrade.
   - The canonical values are: a missing `seed` with a record means 1001, a missing `pairing` means g1, and `seed=-1` stays -1.
6. **How `common/` reaches the Space.** The Space builds from `space/` alone, so `common/` has to be copied in.
   - `space/stage_common.sh` copies `common/{s2s_token.py, session.py, data/records_v4.json}` into `space/common/` before the push.
   - `app/main.py` resolves `common/` in this order: `S2S_COMMON`, then `space/common`, then `../common`. It sets `S2S_RECORDS` to `<common>/data/records_v4.json` when that file exists.
   - The Space Dockerfile fails the build if `space/common` is missing.
7. **The wake fails fast on bad credentials.** It ends `failed` at once, with a hint naming the env var to check, instead of polling for 600 s. This happens when:
   - RunPod answers the wake with 401/403;
   - the worker answers the claim with 401.
8. **Queue `sid` check.** In queue mode, a `progress_update` with `state:"ready"` and a `sid` that is not the session's ends the session as `failed`.
9. **Space-only test doubles in `space/tests/`.**
   - `fake_runpod_lite.py`, `test_space_flows.py`, `e2e_browser.mjs` + `run_e2e_browser.sh` (Playwright), and `xcheck_repo_fakes.py` (the Space against C's `tests/fake_runpod.py` + `tests/stub_worker.py`).
   - They use ports 27xxx/28xxx, so they can run alongside `tests/` (17860/18xxx).
   - They are not part of the Space image (`.dockerignore`).

## Amendment S2 (2026-10-04, Space builder S, after review)

1. **Post-upgrade refusals.** When the upstream websocket closes with code 4401, 4409 or 4503 (DESIGN 3.1) and the worker sent no `session_end`, the relay reports `relay_error` with a detail line (`auth`, `busy` or `loading`) and closes with 1011. It does not report `worker_lost`.
2. **Every failed wake releases.** A failed wake now always calls `Upstream.release(sess)`, whether it failed through `WakeFailed` or a crash.
   - In queue mode this cancels a job that may already hold a billed worker, for example a `ready` progress update with no `tcp_port`.
   - In LB mode it frees a claim that may have just succeeded.
   - `release` is idempotent.
3. **Queue `sid` check.** A queue `ready` progress fails the session only when it carries a `sid` that is different from the session's. A missing `sid` is accepted, because the job id already ties the progress to this session. W's `rp_handler.py` sends the `sid`.

## Amendment W1 (2026-10-04, worker builder W, found while building worker/)

The number is "W1" so that builders working in parallel cannot pick the same N. Route changes are additive only. Every frozen status code, header name and JSON field in 3.1 keeps its meaning.

1. **The model resolver runs inside worker_server's load task, not before it in stack.sh.**
   - Section 3.2 orders it first. With that order, a resolver failure leaves nothing listening on `$PORT`, so `/ping` could never answer 500, and the container would exit and restart in a loop.
   - Now `worker_server.py` binds first. Its load task calls `resolve_models.resolve()` and then builds the Engine. Any failure there means `/ping` 500 and `/status` `state:"failed"` with the reason in `detail`. The same happens for config errors: `S2S_SESSION_SECRET` missing or under 32 chars, no `S2S_AUDIENCE`/`RUNPOD_ENDPOINT_ID`, or a bad `S2S_MODE`.
   - `resolve_models.py` can still be run alone for diagnosis (exit 2 = not found).
   - stack.sh waits for `/status` `"engine": true`, then starts ASR, then the router. It cannot wait for `/ping`, because `/ping` needs ASR and router.
2. **Build context, exact layout.** `docker build -f worker/Dockerfile .` runs from the repo root. `ops/prepare_build_context.sh` (builder C) must create:
   - `worker/build/v3_adapter/{config.json,lora.safetensors}`, from the V3_CKPT `path=` dir;
   - `worker/build/tuned_full.cact`;
   - `worker/build/cactus-needle/v3/3.0.2/{libneedle.so,needle3.cact}`, from `~/.cache/cactus-needle`.

   These must be **real files** (`cp -L`), not symlinks: Docker COPY copies a symlink as a link.

   Already in the repo:
   - `common/data/records_v4.json`;
   - **`worker/vendor/moshi`**. This is the personaplex `moshi` package source, 1.1 MB, identical to `/root/deploy/assets/personaplex/moshi` minus build/egg-info. It is installed with `--no-deps`. DEP1's venv-pp had it as an editable install, and section 1 does not list it.

   `python3 worker/tests/check_dockerfile.py [--root DIR]` statically checks every COPY source against `worker/Dockerfile.dockerignore`.
3. **venv-needle has no jax at all**, which is stronger than "CPU jax" in 6.2.
   - The Needle runtime is a ctypes native library. Checked on runpod2: `router.route_raw` imports no jax, flax or optax.
   - `requirements-needle.txt` has 18 pins, 81 MB, compiled against the working `/root/venv-needle`.
   - The full CPU stack (mock engine + faster-whisper-small on CPU + Needle) ran trigger, asr, needle, resolved, executed with it.
4. **The runpod SDK lives in its own `/opt/venv-rp`**, `requirements-rp.txt`, runpod==1.10.0.
   - It pulls aiohttp 3.14.3, boto3 and fastapi. That would move venv-pp off the DEP1-tested pins.
   - `rp_handler.py` is stdlib-only and imports `runpod` lazily. Tests set `rp_handler.PROGRESS_HOOK = callable(job, payload)`, or install a fake `runpod` module as `tests/fake_runpod.py` does.
5. **Paths.**
   - `hinglish/` and `needle/` live under `worker/` (image: `/opt/s2s/worker/{hinglish,needle}`), not `/opt/s2s/{hinglish,needle}` as in table 1.2. The env names are unchanged.
   - Trelis is baked as a local dir `/opt/hf-asr/trelis` (`S2S_ASR_DIR`), at revision `eab1188fd2d0e91f2584229b32b3bfe1901c896c` (runpod2 cache refs/main), with an explicit file list.
   - Ports are env-driven: `S2S_INTERNAL_PORT`, `S2S_ASR_PORT`, `S2S_ROUTER_PORT`. Image defaults are 8999/8996/8995. entrypoint sets `PORT` to 80 (lb) or 8765 (queue) when unset.
   - Other env: `S2S_RUNPOD_VOLUME` (default `/runpod-volume`, for tests), `S2S_TMP`, `S2S_MODELS_ENV`, `S2S_MOCK_{REPLAY,INPUTS,SPM}`.
   - `worker/local/` (`runpod2.env`, `run_local.sh`) is the only place in worker/ that names runpod2 paths. It is exempt from the 1.2 grep rule and excluded from the image.
6. **Additive routes and fields.**
   - `/status` also returns `engine`, `mode`, `detail`, `uptime_s`, `claim_ttl_left_s`, `asr_backend`, `router_mode`, `health`, and `active.elapsed_s`.
   - Internal listener (queue handler): `POST /internal/claim_release {"sid"}`, `GET /internal/claim/{sid}` → `{"sid","phase":claimed|busy|ended|expired|released|unknown,"end_reason","summary"}`, `GET /internal/status`.
   - If the endpoint sets `HEALTH_CHECK_PATH`, the `/ping` handler is also served there.
7. **Error bodies, as built.**
   - 401: `{"error":"bad_token"|"expired","detail":<TokenError code or "missing">}`, `{"error":"replayed"}`, `{"error":"cfg_mismatch"}`. A re-claim of the same sid with a different config also gets 401 `cfg_mismatch`.
   - 409: `{"error":"busy"}`. The websocket also gets 409 `{"error":"no_claim"}` when the worker is idle.
   - 400: `{"error":"free_prompt_disabled"}`.
   - 503: `{"error":"loading"}`, or `{"error":"failed"}` after a failed load.
   - 403: `{"error":"not_active"}` for `/metrics` with a valid token of a sid that is not the claimed or busy one.
   - The websocket upgrade refusal order is: loading → token → config (400) → free prompt → cfg_mismatch → replayed → busy/no_claim.
8. **Config normalisation (D-W-1, same as S1.5).** The token's and the query's (record_id, pairing, seed) are compared after DEP1 defaults:
   - with a record_id: pairing missing → g1, seed missing → 1001, and -1 stays -1;
   - without a record_id: pairing None, seed None or -1 → None.
9. **Behaviour details.**
   - `POST /session/release` during a call ends it with `client_closed`.
   - If `time_limit` or SIGTERM hits during the prompt phase, the worker sends `session_end` with `frames:0` before closing.
   - A prompt-phase exception gives `error`, not `client_closed`.
   - Each session summary is also printed as one stdout line `SESSION_SUMMARY {json}`, with `sid`, `worker_id`, `gpu` and `mode`; the token is never logged.
   - The base image is `nvidia/cuda:13.0.3-base-ubuntu24.04` (tag checked on Docker Hub; it matches cuda-toolkit 13.0.3 in the venvs). Bytecode is compiled at build time (`UV_COMPILE_BYTECODE=1`) to save import time on a cold start.
   - The image sets `NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1` and `HF_HUB_DISABLE_TELEMETRY=1`. cactus-needle otherwise sends telemetry.
10. **ASR fallback.** `ASR_BACKEND=fw-small`/`fw-medium` (6.1) is NOT baked into the image. With `HF_HUB_OFFLINE=1` it fails at start. To use it, bake `Systran/faster-whisper-small` into `/opt/hf-asr` and set `S2S_ASR_HF_HOME`, or use `ASR_BACKEND=off`.
11. **Tools added.**
    - `worker/server/ws_feed.py` gains `--token`, `--sid` and `--header` for the GPU smoke test.
    - `worker/tools/premerge.py` implements option A of 6.4. It is not run in this stage.

## Amendment W2 (2026-10-04, worker builder W, after review)

1. **The image needs a C compiler.**
   - moshi wraps `gating`, `rope` and `seanet` in `torch_compile_lazy`. At the first warmup, inductor/Triton compiles a small C launcher. Triton's `runtime/build.py` raises "Failed to find C compiler" when there is no `gcc`/`cc`.
   - The `-base-` CUDA image has none, so the apt line now includes `gcc libc6-dev`. The final self-check RUN asserts `command -v gcc` and that `Python.h` exists.
   - runpod2 has gcc, so no outside-Docker run can catch this.
   - `NO_TORCH_COMPILE` is deliberately NOT used: it would change the per-frame path that DEP1's p95 was measured on.
2. **Cold start includes the torch.compile of those kernels.**
   - On runpod2 the caches are warm (`/tmp/torchinductor_root`, `~/.triton/cache`). A fresh serverless worker starts with empty ones. The 6.5 estimate "PersonaPlex load 40-90 s" may therefore be low.
   - `run_local.sh gpu` with `S2S_FRESH_COMPILE_CACHE=1` points `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at empty dirs, so a GPU stage can measure a realistic cold start.
   - The caches cannot be baked at build time: there is no GPU at build time, and they depend on the GPU architecture.
3. **`TORCH_INDEX=cu128` also needs `--build-arg CUDA_BASE=nvidia/cuda:12.8.2-base-ubuntu24.04`** (tag checked on Docker Hub).
   - The 13.0.3 base sets `NVIDIA_REQUIRE_CUDA=cuda>=13.0`, so the container runtime would refuse older drivers whatever torch build is inside.
   - Both options are still untested.

## Amendment C1 (2026-10-04, common/ops/tests builder C, found while building common/, ops/ and tests/)

The number is "C1" so that builders working in parallel cannot pick the same N. Contracts in sections 3.1, 4 and 7 are unchanged.

1. **File names (ops task vs section 1).**
   - `tests/hold_test.py` is named **`tests/ws_hold_test.py`**.
   - `ops/BUILD.md` remains, but only as an index. The commands live in **`ops/build_and_push.sh`** (docker build/push, dry run unless `--run`/`--push`), **`ops/create_endpoint.sh`** (RunPod API, dry run unless `--yes`) and **`ops/space_push.md`**.
   - New files: `tests/TEST_PLAN.md` (the order of section 9 plus the deploy steps), `tests/rt_common.py` (shared code of the real-test scripts), `tests/stub_worker.py` (a CPU test double of the 3.1 public contract) and `tests/test_fake_runpod.py` (self-test of the harness, ports 38xxx).
   - `tests/test_mock_e2e.py` takes `--base` (default 18000 as in section 7). Use another base while other builders test on 18xxx.
2. **Every real-test script is a dry run by default.** Without `--space URL` or `--direct lb|queue` it prints its plan and exits 0. With `--direct` and missing env it exits 2 and sends nothing. Secrets come from env only and are printed masked.
   - `--direct lb` does the Space's job from the test machine: `/status` wake, token claim, strict-pinned websocket with Bearer, and its own keepalive (`--keepalive-path /status|/ping`, `--keepalive-s`, where 0 = off). This is the preferred way to run the 3-run hold matrix of section 9 #3, because it needs no Space restarts.
   - Through the Space, the Space's `KEEPALIVE_*` variables apply.
   - Keepalive defaults follow this DESIGN (`/status`, 20 s, strict), not the ops task's "/ping every 30 s". That variant is run C of the matrix.
3. **A hold "PASS" includes `context_full`.** The context ends at about 221 s (DEP1 A1.4), so a 4-minute hold normally ends with `context_full` (or `time_limit`). A DROP is a close without `session_end`, or reason `worker_lost`, `worker_shutdown` or `error`.
4. **`PORT_HEALTH` must be set explicitly on LB endpoints (`PORT=80`, `PORT_HEALTH=80`).** This contradicts section 1.1's "PORT_HEALTH unset". runpod/docs issue #853 (open) reports that RunPod injects `PORT_HEALTH=80` when `PORT` is not 80, and that `HEALTH_CHECK_PATH` is ignored. Our worker serves `/ping` on `$PORT`, so `PORT=80` (W's entrypoint default) plus an explicit `PORT_HEALTH=80` is safe either way.
5. **API facts for ops (checked 2026-10-04).**
   - LB endpoints can be created with REST API v2 `POST https://api.runpod.io/v2/serverless` (`type: "LOAD_BALANCER"|"QUEUE"`) or with GraphQL `saveEndpoint(type:"LB")`. REST v1 (`rest.runpod.io/v1/endpoints`) has no type field.
   - API v2 defaults differ from the console: `flashboot` defaults to `OFF`, `workers.idleTimeout` to 10 s and `timeout` to 300000 ms. create_endpoint.sh sets `FLASHBOOT`, idle timeout 120 s, and timeout 330000 ms (LB) or 600000 ms (queue).
   - **No API field exists for the cached Model** (v1, v2, GraphQL). It is a console step, printed by the script.
   - GPU pools: `AMPERE_48`, `ADA_24`, `ADA_48_PRO` (docs gpu-types page). `AMPERE_24` (L4/A5000/3090) is excluded. The `ADA_24` = "4090 PRO" label is unconfirmed; the script has a `--list-gpus` read-only check.
   - Endpoint CUDA filter: `gpu.minCudaVersion: "13.0"`.
   - Scaling: the LB endpoint uses `REQUEST_COUNT` 1, one worker per in-flight call/claim. The queue endpoint uses `QUEUE_DELAY` 1 s, because API v2 says `idleTimeout` is "not applicable to queue-based endpoints scaling on requestCount", and we want the 120 s warm tail there too.
6. **`S2S_AUDIENCE`: set an explicit name on both sides** (e.g. `hinglish-lb-1`). Do not rely on the default (the endpoint id). Our sources do not confirm that RunPod injects `RUNPOD_ENDPOINT_ID` into the worker, and an explicit value avoids editing the endpoint after its id is known.
7. **`{{ RUNPOD_SECRET_name }}`** is documented for Pod templates only. ops treats it as optional and unverified for serverless (`create_endpoint.sh --secret-ref`). The default is the plain env value.
8. **Token test vectors** are frozen in `common/test_vectors.json`, computed once with `common/s2s_token.py` on 2026-10-04. `python3 common/s2s_token.py` self-checks them.
   - Two decisions inside 4.1/4.3: `payload_json` uses `json.dumps(..., sort_keys=True, separators=(",",":"))` with the default `ensure_ascii=True`.
   - `verify` also raises `malformed` for a validly signed payload that lacks a field, or has non-int `iat`/`exp`.
9. **Measured harness baseline.** Opus encode + decode in the test client (sphn) adds about 240 ms to `pipeline_lag_ms` on localhost, measured against the echo stub. latency_probe results must therefore be compared with `--baseline-url` (DEP1), not with 0.

## Amendment R-ops1 (2026-10-05, ops reviewer)

Section 2.3 row "queue execution timeout" is now **900 s** (was 600 s). The runpod SDK picks the job up when the
container starts, so the model load counts against it: load cap 420 s (`S2S_QUEUE_LOAD_TIMEOUT_S`) + claim retry
30 s + `CLAIM_TTL_S` 90 s + `CALL_MAX_S` 300 s + 30 s margin = 870 s. Space `QUEUE_EXEC_TIMEOUT_MS` default 900000;
endpoint env `S2S_QUEUE_EXEC_TIMEOUT_S=900`. Details: DECISIONS D-REVIEW-ops, REVIEW-ops-3.

## Amendment R-space (2026-10-05, Space reviewer; additive)

1. **Non-upgrade request.** `GET /api/chat` without a websocket upgrade returns `400 {"error":"websocket_required"}` before any state change, and the session stays `ready`.
2. **Every relay exit ends the session.**
   - Every relay exit after `in_call` ends the session.
   - Ends other than a worker `session_end` or `client_closed` also call `Upstream.release` (LB: release; queue: cancel).
   - A DELETE during the upstream open is honoured when the open returns.
3. **New Space env `ABANDON_S`** (default 120, 0 = off). A `waking`/`busy` session that the browser has not polled for that long is cancelled, with the new Space-side `end_reason` `abandoned`.
4. **Janitor backstop.** An `in_call` session older than `CALL_MAX_S` + `UPSTREAM_OPEN_TIMEOUT_S` + 60 s is ended and released.
5. **`space/README.md` `short_description`** is now "Hinglish support-agent voice demo on RunPod Serverless", 60 characters or fewer, replacing the 5.2 text.
