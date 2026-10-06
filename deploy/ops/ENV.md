# S2S serverless: environment variables and secrets (both sides)

Written 2026-10-04 by the ops builder (C) from DESIGN.md sections 4.4, 5 and 6, and from the code as built:
`space/app/sessions.py` `Config.from_env`, `worker/paths.py`, `worker/stack.sh`, `worker/entrypoint.sh`,
`worker/server/worker_server.py`, `worker/rp_handler.py` and `worker/resolve_models.py`. If code and this file
disagree, the code wins; please fix this file.

Rules:
- **No secret in any image, file or repo.** Secrets are set in the HF Space *Secrets* and the RunPod endpoint *environment
  variables* only. Generate the session secret once:
  `python3 -c "import secrets;print(secrets.token_urlsafe(48))"`.
- The **same** `S2S_SESSION_SECRET`, `S2S_MODE` and `S2S_AUDIENCE` must be set on both sides, otherwise every call is
  refused with 401.
- Never print `RUNPOD_API_KEY`, `HF_TOKEN` or `S2S_SESSION_SECRET` in logs, issues or chat. The ops scripts mask them.

## 1. The matching set (must be equal on both sides)

| var | Space | worker endpoint | value |
|---|---|---|---|
| `S2S_SESSION_SECRET` | **Secret** | env (see 3.1 on RunPod secrets) | ≥ 32 chars, same value |
| `S2S_MODE` | Variable | env | `lb` (primary) or `queue` (fallback endpoint) |
| `S2S_AUDIENCE` | Variable | env | **Recommended: a fixed name you choose, e.g. `hinglish-lb-1`** (or `hinglish-q-1` for the queue endpoint). Both sides default to the endpoint id (`RUNPOD_ENDPOINT_ID`), but the worker gets that only if RunPod injects `RUNPOD_ENDPOINT_ID` into the container, and our sources do not confirm that. An explicit value also means you need not edit the endpoint after you learn its id. |
| `CALL_MAX_S` | Variable (optional) | env (optional) | 300. The worker's value is the one enforced; the Space closes at this + 15 s. Keep it below 330 s, the LB per-request processing cap ([LB-OV]). |
| `CLAIM_TTL_S` | Variable (optional) | env (optional) | 90 |
| `S2S_ALLOW_FREE_PROMPT` | Variable (optional) | env (optional) | 0. Set it to 1 on both sides to allow stock free prompts. |

## 2. HF Space (Settings → Variables and secrets)

| var | kind | default | meaning |
|---|---|---|---|
| `RUNPOD_API_KEY` | **Secret** | — | RunPod API key. Create a dedicated key; if the console offers scoping, limit it to this endpoint / serverless access. Used as `Authorization: Bearer` for the LB proxy and the queue API. |
| `S2S_SESSION_SECRET` | **Secret** | — | see section 1 |
| `S2S_PASSCODE` | **Secret** | empty | when set, Connect asks for it. **Set it** on a public Space: it is the main cost brake (R7). |
| `RUNPOD_ENDPOINT_ID` | Variable | — | the endpoint id (from create_endpoint.sh output or the console) |
| `S2S_MODE` | Variable | `lb` | section 1 |
| `S2S_AUDIENCE` | Variable | = endpoint id | section 1 |
| `MAX_CONCURRENT_CALLS` | Variable | 1 | set it **equal to the endpoint's max workers** |
| `RATE_PER_IP_PER_HOUR` | Variable | 6 | session creations per client IP per hour |
| `MAX_CALLS_PER_DAY` | Variable | 100 | total calls per UTC day (cost ceiling, about 100 × $0.15 = $15/day worst case, see COSTS.md) |
| `S2S_TOKEN_TTL_S` | Variable | 120 | token lifetime |
| `WAKE_POLL_S` / `WAKE_TIMEOUT_S` / `WAKE_HTTP_TIMEOUT_S` | Variable | 3 / 600 / 130 | wake polling of `GET /status` (LB) |
| `UPSTREAM_OPEN_TIMEOUT_S` | Variable | 60 | upstream websocket open timeout ([LB-BUILD]) |
| `KEEPALIVE_S` | Variable | 20 | LB keepalive during a call; **0 = off** (hold test run A) |
| `KEEPALIVE_PATH` | Variable | `/status` | `/status` or `/ping` (hold test run C) |
| `KEEPALIVE_TIMEOUT_S` | Variable | 10 | non-fatal (R16) |
| `QUEUE_EXEC_TIMEOUT_MS` / `QUEUE_TTL_MS` / `QUEUE_POLL_S` | Variable | 900000 / 1200000 / 2 | queue mode only; must be ≤ the endpoint execution timeout (900 s, ENDPOINT_SETTINGS 2) |
| `ABANDON_S` | Variable (optional) | 120 | Space only: cancel a `waking`/`busy` session whose browser stopped polling `GET /api/session` for this long (tab closed, pagehide DELETE lost); 0 = off (REVIEW-space-3) |
| `RUNPOD_LB_URL` / `RUNPOD_API_URL` | Variable | `https://{id}.api.runpod.ai` / `https://api.runpod.ai/v2/{id}` | overrides; the tests point them at localhost |
| `S2S_TOKEN_IN_QUERY` | Variable | 0 | LB only. 1 = also send the session token as `?token=` on the claim, the websocket upgrade and the release. Set it if a session fails with "token did not reach the worker" or the relay gets 401 `missing`: the RunPod proxy may drop the custom `X-S2S-Token` header (O-W1). The token is single-use on the upgrade and lives 120 s. The worker always accepts both (CRIT-3) |
| `S2S_SELFTEST` | Variable | 0 | 1 enables `/api/selftest/outbound` (test #1, R17). Needs `S2S_PASSCODE`. **Turn it off again** after the test. |
| `S2S_BUILD` | Variable | empty | free text shown in `/api/config` |
| `LOG_LEVEL` | Variable | INFO | |
| `HOST` / `PORT` | (image) | `0.0.0.0` / 7860 | `app_port: 7860` in the Space README; do not change |
| `S2S_COMMON` / `S2S_STATIC` / `S2S_SAMPLES` / `S2S_RECORDS` | (image) | `space/common`, `space/static`, `space/samples`, `common/data/records_v4.json` | layout overrides for local runs only |
| `SPACES_ZERO_GPU` | set by HF | — | launcher (b) only: declares a dummy `@spaces.GPU` function |

Use HF *Secrets* for the three secrets, never *Variables*: Variables are visible to anyone who can see the Space
settings and, in a Static Space, to client JavaScript ([HF-OV] "Managing secrets").

## 3. RunPod worker endpoint (console: endpoint → Manage → Edit → Environment Variables; API: `env`)

### 3.1 Set these per endpoint

| var | LB endpoint | queue endpoint | note |
|---|---|---|---|
| `S2S_MODE` | `lb` | `queue` | |
| `S2S_SESSION_SECRET` | the secret | the secret | Plain value, or `{{ RUNPOD_SECRET_s2s_session_secret }}` if you store it under RunPod *Secrets*. That syntax is documented for Pod templates (docs.runpod.io/pods/templates/secrets); serverless support is **unverified**. If the worker logs `S2S_SESSION_SECRET is not set` or `/ping` answers 500, use the plain value. |
| `S2S_AUDIENCE` | e.g. `hinglish-lb-1` | e.g. `hinglish-q-1` | section 1 |
| `PORT` | **`80`** | `8765` | LB: the HTTP+WS port; queue: the TCP port to expose |
| `PORT_HEALTH` | **`80`** (= `PORT`) | n/a | **Set it explicitly.** [LB-OV] says it defaults to `PORT`, but runpod/docs issue #853 (open, 2026) reports that RunPod injects `PORT_HEALTH=80` when `PORT` is not 80, and that `HEALTH_CHECK_PATH` is ignored (the LB always polls `/ping`). Using 80 for both makes the bug harmless; our `/ping` is the default path anyway. |
| `CALL_MAX_S` | 300 | 300 | optional |
| `CLAIM_TTL_S` | 90 | 90 | optional |

### 3.2 Model access (not env vars in the container)

| item | where | note |
|---|---|---|
| HF token for `nvidia/personaplex-7b-v1` (gated) | console: endpoint → **Model** field → HF access token ([CACHE]) | Use a **read-only, fine-grained** token for that one repo, from the account that accepted the model licence. The image never sees it. |
| `HF_TOKEN` + `S2S_ALLOW_DOWNLOAD=1` | endpoint env, **debug only** | resolve_models.py fallback 4: downloads 16 GB per cold start (billed). Leave both unset in normal use. `HUGGING_FACE_HUB_TOKEN` is accepted as an alias. |

### 3.3 Optional worker tuning (image defaults are right; change only to debug)

| var | default | meaning |
|---|---|---|
| `ASR_BACKEND` | `trelis` | `off` = no ASR (the router then has no transcript). `fw-small` / `fw-medium` (CPU faster-whisper, DESIGN 6.1 fallback if 24 GB is tight) are **not baked** into the image and fail at start under `HF_HUB_OFFLINE=1` (Amendment W1.10): bake `Systran/faster-whisper-small` into `/opt/hf-asr` first |
| `ASR_DEVICE` | `cuda` | |
| `ROUTER` | `on` | `off` = no Needle router (no tool actions) |
| `S2S_ROUTER` | `v2` | `n1` = roll back to the Needle N1 router (its weights stay in the image; D-ROUTER-V2 2026-10-06) |
| `S2S_THREADS` | 4 | BLAS/OpenMP/torch threads per GPU process (DEP1 A1.3; R13) |
| `S2S_SLOW_GPUS` | `L4,A4000,A4500,RTX 4000,A2000` | startup warning if the GPU name matches (R4) |
| `S2S_LOAD_TIMEOUT_S` | 900 | stack.sh gives up if the engine is not loaded by then |
| `S2S_QUEUE_EXEC_TIMEOUT_S` | 900 (CRIT-4; was 600); set it explicitly on the queue endpoint anyway | the job's execution timeout in s; must equal the endpoint setting and the Space `QUEUE_EXEC_TIMEOUT_MS`/1000 (REVIEW-worker-3, REVIEW-ops-3) |
| `S2S_QUEUE_LOAD_TIMEOUT_S` | derived: exec − `CLAIM_TTL_S` − `CALL_MAX_S` − 30; **set 420** | queue handler: wait for `/ping` 200. The load counts against the job's execution timeout, so keep this + 30 (claim retry) + `CLAIM_TTL_S` + `CALL_MAX_S` + 30 ≤ that timeout |
| `S2S_ADAPTER` | `/opt/s2s/assets/v3_adapter` | `''` = base model; `premerged` = DESIGN 6.4 option A |
| `S2S_PP_DIR` / `S2S_VOICES` | resolved | explicit PersonaPlex dir (network volume or pre-merged snapshot; DESIGN 6.3 order 1) |
| `S2S_PP_REPO` | `nvidia/personaplex-7b-v1` | the repo the resolver looks for in the model cache (set it to your private repo for option A/B) |
| `S2S_RUNPOD_VOLUME` / `S2S_MODEL_CACHE` / `S2S_VOLUME_HF` | `/runpod-volume`, `…/huggingface-cache/hub`, `…/hf/hub` | resolver search paths ([CACHE], [VOLCACHE]) |
| `S2S_ASR_DIR` / `S2S_ASR_HF_HOME` | `/opt/hf-asr/trelis` / `/opt/hf-asr` | baked Trelis |
| `S2S_NEEDLE_WEIGHTS` / `S2S_NEEDLE` / `S2S_RECORDS` / `S2S_NEEDLE_RECORDS` | image paths | |
| `S2S_NEEDLE_V2_WEIGHTS` / `S2S_NEEDLE_V2` | image paths | Needle v2 `.cact` (`$S2S_ASSETS/needle_v2/tuned_full.cact`) and code dir (`worker/needle_v2`) |
| `S2S_TMP` / `S2S_LOGS` / `S2S_MODELS_ENV` / `S2S_DOWNLOAD_DIR` | `/tmp/s2s`, `/tmp/s2s/logs`, … | scratch paths |
| `S2S_INTERNAL_PORT` / `S2S_ASR_PORT` / `S2S_ROUTER_PORT` | 8999 / 8996 / 8995 | localhost-only ports |
| `S2S_WORKER_ARGS` | empty | extra `worker_server.py` args (`--mock` for CPU tests) |
| `HEALTH_CHECK_PATH` | `/ping` | if set, worker_server also serves the health handler there (W1.6); leave it unset, the LB polls `/ping` anyway (issue #853) |
| `S2S_ROOT`, `S2S_COMMON`, `S2S_ASSETS`, `S2S_HINGLISH`, `S2S_PY_PP`, `S2S_PY_ASR`, `S2S_PY_NEEDLE`, `S2S_PY_RP`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `JAX_PLATFORMS` | set in the Dockerfile | do not override on RunPod |

### 3.4 Set by RunPod (read, never set)

| var | used by | note |
|---|---|---|
| `RUNPOD_POD_ID` | worker_server, rp_handler (`worker_id`) | fallback: hostname |
| `RUNPOD_ENDPOINT_ID` | worker_server `aud` fallback | injection not confirmed in the docs we fetched; set `S2S_AUDIENCE` instead |
| `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_8765` | rp_handler (queue mode) | present only when the TCP port is exposed ([EX-QWS]) |

## 4. Test scripts (your machine; `tests/`)

| var | used by | note |
|---|---|---|
| `S2S_PASSCODE` | all `--space` runs | or `--passcode` |
| `RUNPOD_API_KEY`, `S2S_SESSION_SECRET`, `RUNPOD_ENDPOINT_ID`, `S2S_AUDIENCE` | `--direct lb|queue` runs | export them in your shell only for the run; the scripts print them masked |
| `CALL_MAX_S`, `QUEUE_EXEC_TIMEOUT_MS` | direct runs | token `max_s`, queue job policy |

## 5. Ops scripts (`ops/`)

| var | used by | note |
|---|---|---|
| `IMAGE`, `TAG` | build_and_push.sh, create_endpoint.sh | e.g. `docker.io/<you>/s2s-worker`, `v1` |
| `TORCH_INDEX`, `TRELIS_BF16` | build_and_push.sh | Docker build args (`cu130`, `0`) |
| `RUNPOD_API_KEY` | create_endpoint.sh `--yes` | only for the real create |
| `S2S_SESSION_SECRET`, `S2S_AUDIENCE` | create_endpoint.sh | put into the endpoint env |
| `RUNPOD_REGISTRY_AUTH_ID` | create_endpoint.sh | only for a private image (RunPod console → Settings → Container registry auth) |
| `HF_SPACE` | space_push.md commands | `<owner>/<space-name>` |


## 5. Added 2026-10-06 (D-GOLIVE-PREP)

| var | side | default | meaning |
|---|---|---|---|
| `HF_TOKEN` | build secret only since D-BAKE | — | HF **read** token for the image build (gated PersonaPlex + the model repos while private) |
| `S2S_ASSETS_FROM_HF` | build (Dockerfile assets RUN) | unset | 1 = `fetch_assets.py` downloads the manifest's pinned files from their model repos (D-ASSET-SWITCH 2026-10-07; replaces `S2S_ASSETS_REPO`/`S2S_ASSETS_REVISION` and `ops/upload_assets.sh`, both removed) |
| `S2S_VOLUME_ASSETS` | worker env | `/runpod-volume/s2s-assets` | network-volume fallback for the assets |
| `S2S_SKIP_FETCH_ASSETS` | worker env | 0 | 1 = trust `S2S_ASSETS` / `S2S_ADAPTER` as they are (custom images) |
| `S2S_INPUT_AGC`, `S2S_AGC_TARGET_DB` (-21), `S2S_AGC_MAX_GAIN_DB` (24), `S2S_AGC_GATE_DB` (-52), `S2S_AGC_NOISE_GAIN_DB` (6) | worker env | on | input AGC (demo D-AGC) |
| `S2S_INPUT_GATE`, `S2S_GATE_OPEN_DB` (-55), `S2S_GATE_HANGOVER_FRAMES` (4) | worker env | on | noise gate (demo D-GATE). **Keep it on**: without it the model stays silent on a real mic |
| `S2S_TURN_FILL` (ticker), `S2S_TURN_FILL_S` (1.0), `S2S_TURN_FILL_QUIET_MS` (400), `S2S_TURN_FILL_DB` (per mode; ticker -34) | worker env | ticker | turn filler defaults (demo D-FILL2); the Space toggle overrides the mode per call |
| `S2S_TURN_FILL_JSON` | worker env | empty | optional JSON override file (demo turn_fill.json format) |
| `S2S_RECORD_SESSIONS`, `S2S_SESSIONS_DIR` | worker env | 0, `$S2S_LOGS/sessions` | session recordings (debug) |
| `S2S_TURN_FILL_DEFAULT` | Space variable | ticker | the toggle's initial mode (`ticker` or `off`) |
