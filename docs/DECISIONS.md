# S2S serverless: DECISIONS (append-only; dated entries; never edit earlier entries)

## D-DESIGN 2026-10-04 architect: serverless design frozen (DESIGN.md)

Sources: RunPod docs (load-balancing overview and build-a-worker, worker-affinity, endpoint-configurations, model-caching, huggingface-models, pricing), the example repos runpod-workers/worker-lb-websocket and worker-websocket, the HF Spaces docs (docker, overview, config reference, gpus/sleep, zerogpu) and HF forum thread 159865. Each claim is cited per URL in the DESIGN.md source table.

**Architecture**
- HF Space front: one aiohttp app (static client, records, session API, websocket relay).
- RunPod LB endpoint as the primary GPU path. The same image runs a queue-mode fallback with `S2S_MODE=queue`.
- Repo layout: `common/ worker/ space/ client/ ops/ tests/` (DESIGN section 1).

**Relay is mandatory in both modes**
- LB auth is documented only as `Authorization: Bearer <RunPod key>`. Browsers cannot set websocket headers, and the key must never reach the browser.
- Queue mode exposes plain `ws://IP:PORT`, which an https page cannot open.
- So the browser only talks to the Space (same-origin `/api/chat`), and the client fork stays minimal.

**Worker affinity and keepalive**
- The Space claims a worker (`POST /session/claim`) and records its `X-Runpod-Worker-Id`.
- It opens the websocket with `X-Runpod-Worker-Id: strict <id>`.
- It sends a keepalive `GET /ping`, pinned the same way, every 20 s.
- The endpoint idle timeout must be 120 s (default 5 s).

**Worker health and call limits**
- `/ping`: 204 while loading, 200 when ready, and 200 even when busy (an unhealthy worker is dropped from routing). A second call is refused with 409 / close code 4409.
- `CALL_MAX_S` = 300, under the 5.5 min LB processing cap. The context ends at about 221 s anyway.

**Session token**
- Format: `v1.<b64url payload>.<b64url HMAC-SHA256>`.
- Fields: sid, iat, exp (120 s), mode, aud, record_id, pairing, seed, max_s.
- Shared secret `S2S_SESSION_SECRET` on both sides. The replay set lives per worker. The token's config must match the websocket params.

**Models**
- RunPod caches only one model per endpoint. Cache `nvidia/personaplex-7b-v1` (gated; token in the endpoint model setting).
- Bake into the image: Trelis (public), the V3 adapter, the `.cact`, the Needle native-lib cache and the code.
- Voices are extracted to `/tmp`, because the cache is treated as read-only.
- Pre-merged weights and an all-in-one private repo are documented as options (DESIGN 6.4).

**Image**
- cu130 torch, as tested on DEP1, so the endpoint needs the CUDA 13.0+ filter. A cu128 build arg is an untested option.
- venv-pp and venv-asr share one torch copy (built in one RUN with hardlinks). The Needle venv uses CPU jax.
- Estimated size about 16-17 GB.

**GPU**
- Priority: 48 GB A6000/A40, then 4090 PRO, then L40/L40S.
- Avoid the "L4, A5000, 3090" pool. The L4's bandwidth is likely too low for the 80 ms frame (inferred, not measured).

**Premise corrections (DESIGN section 0)**
- Docker/Gradio Spaces need an HF PRO plan. Free accounts get only Static Spaces, plus 2 ZeroGPU Gradio Spaces.
- cpu-basic sleeps after 48 h.
- RunPod idle endpoints drop to max workers 2 after 3 days and 0 after 7 days.
- Websocket-404 on Spaces is unresolved upstream. Test #1 is a wss echo, and the fallback front is the same container on another host.

**Builders**
- W: worker. S: Space + client. C: common/ops/tests and fake_runpod.
- Their contracts are DESIGN section 7.

## D-DESIGN addendum 2026-10-04 architect (after review, before the freeze)

These were folded into DESIGN.md before the freeze. No amendment number is needed.

1. **Wake and keepalive use `GET /status`, not the health path `/ping`.** The [EX-LBWS] test_scaling.py docstring says `/ping` polling is the behaviour "that does NOT scale workers". `/ping` stays RunPod's health contract only. Hold test #3 also runs with `KEEPALIVE_PATH=/ping` to compare.
2. **Queue mode: the `/run` job carries `sid`, `record_id`, `pairing` and `seed`, with no token.** A 120 s token would expire during a cold load of 1-8 min. The job input is trusted because `/run` needs the RunPod key. The handler claims through the new internal route `POST 127.0.0.1:8999/internal/claim`. The websocket still requires a fresh relay-minted token.
3. **The replay set is checked only on the websocket upgrade.** Claim, release and metrics tokens for the same sid stay valid after the socket has started.
4. **`/api/diag` works only for an active sid and only sends a strict-pinned request.** It can never wake a worker or keep one alive on its own (cost hole, R7). The outbound-port self-test route is off by default (`S2S_SELFTEST`).

Also:
- Strict-pinned requests use a 10 s non-fatal client timeout (R16).
- New risk: HF outbound non-443 ports for queue mode (R17).
- The image base is `nvidia/cuda:13.0.x-base` (torch wheels carry their own CUDA and cuDNN) plus `libopus-dev` and `ffmpeg`. The size estimate is now about 13-14 GB.

## D-SPACE 2026-10-04 Space builder (S): space/ + client/ built; CPU mock tests pass

### What was built

**space/**
- `app/main.py` (aiohttp)
  - Serves the static client, plus records and samples through `common/session.py`.
  - Session API and the relay.
  - `/ws-echo`, `/api/diag`, `/metrics?sid=`, `/healthz`, and an opt-in selftest route.
- `app/sessions.py`
  - Config read from env.
  - The session state machine.
  - Cost brakes: passcode, capacity, per-IP rate, daily cap.
  - Claim-TTL expiry and garbage collection.
- `app/upstream.py`
  - LB mode: wake via `/status`, claim with a token, strict pinning, keepalive, release, diag.
  - Queue mode: `/run`, then `/status/{job}` progress, then `ws://ip:port`; cancel.
- Packaging:
  - `app.py`: Gradio-SDK shim.
  - `Dockerfile`: python:3.11-slim, uid 1000, runs `-m app.main`.
  - `README.md`: YAML header exactly as in DESIGN 5.2.
  - `requirements.txt`: aiohttp==3.14.3.
  - `stage_common.sh` and `.dockerignore`.

**client/**
- Fork of `/root/deploy/client`, copied without node_modules, dist, review_backup or e2e outputs. Dependencies reinstalled with `npm ci`. Changes are listed in `client/README_SERVERLESS.md`:
  - buildURL uses `window.location.host` and appends `sid`.
  - New warm-up panel, `dep1/Warmup.tsx`. The mic and AudioContext are acquired in the Connect click, before `POST /api/session`. `<Conversation>` mounts only at `ready`.
  - Passcode field and start errors. The stock panel is shown only when `/api/config` allows it.
  - Metrics come from the 0x07 events instead of the 1 s poll.
  - End-reason texts. DELETE is sent on cancel, disconnect, new call and pagehide.
- Built to `client/dist` and copied to `space/static`.

**Test venv:** `/root/deploy_serverless/.venv-space`, built with uv on py3.11 from `space/requirements.txt` (aiohttp 3.14.3).

### Tests run (CPU only; no RunPod, no GPU)

**`space/tests/test_space_flows.py`: 60/60 checks passed.**
- LB mode:
  - Cold wake: 503s, then ready.
  - The token is bound to the sid and the session config.
  - The upstream websocket carries Bearer and `strict <id>`.
  - The proxy's worker-id header wins over the worker_id in the claim body.
  - The relay passes frames both ways and peeks at metrics.
  - The worker's `time_limit` passes through to the browser.
  - Keepalive and diag work.
  - Claim 409s show as busy, then ready.
  - Refusals: passcode, capacity, rate, bad record, config mismatch.
  - A second socket on the same sid gets 409.
  - Killing the worker mid-call gives `worker_lost` and close code 1011.
  - The Space call cap fires at 16 s.
  - Claim-TTL expiry triggers a release.
  - DELETE while waking cancels the wake.
  - A wrong secret or a wrong key ends `failed`.
  - The wake times out.
- Queue mode:
  - `/run` reaches ready, then the plaintext relay works with a queue token and no Bearer, and the job ends COMPLETED.
  - Cancel while waking works.
- Misc: `/ws-echo`, `/api/config`, records, static files.

**`space/tests/run_e2e_browser.sh`** (Playwright Chromium, fake mic, real Opus from the fake worker)
- Flow seen: setup, then the warm-up panel (`waking`), then ready, then live, then "Session ended (time_limit …)".
- No console errors. One POST plus GET polls.

**`space/tests/xcheck_repo_fakes.py`** (against C's `tests/fake_runpod.py` + `tests/stub_worker.py`): 4/4 checks passed.

**Launchers:** the Gradio shim (`python app.py`), `python -m app.main`, and the `stage_common.sh` layout were each run once.

### Deviations

See DESIGN Amendment S1.

### Known limits

- The `X-Forwarded-For` first hop (DESIGN 5.1) can be spoofed by the client, so the per-IP rate is only a soft brake. The hard brakes are the passcode, `MAX_CONCURRENT_CALLS` and `MAX_CALLS_PER_DAY`.
- Nothing was run against the real RunPod or HF. Those are the user's tests in DESIGN section 9.

## D-SPACE follow-up 2026-10-04 Space builder (S)

- DESIGN Amendment S2 was added after review. It covers:
  - the 4401/4409/4503 close codes, now reported as `relay_error`;
  - a `release` on every failed wake, so no billed queue job is left running;
  - a looser queue `sid` check.
- Test results after the fixes:
  - `space/tests/test_space_flows.py`: 63/63 checks passed. New: a 4409 close gives `relay_error`, and a queue `ready` with no port fails and sends `/cancel`.
  - `space/tests/xcheck_repo_fakes.py` (run with DEP1 venv-pp, i.e. aiohttp 3.10.11 + sphn, sending real Opus): 12/12 checks passed. The three setups were:
    - C's `fake_runpod` + `stub_worker` (LB);
    - C's `fake_runpod` + W's `worker_server.py --mock` (LB);
    - C's `fake_runpod --handler` W's `rp_handler.py` + W's `worker_server.py --mock` (queue).
  - `space/tests/run_e2e_browser.sh`: passed again, with no console errors.
  - W's env-driven `common/session.py`, staged with `stage_common.sh`, resolves `RECORDS_V4` to `space/common/data/records_v4.json`.

## D-WORKER 2026-10-04 worker builder (W): worker/ built; CPU tests pass; Dockerfile written, not built

### What was built (`/root/deploy_serverless/worker/`, plus the `common/session.py` fork with env record paths)

- `server/worker_server.py`: fork of DEP1 server.py.
  - It binds first, then loads.
  - Readiness = engine + ASR + router health. `/ping` 204/200/500, and 200 also while busy.
  - `/status`, `/session/claim`, `/session/release`, the websocket `/api/chat` and the token-gated `/metrics`.
  - Single-session state machine. HMAC token via `common/s2s_token.py`. Replay set checked on the websocket only. Claim TTL.
  - `CALL_MAX_S` watchdog, including the prompt phase. SIGTERM ends a call with `worker_shutdown`.
  - The DEP1 chat body, Hub and internal API are otherwise kept. `--mock` runs DEP1's MockEngine.
- `server/engine.py`: explicit PersonaPlex file paths; `S2S_ADAPTER=premerged`.
- `resolve_models.py`, `stack.sh`, `entrypoint.sh` (lb | queue), `rp_handler.py` (queue).
- `paths.py`: every path and port from env.
- `router/`, `hinglish/` and `needle/` copies, with path edits only. `vendor/moshi`.
- `requirements-{pp,asr}.txt`: the runpod2 freezes. `requirements-needle.txt`: no jax. `requirements-rp.txt`: runpod 1.10.0.
- `Dockerfile` + `Dockerfile.dockerignore`.
- `local/run_local.sh` + `local/runpod2.env`: run outside docker on runpod2 with the DEP1 venvs, on ports 18000/18999/18996/18995.
- `tests/test_worker_mock.py`, `tests/check_dockerfile.py`, `tools/premerge.py`, `README.md`.
- Test venvs, not part of the image: `/root/deploy_serverless/.venvs/{needle,rp}`, built from the requirement files.

### Tests run (CPU only; no GPU, no docker, no RunPod/HF writes)

- **`worker/tests/test_worker_mock.py`: 50/50 checks passed.**
  - `/ping` 204→200; 500 on a failed load or a missing secret.
  - Claim 200, idempotent re-claim, 409, and 401 for missing, garbage, expired, wrong-mode, wrong-aud and wrong-secret tokens.
  - Websocket refusals: 401 no token, 409 not claimed, 401 cfg_mismatch, 400 free prompt, 400 no record, 401 replayed (mid-call and after).
  - A full mock call: handshake, 0x01/0x02/0x07, query without pairing/seed matched by normalisation.
  - `/status` mid-call: busy, active sid, 1-2 ms.
  - `/metrics` 200 for the active sid, 403 for another sid.
  - Claim-TTL expiry. Release before and during a call.
  - `CALL_MAX_S` → `time_limit` at 8 s. Token `max_s=3` → `time_limit` at about 3 s.
  - SIGTERM mid-call → `worker_shutdown` and exit 0.
  - Queue mode via `rp_handler.handler`: progress loading→ready {127.0.0.1, port, worker_id, sid}, call through it, summary returned. An lb token is refused on a queue worker.
  - `resolve_models`: fake `/runpod-volume` cache with refs/main and voices.tgz extracted to `S2S_TMP`; network-volume fallback; not found → exit 2; explicit `S2S_PP_DIR`.
  - Path grep.
- **Full CPU stack** (`run_local.sh mock` with `ASR_BACKEND=fw-small ASR_DEVICE=cpu ROUTER=on`, Needle from the no-jax venv; `ws_feed.py --token`, 62 s of the food_07_g1 input):
  - 2 triggers, each with asr → needle → resolved → executed (`change_delivery_address`).
  - ring_check corr 1.0. `/ping` 200 about 8 s after start.
- **Space builder's `space/tests/xcheck_repo_fakes.py`: 12/12.**
  - B = the Space + C's `fake_runpod` LB + this worker.
  - C = queue via `fake_runpod --handler worker/rp_handler.py`.
- **GPU-free check of the engine fork:**
  - `engine.py` imports.
  - `pp_files_from_dir` on the runpod2 snapshot.
  - `merge_lora` and `lora_merge` load from `worker/hinglish`.
  - The V3 adapter loads: 506 tensors, 253 LoRA, ft_embed false.
- **`check_dockerfile.py`:** OK on a staged context. On the repo it reports only the missing `worker/build/*`, which `ops/prepare_build_context.sh` (C) still has to create.

### Not verified here (left for the Integrate GPU smoke test, or for the user)

- A real Engine load, merge and frame loop through worker_server, and Trelis on GPU inside this stack: `run_local.sh gpu`.
- `docker build` itself.
- The `TORCH_INDEX=cu128` and `TRELIS_BF16=1` build args.
- `tools/premerge.py`.

### Housekeeping

Two `hf --help` calls with venv-asr's CLI wrote 110 `.pyc` files under `/root/deploy/venv-asr/.../huggingface_hub|click`. They were deleted again: same files, same timestamps window. Nothing else under `/root/deploy` or `/root/needle` changed.

### Deviations

See DESIGN Amendment W1.

## D-WORKER follow-up 2026-10-04 worker builder (W)

DESIGN Amendment W2 records three changes made after review:
- `gcc libc6-dev` added to the image (torch.compile/Triton needs a C compiler);
- a cold-start note on the empty torch.compile caches, with `S2S_FRESH_COMPILE_CACHE=1` in `run_local.sh gpu`;
- `TORCH_INDEX=cu128` also needs the 12.8.2 CUDA base.

Amendments S1/S2 were read: no new worker behaviour is required. W's handler never publishes `ready` without a `tcp_port`.

Re-run after the edits: `worker/tests/test_worker_mock.py` 50/50; `check_dockerfile.py` OK on the staged context.

## D-OPS 2026-10-04 common/ops/tests builder (C): common/s2s_token.py, ops/, tests/ built; CPU tests pass; nothing deployed

**What was built (all under /root/deploy_serverless)**
- `common/s2s_token.py`: the DESIGN 4.3 API, stdlib only. Its self-check passes in /usr/bin/python3, venv-pp, venv-asr and venv-needle.
- `common/test_vectors.json`: frozen vectors.
- `ops/`:
  - `ENV.md`: every variable on both sides, and where to set it.
  - `ENDPOINT_SETTINGS.md`: console and API settings for LB and queue, GPU pools, CUDA 13 filter, data centers including the AP-JP-1 note, spend controls.
  - `COSTS.md`: per call about $0.15 on A6000 from zero, including the 120 s idle; about $0.085 warm; $0 at idle.
  - `BUILD.md`: index only.
  - `build_and_push.sh` and `create_endpoint.sh`: dry run by default.
  - `space_push.md`.
  - `prepare_build_context.sh`: run; it filled worker/build (470 MB, lora md5 = V3_CKPT) and can write a tar bundle.
  - `build_client.sh`.
- `tests/`:
  - test doubles and self-tests: `fake_runpod.py`, `stub_worker.py`, `test_token.py`, `test_fake_runpod.py`, `test_mock_e2e.py`;
  - real-test scripts: `ws_hold_test.py`, `cold_start_timer.py`, `latency_probe.py` and `space_ws_echo.py`, with shared code in `rt_common.py`;
  - `TEST_PLAN.md`.

**Verified (CPU only; runpod2; no network beyond 127.0.0.1; no GPU)**
- `test_token.py`: ALL OK.
- `test_fake_runpod.py`: 26/26.
  - Proxy behaviour: Bearer 401; 503 while the worker is loading; the worker-id header; strict 404 `affinity_worker_gone`; websocket proxying with refusal codes; 409 busy.
  - The queue API.
  - All four real-test scripts run end to end in `--direct lb` mode against the stubs.
  - The same scripts run in `--space` mode against the REAL Space code.
- `test_mock_e2e.py --base 48000`: 18/18. It uses the REAL worker code (`run_local.sh mock`), fake_runpod and the REAL Space, in both modes.
  - Checked: the state sequence, the relayed events, busy on a 2nd session, 401 for tampered/expired/wrong-aud tokens, `time_limit` from a token `max_s`, `worker_lost` on a kill mid-call, queue mode via /run + ws://ip:port, and the 1.2 path-grep rule.
- Dry-run defaults were checked with shim `curl`/`docker`/`git`/`hf`/`npm` binaries first on PATH: zero calls. `--yes` without env refuses with exit 2. `bash -n` passes on all ops scripts.

**Decisions**
- Names follow the ops task; see DESIGN Amendment C1.1.
- Keepalive defaults follow DESIGN (`/status`/20 s/strict); `/ping`/30 s is run C of the hold matrix.
- A hold PASS includes `context_full`.
- LB endpoint env sets `PORT=80` and `PORT_HEALTH=80` (runpod/docs#853).
- `S2S_AUDIENCE` is an explicit name on both sides.
- The model cache is a manual console step (no API field).
- API v2 defaults are overridden: FLASHBOOT, idle 120 s, timeout 330 s LB / 600 s queue.
- COSTS: a Docker Space needs HF PRO. The ops task's "HF CPU Space free" is wrong except for Static or ZeroGPU Spaces.
- Self-test ports: 38xxx (harness) and `--base` for e2e. S's xcheck already uses 28xxx with tests/fake_runpod.py.

**Not done (by design)**
- No docker build or push, no endpoint, no Space, no RunPod/HF API calls.
- No GPU run. The GPU smoke test belongs to the Integrate stage.

## D-REVIEW-ops 2026-10-05 adversarial reviewer for ops/ (and its interfaces)

Re-ran on runpod2, CPU only, ports 51xxx/52xxx: `test_token.py` ALL OK, `test_fake_runpod.py` 26/26,
`test_mock_e2e.py` 18/18 (both modes), before and after the fixes. Ops scripts were dry-run with shim
`curl`/`docker`/`git`/`hf`/`npm` first on PATH: 0 calls. Nothing deployed, built, pushed or created; no RunPod/HF
write API called; the GPU was not used; /root/deploy unchanged.

Verified against primary sources (fetched 2026-10-05):
- API v2 `POST https://api.runpod.io/v2/serverless` body (type LOAD_BALANCER/QUEUE, gpu.pools, minCudaVersion,
  workers.idleTimeout 1-3600 default 10, scaling REQUEST_COUNT / QUEUE_DELAY min 0.5, timeout default 300000 ms,
  flashboot default OFF, ports "80/http", env object; LB env PORT/PORT_HEALTH default 80, HEALTH_CHECK_PATH /ping)
  matches create_endpoint.sh.
- GPU pool ids (references/gpu-types): AMPERE_48 = A6000/A40, ADA_24 = 4090, ADA_48_PRO = L40/L40S/6000 Ada,
  AMPERE_24 = L4/A5000/3090. The script's pools are right.
- LB overview: /ping 200/204/other, 2 min no-worker hold, 5.5 min processing cap, 8 min/502 on port misconfig.
- Worker affinity: strict at capacity waits ~5 min then 400 "timed out waiting for worker"; gone -> 404
  affinity_worker_gone. (The Space's 10 s keepalive timeout means it never waits for the 400.)
- Model caching: `/runpod-volume/huggingface-cache/hub/`, one cached model per endpoint, token for gated models,
  console only. Matches resolve_models.py and ENDPOINT_SETTINGS.
- Worker Dockerfile fetches: `nvidia/cuda:13.0.3-base-ubuntu24.04` and `12.8.2-base-ubuntu24.04` tags exist;
  `torch-2.14.1+cu130-cp311` and `torchaudio-2.11.0+cu130-cp311` manylinux_2_28 wheels exist; uv 0.9.0 installer
  URL 200; DEP1 venvs are py3.11.13 (pp, asr) and 3.12.3 (needle), as the Dockerfile assumes; TRELIS_REVISION
  eab1188f… equals runpod2's refs/main and the 12-file list equals that snapshot's file list.
- `uv pip compile` (metadata only, python 3.11/3.12, x86_64-manylinux_2_28, same index flags as the Dockerfile)
  resolves requirements-pp (58 pkgs, torch 2.14.1+cu130), -asr (65), -needle (18), -rp (99).
- ENV.md variable names cross-checked against every env read in space/app and worker/: no undocumented or stale
  name. COSTS.md arithmetic re-computed: correct.

Fixes (each a confirmed defect):
- **REVIEW-ops-1** `ops/space_push.md`: the primary push was plain git with "no LFS is needed". HF rejects binary
  files pushed over plain git regardless of size, and the Space has `*.wasm` plus favicon PNGs, so the first push
  would fail. Now `hf upload` is primary (stores binaries via Xet/LFS); the git path tracks binaries with LFS
  before the first commit.
- **REVIEW-ops-2** `ops/BUILD.md` short path ran `--bundle` before `build_client.sh --copy-only` and
  `stage_common.sh`, so the bundle could carry a stale or unstaged space/; order swapped. "Amendment 1" -> "Amendment
  C1" in BUILD.md and TEST_PLAN.md.
- **REVIEW-ops-3** queue timeout arithmetic. The runpod SDK picks the job up when the container starts, so the model
  load counts against the execution timeout. 600 s minus claim TTL 90, call 300 and margins left about 150-180 s
  for the load, which a fresh worker (torch.compile warmup, W2.2) can exceed; the job would then be killed mid-call
  while rp_handler's own load wait (900 s default before REVIEW-worker-3) was longer than the job itself. Now:
  queue endpoint execution timeout **900 s** (create_endpoint.sh `timeout: 900000`, ENDPOINT_SETTINGS 2), endpoint env
  `S2S_QUEUE_EXEC_TIMEOUT_S=900` + `S2S_QUEUE_LOAD_TIMEOUT_S=420` (420 + 30 claim retry + 90 + 300 + 30 = 870 ≤ 900),
  Space `QUEUE_EXEC_TIMEOUT_MS` default 900000 (space/app/sessions.py, space/README.md, its test assertion) and
  tests/rt_common.py default 900000. ENV.md updated. rp_handler.py itself was being fixed in parallel by the worker
  reviewer (REVIEW-worker-3: load cap derived from `S2S_QUEUE_EXEC_TIMEOUT_S`, default 600); not touched here.
  Note for W: the derived cap `exec − claim_ttl − call_max − 30` omits the up-to-30 s claim retry, so with only
  `S2S_QUEUE_EXEC_TIMEOUT_S` set the worst case is 30 s over; ops sets the load cap explicitly to avoid that.
  The Space `WAKE_TIMEOUT_S` 600 s remains above load 420 + queue wait. DESIGN 2.3 table value superseded
  (Amendment R-ops1).
- **REVIEW-ops-4** (docs only, a cost risk not a code defect) TEST_PLAN step 6 and COSTS 6: if a strict keepalive
  waits behind the open websocket, the request-count scaler may see 2 requests for 1 call and start a phantom second
  worker each call (max workers ≥ 2). The hold test now says to watch the worker count; COSTS lists it as a possible
  multiplier.

Not fixed here (other owners, or needs the real platform):
- `worker/entrypoint.sh` header says "PORT_HEALTH unset"; ops sets PORT_HEALTH=80 explicitly (Amendment C1.4). Comment
  only; harmless with PORT=80. Worker reviewer's call.
- `space/samples` is empty and nothing stages samples; `/api/samples` returns an empty list. Space owner to confirm
  that is intended.
- Unverifiable without RunPod: whether the LB proxy forwards the `X-S2S-Token` header on the websocket upgrade (if
  not, the relay has no fallback that hides the token: `?token=` in the upstream URL would work since the URL is
  server-to-server, but is not wired), whether the websocket counts as work / against the 330 s cap (R2), strict
  keepalives behind the websocket (R16), `ADA_24` = console "4090 PRO" label, data-center ids. The cu128 build path
  is untested.

## D-REVIEW-space 2026-10-05 adversarial reviewer for space/ + client/ (and their interfaces)

**Checked and found correct (no change):**
- Auth: the RunPod key and the HMAC tokens never reach the browser. The browser only talks to the Space, and the relay drops `sid`/`token` from the upstream query.
- Tokens: a fresh token per upstream action, TTL 120 s, config bound into the token, worker replay check on the upgrade.
- Single-session guard: the Space sets `in_call` synchronously (a second socket gets 409); worker 409 / 4409.
- LB:
  - `/status` wake polling with a 130 s per-poll timeout and a 600 s wake cap;
  - strict pinning taken from the proxy's `X-Runpod-Worker-Id` response header;
  - keepalive every 20 s with a 10 s non-fatal timeout;
  - Space cap `CALL_MAX_S`+15 = 315 s, under the 330 s LB cap.
- Queue: `/run` carries no token; on failure the job is cancelled; the relay goes to `ws://ip:port` with a token.
- Re-fetched docs.runpod.io load-balancing/overview and worker-affinity: Bearer auth only; `strict <id>`; the ~5 min wait at capacity; 404 `affinity_worker_gone`; 200/204 health codes; `PORT`/`PORT_HEALTH` default 80. All match DESIGN.
- Space Dockerfile: python:3.11-slim, uid 1000, aiohttp pinned, build fails without `common/`. The CUDA/torch match is a worker-image concern and does not apply here.
- `space/static` is byte-identical to `client/dist`, and no `client/src` file is newer than the build.

**Fixed:**
- **REVIEW-space-1 (capacity lockout, confirmed by repro).**
  - Cause: `handle_chat` set `in_call` before `WebSocketResponse.prepare()`, and the cleanup `try/finally` started only after the upstream open succeeded.
  - Effect: one plain non-upgrade `GET /api/chat?sid=<ready sid>` (curl, a crawler, a link preview) got 400, and the session stayed `in_call` until the Space restarted. DELETE did nothing because `relay_close` was None. With `MAX_CONCURRENT_CALLS=1`, every later visitor got 429 `capacity`.
  - Fix:
    - `can_prepare()` is checked before any state change; a non-upgrade request gets 400 `websocket_required` and the session stays `ready`.
    - Everything after `in_call` runs in a `try/finally` that always ends the session.
    - The relay body moved to `_relay()`.
    - A DELETE that arrives during the upstream open sets `cancel_pending`, honoured when the open returns.
    - The janitor got a backstop that ends any `in_call` session older than `CALL_MAX_S` + open timeout + 60 s.
    - A single janitor exception no longer kills the janitor.
- **REVIEW-space-2 (release after an unclean end).**
  - Before: after `relay_error`, `worker_lost`, the Space cap, or a crash, the claim was not released (LB) or the job was not cancelled (queue). The worker self-heals after `CLAIM_TTL_S`/`CALL_MAX_S`, but a queue job could keep billing until then.
  - Now the relay calls `Upstream.release()` on every end except a worker `session_end` or `client_closed`. Those are clean ends the worker sees itself, and cancelling would turn a queue job's COMPLETED output into CANCELLED.
- **REVIEW-space-3 (abandoned wakes, a cost hole).**
  - Before: if the tab closed during the 1-8 min warm-up and the `pagehide` DELETE was lost (mobile Safari, network), the Space kept waking a worker for up to 600 s, then claimed it and held it for 90 s, all for nobody.
  - Now the Space records `last_seen` on each `GET /api/session/{sid}`. A `waking`/`busy` session not polled for `ABANDON_S` (default 120 s, 0 = off) is cancelled with end reason `abandoned`.
  - 120 s is above Chrome's 60 s timer throttling for hidden tabs; the client polls every 1.5 s.
  - Documented in `space/README.md` and `ops/ENV.md`.
- **REVIEW-space-4 (README header).** `short_description` was 74 characters.
  - The Hub's README metadata validation rejects more than 60 (from memory of the Hub's push validation; the config-reference page states no limit). It is now 54 characters: "Hinglish support-agent voice demo on RunPod Serverless".
  - DESIGN 5.2 still shows the old text; the README wins.

**Tests (runpod2, CPU):**
- `space/tests/test_space_flows.py`: 72/72, including 9 new `t_review_space` checks.
- `space/tests/xcheck_repo_fakes.py`: 12/12.
- `tests/test_mock_e2e.py --base 48000`: ALL OK.
- `space/tests/run_e2e_browser.sh`: live, then `time_limit`, with no console errors.
- The aiohttp tracebacks in test output come from the test fake LB reading a release POST while the test tears it down. They are harmless.

**Not fixed, listed:**
- The per-IP limit trusts the first `X-Forwarded-For` hop, which a client can spoof. The passcode and the concurrency and daily caps are the hard brakes.
- Cancelling during the claim POST itself, before `worker_id` is known, cannot release; the worker frees the claim after 90 s.
- The Space and worker `CLAIM_TTL_S` start a few hundred ms apart, so a socket opened in the last second of the TTL can get `relay_error` (`no_claim`).
- The client shows `abandoned` as the raw reason. No client rebuild was done for one label.
- Everything on real HF/RunPod is untested: the websocket through the HF proxy, the Gradio shim, the Docker build.

## D-REVIEW-worker 2026-10-05 adversarial reviewer for worker/ (and its interfaces)

Scope: worker/ against DESIGN 2-3, 6 and 7, the LB, model-caching and worker-websocket docs (re-fetched 2026-10-05), and the Space and fake interfaces. Nothing was deployed, built or pushed. No GPU was used. /root/deploy was not touched. Backups of the edited files are in /tmp/*.bak_review on runpod2.

**Re-checked against the docs and found correct**
- LB `/ping`: 200 healthy, 204 initializing, anything else unhealthy and "removed from the routing pool".
- `PORT` default 80; `PORT_HEALTH` defaults to `PORT`.
- Request timeout: 2 min with no worker. Processing timeout: 5.5 min per request. CALL_MAX_S=300 counts from the upgrade, including the prompt phase.
- Model cache:
  - path `/runpod-volume/huggingface-cache/hub/models--org--name/snapshots/{hash}/`;
  - one model per endpoint;
  - the gated token goes in the endpoint setting;
  - the docs say nothing on read-only, and the resolver already treats it as read-only.
- worker-websocket example: `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_8765`, `progress_update(job, …)`.

**Verified by running (CPU)**
- The real voices.tgz from the runpod2 snapshot: 19 members under `voices/`, including NATF2.pt. It extracts through `resolve_models.voices_dir()` with `filter="data"` into S2S_TMP.
- torch, torchaudio, triton, nvidia-* and cuda-* pins are identical in requirements-pp.txt and requirements-asr.txt, so the one-layer hardlink dedupe premise holds.
- ldd of libneedle.so and sphn: only libc, libm, libpthread, libdl, libgcc_s and libstdc++, all present on ubuntu24.04 with gcc installed.
  - ctranslate2's bundled libgomp shows "not found" under plain ldd; that is an auditwheel RPATH artefact, and the fw-* fallback is not baked anyway.
  - No ldd assertion was added to the Dockerfile, because that false positive would fail the build.
- SIGTERM through the real chain `run_local.sh entrypoint → entrypoint.sh → exec stack.sh` (mock engine, LB mode): the client got session_end `worker_shutdown`, the entrypoint exited 0, and no listener was left.
- Queue mode outside RunPod: the runpod SDK exits at once ("test_input.json not found"), and entrypoint.sh then stops the stack (exit 1). That is the expected local behaviour.
- The client fork passes `tsc --noEmit`, and dist is newer than src. vite build was skipped while DEP1 is live.

**Fixes (all in rp_handler.py, except item 5)**

REVIEW-worker-1 (queue): stale claim versus the claim retry.
- Problem: if the Space cancels a job before the browser connects, the worker slot stays `claimed` for up to CLAIM_TTL_S (90 s). rp_handler retried 409 for only 30 s, so the next job on that worker failed with `claim_failed`.
- Fix: retry for CLAIM_TTL_S + 5 s.
- This wait and the load wait cannot both happen in one job: a stale claim exists only on a worker that has already loaded. So the worst-case budget is max(load, claim wait) + CLAIM_TTL_S + CALL_MAX_S + 30.

REVIEW-worker-2 (queue): a worker whose load failed ate every job.
- Problem: queue mode has no LB health check, and stack.sh keeps the server alive answering /ping 500. Every job routed to that worker failed instantly.
- Fix: `load_failed` and `worker_lost` now return `"refresh_worker": true`. runpod 1.10.0 `rp_job.py` pops it and sets `stopPod`, so RunPod replaces the worker.
- `load_timeout` deliberately does not refresh: the load is still running, and the next job on the same worker can use it.

REVIEW-worker-3 (queue): the load wait did not fit the job execution timeout.
- Problem: the job's timer runs from pickup, which happens before the model load. The handler's 900 s load wait was longer than the 600 s execution timeout.
- Fix: the load cap now defaults to `S2S_QUEUE_EXEC_TIMEOUT_S` (default 600, must equal the job policy) − CLAIM_TTL_S − CALL_MAX_S − 30, with a floor of 60 s. `S2S_QUEUE_LOAD_TIMEOUT_S` still overrides it.
- The ops reviewer set the endpoint and Space to 900 s, with `S2S_QUEUE_LOAD_TIMEOUT_S=420` (REVIEW-ops-3). That fits the budget above.

REVIEW-worker-4 (queue): a missing `RUNPOD_PUBLIC_IP` published `ready` with `public_ip: null`.
- Fix: it is now handled like a missing TCP port. The claim is released, the job ends with `no_tcp_port`, and the error names the missing variable.

REVIEW-worker-5 (queue; worker_server.py): the public `/status` leaked the active sid.
- Problem: in queue mode the public TCP port has no RunPod Bearer in front of it, and the Space treats the sid as the session handle (`DELETE /api/session/{sid}`, `/metrics?sid=`). So anyone who found ip:port could end or observe a call.
- Fix: in queue mode the public `/status` returns `active.sid: null`. LB mode, which sits behind the RunPod key, and `/internal/status` keep the sid.
- No consumer reads `active.sid`; checked across space/app, tests/ and client/src.

**Tests after the fixes**
- worker/tests/test_worker_mock.py: 54/54. That is the 50 existing checks plus 4 new ones for REVIEW-worker-1, -2, -4 and -5.
- check_dockerfile.py: OK.
- space/tests/xcheck_repo_fakes.py: 12/12.
- tests/test_mock_e2e.py --base 44000: ALL OK, both modes.

**Open (not fixable in this stage; for the user's tests)**
- O-W1: nobody has verified that the LB proxy forwards the custom `X-S2S-Token` header on a websocket upgrade.
  - The worker also accepts `?token=`. The Space relay sends only the header.
  - If test #1 or #3 shows 401 `missing`, make the relay also send `token=` in the upstream query string. That is a Space change.
- O-W2 (LB): after a failed load, a worker stays up answering /ping 500, by design (Amendment W1). The docs say only that unhealthy workers are "removed from the routing pool", not that they are stopped.
  - If the console shows such a worker running and billing, add an exit after a grace period in stack.sh. The trade-off is a restart loop on config errors.
- O-W3 (LB): `/ping` stays 200 while busy, so claims can land on the busy worker and get 409 (R11).
  - Quick 409s may not raise the request-count scaler.
  - Watch for this in test #3 with 2 concurrent sessions.
- Still unverified: docker build, the real GPU load, merge and frame loop in this stack (Integrate), the cu128 and TRELIS_BF16 options, and premerge.py.

## D-REVIEW-space follow-up 2026-10-05

- **REVIEW-space-3 trade-off.** Mobile browsers, especially iOS Safari, pause timers completely while the app is in the background. A phone user who switches apps during a long cold start can come back to "ended (abandoned)" and must click Connect again.
  - In LB mode the saving is mostly the 90 s claim hold, because the worker boot is already billed. In queue mode the cancel really stops the job.
  - If mobile users complain, raise `ABANDON_S` or set it to 0. The `space/README.md` row says this.
- **Concurrent edits.** The concurrent REVIEW-ops-3 edit (`QUEUE_EXEC_TIMEOUT_MS` 900000 in `space/app/sessions.py` and its test assertion) is intact after the REVIEW-space edits.
- **Secret scan.** `space/` (excluding the test dir) and `client/src` contain no `rpa_`/`hf_` tokens and no `Bearer` literals. `client/.env.local` holds only `VITE_QUEUE_API_PATH`.

## D-INTEG 2026-10-05 (Integrate stage)

The pieces were proven together on runpod2 with no RunPod or HF involved. Results: `results/integration.md`.
- CPU: the new `tests/test_integration.py` passed 37/37 (LB + queue) and 4/4 (router on the CPU stack). C's `test_mock_e2e.py`, `test_worker_mock.py` 54/54, `check_dockerfile.py`, `test_token.py` and `test_fake_runpod.py` all still pass.
- GPU smoke (real worker outside docker, model-cache resolver path, empty compile caches, behind fake_runpod LB + the real Space):
  - `/ping` 200 at T0+39.6 s (engine 20 s, Trelis 17.4 s, router 2 s; weights probably in the page cache);
  - a 75 s food_07_g1 call through the Space: step p95 69.05 ms (DEP1, same call: 66.66), VRAM 23 129 MiB, 186 text tokens, 2 router triggers relayed, keepalive 3x200, clean `client_closed`, worker back to idle;
  - DEP1 restarted and healthy; /root/deploy unchanged.

INTEG-1 (ops, runpod2): `/root/deploy/stop.sh` kills every tmux session.
- Cause: the tmux server keeps the argv of the first `tmux new-session` (DEP1's `... bash /root/deploy/gpu_stack.sh ...`), and stop.sh runs `pkill -f "bash /root/deploy/gpu_stack.sh"`.
- The first smoke attempt ran in tmux and died at stop.sh before its restore trap ran.
- Fix, with /root/deploy left untouched: `tests/gpu_smoke.sh` runs with `setsid nohup`, never in tmux, and its trap also covers HUP/INT/TERM. Documented in the script header and `worker/README.md`.

INTEG-2 (worker/local/run_local.sh): SIGTERM to `run_local.sh gpu` never reached stack.sh.
- Cause: `exec flock -n /root/gpu.lock bash stack.sh` meant the signal hit flock(1), which exited 143 without forwarding it, leaving the stack and its children running and holding the lock and ports.
- Fix: `exec 9>>"$S2S_GPU_LOCK"; flock -n 9 || refuse; exec bash stack.sh`. That pid is now stack.sh and its TERM trap runs. The lock frees when the last fd-9 holder exits. `S2S_GPU_LOCK` defaults to /root/gpu.lock.
- Verified: a stand-in stack held the lock, a second run was refused, TERM reached the trap and the lock was released; `run_local.sh gpu` still refuses at once while DEP1 holds the lock.
- The image has no flock and is unaffected.

Test tooling (additive, not a defect): `tests/rt_common.py stream_call(..., on_event=None)` passes each 0x07 event to a callback; existing callers are unchanged.

Not changed, for the user's real tests: O-W1 (X-S2S-Token on the LB websocket upgrade), R2, R16 / REVIEW-ops-4, O-W3, websocket through the HF proxy, docker build, cold start on RunPod hardware.

D-INTEG follow-up 2026-10-05:
- `test_integration.py` now also asserts that `/ping` returns 200 mid-call in both modes: 39/39.
- On GPU, /ping stayed 200 through the call (the fake proxy health loop logged no change).
- `/root/deploy` is unchanged since 18:40 UTC, which covers every run in this stage.
- DEP1 was restarted in live mode. Its mode before the stop was not logged.
- The fixed `gpu_smoke.sh` and the INTEG-2 `run_local.sh gpu` path were not rerun on GPU. A GPU SIGTERM clean exit is shown only by the stack.sh "stopped" line and the lock release.

---------------------------------------------------------------------------------------------------------------------
D-CRIT 2026-10-05 (completeness critic: what would block the user on deploy/test day). Nothing was deployed, built or
pushed; no RunPod/HF API call; no GPU; /root/deploy unchanged. Backups of every edited file: /root/rv_crit/bak/.

CRIT-1 (docs; possible deploy blocker): the RunPod docs do not confirm that the endpoint **Model** (cache) field exists for
load-balancing endpoints. The model-caching, LB build-a-worker and endpoint-configurations pages were re-read on
2026-10-05 and none of them says so either way. The only fallback, DESIGN 6.4 C (network volume), had no procedure.
- New `ops/NETWORK_VOLUME.md`:
  - choose the data center, create a 25 GB volume, fill it from a cheap pod (mounted at /workspace on a pod and
    /runpod-volume on serverless) with `HF_HOME=/workspace/hf hf download …`;
  - **no `hf auth login`**, because it would write the token onto the volume that workers mount;
  - leave the Model field empty; no extra env (`S2S_VOLUME_HF` defaults to /runpod-volume/hf/hub; resolver step 3,
    same snapshot lookup as the cache);
  - the pre-merged variant via `S2S_PP_DIR` + `S2S_ADAPTER=premerged`.
- Pointers added in ENDPOINT_SETTINGS.md (Model and Network volume rows) and in TEST_PLAN step 3's failure column.

CRIT-2 (persistence): the build context's baked assets existed only in runpod2 `worker/build` (ephemeral disk). The
laptop copy excludes them, and `prepare_build_context.sh` reads runpod2-only sources (the Needle lib cache is on runpod2
alone).
- Assets-only tar (`worker/build/**` + `common/data/records_v4.json`, 466 MB, md5
  f620575d6357e9a20fd743213c803c0b) parked on pod1's persistent network volume at
  `/workspace/hinglish/deploy_serverless_assets/s2s_build_assets.tar` (md5 verified after the copy).
- `ops/BUILD.md` gained "If runpod2 is gone": laptop code + that tar, then check_dockerfile.
- The full `--bundle` path was rehearsed in a scratch copy (/root/rv_crit/repo → ctx.tar 471 MB → extracted):
  `check_dockerfile.py --root <extracted>` OK, and space/common + space/static were present.
- `ops/build_and_push.sh` and `ops/create_endpoint.sh` dry runs were NOT executed: a permission block stopped them in
  this stage. BUILD.md now says they were only statically reviewed. The create body was read: PORT/PORT_HEALTH 80,
  pools, CUDA 13.0, idle 120, flashboot, timeouts all as ENDPOINT_SETTINGS.

CRIT-3 (code; O-W1 made switchable instead of a test-day code change):
- Space env `S2S_TOKEN_IN_QUERY` (default 0). When it is 1, the LB upstream also sends the token as `?token=` on
  `/session/claim`, the websocket upgrade and `/session/release`. Code: `space/app/sessions.py` Config,
  `space/app/upstream.py` `LBUpstream.tq()`.
- The worker's claim, release and metrics now accept `?token=` too; the header still wins. The websocket already did.
  Code: `worker/server/worker_server.py`.
- A claim 401 `missing` with the switch off now says "the token did not reach the worker … set S2S_TOKEN_IN_QUERY=1"
  instead of pointing at the secret.
- `tests/fake_runpod.py --strip-headers` simulates a header-stripping proxy.
- New `tests/test_crit_token_query.py`: 8/8. With the switch off, the session fails and the hint names the switch.
  With it on, a full call runs through the stripping proxy.
- Documented in ENV.md 2, space/README.md, TEST_PLAN step 3 and README.md.

CRIT-4 (`worker/rp_handler.py`): the default `S2S_QUEUE_EXEC_TIMEOUT_S` went from 600 to 900, matching the endpoint
(900 s) and the Space (`QUEUE_EXEC_TIMEOUT_MS` 900000). A queue endpoint made in the console without that env would
otherwise have derived a load cap of about 180 s. ENV.md updated.

CRIT-5 (comment only): `worker/entrypoint.sh` said "PORT_HEALTH unset"; it now says to set PORT=80 and PORT_HEALTH=80
explicitly (runpod/docs#853), matching ENV.md and create_endpoint.sh.

CRIT-6 (docs): new top-level `README.md` ("start here"):
- the reading order;
- a day-of-deploy checklist;
- where to see logs (HF Space → Logs; RunPod endpoint → Workers → Logs; `SESSION_SUMMARY` lines);
- a symptom → fix table for the first expected failures;
- the CPU test commands.
The `--bundle` tar picks README.md up automatically.

Tests after the CRIT changes (runpod2, CPU):
- test_worker_mock 54/54;
- test_space_flows 72/72;
- test_mock_e2e --base 52000 ALL OK;
- xcheck_repo_fakes 12/12;
- test_integration --base 45000 --phases lb,queue 39/39;
- test_fake_runpod ALL OK; test_token ALL OK;
- check_dockerfile OK;
- test_crit_token_query 8/8.

Left for the user's test session (unchanged): O-W1 itself (whether the switch is needed), R2, R16/REVIEW-ops-4, O-W2,
O-W3, the websocket through the HF proxy, docker build, the cu128 and TRELIS_BF16 build args, premerge.py, the
data-center ids and the ADA_24 label, and two concurrent calls.

D-CRIT follow-up 2026-10-05:
- Correction to CRIT-2. `build_and_push.sh` and `create_endpoint.sh` WERE dry-run by the earlier ops stages (shimmed
  PATH, 0 external calls). Only this stage could not re-run them, because of a permission block. BUILD.md and README.md
  are corrected.
- The "If runpod2 is gone" recipe was rehearsed on runpod2: the laptop copy plus `s2s_build_assets.tar`, then
  stage_common, then check_dockerfile `--root` printed OK. The scratch copy was deleted.
- NETWORK_VOLUME.md:
  - adds `HF_XET_CACHE=/tmp/hf-xet`, so the download's chunk cache does not land on the volume;
  - adds a `du -sh` check;
  - raises the suggested volume size to 30 GB.
- CRIT-3, extended:
  - The more likely real case is that the HTTP claim passes and only the websocket upgrade loses the header. In that
    case the relay's `relay_error` detail ("worker refused the call (HTTP 401)") now also names S2S_TOKEN_IN_QUERY.
    This applies in LB mode with the switch off (`space/app/main.py`).
  - `tests/fake_runpod.py --strip-ws-only` simulates it.
  - `tests/test_crit_token_query.py`: 11/11.
- Re-run after these edits:
  - test_space_flows 72/72;
  - test_mock_e2e ALL OK;
  - xcheck_repo_fakes 12/12;
  - test_fake_runpod ALL OK. Its first parallel run failed only because its Space port collided with test_mock_e2e's
    51860; it passed when run alone.

## D-README (2026-10-05, README/sync stage)
- README.md rewritten as the single start page: the CRIT-6 content (reading order, day-of-deploy checklist, quick fixes, CPU tests) is kept verbatim, and these are added: what this is, an ASCII architecture diagram, both modes, the security model, step-by-step deploy, the test-plan order, measured numbers (from results/integration.md, with the page-cache caveat), known risks, costs, file map, and where the copies live. Previous version: runpod2:/root/rv_readme/README.md.bak. DESIGN.md untouched.
- Sync: code/docs/results copied to pod1 /workspace/hinglish/deploy_serverless and the laptop; excluded .venv-space/, .venvs/, node_modules/, __pycache__/, worker/build/ (baked assets; parked tar on pod1) and files > 50 MB. No --delete on any destination.

## D-GOLIVE-PREP 2026-10-06 (user: "do the rest of the work, make the docker images gh repos etc etc")
Decided by the user: the GPU worker runs on a RunPod Serverless LB endpoint, built by RunPod's "Deploy from a GitHub
repository"; the front is an HF Docker Space (PRO). Later orchestrator updates:
- The user creates the endpoint in the console, on a **personal** RunPod account with $20 credit. The org account
  cannot create keys or connect GitHub.
- The RunPod key goes only into the Space secrets.
- Every later test goes through the Space URL + passcode.
GO_LIVE.md has the remaining steps.

- **PORT** (the live-demo fixes; behaviour identical to the pod1 demo; see `deploy_pod1/DECISIONS.md` D-AGC, D-BUFFER,
  D-GATE, D-FILL, D-FILL2, D-TOGGLE):
  - `worker/server/core.py` = the demo's `server/core.py` as of 2026-10-06 (md5 70771832), plus serverless edits:
    - `FILL_JSON` comes from `S2S_TURN_FILL_JSON` (default off).
    - `TurnFiller.reset(session_mode)` / `request_mode()`: the toggle is applied by the GPU thread at the next frame,
      never on the event loop.
    - Recording is gated by `S2S_RECORD_SESSIONS` (default 0; nothing is buffered when off). The dir is
      `S2S_SESSIONS_DIR` (default `$S2S_LOGS/sessions`).
    - `apply_now` also refreshes `buf_rms_db`/`buf_peak_db`. This is metrics only; the demo left them None after an
      off → ticker switch.
    - The env names and defaults are unchanged: `S2S_INPUT_AGC`, `S2S_AGC_*`, `S2S_INPUT_GATE`, `S2S_GATE_OPEN_DB=-55`,
      `S2S_GATE_HANGOVER_FRAMES=4`, `S2S_TURN_FILL=ticker`, `S2S_TURN_FILL_S/_QUIET_MS/_DB`.
  - **Toggle over serverless (decided):** in-band. A strict-pinned HTTP request could queue behind the websocket
    (R16), and the in-band path works the same in queue mode.
    - The Space keeps one mode per process (`S2S_TURN_FILL_DEFAULT`, default ticker; = the demo's global
      turn_fill.json).
    - `GET/POST /api/filler?mode=ticker|off`: the same paths and JSON as the demo, so the demo's `filler-toggle.js` is
      reused byte-identical.
    - A new call: the Space adds the query param `turn_fill=` to the upstream websocket (a browser-supplied one is
      dropped).
    - A live call: the Space sends `b"\x08"+{"type":"turn_fill","mode":…}` on the relay's upstream socket, under a
      per-call send lock shared with the audio pump.
    - Worker: `resolve_session_config` validates `turn_fill` (400 on a bad value). It is not part of the token's
      (record_id, pairing, seed) triple.
    - Worker `recv_loop`: kind 8 → `_control` → `filler.request_mode`, plus a `turn_fill` hub event.
  - **Buffer:** in the client source (`client/src/audio-processor.ts`, both the constructor and `initState`):
    4 frames / 80 / 1500 / 40 / 400 / 3000 ms.
  - **Toggle UI:** `client/public/filler-toggle.js` + a script tag in `client/index.html`.
- **CLIENT:** built on pod1 with Node v20.12.2 (`.nvmrc`) installed under `/root/node`, using `npm ci && npm run build`
  (nice + taskset to 16 cores, with the demo live).
  - The emitted worklet `audio-processor-CwnWMvpX.js` differs from the demo's hand-edited `audio-processor-S2Sbuf1.js`
    only in one literal: `i(3e3)` vs `i(3000)`. So the buffer is identical.
  - `client/dist` → `space/static`.
- **WEIGHTS** (decided): the repos are code-only (GitHub rejects > 100 MB; no LFS).
  - `worker/fetch_assets.py` runs in the load task before `resolve_models`, with the venv-asr interpreter (hub 1.33 +
    hf-xet), so a failure shows as `/ping` 500.
  - Order of sources: already in place → `/runpod-volume/s2s-assets` → the private HF model repo `$S2S_ASSETS_REPO`
    with `$HF_TOKEN`.
  - Every file is md5- and size-checked against `worker/assets_manifest.json`:
    - V4_A2 step 600 `lora.safetensors` 627bc77a… + `config.json` 6c4fab73…;
    - Needle N1 `tuned_full.cact` 92ddd0d7…;
    - the optional `libneedle.so` 679dc570….
  - `paths.ADAPTER` defaults to `$S2S_ASSETS/v4_adapter`. The V3 LoRA is no longer the default.
  - PersonaPlex comes from the RunPod model cache, or else the network volume (unchanged).
  - The Dockerfile no longer COPYs `worker/build/`. It pre-fetches the cactus-needle lib at build time (best effort).
  - The Needle router stays N1. **TODO:** Needle v2 is not wired in.
  - `ops/upload_assets.sh` (dry run by default; `--remote runpod` sends the token over ssh stdin):
    - creates `<hf_user>/s2s-v4-assets` as **private** and refuses a public one;
    - uploads the 4 manifest files + `data/records_v4.json` + MANIFEST.json;
    - verifies size + LFS sha256.
    - The dry run on pod1 passed: all 5 sources match their md5.
  - `ops/prepare_build_context.sh`, `ops/build_and_push.sh` and `ops/BUILD.md` (the registry-image route with baked
    V3 assets) are superseded. They are kept for reference.
- **RunPod GitHub-build limits** (docs.runpod.io github-integration, read via the runpod/docs repo on 2026-10-06):
  - `docker build` ≤ 30 min; the whole build ≤ 160 min; image ≤ 80 GB.
  - A Dockerfile Path is settable; there is no build-context field (we assume the repo root).
  - No documented build args or BuildKit, so: no `# syntax=` line and a plain `.dockerignore`.
    `check_dockerfile.py` now enforces this, plus no `worker/build` COPY and no file > 95 MB.
  - Builds trigger on a **GitHub release**, not a push.
  - GPU-requiring builds are not supported.
  - One GitHub account per RunPod account.
  - LB endpoint type is supported for GitHub deploys. The Model (cache) field is documented for GitHub deploys, but
    not confirmed for LB.
  - The 30-min fallback is `TRELIS_BF16=1` (edit the ARG default).
- **Space extra:** `GET /api/endpoint_health?passcode=` passes through RunPod `GET /v2/{id}/health`, which does not
  wake a worker.
  - It exists only when `S2S_PASSCODE` is set, and is rate-limited to 1 per 5 s.
  - Purpose: cold starts can be debugged without anyone but the Space holding the key.
  - Not verified on LB endpoints.
- **TESTS** (pod1 `/root/dep2`, CPU only, `CUDA_VISIBLE_DEVICES=` and ports 18xxx/27xxx/28xxx/38xxx/5xxxx, away from
  the demo's 899x; `/root/deploy` untouched; `bash tests/run_all_cpu_pod1.sh`; logs in `results/cpu_tests_2026-10-06/`):
  - New `worker/tests/test_fixes.py`: 31 checks.
    - gate zeros + hangover; AGC → −21 dBFS;
    - ticker buffer −34 dBFS / 1.0 s fed at frame 15 after 400 ms quiet; check-line skip; user cancel;
    - session mode, the deferred `request_mode`, a mid-fill switch to off;
    - **bit-exact equivalence with the live demo's core.py**: AGC output, ticker buffer, and frame-by-frame filler
      decisions + model input;
    - through `worker_server --mock`: query `turn_fill`, 400 on a bad value, the mid-call 0x08 switch shown in the
      metrics, fills only after the switch, recording off/on.
  - `space/tests/test_space_flows.py` gains `t_filler_toggle` (toggle API, upstream query, in-band frame,
    browser-param drop, env default, bad env) and `t_endpoint_health`. The fake worker logs control frames, and the
    fake queue has `/health`.
  - The test env for pod1 is `worker/local/pod1.env`, selected with `S2S_LOCAL_ENV`, which test_worker_mock,
    xcheck_repo_fakes and run_local.sh now honour.
  - The Space image was built and run on the laptop: 148 MB, with `/`, `/filler-toggle.js`, hashed assets,
    `/api/records` and `/api/filler` GET/POST working. The worker image was NOT built (no daemon on the pods; too heavy
    for the laptop).
- **REPOS:** `ops/make_repos.sh OUT` assembles `s2s-worker` (repo-root Dockerfile + .dockerignore + README, common/,
  worker/ minus local/build) and `s2s-space` (space/ minus tests, staged common/, built static/). It then scans for
  token-shaped strings, forbidden files and files > 50 MB.
- **Secrets:** `~/.config/s2s/secrets.env` (laptop, 700/600) was generated with `S2S_SESSION_SECRET` and
  `S2S_PASSCODE`, plus empty `HF_TOKEN_READ` / `HF_TOKEN_WRITE`. No RunPod key, by design. Nothing was printed.
- **PUSHED** 2026-10-06: these are private GitHub repos, branch `main`, assembled by `make_repos.sh`.
  - https://github.com/shivamgcodes/s2s-worker, initial commit `f7314f2`.
  - https://github.com/shivamgcodes/s2s-space, initial commit `f554172`.
  - Scans before the commit found no token-shaped strings, no file > 50 MB, and none of the 3 real secret values
    (the session secret, the passcode, the personal RunPod key).
  - Local clones: `hinglish/deploy_repos/`.
  - Later changes need the user's go-ahead before any commit or push.
- **Orchestrator updates handled.**
  - `upload_assets.sh` runs on pod1, reading `/workspace/hf/token` in place. Its default repo is
    `shivamgupta/s2s-v4-assets`. The dry run on pod1 found the token and checked the md5s; no HF call was made.
  - New `ops/push_space.sh`: an HF-hub `upload_folder` to `shivamgupta/hinglish-agent`, dry run by default. git push is
    avoided because a web-created Space has its own first commit and plain git rejects the binaries.
  - Neither the HF token nor the RunPod key was used for any API call.

## D-BAKE 2026-10-06 (user: "continue with the work please"; plan agreed with the orchestrator)
Decided: bake ALL weights into the worker image; GitHub Actions builds it (GitHub runners have Docker, RunPod pods do
not) and pushes it to a PRIVATE Docker Hub repo; RunPod (personal account, $20) uses "Deploy from a Docker image" with
a registry credential. Supersedes the "WEIGHTS" runtime fetch and the RunPod GitHub build + model cache route.
This phase (A) spent no RunPod money: no endpoint, template or Space was created; the RunPod key was not used.

- **Repo verification.** Fresh clones of `shivamgcodes/s2s-worker` (f7314f2) and `s2s-space` (f554172) were
  byte-identical to a fresh `ops/make_repos.sh` assembly of this folder (`diff -rq`: no output). Secret scans of
  the clones: no `hf_`/`rpa_`/`gh*_`/`sk-`/`dckr_pat_`/private-key shapes; none of the real values (session secret,
  passcode, RunPod key, Docker Hub token) present. Live-demo fixes present in the repos: AGC −21 dBFS, gate −55 dBFS
  + 4-frame hangover, ticker −34 dB / 1.0 s / 400 ms quiet with the check-line skip, the Ticker|Off toggle
  (`static/filler-toggle.js` + `/api/filler` + in-band 0x08), and the D-BUFFER values in
  `client/src/audio-processor.ts` (4 frames / 80 / 1500 / 40 / 400 / 3000 ms).
- **push_space.sh guard** (the interrupted item): it already requires `common/s2s_token.py` and
  `common/data/records_v4.json` in `--dir`. Checked: it refuses the raw `space/` folder and accepts a clone of
  s2s-space. Its python search now also tries `/root/dep2/.venvs/needle` (pod1's `/root/deploy` venvs are gone).
- **pod1 env** (pod1 restarted; `/root` was empty): `/root/dep2` re-synced from this folder; venvs rebuilt with uv under
  `/root` (venv-pp with the **CPU** torch 2.14.1 wheel, needle py3.12, rp, space); the paths in `worker/local/pod1.env`
  re-created as symlinks to `/workspace` (read only: adapter, Needle cact, PersonaPlex snapshot); the demo
  `core.py` (md5 70771832) copied from the laptop `deploy_pod1/server/` for the bit-exact test. Nothing written to
  `/workspace`.
- **CPU tests** (pod1, `CUDA_VISIBLE_DEVICES=`, `tests/run_all_cpu_pod1.sh`): all 9 suites rc=0 —
  check_dockerfile OK; test_token ALL OK; test_fixes 31/31; test_worker_mock 54/54; test_crit_token_query ALL OK;
  test_fake_runpod ALL OK; test_mock_e2e ALL OK; test_space_flows 88/88; xcheck_repo_fakes 12/12.
  Logs: `results/cpu_tests_2026-10-06_bake/`.
- **HF assets repo** `shivamgupta/s2s-v4-assets` (**private**), created by `ops/upload_assets.sh --yes` on pod1 (token
  read in place). Uploaded: `v4_adapter/{config.json,lora.safetensors}`, `needle/tuned_full.cact`,
  `needle/libneedle.so` (re-fetched by cactus-needle 3.0.6 on pod1: md5 679dc570 = manifest), `data/records_v4.json`
  (from `/workspace/hinglish/data/V4/records.json`, md5 c6c0cc7a = the repo copy), `MANIFEST.json`. Verified by size +
  LFS sha256, then **re-downloaded through `fetch_assets.py`**: config 6c4fab73, lora 627bc77a, cact 92ddd0d7,
  lib 679dc570 — all equal to `worker/assets_manifest.json`. `upload_assets.sh` defaults updated (records path, python).
- **Gated access preflight**: the token (user shivamgupta, role write) reads `nvidia/personaplex-7b-v1` at
  fdaf4090 (model.safetensors 16,742,874,000 B; a real file download worked), so the licence is accepted.
- **Docker Hub**: user `shivamgupta579`. `shivamgupta579/s2s-worker` created via the Hub API with `is_private: true`
  (read back: true). The account's other repo (`alphafold3`) is public, so the one free private slot is this one.
- **Dockerfile** (BuildKit, `# syntax=docker/dockerfile:1.7`):
  - PersonaPlex: `hf download nvidia/personaplex-7b-v1 --revision fdaf4090…` of exactly the files the engine /
    `resolve_models.NEEDED` use — `model.safetensors`, `tokenizer-e351c8d8-checkpoint125.safetensors`,
    `tokenizer_spm_32k_3.model`, `voices.tgz` (pre-extracted to `/opt/pp/voices`) — into `/opt/pp`, own layer
    (~16.4 GB). `config.json` and the rest of the repo are not baked.
  - Private assets: `fetch_assets.py` itself at build time (`S2S_ASSETS=/opt/s2s/assets`), so the dest layout and the
    md5 checks are the runtime ones; `--check` must pass. At worker start it finds everything "in place".
  - The token: `RUN --mount=type=secret,id=hf_token,required=true`, read from `/run/secrets/hf_token` inside the
    two download RUNs only, which use `set -eu` (no `-x`); never an ARG/ENV; HF caches deleted in the same RUN.
  - Runtime ENV: `S2S_PP_DIR=/opt/pp`, `S2S_VOICES=/opt/pp/voices`, `S2S_ASSETS=/opt/s2s/assets`, `HF_HUB_OFFLINE=1`
    (resolver step 1 = the baked dir). The last RUN also runs `fetch_assets.py --check` and `resolve_models.py`.
  - Weight layers come before the code COPYs. CUDA 13.0.3 base + torch 2.14.1+cu130 unchanged (Blackwell + Ada).
  - `check_dockerfile.py` now enforces: secret mount on every RUN reading the token, no `set -x` there, only
    `type=secret` mounts, the `# syntax=` line, no HF_TOKEN in ARG/ENV, and the baked runtime ENV.
    `docker buildx build --check` on the laptop: no warnings.
- **Workflow** `worker/github/build.yml` → `.github/workflows/build.yml` in s2s-worker (`make_repos.sh` copies it, so a
  re-assembly keeps it). workflow_dispatch (+ optional extra tag) and `v*` tags; ubuntu-latest. Steps: plain disk
  cleanup (rm of the preinstalled toolchains, swap off: 14 → 58 GB free); secrets presence check; **refuse to push
  unless the Docker Hub repo exists and is private**; static Dockerfile check; then two phases:
  1. BuildKit (build-push-action v6, `secrets: hf_token=…`, `BAKE_PP=0`, `provenance/sbom: false`) builds + pushes
     everything except PersonaPlex as `:base-<sha12>`; the builder and its state are then removed (`df` back to 58 GB);
  2. the 4 PersonaPlex files are downloaded on the runner with curl (token only in that step's env, sent only to
     huggingface.co), **sha256-checked against the HF LFS oids** (= the pod1 cache blob names), voices extracted;
  3. `tar … opt/pp | crane append -f - -b :base-<sha12> -t :<sha12>` streams `/opt/pp` as ONE appended layer (no
     local blob; owner 0:0), then `crane tag … latest`;
  4. verify: the final config has `S2S_PP_DIR`/`S2S_ASSETS`/`HF_HUB_OFFLINE`, diff_ids = base + 1; print the size.
  Why two phases (**attempt 1, run 37417110171, failed**): ENOSPC in the PersonaPlex RUN. The runner has one 72 GB
  disk (no separate `/mnt`, 14 GB free); the easimon LVM step gave a 30 GB `/var/lib/docker`, and the xet chunk cache
  doubled the 16 GB download. Even with full cleanup (58 GB) a BuildKit build must hold the 16 GB layer AND its
  compressed blob plus the base (~55 GB peak), so the PersonaPlex layer now bypasses BuildKit. A local
  `docker build` (BAKE_PP=1 default) still bakes everything in one go. `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` added.
  crane stdin streaming was first checked locally against a throwaway `registry:2` (1 MB layer, 0/0 owners).
- **BUILD RESULT (attempt 2, run 37418330123, commit 4e950bb): SUCCESS in 28 min** (05:23:56 → 05:52:14 UTC):
  cleanup 2.5 min, phase 1 build + push 10.1 min, PersonaPlex download 3.2 min (16.7 GB at ~87 MB/s) + sha256 1.3 min,
  crane append + push 10.3 min, verify OK.
  - Image: **`docker.io/shivamgupta579/s2s-worker:4e950bb7bbc6`** (= `:latest`), digest
    `sha256:bbb9b24b0a4a21314b40640f48fea6c8d559947f098f5978d48d57cc11efef67`.
  - **23.41 GB compressed**, 27 layers, largest 13.43 GB (PersonaPlex). Unpacked ≈ 31 GB (base ~14 GB incl. Trelis
    5.8 GB + PersonaPlex 17.1 GB). `:base-4e950bb7bbc6` (9.98 GB compressed) also stays in the repo (shared layers;
    harmless, deletable).
  - Docker Hub repo read back after the push: private = true. No HF-token-shaped string and not the real token value
    in the image config/history; the build log has no token-shaped string.
  - Phase 1 log: the private assets RUN printed `missing_required: [], missing_optional: []` (adapter, cact and
    libneedle all md5-OK in the image).
  - **Not yet run on a GPU** (no GPU on Actions; first real start = GO_LIVE T3).
- **Actions secrets** on s2s-worker: `HF_TOKEN` (piped from pod1 `/workspace/hf/token`; the write token, the only one
  there is), `DOCKERHUB_USER`, `DOCKERHUB_TOKEN` (piped from the laptop `.env`). No value was printed.
- `make_repos.sh` scans now include the `dckr_pat_` shape and the Docker Hub token value.
- Size levers if pulls are too slow: `TRELIS_BF16=1` would save ~3 GB (untested).
- Laptop: crane v0.22.1 binary in the session scratch only; its Docker Hub login was logged out again (the laptop's
  pre-existing docker credentials untouched).
- **Spec deviation (stated):** the PersonaPlex download is no longer a BuildKit-secret RUN in the workflow. It runs in
  a plain runner step with the token in that step's env (masked; sent only to huggingface.co; curl drops it on the
  CDN redirect; never in a layer; the run log grep found no token-shaped string). The private-assets download is
  still a BuildKit secret mount.
- PersonaPlex `config.json` is deliberately not baked: `moshi/models/loaders.py` uses the hardcoded `_lm_kwargs`, and
  `engine.pp_files_from_dir` needs only the 3 weight/tokenizer files (+ voices). The only `config.json` read is the
  adapter's own (`v4_adapter/config.json`, baked).
- Repo HEAD after this phase: s2s-worker `f9d08f4` (README-only commit after the image's `4e950bb`; the README is
  not in the image), s2s-space `f554172` (unchanged). The laptop folder re-assembles byte-identical to both.
- **Open:**
  - The first GPU start of the image is untested (GO_LIVE T3).
  - The `HF_TOKEN` Actions secret is the write token; replace it with a read token.
  - RunPod needs a read-only Docker Hub PAT for its registry credential.
  - Each rebuild re-pushes a new 13.4 GB PersonaPlex layer, because the tar is not reproducible: fresh mtimes give a
    new digest, and RunPod hosts re-pull it. Fix later with `tar --sort=name --mtime=@0 --pax-option=…delete=atime,
    delete=ctime`.
  - Each run leaves a ~10 GB `:base-<sha>` tag.
  - The pod1 `/workspace/hinglish/deploy_serverless` mirror is stale: `/workspace` is at quota and was off-limits.
    `/root/dep2` on pod1 is ephemeral and predates these doc edits.

## D-GOLIVE 2026-10-06 (user: "send the read-only token and say go"; phase B)
Deployed and tested. Everything was created by API on the personal RunPod account and the HF account `shivamgupta`.
Full results: `results/golive_2026-10-06.md`, raw data in `results/golive_2026-10-06/`.
- **RunPod:**
  - Registry credential `cmuwbpeqa00cx3b25jfu5s9go` (`dockerhub-s2s-ro`, Docker Hub read-only token), via REST
    `/containerregistryauth`.
  - Template `gymgxjcf93`, via REST `/templates`: image `:4e950bb7bbc6`, disk 50 GB, `80/http`, env per GO_LIVE 3,
    no HF token.
  - **LB endpoint `x8842o65hgefdd`**, via GraphQL `saveEndpoint` with `type: "LB"`. REST v1 cannot set the type, but
    GraphQL can, so the "console only" note in GO_LIVE appendix A / ENDPOINT_SETTINGS is outdated.
  - Endpoint settings: `gpuIds ADA_24,AMPERE_48` (no 5090 pool id is documented), `allowedCudaVersions 13.0`,
    min 0 / max 1, idle **90 s**, REQUEST_COUNT 1, FlashBoot.
  - The account spend limit ($80/h) cannot be changed by API.
- **Space** `shivamgupta/hinglish-agent`: public, cpu-basic, contents s2s-space f554172. Secrets and variables were set
  before `upload_folder`, so the first build was already configured.
- **Measured:**
  - The image pull happens on standby workers right after the endpoint is created. It is **not billed** (spend/h 0)
    and was done within 4 min 16 s.
  - Cold start (cached host): 71 s to ready, 78.5 s to handshake. Warm reuse: 0.5 s.
  - 4-min hold through the Space: PASS (`context_full` 222 s, no gap > 500 ms, step p95 56 ms on a 4090, VRAM
    23.4 GB).
  - The router executed 4 of 5 triggers. The filler toggle reached the live call.
  - Hinglish text was confirmed.
  - Laptop → Space ws RTT 352 ms (the Space is in us-east). Pipeline lag p50 856 ms from India, 513 ms from pod1.
- **Problems:** strict `/status` keepalives and `/api/diag` time out while the websocket is open (R16). This is
  harmless: the websocket keeps the worker. `/api/filler` is not passcode-gated.
- **Cost:** $0.19 ($20.00 → $19.81). Left at 0 running workers, min 0, max 1. No code changes, no commits.

## D-ROUTER-V2 2026-10-06 (user: "do only 1." = wire Needle v2 into the serverless router)
**Why:** a live serverless call (record food_01, session s-20261006T084506-47103a) dropped the cancel.
- The customer said "mujhe order cancel karna hai. order cancel karna hai. I want to cancel the order".
- N1 returned `cancel_order(order_ref="Karna")`.
- The N1 guard (`default_active_unmatched` = unbound) dropped it, so FD4124 was never cancelled.

**What changed** (the 5 deploy changes of `hinglish/needle_v2/REPORT.md` section 3):
- **Code.** `worker/needle_v2/{schema,numconv}` are path-edit copies of `hinglish/needle_v2`: `router_v2`, `resolver_v2`, `tools_v2`, `targets`, `n1path` and `numconv.py`.
  - `score_map.py` is not copied: it needs N1 `evaluate.py`.
  - `n1path` searches only `$N1_DIR` and `worker/needle`. `router_core` sets `N1_DIR=paths.NEEDLE` before the import, so a dev-machine needle tree can never shadow it.
- **`router_core.RouterCore`.**
  - Router = `S2S_ROUTER` (`v2` by default; `n1` = the old path, unchanged).
  - v2 gets `asr["text_raw"]`, falling back to `text`.
  - Guard = `call["ask"]`. An ask call is NOT executed; it emits the new action stage `needs_clarification` with `name`, `ask_reasons`, `refs`, `server_args`, `reason`.
  - Otherwise it executes `server_args` on `resolved_id`, or the record's `primary_id` for tools without a reference (`update_contact_number` / `update_email_address`).
  - `executed.arguments` = `server_args`; the model's own arguments are in `model_arguments`.
  - `router_service` loads the router at start (a broken install fails `/health`) and reports `router` in `/health` and `/status`.
- **The new stage** is in `worker_server.ACTION_STAGES` (`/internal/action` returns 400 for unknown stages), the core and router counts, and the client.
- **Weights.** `needle_v2/tuned_full.cact` (R_e15, 63,437,076 B, md5 5361a65ba4703689e0ff1428a8f3dea4) is uploaded to `shivamgupta/s2s-v4-assets` with `ops/upload_assets.sh --only needle_v2/tuned_full.cact --yes` on pod1 (new `--only` option).
  - md5 was checked before the upload and sha256 verified against LFS after it.
  - It is in `assets_manifest.json` and baked at `/opt/s2s/assets/needle_v2/tuned_full.cact`.
  - N1 stays in the image for rollback.
  - The Dockerfile build step imports `router_v2` and asserts `N1_DIR` and the weights.
- **Agent types.** The demo stays on the 5 types (`common/session.SUPPORTED_AGENT_TYPES`). v2 also knows bank/telecom, but the stub tools (no `block_card` etc.) and the shared demo record list do not. Enabling them is a separate change.
- **Tests (pod1 CPU, real Needle).** `worker/tests/test_router_v2.py`: 27/27.
  - **K1:** the Karna transcript → v2 `cancel_order(order_ref="FD4124")`, rule `exact`, executed on FD4124.
  - **K2:** N1 on the same input → `order_ref "karna"`, unbound. This reproduces the bug.
  - **R1–R5:** v2-eval rows that were correct → executed with the expected `server_args`:
    - address change on food_07
    - phone updates on ecom_07 / air_11 (no record reference)
    - email update on air_11
    - cancel by spoken ID on ecom_03
  - **A1:** the model hears a 9-digit phone → `needs_clarification`.
  - **A2:** deterministic ask paths, plus the event path.
  - Full suite: `results/cpu_tests_2026-10-06_final/`, 10/10 suites green.
- **Deploy.**
  - s2s-worker 3957130 + s2s-space 092733a were pushed (both private GitHub repos). The Space was uploaded with `ops/push_space.sh --yes` on pod1 (HF commit 34932f37).
  - Image `docker.io/shivamgupta579/s2s-worker:3957130716eb`: 23.47 GB, built in 26 min (GitHub Actions run 37443533692).
  - `build.yml` now writes the PersonaPlex layer tar reproducibly (`--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=posix`, pax atime/ctime deleted). The same 4 files then give the same layer digest, so a later rebuild can skip re-pushing it. This build still pushed it once (286 s).
  - Template `gymgxjcf93`: `imageName` was REST-PATCHed. A before/after GET showed only imageName changed; env was identical. The standby pre-pull was free.
- **Live check (public Space, $0.12):**
  - food_08_g1 → `change_delivery_address` on FD2035 with address "flat 1801 palm drive sector 66 gurgaon" (the post-correction address).
  - The same call, trigger 2 → an invented `add_delivery_instruction` (open issue, v2 pattern F4).
  - food_21_g1 → `cancel_order` on FD5471.
  - Gate metrics on the live worker: threshold −55 for the scripted digital-silence audio.
  - Full record: `results/router_v2_2026-10-06.md`.

## D-TICKER-UI 2026-10-06 (user: "i want to see the ticker playing ... also put one for the ticker")
- **Worker.** `EngineBase.step_frame` emits a 0x07 event on every real TurnFiller transition: `{"type":"filler","state":"start|stop|cancel|skip_check","reason":model_paused|done|off|user_speech|check_line,"frame","t","mode","seconds"}`.
  - The transitions are detected around `input()`/`after()`. TurnFiller itself is untouched, and still bit-identical to the demo.
  - The events are sent through `eng.listeners` (the hub), and the Space relays them unchanged.
- **Client.** A third visual next to agent / customer: `TickerVisualizer.tsx`, labelled "ticker (model only)" with a fill count.
  - While filling it pulses at the ticker's 5 clicks/s.
  - Otherwise it shows a grey dot when idle, "skipped (check-line)" / "stopped: you spoke" briefly, and grey "off" when the filler is Off.
  - The mode comes from the newest of the `turn_fill` event, the `metrics.turn_fill.mode` and the `filler` event.
  - A start with no stop is treated as ended after `seconds` + 1.5 s, in case an event is lost.
  - The client was built on pod1 (Node 20.12.2 reinstalled at `/root/node`) and copied into `space/static`.
- **Tests.**
  - test_fixes B: the filler events arrive only while the ticker is on, start count == metrics fills, each start is closed after 13 frames, no events when off.
  - test_mock_e2e: the events are relayed by the Space.
  - Browser check `space/tests/run_e2e_ticker.sh` (Playwright on pod1; real worker `--mock` + fake LB + Space): saw `filling`, `idle`, `cancelled`, `skipped`; `off` after POST `/api/filler?mode=off`, and back after `ticker`; no page errors. Screenshots: `results/e2e_ticker/`.

## D-GATE2 2026-10-06 (user-approved: adaptive noise gate)
**The case:** a live serverless call (`hinglish/demo_debug/serverless/call_0851.wav`; ch1 = the browser mic).
- Speech was −20..−31 dBFS and the room noise −43..−57 dBFS.
- The fixed −55 gate let 89.8 % of the frames through, so the model went silent for 136 s.

**`core.InputAGC`:**
- Per session, noise floor = the 10th percentile of the raw frame dB over the last 5 s (62 frames).
- Open threshold = `clip(floor + 12, −55, −38)`. The fixed −55 is used for the first 12 frames. The 4-frame hangover is unchanged.
- Env: `S2S_GATE_ADAPTIVE` (1), `S2S_GATE_MARGIN_DB` (12), `S2S_GATE_FLOOR_WIN_S` (5), `S2S_GATE_FLOOR_PCT` (10), `S2S_GATE_MAX_OPEN_DB` (−38), `S2S_GATE_WARMUP_FRAMES` (12). `S2S_GATE_ADAPTIVE=0` gives the bit-identical D-GATE gate.
- New metrics in `input_agc`: `gate_adaptive`, `gate_margin_db`, `noise_floor_db`, `gate_threshold_db`.
- The −38 upper bound was added beyond the spec, so that a window full of speech (8 s at −32 dBFS) is never gated.

**Test results (test_fixes A, 46/46):**

| case | fixed gate | adaptive gate |
|---|---|---|
| call_0851 ch1: speech frames passed | 100 % | 100 % |
| call_0851 ch1: noise frames gated | 10.6 % | 83.4 % |
| call_0851 ch1: frames passed while the customer was silent (70–120 s) | 76.5 % | 7.7 % |
| synthetic −50 dBFS room | passes everything | gated after warm-up; −30 dBFS speech 100 % passed |

- **pod1 case** (−66 dBFS noise, −41 dBFS speech): bit-identical to the demo gate (threshold −54..−55).
- **Digital silence** (scripted calls): threshold exactly −55, all zeros.
- `/root/user_raw.wav` is gone (pod restart), so the synthetic −66 case stands in for it.

## D-LIMITS 2026-10-06 (user, via the orchestrator; set by the orchestrator, recorded here)
- **Space variables:** `MAX_CONCURRENT_CALLS=3`, `MAX_CALLS_PER_DAY=1000000`, `RATE_PER_IP_PER_HOUR=1000000` (in effect, no limit; only `CALL_MAX_S` caps a call).
- **The `S2S_PASSCODE` secret was deleted** (`passcode_required: false`). The passcode-gated helper routes `/api/endpoint_health` and `/api/diag` are therefore disabled.
- **RunPod endpoint `x8842o65hgefdd`:** `workersMax=3` (min 0, idle 90 s, same GPUs and template).
- This agent changed only the template `imageName` (D-ROUTER-V2 deploy). No `saveEndpoint` call was made.

## D-LOCAL 2026-10-06 (user: "add a local server or local running guidelines to the GitHub repos")
**Goal:** someone with one NVIDIA GPU can run the full demo (page + PersonaPlex/V4 LoRA + Trelis + Needle v2) without
RunPod or the HF Space.
- **Space backend `S2S_MODE=local`** (`space/app/sessions.py`, `space/app/upstream.py` `LocalUpstream`):
  - same claim / websocket / release flow as LB, straight to `S2S_WORKER_URL` (default `http://127.0.0.1:8000`);
  - no RunPod key, Bearer, `X-Runpod-Worker-Id` pin or keepalive;
  - the token is still minted with mode `lb` (the worker checks it); audience defaults to `local`;
  - per-IP / per-day brakes default to 100000 in local mode. `problems()` only needs the secret (≥ 32) and a URL.
- **`worker/run_local.sh`** (copied to the s2s-worker repo root by `ops/make_repos.sh`; `.local-run/` gitignored):
  - `setup`: uv venvs pp/asr/needle/space, same requirement files as the image. Torch is auto: cu130 if the driver
    reports CUDA ≥ 13, else cu128 = torch/torchaudio 2.11.0 (the newest cu128 wheels), with the `nvidia-*`, `cuda-*`,
    `triton` and `setuptools` pins dropped (torch 2.11 needs setuptools < 82);
  - downloads PersonaPlex (pinned revision, 4 files, the user's `HF_TOKEN`), Trelis (pinned), the LoRA from
    `shivamgupta/personaplex-hinglish-v4-lora` and `tuned_full.cact` from `shivamgupta/needle-hinglish-router-v2`
    (both configurable; md5s on pod1 = `assets_manifest.json`: lora 627bc77a, config 6c4fab73, v2 cact 5361a65b),
    plus the cactus-needle lib;
  - `run`: `stack.sh` (`S2S_SKIP_FETCH_ASSETS=1` + explicit paths, so `fetch_assets.py` and the private assets repo
    are not used) and the Space backend in local mode, with a generated shared secret; waits for `/ping` 200;
    TERM/Ctrl-C stops both; `check` lists what is missing.
- **No worker code changed.** The RunPod path and image are untouched, so **no image rebuild**. The worker repo diff
  is `run_local.sh`, the README and `.gitignore`.
- **Docker for outsiders: not offered.** The Dockerfile's asset step needs the private `s2s-v4-assets` (secret
  `required=true` even with `BAKE_PP=0`). Its untested cu128 branch pins `torch==2.14.1`, which has no cu128 wheel,
  and keeps the setuptools 84 pin. Open item; the README tells outsiders to use `run_local.sh`.
- **Verified on pod1** (RTX 5090, driver 570 = CUDA 12.8, so cu128) from the assembled repos (`/root/lt/s2s-worker` +
  `/root/lt/s2s-space`):
  - `setup` ok: 7.2 GB venvs; PersonaPlex read in place via `S2S_PP_DIR`; Trelis, LoRA and cact downloaded.
  - `run`: READY with engine load 48.6 s; 23.75 GB VRAM.
  - Scripted call (`tests/gpu_call.py`, `food_07` g3, seed 1001, 70 s of `inputs/V4/food_07_g3.wav`) through the local
    page backend: ready 0.5 s, handshake 5.9 s; 194 text tokens, 15 filler events; 3 router actions executed
    (`change_delivery_address` ×2, `add_delivery_instruction`); step p95 ≤ 56.8 ms; recv gap p99 87 ms.
  - Playwright `e2e_script.mjs` against the local page: SCRIPT UI OK.
  - TERM stops everything (GPU back to 2 MiB, ports closed); a second start loaded in 42.6 s.
  - Results: `results/local_gpu/`.

## D-SCRIPT-PANEL 2026-10-06 (user: "a script on the right of the text ... what the model expects me to say")
- **Data:** `common/data/scripts_v4.json` (905 KiB), built by `ops/build_scripts.py` from V4 `calls.jsonl` (md5 7f540269)
  and `holdout.json` (0a4d5483), both read from pod1 `/workspace/hinglish/data/V4` (laptop copies have the same md5).
  - Every demo record (the 5 Needle agent types, 122 records) × g1..g4 → call `<record>_<gN>`. The role prompt is
    checked equal to `session.session_config`. Per turn it keeps speaker, `text_roman` and tags, plus the expected writes.
  - 485 scripts. 3 calls were dropped at generation (`food_18_g1`, `sub_12_g2`, `sub_12_g3`): "no script for this pairing".
  - Split per call:
    - `train`: 357 (fine-tuned on this exact dialogue);
    - `test`: 21 (the scored held-out call);
    - `test_scenario`: 63 (other pairings of test scenarios; never trained on, not scored; shown as "test");
    - `val`: 44.
- **Backend:** `GET /api/script/{record_id}?pairing=gN` (`space/app/main.py`).
  - Unknown or non-demo record → 404; bad pairing → 400; dropped call → 200 `available:false`.
  - Staged into the Space by `stage_common.sh`, checked by the Space Dockerfile and `push_space.sh`. Served in local
    mode too.
- **Client:** `client/src/dep1/ScriptPanel.tsx`, in the same row to the right of "Live text stream" (`lg:grid-cols-2`;
  stacked below it when narrow).
  - "you say" customer lines are highlighted; "agent (expected)" lines are dimmed italics; `check` / `write` / `read`
    tags are small pills; the expected action is listed; a note gives the call id and split.
  - The next customer line is guessed by matching, in order, the content words of the agent lines (a stoplist of
    common Hinglish words is ignored) against the model's text stream. It is approximate.
  - The Ticker | Off toggle (`public/filler-toggle.js`) and the ticker indicator are unchanged.
- **Tests:**
  - `space/tests/test_space_flows.py` adds `t_local` (12 checks) and `t_script` (11): 111/111.
  - All 10 CPU suites rc=0 on pod1 (`results/cpu_tests_2026-10-06_local_script/`).
  - Playwright `space/tests/e2e_script.mjs` (`run_e2e_script.sh`, mock worker): panel right of the text at 1440 px;
    stacked with no horizontal scroll at 390 px; dropped-call message; toggle Off → indicator "off" → Ticker. SCRIPT UI OK.
  - `run_e2e_ticker.sh` unchanged: TICKER UI OK.
  - Screenshots: `results/e2e_script/`, `results/local_gpu/e2e/`.
- **Deploy:**
  - s2s-worker `12dee81` (`run_local.sh` + README; no image rebuild, the endpoint stays on `:3957130716eb`);
  - s2s-space `15a9cc3`;
  - HF Space upload commit `29f7a36a` (from the pod1 copy of the assembled tree). `/api/script` was live about 20 s later.
- **Live check** (one call, food_08 g1, Playwright on pod1, fake mic; `results/live_script_2026-10-06/`):
  - Cold start: 88 s to "Live". The panel sits right of the text, 7 customer lines, `train` badge, expected write
    shown. Toggle Off → `/api/filler` off and indicator "off"; Ticker → back. The mode was ticker before and after.
    Disconnect at 104 s.
  - RunPod balance $19.4980 → $19.4388 (**$0.059**). The endpoint was at 0 workers (idle 0 / running 0) at 12:51 UTC.
  - **Observation, not investigated:** in this live call the agent produced no text in the ~11 s after "Live", with
    Chrome's synthetic mic. The local GPU run with the same fake mic greeted at 1.4 s, and the relay code is
    unchanged for lb mode. Recheck with a real voice.
