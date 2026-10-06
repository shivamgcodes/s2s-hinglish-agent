# GO LIVE: Hinglish agent on RunPod Serverless + Hugging Face Space

Written 2026-10-06; **revised 2026-10-06 for D-BAKE** (DECISIONS): all weights are baked into the worker image, which
GitHub Actions builds and pushes to a **private Docker Hub repo**; RunPod uses **"Deploy from a Docker image"**. The
earlier "Deploy from a GitHub repository" + RunPod model cache route is superseded (kept below only where noted).

**Credentials status (2026-10-06):**
- HF account `shivamgupta` (PRO). Its write token is at pod1 `/workspace/hf/token`: use it only on pod1, never copy
  or print it. It is also the `HF_TOKEN` Actions secret on `s2s-worker` (build-time only, BuildKit secret).
- Docker Hub user `shivamgupta579`; its access token is in laptop `~/Desktop/S2S/.env` (`DOCKERHUB_TOKEN`) and in the
  `s2s-worker` Actions secrets. The repo `docker.io/shivamgupta579/s2s-worker` exists and is **private**.
- The personal RunPod key is on the laptop in `~/Desktop/S2S/.env` as `RUNPOD_API_KEY_PERSONAL`. It belongs in the
  Space secret only.
- **U** = you (the user), in a browser.
- **A** = an agent session, with your go-ahead.

```
browser ──https/wss──> HF Space (s2s-space, CPU, holds the RunPod key) ──wss + Bearer──> RunPod LB endpoint
                                                                                         (private image, 1 GPU)
image (GitHub Actions, s2s-worker/.github/workflows/build.yml) = code + venvs + ALL weights:
   PersonaPlex 7B (gated)  <- nvidia/personaplex-7b-v1 @ fdaf4090 (4 files)       ┐ downloaded at BUILD time with
   V4 adapter + Needle N1  <- private HF repo shivamgupta/s2s-v4-assets (md5)      ┘ the HF token as a BuildKit secret
   Trelis ASR              <- Trelis/whisper-hinglish-preview (public, pinned)
worker start: fully offline. No HF token, no Model field, no network volume.
```

> **Cost box.** You start with $20 of credit on a personal RunPod account.
> - **GPU price:** RTX 4090 (24 GB, "4090 PRO") about $1.12/h; RTX A6000 (48 GB) about $1.22/h. Billed per second,
>   only while a worker is up.
> - **What $20 buys:** about 16-18 GPU-hours.
> - **One call:** a ~4-min call costs about $0.10-0.17. That covers the cold start (~1.5 min), the call, and the 120 s
>   idle tail.
> - **First call on a fresh host:** the 23.4 GB (compressed) image pull adds several minutes. RunPod pulls the image when a worker
>   is placed on a host; whether that time is billed is not confirmed. Budget up to about $0.30 for the first call.
> - **At idle:** $0. Active workers = 0 means nothing runs when nobody calls.
> - **HF PRO:** $9/month. It is needed for a Docker Space. The CPU-basic Space hardware is free.
> - **Worst case per day:** limited by the Space brakes. With `MAX_CALLS_PER_DAY=100` it is about $15; set it lower
>   (for example 30) while testing.

## UPDATE 2026-10-06 (D-LOCAL, D-SCRIPT-PANEL)
- **Script panel:** "Script: what to say", to the right of the live text, shows the expected V4 conversation for the
  record + pairing, with its train/val/test split. Route: `GET /api/script/{id}?pairing=`; data:
  `common/data/scripts_v4.json` (`ops/build_scripts.py`). Space-only change. The ticker toggle and indicator are
  unchanged.
- **Run locally:**
  - `s2s-worker/run_local.sh setup|run`: one GPU, no RunPod; the Space backend runs with `S2S_MODE=local`.
  - Documented in both repo READMEs ("Run it locally"); tested on pod1 (cu128).
  - Worker image **not rebuilt**: no worker code changed. The endpoint keeps `:3957130716eb`.
- **Rebuild the Space client:** pod1 `/root/dep2/client` (`npm run build`), then `client/dist`, `ops/make_repos.sh`,
  `ops/push_space.sh`.

## UPDATE 2026-10-06 (D-ROUTER-V2, D-TICKER-UI, D-GATE2, D-LIMITS)
- **Image now: `docker.io/shivamgupta579/s2s-worker:3957130716eb`** (s2s-worker 3957130, 23.47 GB, 26 min build).
  Template `gymgxjcf93` `imageName` was patched by REST (nothing else changed). Space = s2s-space 092733a (HF commit 34932f37).
- **Router: Needle v2** (`S2S_ROUTER=v2` default; set the endpoint env `S2S_ROUTER=n1` to roll back to N1 without a
  rebuild). A call v2 cannot fill safely is not executed; it is shown as the action stage `needs_clarification`.
- **Ticker indicator** next to the agent / customer visuals, driven by the worker's 0x07 `filler` events.
- **Adaptive noise gate** (D-GATE2): threshold = clip(noise floor + 12, −55, −38) dBFS; `S2S_GATE_ADAPTIVE=0` = old gate.
- **Limits (D-LIMITS, user):**
  - No passcode: the `S2S_PASSCODE` secret was deleted, so `/api/endpoint_health` and `/api/diag` are disabled.
  - Space variables: `MAX_CONCURRENT_CALLS=3`, `MAX_CALLS_PER_DAY=1000000`, `RATE_PER_IP_PER_HOUR=1000000`.
  - Endpoint: `workersMax=3`.
- **Live check:**
  - food_08 → `change_delivery_address` on FD2035, address "flat 1801 palm drive sector 66 gurgaon" (the corrected address).
  - food_21 → `cancel_order` on FD5471.
  - Cost $0.12. Details: `results/router_v2_2026-10-06.md`.
- Sections below are the original go-live record; where they say 4e950bb7bbc6, passcode, max 1 or N1, the update
  above wins.

## LIVE (2026-10-06, phase B, D-GOLIVE)
- **Space:** https://shivamgupta-hinglish-agent.hf.space/ (public, passcode = `S2S_PASSCODE` in secrets.env).
- **RunPod LB endpoint:** `x8842o65hgefdd` (`https://x8842o65hgefdd.api.runpod.ai`). Template `gymgxjcf93`, registry
  credential `dockerhub-s2s-ro` (read-only token). GPUs 4090 → A40/A6000, CUDA 13.0, min 0 / max 1, idle 90 s,
  FlashBoot. All created by API (REST + GraphQL `saveEndpoint type:"LB"`), so sections 2-4 below are done.
- **Measured:** first cold start 71 s to ready and 78.5 s to handshake (the image was pre-pulled for free when the
  endpoint was created); warm 0.5 s; 4-min hold PASS (`context_full` at 222 s); Hinglish speech, router and toggle OK.
  Cost of all tests: **$0.19** (balance $20.00 → $19.81). Details: `results/golive_2026-10-06.md`.
- **Still for you (U):** T4, a real browser call with a microphone. Account spend limit: lower it in the console if
  offered (no API field). When done, set max workers to 0.

## 0. Done (2026-10-06)
| item | where |
|---|---|
| The live-demo fixes, ported and tested identical to the pod1 demo: input AGC, the −55 dBFS noise gate, the ticker turn filler + its toggle, the loosened playback buffer, recording off by default | `worker/server/core.py`, `worker/server/worker_server.py`, `space/app/main.py`, `client/src/audio-processor.ts`, `client/public/filler-toggle.js`; DECISIONS "PORT" |
| V4_A2 step-600 adapter + Needle N1 router (since D-ROUTER-V2: + Needle v2, the default) (the repos are code-only); since D-BAKE fetched at image BUILD time, not at worker start | `worker/fetch_assets.py`, `worker/assets_manifest.json`; DECISIONS "WEIGHTS", D-BAKE |
| Upload script for the private HF assets repo (dry run passed on pod1: all 5 md5s match) | `ops/upload_assets.sh` |
| Client built from source (Node 20.12.2 on pod1) and copied to `space/static` | DECISIONS "CLIENT" |
| CPU tests: 9 suites green on pod1 `/root/dep2` (logs copied back); the Space image was built and run on the laptop | `results/cpu_tests_2026-10-06/`, DECISIONS "TESTS" |
| Private GitHub repos pushed: https://github.com/shivamgcodes/s2s-worker (GitHub Actions builds the image) and https://github.com/shivamgcodes/s2s-space (the HF Space contents), branch `main`, initial commits `f7314f2` (worker) / `f554172` (space) | `ops/make_repos.sh` assembles both from this folder; the local clones are in `hinglish/deploy_repos/` |
| **D-BAKE (2026-10-06 later):** private HF repo `shivamgupta/s2s-v4-assets` created + uploaded (5 files + MANIFEST; sizes + LFS sha256 verified, then re-downloaded through `fetch_assets.py`: all 4 md5s match). Private Docker Hub repo `shivamgupta579/s2s-worker` created. Dockerfile bakes all weights; `.github/workflows/build.yml` builds + pushes; Actions secrets `HF_TOKEN`, `DOCKERHUB_USER`, `DOCKERHUB_TOKEN` set | DECISIONS D-BAKE |
| `~/.config/s2s/secrets.env` (laptop, mode 600) holds a generated `S2S_SESSION_SECRET` and `S2S_PASSCODE`, plus empty `HF_TOKEN_READ` / `HF_TOKEN_WRITE` lines (no longer needed for the endpoint under D-BAKE) | never printed; open it yourself to copy the values |

## 1. Done by agents: assets repo, Docker Hub repo, image build (no RunPod money)
1. `shivamgupta/s2s-v4-assets` (private HF model repo): `ops/upload_assets.sh --yes` on pod1 (done 2026-10-06).
2. `docker.io/shivamgupta579/s2s-worker` (private Docker Hub repo; a free account has 1 private repo): created via the
   Hub API, `is_private: true` read back. The workflow refuses to push if it is ever public or missing.
3. Image: GitHub → `shivamgcodes/s2s-worker` → Actions → **build-worker-image** → Run workflow (or push a `v*` tag).
   Tags pushed: `:<first 12 of the commit sha>` and `:latest` (plus the intermediate `:base-<sha12>`).
   **Built 2026-10-06: `docker.io/shivamgupta579/s2s-worker:4e950bb7bbc6`** (= `:latest`), 23.4 GB compressed,
   ~31 GB unpacked, 28 min on GitHub Actions (DECISIONS D-BAKE). Not yet started on a GPU.

## 2. U: RunPod account bits (console, ~5 min)
1. Personal RunPod account with **$20** credit, **auto top-up off**.
2. **Settings → Container Registry Auth → Add credential**: name `dockerhub-s2s`, username `shivamgupta579`, password
   = a Docker Hub access token with **read** scope (make a separate read-only one at hub.docker.com → Account settings
   → Personal access tokens; do not reuse the read/write build token).
3. **Settings → API Keys → Create API Key** (read/write). Do not save it anywhere: it goes only into the Space secret
   `RUNPOD_API_KEY` in step 4.2.

## 3. U: RunPod endpoint (console, ~10 min; billed only while a worker runs)
**Serverless → New Endpoint → Deploy from a Docker image** (Docker Image / "Import from Docker Registry"). Then:

| field | value |
|---|---|
| Container image | `docker.io/shivamgupta579/s2s-worker:3957130716eb` since D-ROUTER-V2 (was `:4e950bb7bbc6`; pin the sha tag; `:latest` only for quick tests) |
| Container registry credentials | `dockerhub-s2s` (step 2.2) |
| Endpoint Name | `s2s-hinglish-lb` |
| **Endpoint Type** | **Load Balancer** |
| GPU Configuration | priority 1 **48 GB "A6000, A40"** ($1.22/h; headroom), priority 2 **24 GB "4090 PRO"** ($1.12/h; tight: the V4 stack measured 23.4 of 24.5 GB). Not the 24 GB "L4/A5000/3090" pool (too slow). Blackwell (5090) also works with the cu130 image |
| GPUs per worker | 1 |
| **CUDA version filter** (Advanced) | **13.0 and newer only** (the image is cu130) |
| **Active workers** | **0** (scale to zero) |
| **Max workers** | **1** (frugal; the Space `MAX_CONCURRENT_CALLS` must equal it) |
| **Idle timeout** | **60-120 s**. 120 s is the safe default; not below 60 s (the keepalive runs every 20 s, R2) |
| FlashBoot | on |
| Container disk | **50 GB** (the image is ~31 GB unpacked; plus logs/tmp) |
| Expose HTTP ports | `80` |
| Health check endpoint (if shown) | `/ping` |
| **Model** | **leave empty** (the weights are in the image) |
| Network volume | none |
| Data centers | all (default) |

**Environment variables** (the ✱ values come from `~/.config/s2s/secrets.env`; nothing else is secret; **no
`HF_TOKEN`**):
```
S2S_MODE=lb
S2S_AUDIENCE=hinglish-lb-1
S2S_SESSION_SECRET=✱ (same value as in the Space)
PORT=80
PORT_HEALTH=80
CALL_MAX_S=300
CLAIM_TTL_S=90
```
Optional (the defaults are right): `S2S_TURN_FILL=ticker`, `S2S_INPUT_AGC=1`, `S2S_INPUT_GATE=1`,
`S2S_GATE_OPEN_DB=-55`, `S2S_RECORD_SESSIONS=0`.

**Spend controls (day 1; you pay from the personal $20).**
- Account spend limit **low** (Billing, if offered; the default is $80/h).
- Balance $20 with auto top-up off = the hard cap.
- Max workers 1, active workers 0, idle 60-120 s.
- After 7 days without requests RunPod sets max workers to 0; raise it again before a demo.
- When you are done testing, set max workers to **0**.

**Copy the endpoint ID** (shown on the endpoint page) for step 4.

## 4. Orchestrator (or U): the HF Space (about 5 min)
1. Create the Space `shivamgupta/hinglish-agent`. Either:
   - `ops/push_space.sh` in step 4.3 creates it (public, Docker, cpu-basic); or
   - in the web UI: New Space, SDK **Docker** (blank), hardware **CPU basic**.
   - Visibility: **Public** (or Protected). **Not private**: a private Space returns 404 to visitors and to their
     websockets.
2. **Settings → Variables and secrets.** Set these **before** the first push.

   | name | kind | value |
   |---|---|---|
   | `RUNPOD_API_KEY` | **Secret** | the key from step 1.3, pasted directly |
   | `S2S_SESSION_SECRET` | **Secret** | ✱ from secrets.env (the same as the endpoint) |
   | `S2S_PASSCODE` | **Secret** | ✱ from secrets.env (share it only with demo users) |
   | `RUNPOD_ENDPOINT_ID` | Variable | the endpoint ID from step 3 |
   | `S2S_AUDIENCE` | Variable | `hinglish-lb-1` |
   | `S2S_MODE` | Variable | `lb` |
   | `MAX_CONCURRENT_CALLS` | Variable | `1` (= endpoint max workers) |
   | `MAX_CALLS_PER_DAY` | Variable | `30` while testing |
   | `RATE_PER_IP_PER_HOUR` | Variable | `6` |
   | `S2S_TURN_FILL_DEFAULT` | Variable | `ticker` (the page toggle switches it) |
3. Upload the Space contents with huggingface_hub, **not git push**. A web-created Space has its own first commit,
   and HF rejects plain-git binaries (.wasm/.png/.ico). On pod1:
   ```bash
   rsync -a <laptop>:Desktop/S2S/hinglish/deploy_repos/s2s-space/ /root/dep2/repos/s2s-space/   # or clone the GitHub repo
   # (pod1 /root is wiped on restart: re-create /root/dep2 + its venvs first; DECISIONS D-BAKE "pod1 env")
   cd /root/dep2 && bash ops/push_space.sh --dir /root/dep2/repos/s2s-space            # dry run
   bash ops/push_space.sh --dir /root/dep2/repos/s2s-space --yes                       # token from /workspace/hf/token
   ```
   If the Space was created by the script, set the variables and secrets of step 4.2 right after it, then use
   **Settings → Factory rebuild**.
4. Wait for the Space to show **Running**, then open **`https://shivamgupta-hinglish-agent.hf.space/`**. Use the direct
   URL, not the huggingface.co/spaces page, whose iframe can block the microphone.

## 5. A: tests, in order, through the Space only (needs only the Space URL + passcode)
Setup (laptop): `python3 -m venv ~/s2s-test && ~/s2s-test/bin/pip install aiohttp numpy sphn==0.1.12`, then
`cd deploy_serverless/tests; P=~/s2s-test/bin/python; SPACE=https://shivamgupta-hinglish-agent.hf.space`, then
`export S2S_PASSCODE=…` (from secrets.env). The ✱ costs below assume a 4090.

| # | what | command | cost | pass |
|---|---|---|---|---|
| T1 | config sane | `curl $SPACE/api/config` and `curl "$SPACE/api/endpoint_health?passcode=$S2S_PASSCODE"` | free | `"ok": true`; `runpod_health_status` 200 with worker counts. If it is not 200, the key or endpoint id is wrong. On LB endpoints the /health route is unverified, so a 404 there is not fatal |
| T2 | websocket through the HF proxy | `$P space_ws_echo.py $SPACE` | free | `ws_echo OK`, 20 s hold |
| T3 | **first cold start** | `$P cold_start_timer.py --space $SPACE --passcode $S2S_PASSCODE --runs 1` | ~$0.15-0.26 | reaches `ready` in ≤ 600 s. While it runs: `endpoint_health` shows initializing → running |
| T4 | **real call in the browser (you)** | open the Space with headphones, enter the passcode, pick a record, Connect, talk | ~$0.15 | the agent greets and answers. The mic works because the gate passes speech and zeroes the room noise. Audio is smooth (the loose buffer). The router fires on an order request |
| T5 | toggle | during T4 click **Turn filler: Off**, then **Ticker** | — | the button turns green and the POST answers `live_calls_updated: 1`. `curl "$SPACE/metrics?sid=<sid>"` shows `turn_fill.mode` following the click; the sid comes from `space_sessions_active` in `/api/endpoint_health`. The page's panel does not show this field. Space log: "sent to 1 live call(s)" |
| T6 | warm reuse | `$P cold_start_timer.py --space $SPACE --passcode $S2S_PASSCODE --runs 2 --gap-s 10` | ~$0.10 | run 2 is ready in seconds |
| T7 | latency | `$P latency_probe.py --space $SPACE --call --talk-s 30` | ~$0.12 | step p95 < 80 ms; record the RTT legs |
| T8 | 4-min hold | `$P ws_hold_test.py --space $SPACE` | ~$0.15 | ends `context_full` (~221 s) or `time_limit`; never `worker_lost` |
| T9 | clean-up | RunPod: max workers 0 if you are done; Space `MAX_CALLS_PER_DAY` as wanted | — | nothing billed at idle |

If a step fails, see `tests/TEST_PLAN.md` "if it fails". In short:
- `/ping` 500 with "fetch_assets failed" or "PersonaPlex not found": the image is not the baked one (check the tag);
  the baked image needs neither a token nor the Model field.
- Worker stuck "initializing" / image pull errors: the registry credential is missing or wrong (step 2.2).
- 502s: a PORT problem.
- Relay 401 `missing`: set the Space variable `S2S_TOKEN_IN_QUERY=1`.

**Total time once the image exists:** about 30-45 min, mostly the first image pull.
**Total test cost:** about $1.

## 6. Open items (not blocking)
- **The image has never run on a GPU.** It is built by GitHub Actions (no GPU there), so the first real start is T3.
  The code inside it is the code the GPU smoke ran (results/gpu_smoke), plus the D-BAKE paths.
- **Image pull time** of 23.4 GB on a fresh RunPod host is unmeasured; T3 measures it. FlashBoot/host caching should
  make later cold starts on the same host fast.
- **The `HF_TOKEN` Actions secret is the write token.** It is build-time only, but a read token (fine-grained:
  `nvidia/personaplex-7b-v1` + `shivamgupta/s2s-v4-assets`) is the better long-term value.
- **The toggle state is per Space process,** not per browser. It is the same as the demo's global switch. Two
  concurrent users would share it; with `MAX_CONCURRENT_CALLS=1` this does not matter.
- **Needle v2 is wired in** since 2026-10-06 (D-ROUTER-V2); N1 is the rollback (`S2S_ROUTER=n1`). Known v2 weakness seen live: an invented `add_delivery_instruction` on a later trigger (REPORT F4).
- **Records.** The worker and the Space use the V4 records (162). The demo filters them to the 5 Needle agent types.
  The records are low-diversity (see the memory note).
- **Untested at runtime:** the `RUNPOD_API_KEY` scope and `/api/endpoint_health` on LB endpoints.

## Appendix A: optional API route (not the primary path)
`ops/create_endpoint.sh` (API-based, written for a registry image, never run) is kept for reference only. With a key
that only the Space holds, use the console (section 3).

## Appendix B: updating
Since D-MONOREPO (2026-10-06), everything is in ONE private repo, `github.com/shivamgcodes/s2s-hinglish-agent`.
`s2s-worker` and `s2s-space` are retired.
- **Code change in this folder:**
  1. `bash ops/export_monorepo.sh <checkout of s2s-hinglish-agent>`. This runs `ops/scan_secrets.sh`: token shapes,
     the real secret values incl. the passcode, forbidden files and files > 50 MB.
  2. Commit + push the monorepo (only with your go-ahead).
  3. Worker: run the **build-worker-image** workflow in the monorepo (or push a `v*` tag), then set the new
     `:<sha12>` image on the RunPod endpoint (Manage → Edit endpoint → container image).
  4. Space: `bash ops/push_space.sh --build-client --out /root/space-tree` (dry run: builds the client, stages and scans
     the tree), then the same command with `--yes`. pod1: `PATH=/root/node/bin:$PATH`; the token is read in place from
     `/workspace/hf/token`.
- **Client change:** nothing to copy by hand. `push_space.sh --build-client` builds `client/` → `client/dist` →
  `static/`, and `client/dist` is not committed.
- **Pre-merge (historical):** `ops/make_repos.sh` split this folder into the two repos. It was removed in D-MONOREPO.
