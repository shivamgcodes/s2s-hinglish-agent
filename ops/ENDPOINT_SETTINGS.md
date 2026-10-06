# S2S serverless: RunPod endpoint settings (LB primary, queue fallback)

Written 2026-10-04 by the ops builder (C). Sources are the DESIGN.md tags plus these pages, fetched 2026-10-04:

| tag | URL |
|---|---|
| [API-V2] | https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint (`POST https://api.runpod.io/v2/serverless`) |
| [API-V1] | https://docs.runpod.io/api-reference/endpoints/POST/endpoints (REST v1; **no endpoint-type field**) |
| [GQL] | https://docs.runpod.io/sdks/graphql/manage-endpoints (`saveEndpoint`, `type: "LB"`/`"QB"`) |
| [GPU-T] | https://docs.runpod.io/references/gpu-types (pool ids) |
| [DOCS-853] | https://github.com/runpod/docs/issues/853 (open: `HEALTH_CHECK_PATH` ignored; `PORT_HEALTH` injected as 80) |
| [SECRETS] | https://docs.runpod.io/pods/templates/secrets (`{{ RUNPOD_SECRET_name }}`, documented for Pods) |

`ops/create_endpoint.sh` encodes the API version of this table. **The console is the primary path**, because the model
cache can only be set there (see "Model").

## 1. LB endpoint (primary)

| setting | console label | value | API-V2 field | why / source |
|---|---|---|---|---|
| Source | New Endpoint → **Import from Docker Registry** | `<registry>/s2s-worker:<tag>` | `image` | [LB-BUILD] |
| Name | Endpoint name | `s2s-hinglish-lb` | `name` | |
| **Endpoint type** | **Endpoint Type → Load Balancer** | Load Balancer | `type: "LOAD_BALANCER"` | [LB-BUILD]; REST v1 cannot create it [API-V1] |
| GPU priority | GPU configuration (up to 3, in order) | **1: 48 GB A6000/A40 · 2: 24 GB 4090 PRO · 3: 48 GB L40/L40S/6000 Ada PRO** | `gpu.pools: ["AMPERE_48","ADA_24","ADA_48_PRO"]` | DESIGN 6.1. **Do not select the 24 GB "L4, A5000, 3090" pool (`AMPERE_24`).** The L4 has about 300 GB/s of memory bandwidth and will probably miss the 80 ms frame (R4; an inference, not a measurement). Pool ids are from [GPU-T]. The [GPU-T] table lists `ADA_24` as "4090" while the console price table says "4090 PRO"; check the console's label before you rely on it. |
| GPUs per worker | GPUs / worker | 1 | `gpu.count: 1` | |
| **CUDA versions** | Advanced → **CUDA version** | **13.0 and newer only** | `gpu.minCudaVersion: "13.0"` | torch 2.14.1+cu130 and the 13.0.3 CUDA base in the image (R5). Only an image built with `TORCH_INDEX=cu128` **and** the 12.8.2 base (Amendment W2.3; `build_and_push.sh` adds it; untested) could allow 12.8+. |
| Active workers | Active (min) workers | **0** | `workers.min: 0` | scale to zero. Set 1 only for a scheduled demo window (billed 24/7 while set; COSTS.md) |
| **Max workers** | Max workers | **N = concurrent calls you allow** (start with 1-2) | `workers.max: N` | one call per worker. Set the Space `MAX_CONCURRENT_CALLS` = N. |
| Worker type | (flex is the default for non-active workers) | flex | — | [PRICE] "Flex workers: scale to zero when idle" |
| Auto-scaling | Auto-scaling type | **Request count**, value **1** | `scaling: {"type":"REQUEST_COUNT","requestCount":1}` | [EP-CFG] `ceil((inQueue+inProgress)/scalerValue)`; 1 means one worker per in-flight call/claim |
| **Idle timeout** | Idle timeout | **120 s** | `workers.idleTimeout: 120` | must be above the 20 s keepalive (DESIGN 2.1 step 7). The default is 5 s in the console ([EP-CFG]) and 10 s in API-V2. |
| Request timeout | (API only) | 330000 ms | `timeout: 330000` | API-V2 "per-request timeout, default 300000". The worker's call cap is 300 s from the upgrade, so a default of exactly 300 s could cut the call at its very end. 330 s matches the documented LB processing cap ([LB-OV]). Whether the field applies to LB websockets is unverified (R2). |
| **FlashBoot** | FlashBoot | **on** | `flashboot: "FLASHBOOT"` | the console default is on ([EP-CFG]), but **API-V2's default is `OFF`**: set it explicitly |
| Container disk | Container disk | **40 GB** | `disk: 40` | image about 13-14 GB, plus /tmp (voices, logs). DESIGN says ≥ 30 GB. |
| Exposed HTTP port | Expose HTTP ports | `80` | `ports: ["80/http"]` | the LB proxy forwards to `$PORT` ([LB-OV]) |
| **Environment** | Environment variables | `S2S_MODE=lb`, `S2S_SESSION_SECRET=…`, `S2S_AUDIENCE=hinglish-lb-1`, **`PORT=80`, `PORT_HEALTH=80`**, `CALL_MAX_S=300`, `CLAIM_TTL_S=90` | `env` | ENV.md 3.1; PORT_HEALTH because of [DOCS-853] |
| **Model (cache)** | **Model** field: `nvidia/personaplex-7b-v1`, plus the **Hugging Face access token** field | gated repo; read-only fine-grained token | **no API field** in API-V1, API-V2 or GQL (checked 2026-10-04) | [CACHE]: one cached model per endpoint; stored under `/runpod-volume/huggingface-cache/hub/`; download time not billed. **Do not attach a network volume as well** (same `/runpod-volume` mount). The docs do not confirm that the Model field exists for **LB** endpoints. If it is missing, use `ops/NETWORK_VOLUME.md` (CRIT-1) |
| Network volume | Network volume | none | — | only for DESIGN 6.4 option C (then drop the Model field); step by step in `ops/NETWORK_VOLUME.md` |
| Data centers | Advanced → Data centers | **All (unrestricted)** to start | `dataCenterIds` omitted | see section 3 |
| Registry auth | Container registry credentials | only for a private image | `registry: <credential id>` | |

After creation: open the endpoint page and confirm the **Model** field shows `nvidia/personaplex-7b-v1` with the cache
ready. Then copy the endpoint id into the Space variable `RUNPOD_ENDPOINT_ID`. The LB base URL is
`https://<ENDPOINT_ID>.api.runpod.ai` ([LB-BUILD]); API-V2 returns it as `requestUrls.base`.

## 2. Queue endpoint (fallback, same image)

Create it as a **separate** endpoint. Do not switch the LB one, because the endpoint type cannot be relied on to be
editable.

| setting | value | API-V2 | note |
|---|---|---|---|
| Endpoint type | **Queue** | `type: "QUEUE"` | |
| GPU priority, CUDA, active/max workers, FlashBoot, disk, model cache, data centers | **as for LB** | same | |
| Auto-scaling | **Queue delay, 1 s** | `scaling: {"type":"QUEUE_DELAY","queueDelay":1}` | one job = one call = one worker (handler concurrency 1), so a waiting job means a new worker is needed. Not request count: [API-V2] says `idleTimeout` is "not applicable to queue-based endpoints scaling on requestCount", and we want the idle tail |
| **Execution timeout** | **900 s** | `timeout: 900000` | The job starts when the container starts (the runpod SDK picks it up before the model is loaded), so the load counts against it. Worst case: load cap `S2S_QUEUE_LOAD_TIMEOUT_S` 420 + claim retry 30 + `CLAIM_TTL_S` 90 + `CALL_MAX_S` 300 + handler margin 30 = 870 s < 900 s. The earlier 600 s left only about 150 s for the load, which a fresh worker's torch.compile warmup (W2.2) can exceed and which would kill the job mid-call (REVIEW-ops-3). The Space also sends `policy.executionTimeout` = `QUEUE_EXEC_TIMEOUT_MS` (default 900000) per job ([SEND]); the endpoint setting must be ≥ that |
| Job TTL | default (24 h) | — | the Space sends `policy.ttl` = `QUEUE_TTL_MS` (20 min) |
| Idle timeout | 60-120 s | `workers.idleTimeout: 120` | after the handler returns |
| **Expose TCP ports** | **`8765`** | `ports: ["8765/tcp"]` | [EX-QWS] "Expose TCP Ports". The worker then sees `RUNPOD_PUBLIC_IP` + `RUNPOD_TCP_PORT_8765` |
| Environment | `S2S_MODE=queue`, `S2S_SESSION_SECRET=…`, `S2S_AUDIENCE=hinglish-q-1`, `PORT=8765`, `CALL_MAX_S=300`, `CLAIM_TTL_S=90`, `S2S_QUEUE_EXEC_TIMEOUT_S=900`, `S2S_QUEUE_LOAD_TIMEOUT_S=420` | `env` | `S2S_QUEUE_EXEC_TIMEOUT_S` = the execution timeout in s (= Space `QUEUE_EXEC_TIMEOUT_MS`/1000); keep load + 30 + claim TTL + call + 30 ≤ it |

The queue relay leg is **plaintext** `ws://IP:PORT` with the HMAC token as the only auth (R6). It also needs the Space
to reach a random non-443 port, which test #1's outbound self-test checks (R17). Use queue mode only if LB fails test #3.

## 3. Data centers and the AP-JP-1 note

- Data-center ids, per a web-search summary of the RunPod API reference (the API-V1 page we fetched shows only a few
  ids plus "etc."; **verify in the console**), include EU-RO-1, CA-MTL-1, EU-SE-1, US-IL-1, EUR-IS-1, EU-CZ-1, US-TX-3, EUR-IS-2,
  US-KS-2, US-GA-2, US-WA-1, US-TX-1, CA-MTL-3, EU-NL-1, US-TX-4, US-CA-2, US-NC-1, OC-AU-1, US-DE-1, EUR-IS-3,
  CA-MTL-2, **AP-JP-1**, EUR-NO-1, EU-FR-1, US-KS-3 and US-GA-1. There is **no India DC** in that list. AP-JP-1 (Japan,
  reportedly Fukushima; same search summary) is the geographically closest to the users.
- The audio path is browser (India) → HF Space (the Space region; HF does not let you choose it on cpu-basic) →
  RunPod DC. The relay hop goes through the Space, so a DC near the **Space** matters as much as one near the user (R10).
- The docs do not say which GPU types or CUDA-13 hosts each DC has. Pinning to one DC, especially a small one like
  AP-JP-1, together with the 48 GB + CUDA 13 filters, may leave **no** eligible host. Then the wake times out (R5).
- **Procedure:**
  1. Start unrestricted.
  2. Run `tests/latency_probe.py --call` from India. leg2 shows Space→worker RTT and `/status` shows the GPU.
  3. Try a restriction (e.g. AP-JP-1, or a US DC near the Space region) and repeat test #2 (cold start). Keep the
     restriction only if the wake time stays acceptable.
  4. Note that restricting DCs also changes which hosts already have the image cached, so the first cold start there
     is slow again.

## 4. Spend controls (do these on day 1)

1. **Account spend limit.** The default is $80/h across all resources ([PRICE] "Account limits"). Ask RunPod to lower
   it if you can; otherwise rely on the next items. Keep a modest balance and turn off auto top-up while testing.
2. **Max workers** small (1-2). This caps parallel billing: N × $1.22/h on A6000.
3. **Active workers 0** except during a planned demo.
4. **Idle timeout 120 s, not more.** Every call pays it once at the end (about $0.04 on A6000).
5. Space brakes: `S2S_PASSCODE`, `MAX_CONCURRENT_CALLS` = max workers, `RATE_PER_IP_PER_HOUR`, `MAX_CALLS_PER_DAY`.
6. **Idle-endpoint throttle** ([EP-CFG]): after 3 days without requests, max workers drops to 2; after 7 days it drops
   to **0**, and stays there until you raise it again in the console (R9). If the Space shows `failed` / "wake timed
   out" after a quiet week, check this first.
7. When you are done testing: set max workers to 0 (or delete the endpoint). A Space that stays up cannot then start a
   worker.

## 5. Health and routing facts the worker relies on

- `/ping` on `PORT_HEALTH` (= `PORT` = 80): 204 while loading, 200 when ready **and while in a call**, 500 if the load
  failed ([LB-OV]: 200 healthy, 204 initializing, anything else unhealthy). "If your server ports are misconfigured,
  workers stay up for 8 minutes before terminating, returning 502" ([LB-OV]). A cold start that keeps returning 502 is
  therefore a PORT problem, not a model problem.
- With no ready worker, the proxy holds a request up to 2 min and then answers "no workers available"
  ([LB-OV] timeouts table). The Space keeps polling up to `WAKE_TIMEOUT_S`=600 s.
- `X-Runpod-Worker-Id: strict <id>` pins the websocket and the keepalives ([LB-AFF]).
