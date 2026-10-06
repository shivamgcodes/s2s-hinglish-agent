# S2S serverless: cost notes

Written 2026-10-04 by the ops builder (C). Prices are per second for **flex** (scale-to-zero) workers, from the
[EP-CFG] GPU table (https://docs.runpod.io/serverless/endpoints/endpoint-configurations, fetched 2026-10-04). Billing
rules are from [PRICE] (https://docs.runpod.io/serverless/pricing). The task's research figure for L40S was
$0.00049-0.00053/s; the console table lists the 48 GB "L40, L40S, 6000 Ada PRO" pool at $0.00053/s, so that is used
below. **Check the console before you rely on any number: prices change.**

## 1. What is billed

- **Per second, from worker start until it fully stops:** start time (container init + model load), execution, and
  **idle timeout** ([PRICE] "Compute cost breakdown").
- Model-cache download time is **not** billed ([CACHE] "How it works").
- Image pull time on a fresh host is not described in the pages we read. The numbers below assume the worst case: it
  is billed.
- Container disk about $0.10/GB/month; network volume $0.07/GB/month (< 1 TB) ([PRICE]). We use no network volume.
- Scale to zero: with **active workers = 0** and no traffic, the GPU cost is **$0**.
- The HF Space front: cpu-basic hardware costs nothing per hour, **but a Docker/Gradio Space needs a paid HF plan**
  (PRO for a personal account; DESIGN premise 1). The task's "HF CPU Space free" holds only for Static Spaces, which
  cannot run the relay, or for launcher (b) on ZeroGPU, which is untested. Check the current PRO price on
  huggingface.co/pricing.

## 2. Rates

| pool (console) | VRAM | $/s | $/h | use? |
|---|---|---|---|---|
| L4, A5000, 3090 (`AMPERE_24`) | 24 GB | 0.00019 | 0.68 | **no**: L4 is probably too slow (R4). Allow only if a test proves L4 p95 < 80 ms. |
| 4090 PRO (`ADA_24`) | 24 GB | 0.00031 | 1.12 | priority 2 (24 GB fits, measured 23.1 GB on the 3090) |
| A6000, A40 (`AMPERE_48`) | 48 GB | 0.00034 | 1.22 | **priority 1** |
| L40, L40S, 6000 Ada PRO (`ADA_48_PRO`) | 48 GB | 0.00053 | 1.91 | priority 3 |

## 3. Per call

Assumptions:
- cold start about 90 s with the image cached on the host (DESIGN 6.5; 4-8 min the first time on a host). Amendment
  W2.2: a fresh worker also compiles moshi's torch.compile kernels at warmup (empty caches), so the real number may be
  higher; test #2 measures it;
- call about 240 s: prompt phase 9-10 s plus up to about 221 s of context (the cap is 300 s);
- idle timeout 120 s after the call.

| scenario | billed seconds | 24 GB pool | 4090 PRO | A6000/A40 | L40/L40S |
|---|---|---|---|---|---|
| **single call, scale from zero** (90 + 240 + 120) | 450 | $0.086 | $0.14 | **$0.15** | $0.24 |
| first call on a fresh host (+ about 5 min pull, worst case) | 750 | $0.14 | $0.23 | $0.26 | $0.40 |
| extra back-to-back call on a warm worker (240 + about 10 s claim) | 250 | $0.048 | $0.078 | $0.085 | $0.13 |
| aborted wake (user leaves during cold start; the worker loads and then idles out: 90 + 120) | 210 | $0.04 | $0.065 | $0.071 | $0.11 |

The per-call costs are dominated by the call itself (about 55%) and the idle tail (about 27%). Lowering the idle
timeout below 120 s saves about $0.007 per 20 s, but risks a mid-call scale-down if the websocket is not counted as
work (R2). Lower it only after hold test #3 shows the call survives.

## 4. Always-on and caps

| setting | cost |
|---|---|
| 1 active worker (min_workers = 1) on A6000, 24/7 | $1.22/h = **$29/day** = about $880/month (flex rate; active-worker discounts only via sales, [PRICE]) |
| 1 active worker for a 2 h demo window | about $2.45 |
| Space cap `MAX_CALLS_PER_DAY` = 100 at the A6000 single-call price | at most about **$15/day** |
| `RATE_PER_IP_PER_HOUR` = 6 | one IP at most about $0.90/h |
| max workers N | at most N × $1.22/h on A6000 while busy |
| RunPod account spend limit (default) | $80/h ([PRICE] "Account limits") |

## 5. Comparison: the DEP1 pod

The DEP1 demo runs on a dedicated RTX 3090 pod that is billed for every hour it is up, whether used or not. Serverless
pays only for calls (about $0.15 each on A6000) and nothing at idle.

Break-even = (pod $/h) ÷ $0.15 calls per hour of pod uptime. For example, a pod at $0.40/h breaks even at about
2.7 calls per hour. Read the pod's actual hourly price in the RunPod console; it is not in our sources. Below that
call rate, serverless is cheaper. The price you pay for it is the cold start (about 1-2 min) on the first call after
a quiet period.

## 6. Hidden cost holes (R7) and their brakes

- A public Space can start paid workers for anyone. Use the brakes: `S2S_PASSCODE`, the per-IP rate, the daily cap,
  and `MAX_CONCURRENT_CALLS`.
- `/api/diag` works only for an active sid and only sends a strict-pinned request, so it cannot wake or hold a worker.
- `S2S_SELFTEST` is off by default.
- Keepalive pings run only while a call is up (20 s), and they stop when the call ends.
- An endpoint left with active workers ≥ 1 is billed 24/7. The test plan's last step sets it back to 0.
