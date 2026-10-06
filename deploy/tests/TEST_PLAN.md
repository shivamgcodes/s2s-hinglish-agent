# S2S serverless: test plan for the user's later session

> **2026-10-06:** the binding test order is `GO_LIVE.md` section 5. It runs through the public Space URL + passcode
> only, because nobody but the Space holds the RunPod key. The `--direct` variants below need the key, so they are
> optional extras.

Written 2026-10-04 by the ops builder (C). It orders DESIGN section 9's five real tests after the deploy steps in
ops/BUILD.md, and says what each result means. Every script is a **dry run unless you give it a target**: `--space
URL` or `--direct lb|queue` with env. Each run prints a plan first.

**Setup on the laptop:**
```bash
python3 -m venv ~/s2s-test && ~/s2s-test/bin/pip install aiohttp numpy sphn==0.1.12    # python >= 3.10 (tested 3.11)
rsync -a runpod2:/root/deploy_serverless/{tests,common} ./s2s/      # the scripts import ../common/s2s_token.py
cd s2s/tests && P=~/s2s-test/bin/python
export S2S_PASSCODE=...          # if the Space has one
SPACE=https://<you>-<space>.hf.space
```
For `--direct` runs, also export `RUNPOD_API_KEY`, `S2S_SESSION_SECRET`, `RUNPOD_ENDPOINT_ID` and `S2S_AUDIENCE`
(ENV.md section 4). Unset them afterwards.

Save every output: add `--out results/<step>.json`.

| # | step | command | cost | pass criterion | if it fails |
|---|---|---|---|---|---|
| 0 | CPU self-tests (runpod2, any time) | `python3 tests/test_token.py`; `/root/deploy/venv-pp/bin/python tests/test_fake_runpod.py`; `/root/deploy/venv-pp/bin/python tests/test_mock_e2e.py` | free | `ALL OK` | fix code before deploying |
| 1 | **wss echo through the HF proxy** (DESIGN test #1, R1) | `$P space_ws_echo.py $SPACE` | free (no GPU) | `"ws_echo": "OK"`, 64 KB binary echoed, 20 s hold kept | 404: check visibility (public/protected), use the direct `*.hf.space` URL; still 404 → launcher (c) (space_push.md section 6) |
| 1b | outbound non-443 port from the Space (R17, queue mode only) | set Space var `S2S_SELFTEST=1`, then `$P space_ws_echo.py $SPACE --outbound <ip>:<port>`; set it back to 0 | free | `"outbound": {"status": 200, "body": {"ok": true}}` | queue mode cannot work from this Space; keep LB, or run the Space as launcher (c) |
| 2 | latency, leg 1 only | `$P latency_probe.py --space $SPACE` | free | record `tcp_connect_ms`, `tls_handshake_ms`, `ws_echo_rtt_ms` from India | high RTT = Space region far away (R10) |
| 3 | **cold start** (DESIGN test #2) | first run after deploy: `$P cold_start_timer.py --space $SPACE --runs 1` (cold host) | about $0.15-0.26 | reaches `ready`; `to_ready_s` ≤ 600; record the phases | `WAKE_FAIL`: look in the RunPod endpoint logs. A `/ping` 500 means the model was not found (R12: check the Model cache step; if the LB endpoint has no Model field, or the log still says `PersonaPlex not found`, use ops/NETWORK_VOLUME.md). Session `failed` with "token did not reach the worker", or `relay_error` 401 `missing`: set Space variable `S2S_TOKEN_IN_QUERY=1` (O-W1, CRIT-3). Repeated 502s mean a PORT problem (ENDPOINT_SETTINGS 5). "no workers" for minutes means the GPU/CUDA filter is too narrow (R5) |
| 3b | scale from zero, cached host | wait ≥ 4 min (idle timeout 120 s + margin), then `$P cold_start_timer.py --space $SPACE` | about $0.15 | expect about 1-2 min to `ready` | > 3 min every time: consider DESIGN 6.4 options A/B, or `TRELIS_BF16=1` |
| 3c | warm | `$P cold_start_timer.py --space $SPACE --runs 2 --gap-s 10` | about $0.10 | run 1 is a few seconds | slow: the worker is not reused; check `MAX_CONCURRENT_CALLS` / the claim logs |
| 4 | **GPU check** (DESIGN test #5) | inside step 5: `gpu_check` (or `/status` `gpu`, `step_ms_p95_last`) | — | `step_ms_p95_max` < 80 ms | ≥ 80 (R4): remove that GPU type from the endpoint's priority list |
| 5 | **latency, all legs** (DESIGN test #4, run from India) | `$P latency_probe.py --space $SPACE --call --talk-s 30`; for comparison, with the DEP1 pod up and `ssh -N -L 8998:localhost:8998 runpod2`: add `--baseline-url wss://localhost:8998 --insecure` | about $0.12 | leg3 `pipeline_lag_ms.p50` within about 150 ms of the DEP1 baseline | large leg-2 RTT: pick a RunPod DC near the Space region (ENDPOINT_SETTINGS 3) |
| 6 | **4-minute hold, matrix** (DESIGN test #3, R2/R16) | the three runs below | about $0.15 each | each run `RESULT PASS`, with the end reason `context_full` (at about 221 s) or `time_limit` | see the table below |
| 7 | queue-mode fallback (only if step 6 fails in every variant) | create the queue endpoint (`create_endpoint.sh --mode queue`), set the Space to `S2S_MODE=queue` + the new id/audience, then repeat steps 3 and 6 | as above | as above | queue also fails → active workers = 1 for demos |
| 8 | **clean-up** | RunPod: active workers 0 (and max workers 0 if you are done). Space: `S2S_SELFTEST=0`, keepalive vars back to the defaults | — | nothing billed at idle | — |

## Step 6 in detail: the hold matrix

The context ends at about 221 s (DEP1 A1.4), so a "4-minute" hold normally ends **cleanly** with `context_full`
before 4 min. That is a PASS. A DROP is a socket closed without `session_end`, or reason `worker_lost`,
`worker_shutdown` or `error`.

**Preferred: direct mode.** The script itself claims the worker, opens the strict-pinned websocket with the RunPod key
and runs the keepalive. You switch variants with flags, without touching the Space:

| run | command | question it answers |
|---|---|---|
| A | `$P ws_hold_test.py --direct lb --keepalive-s 0` | does RunPod count the open websocket as work? PASS = no keepalive needed (R2) |
| B | `$P ws_hold_test.py --direct lb` (GET `/status` every 20 s, strict) | the design default |
| C | `$P ws_hold_test.py --direct lb --keepalive-path /ping --keepalive-s 30` | the [EX-LBWS] README advice; does health-path traffic count as work? |

**Through the Space** (what users get): `$P ws_hold_test.py --space $SPACE`. Here the *Space* does the keepalive
from its variables. For runs A and C, set Space variables `KEEPALIVE_S=0`, or `KEEPALIVE_PATH=/ping` +
`KEEPALIVE_S=30`. The Space restarts when you change them. Restore the defaults afterwards.

How to read the results:
- `keepalive_summary.codes` all `200`: pinned requests are served next to the open websocket.
- Mostly `timeout`: strict requests queue behind the websocket (R16). If A also fails, the only fix is active
  workers = 1 during demos.
- `recv_gap_ms.max` > 1000: a stall. Compare `step_p95_max` (GPU) with the leg-2 RTT (network).
- A DROP at about 330 s means the LB processing cap applies to the websocket. The worker cap of 300 s should prevent
  that, so investigate.

## Output files and what to send back

Collect `results/*.json` and the RunPod endpoint **Logs** of each worker. Stdout has one `SESSION_SUMMARY {json}` line
per session, with `sid`, `worker_id`, `gpu`, `end_reason` and step stats (Amendment W1.9). Do not paste keys or tokens. The scripts never print them, but
check shell history and screenshots.

## Files

| file | DESIGN 9 test | notes |
|---|---|---|
| `space_ws_echo.py` | #1 | needs only aiohttp |
| `cold_start_timer.py` | #2 | phases from the Space states (`waking:<detail>`, `busy`, `ready`), then ws open, prompt phase and first audio |
| `ws_hold_test.py` | #3 | DESIGN calls it `hold_test.py` (Amendment C1) |
| `latency_probe.py` | #4, #5 | `pipeline_lag_ms` = time from sending user frame k to the arrival of the k-th 80 ms of model audio. It is transport + relay + GPU + Opus buffering, **not** the agent's reply time. The Opus encode/decode alone accounts for about 240 ms: measured against `stub_worker.py` echo on localhost, 2026-10-04. Compare against `--baseline-url`, not against 0 |
| `rt_common.py` | — | shared targets (Space / direct LB / direct queue), audio and dry-run gate |
| `fake_runpod.py`, `stub_worker.py`, `test_fake_runpod.py` | step 0 | local stand-ins; no network |
| `test_token.py`, `test_mock_e2e.py` | step 0 | token vectors; CPU end to end with the real worker (`--mock`) and Space code |
