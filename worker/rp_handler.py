"""S2S serverless worker: queue-mode handler (DESIGN.md 2.2 and 3.3), the [EX-QWS] worker-websocket pattern.

  /opt/venv-rp/bin/python worker/rp_handler.py          (entrypoint.sh, S2S_MODE=queue; stack.sh runs alongside)

One job = one call. Job input {"sid", "record_id", "pairing", "seed"} (no token: a minted token would expire during a
cold load, and /run already needs the RunPod key). Steps:
  (a) wait for the local stack (GET 127.0.0.1:$PORT/ping == 200), progress {"state":"loading"} every 10 s; the wait is
      capped so that load + claim TTL + call + 30 s fits the job execution timeout (REVIEW-worker-3)
  (b) claim the sid in process: POST 127.0.0.1:$S2S_INTERNAL_PORT/internal/claim (internal listener, no token)
  (c) progress {"state":"ready","public_ip":RUNPOD_PUBLIC_IP,"tcp_port":RUNPOD_TCP_PORT_<PORT>,"worker_id","sid"}
  (d) block until the call ends, the claim expires unused (CLAIM_TTL_S) or CALL_MAX_S + CLAIM_TTL_S + 30 s pass
  (e) return {"summary": <DEP1 session summary or null>, "end_reason": str}
Failures that leave this worker unusable (load_failed, worker_lost) return "refresh_worker": true, so the runpod SDK
stops the worker after the job and RunPod starts a fresh one (REVIEW-worker-2). The relay then opens ws://public_ip:tcp_port/api/chat with a fresh X-S2S-Token (DESIGN 2.2 step 4).
Stdlib only (sync urllib, no asyncio). `runpod` is imported lazily, so tests can call handler() without the SDK:
progress goes through PROGRESS_HOOK if a test sets it, else runpod.serverless.progress_update, else the log.
"""
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

log = logging.getLogger("s2s.rp_handler")
PROGRESS_HOOK = None          # tests: callable(job, payload)
LOADING_EVERY_S = 10.0
POLL_S = 1.0


def _env_int(name, default):
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return int(default)


def public_port():
    """(container port, public port). PORT if set; else the one RUNPOD_TCP_PORT_<p> variable."""
    port = os.environ.get("PORT")
    if port:
        return int(port), os.environ.get(f"RUNPOD_TCP_PORT_{port}")
    keys = [k for k in os.environ if k.startswith("RUNPOD_TCP_PORT_")]
    if len(keys) == 1:
        return int(keys[0].rsplit("_", 1)[1]), os.environ[keys[0]]
    return 8765, os.environ.get("RUNPOD_TCP_PORT_8765")


def _http(method, url, body=None, timeout=5.0):
    """-> (status, parsed JSON | None); status 0 = unreachable."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            st = r.status
    except urllib.error.HTTPError as e:
        raw, st = e.read(), e.code
    except Exception:  # noqa: BLE001
        return 0, None
    try:
        return st, json.loads(raw) if raw else None
    except ValueError:
        return st, None


def progress(job, payload):
    log.info("progress %s", json.dumps(payload))
    if PROGRESS_HOOK is not None:
        PROGRESS_HOOK(job, payload)
        return
    try:
        import runpod
    except ImportError:
        return
    runpod.serverless.progress_update(job, payload)


def handler(job):
    inp = job.get("input") or {}
    sid = inp.get("sid")
    if not isinstance(sid, str) or not (8 <= len(sid) <= 64):
        return {"error": "input.sid (str) is required", "end_reason": "bad_input", "summary": None}
    port, tcp_port = public_port()
    base = f"http://127.0.0.1:{port}"
    ibase = f"http://127.0.0.1:{paths.INTERNAL_PORT}"
    worker_id = os.environ.get("RUNPOD_POD_ID") or os.uname().nodename
    claim_ttl = _env_int("CLAIM_TTL_S", 90)
    call_max = _env_int("CALL_MAX_S", 300)
    # REVIEW-worker-3: the job's execution timer runs from pickup, i.e. through the model load. Keep the load wait
    # inside S2S_QUEUE_EXEC_TIMEOUT_S (must equal the job policy executionTimeout = Space QUEUE_EXEC_TIMEOUT_MS / 1000)
    # minus what the call itself may need; S2S_QUEUE_LOAD_TIMEOUT_S overrides.
    exec_s = _env_int("S2S_QUEUE_EXEC_TIMEOUT_S", 900)   # CRIT-4: = endpoint/Space default 900 s
    load_timeout = _env_int("S2S_QUEUE_LOAD_TIMEOUT_S", max(60, exec_s - claim_ttl - call_max - 30))
    base_p = {"worker_id": worker_id, "sid": sid, "public_ip": None, "tcp_port": None}

    # (a) wait for the stack
    t0 = time.time()
    deadline = t0 + load_timeout
    last = 0.0
    while True:
        st, body = _http("GET", base + "/ping", timeout=5)
        if st == 200:
            break
        if st == 500:
            progress(job, dict(base_p, state="ended"))
            return {"error": f"worker load failed: {(body or {}).get('error')}", "end_reason": "load_failed",
                    "summary": None, "refresh_worker": True}
        if time.time() > deadline:
            progress(job, dict(base_p, state="ended"))
            # no refresh: the load is still running, and the next job on this worker can use it
            return {"error": f"worker not ready within {load_timeout} s (S2S_QUEUE_LOAD_TIMEOUT_S)",
                    "end_reason": "load_timeout", "summary": None}
        if time.time() - last >= LOADING_EVERY_S:
            progress(job, dict(base_p, state="loading", elapsed_s=round(time.time() - t0, 1)))
            last = time.time()
        time.sleep(POLL_S)
    log.info("stack ready after %.1f s", time.time() - t0)

    # (b) claim in process. A previous call may still be finishing, or a cancelled job may have left an unused claim
    # that only expires after CLAIM_TTL_S (REVIEW-worker-1): retry 409 for CLAIM_TTL_S + 5 s.
    body = {"sid": sid, "record_id": inp.get("record_id"), "pairing": inp.get("pairing"), "seed": inp.get("seed")}
    t_claim = time.time()
    while True:
        st, res = _http("POST", ibase + "/internal/claim", body)
        if st == 200:
            break
        if st != 409 or time.time() - t_claim > claim_ttl + 5:
            progress(job, dict(base_p, state="ended"))
            return {"error": f"claim failed: {st} {res}", "end_reason": "claim_failed", "summary": None}
        time.sleep(1.0)

    # (c) publish where to connect (REVIEW-worker-4: both the public IP and the TCP port are required)
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP")
    if not tcp_port or not public_ip:
        _http("POST", ibase + "/internal/claim_release", {"sid": sid})
        progress(job, dict(base_p, state="ended"))
        miss = " and ".join(n for n, v in ((f"RUNPOD_TCP_PORT_{port}", tcp_port), ("RUNPOD_PUBLIC_IP", public_ip)) if not v)
        return {"error": f"{miss} not set: enable 'Expose TCP Ports' ({port}) on the endpoint",
                "end_reason": "no_tcp_port", "summary": None}
    ready = dict(base_p, state="ready", public_ip=public_ip, tcp_port=int(tcp_port))
    progress(job, ready)

    # (d) block until the call is over
    deadline = time.time() + claim_ttl + call_max + 30
    o = {"phase": "unknown", "end_reason": None, "summary": None}
    while time.time() < deadline:
        st, res = _http("GET", f"{ibase}/internal/claim/{sid}")
        if st == 200 and res and res.get("phase") not in ("claimed", "busy"):
            o = res
            break
        if st == 0:
            o = {"phase": "worker_gone", "end_reason": "worker_lost", "summary": None}
            break
        time.sleep(POLL_S)
    else:
        o = {"phase": "handler_timeout", "end_reason": "handler_timeout", "summary": None}
    end_reason = o.get("end_reason") or o.get("phase")
    progress(job, dict(ready, state="ended", end_reason=end_reason))
    log.info("job for sid %s done: %s", sid, end_reason)
    out = {"summary": o.get("summary"), "end_reason": end_reason}
    if end_reason == "worker_lost":       # the local stack is gone: let RunPod replace this worker
        out["refresh_worker"] = True
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import runpod
    runpod.serverless.start({"handler": handler})
