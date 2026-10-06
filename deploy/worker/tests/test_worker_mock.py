"""S2S serverless worker: CPU tests of worker_server.py (MockEngine), rp_handler.py and resolve_models.py. No GPU.

  PYTHONDONTWRITEBYTECODE=1 <venv-pp>/bin/python worker/tests/test_worker_mock.py [--only NAME,...] [--port-base 18200]

Needs a python with aiohttp + sphn + numpy + sentencepiece (venv-pp) and the mock replay assets (S2S_MOCK_REPLAY,
S2S_MOCK_INPUTS, S2S_PP_DIR or S2S_MOCK_SPM; worker/local/runpod2.env has the runpod2 values and is read for any
var not set). Every worker is started with CUDA_VISIBLE_DEVICES="" on ports port-base.. (default 18200; DEP1 and the
other builders' tests use 899x / 180xx). Prints one line per check and exits non-zero on any failure.

Covers (DESIGN 3.1, 2.2, 6.3, 8): /ping 204 -> 200 and 500 on a failed load or missing secret; claim 200 / idempotent
re-claim / 409 / 401 bad_token, expired, mode, aud; websocket refusals (no token 401, cfg_mismatch 401, busy 409,
no_claim 409, free prompt 400, replayed 401); a full mock call (handshake, audio, text, events, session_end); /status
fast and correct mid-call; claim TTL expiry; release; CALL_MAX_S -> time_limit; SIGTERM -> worker_shutdown and exit 0;
queue mode via rp_handler.handler (progress loading -> ready -> ended, summary returned); resolve_models against a fake
/runpod-volume (cached-model layout, refs/main, voices.tgz extraction, network-volume fallback, not-found exit 2);
the path-grep rule on worker/ code.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import aiohttp
import numpy as np
import sphn

W = Path(__file__).resolve().parent.parent
REPO = W.parent
sys.path.insert(0, str(REPO / "common"))
import s2s_token  # noqa: E402

SECRET = "test-secret-worker-0123456789-abcdefghijklmnop"
AUD = "ep-worker-test"
SR, FS = 24000, 1920
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return cond


def local_env():
    env = dict(os.environ)
    f = Path(os.environ.get("S2S_LOCAL_ENV") or (W / "local" / "runpod2.env"))   # pod1: worker/local/pod1.env
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip() or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    return env


class Worker:
    def __init__(self, port, extra_env=None, args=(), name="w"):
        self.port, self.iport = port, port + 1
        self.name = name
        env = local_env()
        env.update({"CUDA_VISIBLE_DEVICES": "", "S2S_SESSION_SECRET": SECRET, "S2S_AUDIENCE": AUD, "S2S_MODE": "lb",
                    "ASR_BACKEND": "off", "ROUTER": "off", "PYTHONDONTWRITEBYTECODE": "1",
                    "S2S_LOGS": tempfile.mkdtemp(prefix="s2s-wtest-logs-"), "PORT": str(port),
                    "S2S_INTERNAL_PORT": str(port + 1), "S2S_ASR_PORT": str(port + 2), "S2S_ROUTER_PORT": str(port + 3)})
        env.pop("RUNPOD_POD_ID", None)
        env.update(extra_env or {})
        self.env = env
        py = env.get("S2S_PY_PP", sys.executable)
        cmd = [py, "-u", str(W / "server" / "worker_server.py"), "--mock", "--port", str(port),
               "--internal-port", str(port + 1), "--mock-prompt-s", "0.5", *args]
        self.logf = open(Path(env["S2S_LOGS"]) / f"{name}.log", "w")
        self.p = subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{port}"
        self.iurl = f"http://127.0.0.1:{port + 1}"

    def log_text(self):
        self.logf.flush()
        return (Path(self.env["S2S_LOGS"]) / f"{self.name}.log").read_text()

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.logf.close()


def tok(sid, mode="lb", aud=AUD, record_id="food_07", pairing="g1", seed=1001, max_s=300, ttl_s=120, now=None,
        secret=SECRET):
    return s2s_token.mint(secret, sid=sid, mode=mode, aud=aud, record_id=record_id, pairing=pairing, seed=seed,
                          max_s=max_s, ttl_s=ttl_s, now=now)


async def http(cs, method, url, token=None, json_body=None, timeout=5):
    h = {"X-S2S-Token": token} if token else {}
    t0 = time.perf_counter()
    async with cs.request(method, url, headers=h, json=json_body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        txt = await r.text()
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(txt) if txt else None
        except ValueError:
            body = txt
        return r.status, body, ms


async def wait_ping(cs, w, want=200, timeout=30):
    t0 = time.time()
    seen = []
    while time.time() - t0 < timeout:
        try:
            st, _, _ = await http(cs, "GET", w.url + "/ping", timeout=2)
        except Exception:  # noqa: BLE001
            st = 0
        if not seen or seen[-1] != st:
            seen.append(st)
        if st == want:
            return seen
        await asyncio.sleep(0.2)
    return seen


async def ws_refusal(cs, url, token=None, query=None):
    """-> HTTP status of a refused upgrade (or 101 if it was accepted, closed at once)."""
    h = {"X-S2S-Token": token} if token else {}
    q = {"record_id": "food_07", "pairing": "g1", "seed": "1001", "events": "1"}
    q.update(query or {})
    q = {k: v for k, v in q.items() if v is not None}
    try:
        ws = await cs.ws_connect(url + "/api/chat", params=q, headers=h, timeout=10)
    except aiohttp.WSServerHandshakeError as e:
        return e.status
    await ws.close()
    return 101


async def run_call(cs, url, token, seconds=3.0, query=None, on_started=None, stop_after_end=True):
    """Stream silence for `seconds` (80 ms pacing); returns dict of what came back."""
    q = {"record_id": "food_07", "pairing": "g1", "seed": "1001", "events": "1"}
    q.update(query or {})
    q = {k: v for k, v in q.items() if v is not None}
    out = {"handshake": False, "n_audio": 0, "n_text": 0, "events": [], "session_end": None, "close_code": None}
    writer = sphn.OpusStreamWriter(SR)
    ws = await cs.ws_connect(url + "/api/chat", params=q, headers={"X-S2S-Token": token}, timeout=10, max_msg_size=0)
    hs = asyncio.get_running_loop().create_future()

    async def recv():
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            d = msg.data
            if d[0] == 0 and not hs.done():
                out["handshake"] = True
                hs.set_result(True)
            elif d[0] == 1:
                out["n_audio"] += 1
            elif d[0] == 2:
                out["n_text"] += 1
            elif d[0] == 7:
                ev = json.loads(d[1:])
                out["events"].append(ev.get("type"))
                if ev.get("type") == "session_end":
                    out["session_end"] = ev
                    if not hs.done():
                        hs.set_result(False)
        out["close_code"] = ws.close_code

    rt = asyncio.create_task(recv())
    await asyncio.wait_for(hs, timeout=20)
    if on_started:
        await on_started()
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < seconds and out["session_end"] is None and not ws.closed:
        due = t0 + i * FS / SR
        now = time.perf_counter()
        if due > now:
            await asyncio.sleep(due - now)
        writer.append_pcm(np.zeros(FS, np.float32))
        b = writer.read_bytes()
        if len(b):
            try:
                await ws.send_bytes(b"\x01" + b)
            except Exception:  # noqa: BLE001
                break
        i += 1
    if out["session_end"] is None and stop_after_end:
        await ws.close()
    try:
        await asyncio.wait_for(rt, timeout=8)
    except asyncio.TimeoutError:
        pass
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return out


# --------------------------------------------------------------------------------------------------- tests
async def t_lb(pb):
    w = Worker(pb, extra_env={"CLAIM_TTL_S": "3", "CALL_MAX_S": "8"}, args=["--mock-load-s", "2.5"], name="lb")
    try:
        async with aiohttp.ClientSession() as cs:
            seen = await wait_ping(cs, w, 200, timeout=30)
            check("lb: /ping 204 while loading, then 200", seen[-1] == 200 and 204 in seen, f"sequence {seen}")
            st, body, _ = await http(cs, "GET", w.url + "/status")
            check("lb: /status idle with engine/asr/router", st == 200 and body["state"] == "idle" and body["engine"],
                  json.dumps({k: body.get(k) for k in ("state", "engine", "asr", "router", "worker_id", "gpu")}))
            check("lb: /status has no secret", SECRET not in json.dumps(body))
            # ---- token refusals on claim
            sid_a = s2s_token.new_sid()
            for nm, t, code in (("missing", None, "bad_token"), ("garbage", "v1.xx.yy", "bad_token"),
                                ("expired", tok(sid_a, now=time.time() - 400), "expired"),
                                ("wrong mode", tok(sid_a, mode="queue"), "bad_token"),
                                ("wrong aud", tok(sid_a, aud="other-ep"), "bad_token"),
                                ("wrong secret", tok(sid_a, secret="x" * 40), "bad_token")):
                st, body, _ = await http(cs, "POST", w.url + "/session/claim", t)
                check(f"lb: claim {nm} -> 401 {code}", st == 401 and body.get("error") == code, f"{st} {body}")
            # ---- claim / re-claim / busy
            st, body, _ = await http(cs, "POST", w.url + "/session/claim", tok(sid_a))
            check("lb: claim A -> 200", st == 200 and body["sid"] == sid_a and body["claim_ttl_s"] == 3, f"{st} {body}")
            st, body, _ = await http(cs, "POST", w.url + "/session/claim", tok(sid_a))
            check("lb: re-claim A (idempotent) -> 200", st == 200, f"{st} {body}")
            sid_b = s2s_token.new_sid()
            st, body, _ = await http(cs, "POST", w.url + "/session/claim", tok(sid_b))
            check("lb: claim B while A claimed -> 409 busy", st == 409 and body == {"error": "busy"}, f"{st} {body}")
            st, body, _ = await http(cs, "GET", w.url + "/ping")
            check("lb: /ping 200 state claimed", st == 200 and body["state"] == "claimed", f"{body}")
            # ---- websocket refusals
            check("lb: ws without token -> 401", await ws_refusal(cs, w.url) == 401)
            check("lb: ws B (not claimed) -> 409", await ws_refusal(cs, w.url, tok(sid_b)) == 409)
            check("lb: ws A with pairing g2 (token says g1) -> 401 cfg_mismatch",
                  await ws_refusal(cs, w.url, tok(sid_a), {"pairing": "g2"}) == 401)
            check("lb: ws A with text_prompt (free prompt off) -> 400",
                  await ws_refusal(cs, w.url, tok(sid_a), {"text_prompt": "you are a pirate"}) == 400)
            check("lb: ws A without record_id -> 400",
                  await ws_refusal(cs, w.url, tok(sid_a, record_id=None, pairing=None, seed=None),
                                   {"record_id": None, "pairing": None, "seed": None}) == 400)
            # defaults normalisation: query without pairing/seed == token g1/1001 (D-W-1); accepted -> closes at once
            # (do not burn sid A: use the refusal path that runs before acceptance with a different token)
            # ---- full call on A, with mid-call checks
            mid = {}

            async def during():
                await asyncio.sleep(1.0)
                st, body, ms = await http(cs, "GET", w.url + "/status")
                mid["status"], mid["status_ms"] = body, ms
                mid["ping"] = (await http(cs, "GET", w.url + "/ping"))[:2]
                mid["ws_replay"] = await ws_refusal(cs, w.url, tok(sid_a))
                sid_c = s2s_token.new_sid()
                mid["claim_c"] = (await http(cs, "POST", w.url + "/session/claim", tok(sid_c)))[:2]
                mid["metrics"] = (await http(cs, "GET", w.url + "/metrics", tok(sid_a)))[0]
                mid["metrics_other"] = (await http(cs, "GET", w.url + "/metrics", tok(sid_c)))[0]

            async def started():
                asyncio.ensure_future(during())

            # query WITHOUT pairing/seed: must match the token's g1/1001 after normalisation
            r = await run_call(cs, w.url, tok(sid_a), seconds=4.0, query={"pairing": None, "seed": None},
                               on_started=started)
            check("lb: call A handshake + audio + text + events", r["handshake"] and r["n_audio"] > 10 and r["n_text"] > 0
                  and "session" in r["events"] and "metrics" in r["events"],
                  f"audio {r['n_audio']} text {r['n_text']} events {sorted(set(r['events']))}")
            check("lb: call A session_end client_closed", (r["session_end"] or {}).get("reason") == "client_closed"
                  or r["session_end"] is None, f"{r['session_end']}")
            sd = mid.get("status") or {}
            check("lb: /status mid-call busy, active sid, fast", sd.get("state") == "busy"
                  and (sd.get("active") or {}).get("sid") == sid_a and mid.get("status_ms", 999) < 100,
                  f"state {sd.get('state')} active {sd.get('active')} {mid.get('status_ms', 0):.1f} ms")
            check("lb: /ping mid-call 200 state busy", mid.get("ping", (0,))[0] == 200
                  and mid["ping"][1].get("state") == "busy", f"{mid.get('ping')}")
            check("lb: second ws with the same token mid-call -> 401 replayed", mid.get("ws_replay") == 401,
                  f"{mid.get('ws_replay')}")
            check("lb: claim C mid-call -> 409", mid.get("claim_c", (0,))[0] == 409, f"{mid.get('claim_c')}")
            check("lb: /metrics with active token 200, other sid 403",
                  mid.get("metrics") == 200 and mid.get("metrics_other") == 403, f"{mid.get('metrics')} {mid.get('metrics_other')}")
            await asyncio.sleep(0.5)
            st, body, _ = await http(cs, "GET", w.url + "/status")
            check("lb: idle again after the call, sessions_served 1", body["state"] == "idle" and body["sessions_served"] == 1,
                  f"{body['state']} {body['sessions_served']} p95 {body.get('step_ms_p95_last')}")
            st, body, _ = await http(cs, "GET", f"{w.iurl}/internal/claim/{sid_a}")
            check("lb: /internal/claim/A ended with summary", body.get("phase") == "ended"
                  and body.get("end_reason") == "client_closed" and (body.get("summary") or {}).get("frames", 0) > 10,
                  f"{body.get('phase')} {body.get('end_reason')} frames {(body.get('summary') or {}).get('frames')}")
            check("lb: ws A after the call -> 401 replayed", await ws_refusal(cs, w.url, tok(sid_a)) == 401)
            # ---- claim TTL expiry
            sid_d = s2s_token.new_sid()
            st, _, _ = await http(cs, "POST", w.url + "/session/claim", tok(sid_d))
            await asyncio.sleep(4.5)
            st2, body, _ = await http(cs, "GET", w.url + "/status")
            st3, body3, _ = await http(cs, "GET", f"{w.iurl}/internal/claim/{sid_d}")
            check("lb: unused claim expires after CLAIM_TTL_S -> idle", st == 200 and body["state"] == "idle"
                  and body3.get("phase") == "expired", f"{body['state']} {body3.get('phase')}")
            # ---- release
            sid_e = s2s_token.new_sid()
            await http(cs, "POST", w.url + "/session/claim", tok(sid_e))
            st, body, _ = await http(cs, "POST", w.url + "/session/release", tok(sid_e))
            st2, body2, _ = await http(cs, "GET", w.url + "/status")
            check("lb: release of a claim -> released, idle", st == 200 and body == {"released": True}
                  and body2["state"] == "idle", f"{body} {body2['state']}")
            st, body, _ = await http(cs, "POST", w.url + "/session/release", tok(s2s_token.new_sid()))
            check("lb: release of an unknown sid -> released false", st == 200 and body == {"released": False})
            # ---- time limit (CALL_MAX_S=8 on this worker; the token asks for 300 -> min = 8)
            sid_f = s2s_token.new_sid()
            await http(cs, "POST", w.url + "/session/claim", tok(sid_f))
            r = await run_call(cs, w.url, tok(sid_f), seconds=20.0)
            check("lb: CALL_MAX_S -> session_end time_limit", (r["session_end"] or {}).get("reason") == "time_limit"
                  and r["elapsed_s"] < 12, f"{r['session_end']} after {r['elapsed_s']} s of audio")
            # ---- token max_s smaller than CALL_MAX_S wins
            sid_g = s2s_token.new_sid()
            await http(cs, "POST", w.url + "/session/claim", tok(sid_g, max_s=3))
            r = await run_call(cs, w.url, tok(sid_g, max_s=3), seconds=20.0)
            check("lb: token max_s=3 -> time_limit at ~3 s", (r["session_end"] or {}).get("reason") == "time_limit"
                  and r["elapsed_s"] < 5, f"{r['session_end']} after {r['elapsed_s']} s")
            # ---- release during the call ends it
            sid_h = s2s_token.new_sid()
            await http(cs, "POST", w.url + "/session/claim", tok(sid_h))

            async def rel():
                await asyncio.sleep(1.0)
                await http(cs, "POST", w.url + "/session/release", tok(sid_h))

            r = await run_call(cs, w.url, tok(sid_h), seconds=10.0, on_started=lambda: _spawn(rel()))
            check("lb: release mid-call -> session_end client_closed", (r["session_end"] or {}).get("reason")
                  == "client_closed" and r["elapsed_s"] < 4, f"{r['session_end']} {r['elapsed_s']} s")
        log = w.log_text()
        check("lb: token never logged", SECRET not in log and "v1." not in log)
    finally:
        w.stop()


async def _spawn(coro):
    asyncio.ensure_future(coro)


async def t_sigterm(pb):
    w = Worker(pb, name="sigterm")
    try:
        async with aiohttp.ClientSession() as cs:
            await wait_ping(cs, w, 200)
            sid = s2s_token.new_sid()
            await http(cs, "POST", w.url + "/session/claim", tok(sid))

            async def kill():
                await asyncio.sleep(1.0)
                w.p.send_signal(signal.SIGTERM)

            r = await run_call(cs, w.url, tok(sid), seconds=10.0, on_started=lambda: _spawn(kill()))
            try:
                rc = w.p.wait(6)
            except subprocess.TimeoutExpired:
                rc = None
            check("sigterm: session_end worker_shutdown", (r["session_end"] or {}).get("reason") == "worker_shutdown",
                  f"{r['session_end']}")
            check("sigterm: worker exits 0 within 6 s", rc == 0, f"rc {rc}")
    finally:
        w.stop()


async def t_failures(pb):
    async with aiohttp.ClientSession() as cs:
        w = Worker(pb, args=["--mock-fail", "simulated load failure"], name="fail")
        try:
            seen = await wait_ping(cs, w, 500, timeout=15)
            st, body, _ = await http(cs, "GET", w.url + "/status")
            check("fail: failed load -> /ping 500, /status failed", seen[-1] == 500 and body["state"] == "failed"
                  and "simulated" in body["detail"], f"{seen} {body['detail']}")
            st, body, _ = await http(cs, "POST", w.url + "/session/claim", tok(s2s_token.new_sid()))
            check("fail: claim on a failed worker -> 503", st == 503, f"{st} {body}")
        finally:
            w.stop()
        w = Worker(pb + 10, extra_env={"S2S_SESSION_SECRET": ""}, name="nosecret")
        try:
            seen = await wait_ping(cs, w, 500, timeout=15)
            st, body, _ = await http(cs, "GET", w.url + "/status")
            check("fail: missing S2S_SESSION_SECRET -> /ping 500", seen[-1] == 500
                  and "S2S_SESSION_SECRET" in body["detail"], f"{seen} {body['detail']}")
        finally:
            w.stop()


async def t_queue(pb):
    w = Worker(pb, extra_env={"S2S_MODE": "queue", "CLAIM_TTL_S": "20"}, args=["--mock-load-s", "3"], name="queue")
    os.environ.update({"PORT": str(pb), "S2S_INTERNAL_PORT": str(pb + 1), "RUNPOD_PUBLIC_IP": "127.0.0.1",
                       f"RUNPOD_TCP_PORT_{pb}": str(pb), "RUNPOD_POD_ID": "queue-test-worker", "CLAIM_TTL_S": "20"})
    sys.path.insert(0, str(W))
    import importlib
    import paths  # noqa: F401
    importlib.reload(paths)
    import rp_handler
    importlib.reload(rp_handler)
    prog = []
    rp_handler.PROGRESS_HOOK = lambda job, p: prog.append(dict(p))
    rp_handler.LOADING_EVERY_S = 0.5
    sid = s2s_token.new_sid()
    job = {"id": "job-1", "input": {"sid": sid, "record_id": "food_07", "pairing": "g1", "seed": 1001}}
    res = {}
    th = threading.Thread(target=lambda: res.update(out=rp_handler.handler(job)), daemon=True)
    th.start()
    try:
        async with aiohttp.ClientSession() as cs:
            t0 = time.time()
            while time.time() - t0 < 30 and not any(p.get("state") == "ready" for p in prog):
                await asyncio.sleep(0.2)
            ready = next((p for p in prog if p.get("state") == "ready"), None)
            check("queue: progress loading then ready{public_ip,tcp_port,worker_id,sid}",
                  prog and prog[0].get("state") == "loading" and ready and ready["public_ip"] == "127.0.0.1"
                  and ready["tcp_port"] == pb and ready["worker_id"] == "queue-test-worker" and ready["sid"] == sid,
                  f"{[p.get('state') for p in prog][:6]} ready={ready}")
            check("queue: lb-mode token refused on a queue worker", await ws_refusal(
                cs, f"http://127.0.0.1:{pb}", tok(sid, mode="lb")) == 401)
            url = f"http://{ready['public_ip']}:{ready['tcp_port']}"
            call_t = asyncio.create_task(run_call(cs, url, tok(sid, mode="queue"), seconds=3.0))
            pub = intl = None
            t0 = time.time()
            while time.time() - t0 < 20 and not call_t.done():
                _, pub, _ = await http(cs, "GET", url + "/status")
                if pub.get("state") == "busy" and (pub.get("active") or {}).get("conv_s"):
                    _, intl, _ = await http(cs, "GET", f"http://127.0.0.1:{pb + 1}/internal/status")
                    break
                await asyncio.sleep(0.2)
            check("queue: public /status mid-call hides active.sid, /internal/status keeps it (REVIEW-worker-5)",
                  pub and (pub.get("active") or {}).get("sid", "x") is None and intl
                  and (intl.get("active") or {}).get("sid") == sid, f"pub {pub and pub.get('active')} int {intl and intl.get('active')}")
            r = await call_t
            check("queue: call through the published ip:port", r["handshake"] and r["n_audio"] > 10, f"audio {r['n_audio']}")
            th.join(15)
            out = res.get("out") or {}
            check("queue: handler returns summary + end_reason", out.get("end_reason") == "client_closed"
                  and (out.get("summary") or {}).get("frames", 0) > 10 and prog[-1].get("state") == "ended",
                  f"{out.get('end_reason')} frames {(out.get('summary') or {}).get('frames')} last {prog[-1].get('state')}")
            out2 = rp_handler.handler({"id": "job-2", "input": {}})
            check("queue: handler rejects a job without sid", out2.get("end_reason") == "bad_input")
            # REVIEW-worker-4: no RUNPOD_PUBLIC_IP -> no 'ready', claim given back
            sid3 = s2s_token.new_sid()
            os.environ.pop("RUNPOD_PUBLIC_IP", None)
            try:
                out3 = await asyncio.to_thread(rp_handler.handler, {"id": "job-3", "input": {
                    "sid": sid3, "record_id": "food_07", "pairing": "g1", "seed": 1001}})
            finally:
                os.environ["RUNPOD_PUBLIC_IP"] = "127.0.0.1"
            _, st3, _ = await http(cs, "GET", url + "/status")
            check("queue: no RUNPOD_PUBLIC_IP -> no_tcp_port, claim released (REVIEW-worker-4)",
                  out3.get("end_reason") == "no_tcp_port" and "RUNPOD_PUBLIC_IP" in out3.get("error", "")
                  and st3.get("state") == "idle", f"{out3} {st3.get('state')}")
            # REVIEW-worker-1: an unused claim (cancelled job) blocks the next job only until CLAIM_TTL_S
            sid4, sid5 = s2s_token.new_sid(), s2s_token.new_sid()
            await http(cs, "POST", f"http://127.0.0.1:{pb + 1}/internal/claim",
                       json_body={"sid": sid4, "record_id": "food_07", "pairing": "g1", "seed": 1001})
            prog.clear()
            t5 = time.time()
            th5 = threading.Thread(target=lambda: res.update(out5=rp_handler.handler(
                {"id": "job-5", "input": {"sid": sid5, "record_id": "food_07", "pairing": "g1", "seed": 1001}})),
                daemon=True)
            th5.start()
            while time.time() - t5 < 40 and not any(p.get("state") == "ready" for p in prog):
                await asyncio.sleep(0.2)
            ok5 = any(p.get("state") == "ready" and p.get("sid") == sid5 for p in prog)
            check("queue: next job claims after a stale claim expires (> 30 s would have failed before; TTL 20 s here)",
                  ok5, f"{time.time() - t5:.1f} s {[p.get('state') for p in prog]}")
            await http(cs, "POST", f"http://127.0.0.1:{pb + 1}/internal/claim_release", json_body={"sid": sid5})
            th5.join(15)
    finally:
        w.stop()
    # REVIEW-worker-2: a failed load returns refresh_worker so the SDK stops this worker
    from aiohttp import web as aweb
    app = aweb.Application()

    async def ping500(request):
        return aweb.json_response({"status": "error", "error": "simulated"}, status=500)
    app.router.add_get("/ping", ping500)
    runner = aweb.AppRunner(app)
    await runner.setup()
    await aweb.TCPSite(runner, "127.0.0.1", pb + 5).start()
    os.environ["PORT"] = str(pb + 5)
    try:
        out6 = await asyncio.to_thread(rp_handler.handler, {"id": "job-6", "input": {"sid": s2s_token.new_sid()}})
    finally:
        os.environ["PORT"] = str(pb)
        await runner.cleanup()
    check("queue: load_failed -> refresh_worker true (REVIEW-worker-2)",
          out6.get("end_reason") == "load_failed" and out6.get("refresh_worker") is True, str(out6))


def t_resolve(pb):
    snap_src = Path(local_env().get("S2S_PP_DIR", ""))
    tmp = Path(tempfile.mkdtemp(prefix="s2s-resolve-"))
    try:
        def make_snapshot(hub, with_voices_dir=False, ref=True):
            base = hub / "models--nvidia--personaplex-7b-v1"
            snap = base / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            if ref:
                (base / "refs").mkdir()
                (base / "refs" / "main").write_text("abc123")
            for f in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
                if (snap_src / f).exists():
                    (snap / f).symlink_to(snap_src / f)
                else:
                    (snap / f).write_bytes(b"x")
            vt = tmp / "vsrc" / "voices"
            vt.mkdir(parents=True, exist_ok=True)
            for v in ("NATF2.pt", "NATM1.pt"):
                (vt / v).write_bytes(b"voice")
            if with_voices_dir:
                shutil.copytree(vt, snap / "voices")
            else:
                with tarfile.open(snap / "voices.tgz", "w:gz") as tar:
                    tar.add(vt, arcname="voices")
            return snap

        def run(env_extra):
            env = local_env()
            env.pop("S2S_PP_DIR", None)
            env.pop("S2S_VOICES", None)
            env.update(env_extra)
            p = subprocess.run([env.get("S2S_PY_PP", sys.executable), str(W / "resolve_models.py")], env=env,
                               capture_output=True, text=True, timeout=60)
            try:
                return p.returncode, json.loads(p.stdout)
            except ValueError:
                return p.returncode, {"raw": p.stdout + p.stderr}

        vol = tmp / "runpod-volume"
        snap = make_snapshot(vol / "huggingface-cache" / "hub")
        rc, out = run({"S2S_RUNPOD_VOLUME": str(vol), "S2S_TMP": str(tmp / "t1")})
        check("resolve: RunPod cache layout + refs/main + voices.tgz extracted to S2S_TMP",
              rc == 0 and out.get("source") == "runpod_model_cache" and out.get("pp_dir") == str(snap)
              and out.get("voices") == str(tmp / "t1" / "voices") and out.get("voices_source") == "extracted"
              and (tmp / "t1" / "voices" / "NATM1.pt").exists()
              and (tmp / "t1" / "models.env").read_text().startswith(f"S2S_PP_DIR={snap}"), json.dumps(out)[:300])
        vol2 = tmp / "vol2"
        snap2 = make_snapshot(vol2 / "hf" / "hub", with_voices_dir=True, ref=False)
        rc, out = run({"S2S_RUNPOD_VOLUME": str(vol2), "S2S_TMP": str(tmp / "t2")})
        check("resolve: network-volume fallback, no refs/main, voices/ dir used as is",
              rc == 0 and out.get("source") == "network_volume" and out.get("pp_dir") == str(snap2)
              and out.get("voices_source") == "snapshot", json.dumps(out)[:300])
        rc, out = run({"S2S_RUNPOD_VOLUME": str(tmp / "empty"), "S2S_TMP": str(tmp / "t3")})
        check("resolve: nothing found -> exit 2 with reasons", rc == 2 and "PersonaPlex not found" in out.get("error", ""),
              out.get("error", "")[:200])
        if snap_src.exists():
            rc, out = run({"S2S_PP_DIR": str(snap_src), "S2S_TMP": str(tmp / "t4")})
            check("resolve: explicit S2S_PP_DIR (runpod2 snapshot)", rc == 0 and out.get("source") == "S2S_PP_DIR",
                  json.dumps(out)[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_grep(pb):
    bad = []
    pat = re.compile(r"/root/deploy|/root/needle|/workspace|/root/hf")
    for p in W.rglob("*"):
        code = p.suffix in (".py", ".sh") or p.name.startswith("Dockerfile")
        if not p.is_file() or not code or "local" in p.relative_to(W).parts or "build" in p.relative_to(W).parts:
            continue
        try:
            lines = p.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if pat.search(line) and not s.startswith("#") and p != Path(__file__).resolve():
                bad.append(f"{p.relative_to(W)}:{i}: {s[:100]}")
    check("grep: no DEP1 hard-coded paths in worker/ code (worker/local exempt)", not bad, "; ".join(bad[:5]))


TESTS = {"lb": t_lb, "sigterm": t_sigterm, "failures": t_failures, "queue": t_queue, "resolve": t_resolve, "grep": t_grep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--port-base", type=int, default=18200)
    a = ap.parse_args()
    names = [n for n in TESTS if not a.only or n in a.only.split(",")]
    for i, n in enumerate(names):
        fn = TESTS[n]
        pb = a.port_base + 20 * i
        print(f"==== {n} (ports {pb}..)", flush=True)
        t0 = time.time()
        try:
            if asyncio.iscoroutinefunction(fn):
                asyncio.run(fn(pb))
            else:
                fn(pb)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{n}: raised {type(e).__name__}: {e}", False)
        print(f"     {n} took {time.time() - t0:.1f} s", flush=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
