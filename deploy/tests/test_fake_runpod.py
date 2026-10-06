"""S2S serverless: self-test of the test harness (CPU, about 60 s). Checks fake_runpod.py against stub_worker.py and runs the
REAL-test scripts (ws_hold_test / cold_start_timer / latency_probe) end to end in --direct mode against them.

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/test_fake_runpod.py [--base-port 38000]

Uses ports base (worker), base+1 (2nd worker), base+999 / base+998 (internal), base+80 (LB), base+81 (queue) so it does
not collide with test_mock_e2e.py (DESIGN 7 ports 18xxx). Never touches the network beyond 127.0.0.1.
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import s2s_token  # noqa: E402

PY = sys.executable
SECRET = "fake-runpod-selftest-secret-0123456789abcdef"
FAILS = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   [{info}]" if info and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def spawn(args, env=None, log=None):
    e = dict(os.environ, S2S_SESSION_SECRET=SECRET, S2S_AUDIENCE="ep-test", **(env or {}))
    return subprocess.Popen([PY, "-u"] + args, env=e, stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)


async def wait_http(url, timeout=20):
    t_end = time.time() + timeout
    async with aiohttp.ClientSession() as cs:
        while time.time() < t_end:
            try:
                async with cs.get(url) as r:
                    return r.status
            except Exception:
                await asyncio.sleep(0.2)
    return None


def tok(sid, mode="lb", **kw):
    d = dict(record_id="food_01", pairing="g1", seed=1001, max_s=300)
    d.update(kw)
    return s2s_token.mint(SECRET, sid=sid, mode=mode, aud="ep-test", **d)


async def lb_checks(B):
    LB = f"http://127.0.0.1:{B + 80}"
    H = {"Authorization": "Bearer test-key"}
    async with aiohttp.ClientSession() as cs:
        async with cs.get(LB + "/status") as r:
            check("LB: no Bearer -> 401", r.status == 401, r.status)
        t = time.time()
        async with cs.get(LB + "/status", headers=H) as r:
            j = await r.json(content_type=None)
            check("LB: worker loading -> 503 no workers available after hold",
                  r.status == 503 and j.get("error") == "no workers available" and time.time() - t >= 0.9, (r.status, j))
        st = await wait_http(f"http://127.0.0.1:{B}/ping")
        for _ in range(40):
            async with cs.get(LB + "/status", headers=H) as r:
                if r.status == 200:
                    break
            await asyncio.sleep(0.25)
        async with cs.get(LB + "/status", headers=H) as r:
            j = await r.json()
            check("LB: ready -> /status 200 idle with X-Runpod-Worker-Id", r.status == 200 and j["state"] == "idle"
                  and r.headers.get("X-Runpod-Worker-Id") in ("fw-a", "fw-b"), (r.status, dict(r.headers)))
        sid = s2s_token.new_sid()
        async with cs.post(LB + "/session/claim", headers={**H, "X-S2S-Token": tok(sid)}) as r:
            j = await r.json()
            wid = r.headers.get("X-Runpod-Worker-Id")
            check("LB: claim 200 + worker id header", r.status == 200 and j["sid"] == sid and wid, (r.status, j))
        async with cs.post(LB + "/session/claim", headers={**H, "X-S2S-Token": tok(sid, mode="queue")}) as r:
            check("LB: wrong-mode token -> 401", r.status == 401, r.status)
        pin = {**H, "X-Runpod-Worker-Id": f"strict {wid}"}
        q = "?events=1&record_id=food_01&pairing=g1&seed=1001"
        try:
            await cs.ws_connect(f"ws://127.0.0.1:{B + 80}/api/chat?record_id=food_02&pairing=g1&seed=1001",
                                headers={**pin, "X-S2S-Token": tok(sid)})
            check("LB ws: cfg mismatch refused", False)
        except aiohttp.WSServerHandshakeError as e:
            check("LB ws: cfg mismatch -> 401 passed through", e.status == 401, e.status)
        ws = await cs.ws_connect(f"ws://127.0.0.1:{B + 80}/api/chat" + q, headers={**pin, "X-S2S-Token": tok(sid)})
        m = await asyncio.wait_for(ws.receive(), 10)
        check("LB ws: handshake 0x00 through the proxy", m.data == b"\x00", m)
        await ws.send_bytes(b"\x01hello")
        got = None
        for _ in range(5):
            m = await asyncio.wait_for(ws.receive(), 5)
            if m.data[:1] == b"\x01":
                got = m.data
                break
        check("LB ws: binary echo through the proxy", got == b"\x01hello", got)
        try:
            await cs.ws_connect(f"ws://127.0.0.1:{B + 80}/api/chat" + q, headers={**pin, "X-S2S-Token": tok(sid)})
            check("LB ws: replay refused", False)
        except aiohttp.WSServerHandshakeError as e:
            check("LB ws: replayed token -> 401", e.status == 401, e.status)
        sid2 = s2s_token.new_sid()
        async with cs.post(LB + "/session/claim", headers={**pin, "X-S2S-Token": tok(sid2)}) as r:
            check("LB: second claim on the busy worker -> 409", r.status == 409, r.status)
        async with cs.get(LB + "/status", headers=pin) as r:
            j = await r.json()
            check("LB: strict-pinned keepalive /status 200 busy", r.status == 200 and j["state"] == "busy", j)
        async with cs.get(LB + "/ping", headers=pin) as r:
            check("LB: /ping stays 200 while busy", r.status == 200, r.status)
        await ws.close()
        await asyncio.sleep(0.5)
        async with cs.get(LB + "/_fake/state") as r:
            s = await r.json()
            check("LB: /_fake/state counts", s["counts"]["unauthorized"] >= 1 and s["counts"]["ws"] >= 3, s["counts"])
        return wid


async def strict_gone(B, wid, procs):
    LB = f"http://127.0.0.1:{B + 80}"
    H = {"Authorization": "Bearer test-key", "X-Runpod-Worker-Id": f"strict {wid}"}
    p = procs[wid]
    os.killpg(p.pid, signal.SIGKILL)
    await asyncio.sleep(1.5)
    async with aiohttp.ClientSession() as cs:
        async with cs.get(LB + "/status", headers=H) as r:
            j = await r.json()
            check("LB: strict to a dead worker -> 404 affinity_worker_gone",
                  r.status == 404 and j.get("error") == "affinity_worker_gone", (r.status, j))
        async with cs.get(LB + "/status", headers={"Authorization": "Bearer test-key"}) as r:
            check("LB: unpinned request goes to the surviving worker",
                  r.status == 200 and r.headers.get("X-Runpod-Worker-Id") != wid, r.headers.get("X-Runpod-Worker-Id"))


async def queue_checks(B):
    QA = f"http://127.0.0.1:{B + 81}/v2/ep-test"
    H = {"Authorization": "Bearer test-key"}
    async with aiohttp.ClientSession() as cs:
        async with cs.post(QA + "/run", json={"input": {"sid": "q1"}}) as r:
            check("queue: /run without Bearer -> 401", r.status == 401, r.status)
        async with cs.post(QA + "/run", headers=H, json={"input": {"sid": "q1", "record_id": "food_01"}}) as r:
            jid = (await r.json())["id"]
        seen = []
        for _ in range(40):
            async with cs.get(f"{QA}/status/{jid}", headers=H) as r:
                j = await r.json()
            seen.append(j["status"])
            if j["status"] == "COMPLETED":
                break
            await asyncio.sleep(0.1)
        check("queue: progress visible while IN_PROGRESS, then COMPLETED with output",
              "IN_PROGRESS" in seen and j["status"] == "COMPLETED" and j["output"]["echo"]["sid"] == "q1"
              and j["output"]["tcp_port"] == str(B), (seen[-3:], j))
        async with cs.post(QA + "/run", headers=H, json={"input": {"sid": "q2", "sleep": 5}}) as r:
            jid = (await r.json())["id"]
        async with cs.post(f"{QA}/cancel/{jid}", headers=H) as r:
            check("queue: cancel", (await r.json())["status"] == "CANCELLED")


HANDLER = '''
import os, time, runpod
def handler(job):
    inp = job["input"]
    runpod.serverless.progress_update(job, {"state": "loading"})
    time.sleep(inp.get("sleep", 0.5))
    port = os.environ.get("PORT", "0")
    return {"echo": inp, "public_ip": os.environ.get("RUNPOD_PUBLIC_IP"),
            "tcp_port": os.environ.get("RUNPOD_TCP_PORT_" + port)}
'''


def run_script(name, args, B, timeout=120):
    env = dict(os.environ, RUNPOD_API_KEY="test-key", S2S_SESSION_SECRET=SECRET, RUNPOD_ENDPOINT_ID="ep-test",
               S2S_AUDIENCE="ep-test")
    cmd = [PY, os.path.join(HERE, name), "--direct", "lb", "--lb-url", f"http://127.0.0.1:{B + 80}"] + args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_space_script(name, args, port, timeout=150):
    env = dict(os.environ, S2S_PASSCODE="pc-selftest")
    for k in ("RUNPOD_API_KEY", "S2S_SESSION_SECRET"):
        env.pop(k, None)                      # Space runs must not need any secret
    cmd = [PY, os.path.join(HERE, name)] + ([f"http://127.0.0.1:{port}"] if name == "space_ws_echo.py"
                                            else ["--space", f"http://127.0.0.1:{port}"]) + args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


async def space_stage(a, B, procs, logf):
    """The REAL-test scripts in --space mode against the actual Space backend (space/app) + fake_runpod + stubs."""
    sp_py = a.space_python
    space_dir = os.path.join(HERE, "..", "space")
    if not (sp_py and os.path.exists(sp_py) and os.path.exists(os.path.join(space_dir, "app", "main.py"))):
        print("skip space stage (no space/app or --space-python)")
        return
    port = B + 860
    env = {"PORT": str(port), "HOST": "127.0.0.1", "RUNPOD_ENDPOINT_ID": "ep-test", "RUNPOD_API_KEY": "test-key",
           "RUNPOD_LB_URL": f"http://127.0.0.1:{B + 80}", "S2S_MODE": "lb", "MAX_CONCURRENT_CALLS": "2",
           "RATE_PER_IP_PER_HOUR": "50", "KEEPALIVE_S": "2", "S2S_PASSCODE": "pc-selftest", "S2S_SELFTEST": "1",
           "WAKE_POLL_S": "0.5"}
    e = dict(os.environ, S2S_SESSION_SECRET=SECRET, S2S_AUDIENCE="ep-test", **env)
    procs["space"] = subprocess.Popen([sp_py, "-u", "-m", "app.main"], cwd=space_dir, env=e, stdout=logf,
                                      stderr=subprocess.STDOUT, start_new_session=True)
    st = await wait_http(f"http://127.0.0.1:{port}/healthz")
    check("space: up", st == 200, st)
    rc_, out, err = run_space_script("space_ws_echo.py", ["--hold-s", "6", "--outbound", f"127.0.0.1:{B}"], port)
    try:
        j = json.loads(out)
    except Exception:
        j = {}
    check("space_ws_echo: OK + outbound selftest ok", rc_ == 0 and j.get("ws_echo") == "OK"
          and (j.get("outbound") or {}).get("body", {}).get("ok") is True, (rc_, out[-500:], err[-300:]))
    rc_, out, err = run_space_script("cold_start_timer.py", ["--runs", "1", "--talk-s", "2"], port)
    lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    check("cold_start_timer via Space: OK, phases include ready", rc_ == 0 and lines and lines[0]["result"] == "OK"
          and any(p["phase"].startswith("ready") for p in lines[0]["phases"]), (rc_, out[-600:], err[-300:]))
    rc_, out, err = run_space_script("ws_hold_test.py", ["--minutes", "0.2", "--tick-s", "3"], port)
    last = (out.strip().splitlines() or [""])[-1]
    check("ws_hold_test via Space: PASS", rc_ == 0 and "RESULT PASS" in last, (rc_, last, err[-300:]))
    rc_, out, err = run_space_script("latency_probe.py", ["--call", "--talk-s", "13", "--n", "3"], port)
    try:
        j = json.loads(out[out.index("\n{\n") + 1:])
        ok = (j["leg1_browser_to_space"]["ws_echo_rtt_ms"]["n_ok"] == 12 and j["leg3_full_path"]["pipeline_lag_ms"]["n"] > 50
              and any(d.get("space_to_worker_ms") is not None for d in j["leg2_space_to_worker"]["samples"]))
    except Exception as ex:
        ok, j = False, repr(ex)
    check("latency_probe via Space: legs 1-3 measured (diag through the Space)", ok, (str(j)[:600], err[-300:]))


async def amain(a):
    B = a.base_port
    tmp = tempfile.mkdtemp(prefix="fake_runpod_selftest_")
    hp = os.path.join(tmp, "rp_handler.py")
    open(hp, "w").write(HANDLER)
    logf = open(os.path.join(tmp, "procs.log"), "w")
    procs = {}
    try:
        procs["fw-a"] = spawn([os.path.join(HERE, "stub_worker.py"), "--port", str(B), "--internal-port", str(B + 999),
                               "--load-s", "2", "--prompt-s", "0.3", "--worker-id", "stub-a"], log=logf)
        procs["fw-b"] = spawn([os.path.join(HERE, "stub_worker.py"), "--port", str(B + 1), "--internal-port",
                               str(B + 998), "--load-s", "2", "--prompt-s", "0.3", "--worker-id", "stub-b"], log=logf)
        procs["proxy"] = spawn([os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(B + 80), "--queue-port",
                                str(B + 81), "--workers", f"http://127.0.0.1:{B},http://127.0.0.1:{B + 1}",
                                "--worker-ids", "fw-a,fw-b", "--hold-s", "1", "--handler", hp, "--worker-port", str(B)],
                               env={"PORT": str(B)}, log=logf)
        await wait_http(f"http://127.0.0.1:{B + 80}/_fake/state")
        wid = await lb_checks(B)
        await queue_checks(B)
        # REAL-test scripts in --direct lb mode against the fake (short durations)
        rc_, out, err = run_script("ws_hold_test.py", ["--minutes", "0.25", "--keepalive-s", "2", "--tick-s", "3"], B)
        last = (out.strip().splitlines() or [""])[-1]
        check("ws_hold_test direct: PASS with keepalive 200s", rc_ == 0 and "RESULT PASS" in last and "'200'" in last,
              (rc_, last, err[-400:]))
        rc_, out, err = run_script("cold_start_timer.py", ["--runs", "2", "--talk-s", "2"], B)
        lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
        check("cold_start_timer direct: 2 runs OK with phases", rc_ == 0 and len(lines) == 2 and
              all(x["result"] == "OK" and x["prompt_phase_s"] is not None for x in lines), (rc_, out[-400:], err[-400:]))
        rc_, out, err = run_script("latency_probe.py", ["--talk-s", "4"], B)
        try:
            j = json.loads(out[out.index("\n{\n") + 1:])
            lag = j["leg3_full_path"]["pipeline_lag_ms"]
            ok = rc_ == 0 and lag["n"] > 20 and lag["p50"] is not None and j["gpu_check"]["step_ms_p95_max"] == 1.0
        except Exception as e:
            ok, lag = False, repr(e)
        check("latency_probe direct: pipeline lag measured through the echo stub", ok, (lag, err[-400:]))
        await space_stage(a, B, procs, logf)
        await strict_gone(B, wid, procs)
    finally:
        for p in procs.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        logf.close()
    print(f"\nprocess log: {logf.name}")
    print("ALL OK" if not FAILS else "FAILED: " + "; ".join(FAILS))
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-port", type=int, default=38000)
    ap.add_argument("--space-python", default="/root/deploy_serverless/.venv-space/bin/python",
                    help="python with the Space requirements (aiohttp); '' skips the Space stage")
    sys.exit(asyncio.run(amain(ap.parse_args())))


if __name__ == "__main__":
    main()
