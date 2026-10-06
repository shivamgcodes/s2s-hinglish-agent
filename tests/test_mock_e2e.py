"""S2S serverless: CPU end-to-end test, both modes (DESIGN 8): the REAL worker code (worker/stack.sh -> worker_server.py
--mock, DEP1 MockEngine replay, ASR/router off) + tests/fake_runpod.py + the REAL Space backend (space/app) + a
headless websocket client (tests/rt_common.py). No GPU, no network beyond 127.0.0.1. Runs on runpod2 (needs the DEP1
venv-pp, the mock replay assets and the PersonaPlex tokenizer named in worker/local/runpod2.env).

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/test_mock_e2e.py [--base 18000] [--modes lb,queue]

Ports (DESIGN 7 with --base 18000): worker base, internal base+999, asr base+996, router base+995, LB proxy base+80,
queue API base+81, Space base-140 (17860). Use another --base when other builders are testing on 18xxx.
Checks:
  LB    : /ws-echo; call through the Space: states waking -> ready -> in_call -> ended, session + text + filler events;
          a second session while the call runs -> busy (claim 409); tampered / expired / wrong-aud token -> 401
          directly at the worker; token max_s=12 -> session_end time_limit; worker killed mid-call -> worker_lost
  queue : call through the Space via /run + /status + ws://ip:port: waking -> ready -> in_call -> ended
  repo  : DESIGN 1.2 path-grep rule on worker/ (outside worker/local, comments allowed)
"""
import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import rt_common as rc  # noqa: E402
import s2s_token  # noqa: E402   (rt_common put ../common on sys.path)

PY = sys.executable
SPACE_PY_DEFAULT = os.path.join(REPO, ".venv-space", "bin", "python")
SECRET = "mock-e2e-secret-0123456789-abcdefghijklmnop"
FAILS = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   [{str(info)[:500]}]" if info and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


class Procs:
    def __init__(self, logdir):
        self.p = {}
        self.logdir = logdir

    def start(self, name, cmd, env, cwd=None):
        lf = open(os.path.join(self.logdir, f"{name}.log"), "w")
        e = dict(os.environ, **{k: str(v) for k, v in env.items()})
        self.p[name] = subprocess.Popen(cmd, env=e, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        return self.p[name]

    def kill(self, name, sig=signal.SIGTERM):
        p = self.p.get(name)
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    def stop_all(self):
        for n in list(self.p):
            self.kill(n)


async def wait_status(url, want=200, timeout=90):
    t_end = time.time() + timeout
    async with aiohttp.ClientSession() as cs:
        while time.time() < t_end:
            try:
                async with cs.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == want:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


def worker_env(b, mode, extra=None):
    e = {"PORT": b, "S2S_INTERNAL_PORT": b + 999, "S2S_ASR_PORT": b + 996, "S2S_ROUTER_PORT": b + 995,
         "S2S_TMP": f"/tmp/s2s-e2e-{b}", "S2S_LOGS": f"/tmp/s2s-e2e-{b}/logs", "S2S_SESSION_SECRET": SECRET,
         "S2S_AUDIENCE": "ep-test", "S2S_MODE": mode, "ASR_BACKEND": "off", "ROUTER": "off",
         "S2S_WORKER_ARGS": "--mock-load-s 2 --mock-prompt-s 1", "CALL_MAX_S": 300, "CLAIM_TTL_S": 90,
         "RUNPOD_POD_ID": f"mock-{mode}-{b}"}
    e.update(extra or {})
    return e


def space_env(b, mode):
    return {"PORT": b - 140, "HOST": "127.0.0.1", "S2S_MODE": mode, "RUNPOD_ENDPOINT_ID": "ep-test",
            "RUNPOD_API_KEY": "test-key", "S2S_SESSION_SECRET": SECRET, "S2S_AUDIENCE": "ep-test",
            "RUNPOD_LB_URL": f"http://127.0.0.1:{b + 80}", "RUNPOD_API_URL": f"http://127.0.0.1:{b + 81}/v2/ep-test",
            "MAX_CONCURRENT_CALLS": 2, "RATE_PER_IP_PER_HOUR": 100, "KEEPALIVE_S": 3, "WAKE_POLL_S": 0.5,
            "QUEUE_POLL_S": 0.5, "S2S_PASSCODE": ""}


def args_ns(**kw):
    d = dict(space=None, direct=None, lb_url=None, api_url=None, passcode=None, dry_run=False, insecure=False,
             record_id="food_07", pairing="g1", seed=1001, wake_timeout_s=120.0, keepalive_path="/status",
             keepalive_s=3.0, keepalive_timeout_s=5.0, no_strict=False)
    d.update(kw)
    return argparse.Namespace(**d)


def dedup(xs):
    return [x for i, x in enumerate(xs) if i == 0 or xs[i - 1] != x]


async def poll_states(tgt, seen, stop):
    while not stop.is_set():
        s = await tgt.state()
        st = s.get("state")
        if st and (not seen or seen[-1][0] != st):
            seen.append((st, s.get("detail"), s.get("end_reason")))
        await asyncio.sleep(0.3)


async def space_call(space_url, talk_s, tag, during=None):
    """One call through the Space. Returns (stream stats, states seen, final state)."""
    seen, stop = [], asyncio.Event()
    async with aiohttp.ClientSession() as cs:
        tgt = rc.SpaceTarget(args_ns(space=space_url), cs)
        await tgt.prepare()
        poller = asyncio.create_task(poll_states(tgt, seen, stop))
        url, hdrs = tgt.ws_args()
        side = {}

        async def tick(st):
            if during and "task" not in side:
                side["task"] = asyncio.create_task(during(st))     # never block the 80 ms send loop

        st = await rc.stream_call(cs, url, hdrs, rc.tone_track(talk_s), tag, on_tick=tick, tick_s=3.0)
        if "task" in side:
            side["result"] = await side["task"]
        await asyncio.sleep(1.5)
        final = await tgt.state()
        stop.set()
        await poller
        await tgt.finish()
    pre = [ph["phase"].split(":")[0] for ph in tgt.phases.items]          # states seen during prepare()
    return st, pre + [s[0] for s in seen], final, side.get("result")


async def lb_suite(a, b, procs, space_url):
    W = f"http://127.0.0.1:{b}"
    # /ws-echo through the Space
    r = subprocess.run([PY, os.path.join(HERE, "space_ws_echo.py"), space_url, "--hold-s", "0"], capture_output=True,
                       text=True, timeout=60)
    check("lb: Space /ws-echo", r.returncode == 0, r.stdout[-300:])

    # call A through the Space, and a second session while it runs
    async def second_session(st):
        async with aiohttp.ClientSession() as cs:
            async with cs.post(space_url + "/api/session", json={"record_id": "food_07", "pairing": "g1",
                                                                 "seed": 1001}) as r2:
                j = await r2.json()
            sid2, states = j.get("sid"), []
            for _ in range(30):
                async with cs.get(f"{space_url}/api/session/{sid2}") as r3:
                    s = (await r3.json()).get("state")
                if not states or states[-1] != s:
                    states.append(s)
                if s == "busy":
                    break
                await asyncio.sleep(0.5)
            async with cs.delete(f"{space_url}/api/session/{sid2}") as r4:
                pass
            return states

    st, states, final, second = await space_call(space_url, 12, "lb-A", during=second_session)
    check("lb: handshake + model audio through Space relay", st.get("t_handshake") and st["model_samples"] > 24000, st)
    check("lb: 0x07 session + text events relayed", st["events"].get("session") == 1 and st["n_text"] > 0, st["events"])
    check("lb: 0x07 filler (ticker indicator) events relayed unchanged by the Space (default turn_fill ticker)",
          st["events"].get("filler", 0) >= 1, st["events"])
    want = ["waking", "ready", "in_call", "ended"]
    check("lb: state sequence waking -> ready -> in_call -> ended", dedup([s for s in states if s in want]) == want, states)
    check("lb: second session while the call runs -> busy", second and "busy" in second, second)

    # token refusals directly at the worker (LB proxy not involved)
    sid = s2s_token.new_sid()
    good = s2s_token.mint(SECRET, sid=sid, mode="lb", aud="ep-test", record_id="food_07", pairing="g1", seed=1001,
                          max_s=300)
    h, p, sg = good.split(".")
    tampered = h + "." + p[:-2] + ("A" if p[-2] != "A" else "B") + p[-1] + "." + sg
    expired = s2s_token.mint(SECRET, sid=sid, mode="lb", aud="ep-test", record_id="food_07", pairing="g1", seed=1001,
                             max_s=300, now=time.time() - 1000)
    wrong_aud = s2s_token.mint(SECRET, sid=sid, mode="lb", aud="ep-other", record_id="food_07", pairing="g1",
                               seed=1001, max_s=300)
    async with aiohttp.ClientSession() as cs:
        for name, t in (("tampered", tampered), ("expired", expired), ("wrong aud", wrong_aud)):
            async with cs.post(W + "/session/claim", headers={"X-S2S-Token": t}) as r:
                check(f"lb: worker /session/claim {name} token -> 401", r.status == 401, (r.status, await r.text()))
        try:
            await cs.ws_connect(W + "/api/chat?record_id=food_07&pairing=g1&seed=1001", headers={"X-S2S-Token": expired})
            check("lb: worker ws expired token -> refused", False)
        except aiohttp.WSServerHandshakeError as e:
            check("lb: worker ws expired token -> 401 before upgrade", e.status == 401, e.status)

    # time cap via token max_s (direct LB path through fake_runpod, CALL_MAX_S=12 in this client's env)
    os.environ.update({"RUNPOD_API_KEY": "test-key", "S2S_SESSION_SECRET": SECRET, "RUNPOD_ENDPOINT_ID": "ep-test",
                       "S2S_AUDIENCE": "ep-test", "CALL_MAX_S": "12"})
    async with aiohttp.ClientSession() as cs:
        tgt = rc.LBDirectTarget(args_ns(direct="lb", lb_url=f"http://127.0.0.1:{b + 80}"), cs)
        await tgt.prepare()
        url, hdrs = tgt.ws_args()
        st = await rc.stream_call(cs, url, hdrs, rc.tone_track(25), "lb-cap")
        await tgt.finish()
    os.environ["CALL_MAX_S"] = "300"
    se = (st.get("session_end") or {}).get("reason")
    check("lb: token max_s=12 -> session_end time_limit (about 12 s after upgrade)", se == "time_limit"
          and st["user_s"] < 16, (se, st.get("user_s")))

    # worker killed mid-call -> worker_lost from the relay
    async def kill_worker(st):
        await asyncio.sleep(2)
        procs.kill("worker_lb", signal.SIGKILL)
        return True

    st, states, final, _ = await space_call(space_url, 20, "lb-kill", during=kill_worker)
    se = (st.get("session_end") or {}).get("reason")
    check("lb: worker killed mid-call -> synthetic session_end worker_lost", se == "worker_lost", (se, st.get("close_code")))
    check("lb: Space session ended with worker_lost", final.get("state") == "ended" and
          "worker_lost" in json.dumps(final), final)


async def queue_suite(a, b, space_url):
    st, states, final, _ = await space_call(space_url, 10, "queue-A")
    check("queue: handshake + model audio via ws://ip:port relay", st.get("t_handshake") and st["model_samples"] > 24000, st)
    want = ["waking", "ready", "in_call", "ended"]
    check("queue: state sequence waking -> ready -> in_call -> ended", dedup([s for s in states if s in want]) == want,
          states)
    check("queue: Space session ended", final.get("state") == "ended", final)


def path_grep():
    """DESIGN 1.2: no /root/deploy, /root/needle, /workspace or /root/hf in worker/ code; comments and docstrings that
    cite the DEP1 origin are allowed; worker/local (runpod2-only runner) is exempt (W's amendment)."""
    import ast
    import io
    import tokenize
    bad = []
    pat = re.compile(r"/root/deploy\b|/root/needle\b|/workspace\b|/root/hf\b")
    for dp, dn, fn in os.walk(os.path.join(REPO, "worker")):
        dn[:] = [d for d in dn if d not in ("local", "build", "__pycache__", "vendor", "tests")]
        for f in fn:
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, REPO)
            src = open(path, errors="replace").read()
            if f.endswith(".py"):
                doc_lines = set()
                for node in ast.walk(ast.parse(src)):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and \
                            node.body and isinstance(node.body[0], ast.Expr) and \
                            isinstance(getattr(node.body[0], "value", None), ast.Constant) and \
                            isinstance(node.body[0].value.value, str):
                        doc_lines.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
                for t in tokenize.generate_tokens(io.StringIO(src).readline):
                    if t.type == tokenize.COMMENT or (t.type == tokenize.STRING and t.start[0] in doc_lines):
                        continue
                    if pat.search(t.string):
                        bad.append(f"{rel}:{t.start[0]}: {t.string[:100]}")
            elif f.endswith(".sh") or f.startswith("Dockerfile"):
                for i, line in enumerate(src.splitlines(), 1):
                    if pat.search(line.split("#", 1)[0]):
                        bad.append(f"{rel}:{i}: {line.strip()[:100]}")
    return bad


async def amain(a):
    b = a.base
    logdir = tempfile.mkdtemp(prefix="s2s_mock_e2e_")
    procs = Procs(logdir)
    run_local = os.path.join(REPO, "worker", "local", "run_local.sh")
    space_dir = os.path.join(REPO, "space")
    space_url = f"http://127.0.0.1:{b - 140}"
    bad = path_grep()
    check("repo: DESIGN 1.2 path-grep rule on worker/ (excl. worker/local)", not bad, bad[:5])
    try:
        if "lb" in a.modes:
            procs.start("worker_lb", ["bash", run_local, "mock"], worker_env(b, "lb"))
            procs.start("fake_lb", [PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(b + 80),
                                    "--workers", f"http://127.0.0.1:{b}", "--worker-ids", "fw-0", "--hold-s", "2"], {})
            procs.start("space_lb", [a.space_python, "-u", "-m", "app.main"], space_env(b, "lb"), cwd=space_dir)
            ok = await wait_status(space_url + "/healthz") and await wait_status(f"http://127.0.0.1:{b}/ping", 200, 120)
            check("lb: worker (mock) + fake proxy + Space up", ok)
            if ok:
                await lb_suite(a, b, procs, space_url)
            for n in ("space_lb", "fake_lb", "worker_lb"):
                procs.kill(n)
            await asyncio.sleep(1)
        if "queue" in a.modes:
            wenv = worker_env(b, "queue")
            procs.start("worker_q", ["bash", run_local, "mock"], wenv)
            fenv = {k: wenv[k] for k in ("PORT", "S2S_INTERNAL_PORT", "S2S_SESSION_SECRET", "S2S_AUDIENCE",
                                         "S2S_MODE", "CLAIM_TTL_S", "CALL_MAX_S", "S2S_TMP", "S2S_LOGS")}
            procs.start("fake_q", [PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--workers", "", "--queue-port",
                                   str(b + 81), "--handler", os.path.join(REPO, "worker", "rp_handler.py"),
                                   "--worker-port", str(b)], fenv)
            procs.start("space_q", [a.space_python, "-u", "-m", "app.main"], space_env(b, "queue"), cwd=space_dir)
            ok = await wait_status(space_url + "/healthz") and await wait_status(f"http://127.0.0.1:{b}/ping", 200, 120)
            check("queue: worker (mock) + fake queue API + Space up", ok)
            if ok:
                await queue_suite(a, b, space_url)
    finally:
        procs.stop_all()
    print(f"\nlogs: {logdir}")
    print("ALL OK" if not FAILS else "FAILED: " + "; ".join(FAILS))
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=18000)
    ap.add_argument("--modes", default="lb,queue")
    ap.add_argument("--space-python", default=SPACE_PY_DEFAULT)
    sys.exit(asyncio.run(amain(ap.parse_args())))


if __name__ == "__main__":
    main()
