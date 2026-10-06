#!/usr/bin/env python
"""Cross-check the Space backend against the OTHER builders' code (subprocesses; ports 28xxx; test secret only):

  A. LB:    builder C's tests/fake_runpod.py (LB proxy) + tests/stub_worker.py
  B. LB:    builder C's tests/fake_runpod.py (LB proxy) + builder W's worker/server/worker_server.py --mock (MockEngine)
  C. queue: builder C's tests/fake_runpod.py --handler worker/rp_handler.py (W) + W's worker_server.py --mock (S2S_MODE=queue)

One call each: waking -> ready -> relay (handshake, audio both ways, 0x07 events) -> browser closes -> ended.
Run with a python that has aiohttp + sphn + numpy (real Opus is sent to the worker), e.g. DEP1 venv-pp:
  /root/deploy/venv-pp/bin/python space/tests/xcheck_repo_fakes.py [A|B|C ...]
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
sys.path.insert(0, str(HERE))
import test_space_flows as T  # noqa: E402

import aiohttp  # noqa: E402
from aiohttp import WSMsgType  # noqa: E402

PY = "/root/deploy/venv-pp/bin/python"


def w_env(mode, port):
    """W's worker env, as worker/tests/test_worker_mock.py builds it (worker/local/runpod2.env + CPU overrides)."""
    env = dict(os.environ)
    f = Path(os.environ.get("S2S_LOCAL_ENV") or (R / "worker" / "local" / "runpod2.env"))   # pod1: pod1.env
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    env.update({"CUDA_VISIBLE_DEVICES": "", "S2S_SESSION_SECRET": T.SECRET, "S2S_AUDIENCE": T.AUD, "S2S_MODE": mode,
                "ASR_BACKEND": "off", "ROUTER": "off", "PYTHONDONTWRITEBYTECODE": "1",
                "S2S_LOGS": tempfile.mkdtemp(prefix="s2s-xcheck-"), "PORT": str(port),
                "S2S_INTERNAL_PORT": str(port + 1), "S2S_ASR_PORT": str(port + 2), "S2S_ROUTER_PORT": str(port + 3)})
    env.pop("RUNPOD_POD_ID", None)
    return env


async def opus_call(e, sid, seconds=4.0):
    """Browser stand-in that sends real Ogg/Opus (sphn) at 12.5 Hz after the handshake."""
    import numpy as np
    import sphn
    wr = sphn.OpusStreamWriter(24000)
    out = {"handshake": False, "kinds": {}, "events": [], "close_code": None}
    ws = await e.http.ws_connect(e.chat_url(sid), max_msg_size=0)
    n = 0

    async def sender():
        nonlocal n
        while True:
            t = (n * 1920 + np.arange(1920)) / 24000.0
            wr.append_pcm((0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
            b = wr.read_bytes()
            if b:
                await ws.send_bytes(b"\x01" + b)
            n += 1
            await asyncio.sleep(0.08)
    st = None
    t_live = None
    async for m in ws:
        if m.type != WSMsgType.BINARY:
            break
        k = m.data[:1]
        out["kinds"][k] = out["kinds"].get(k, 0) + 1
        if k == b"\x00" and not out["handshake"]:
            out["handshake"] = True
            t_live = time.time()
            st = asyncio.ensure_future(sender())
        if k == b"\x07":
            out["events"].append(json.loads(m.data[1:]))
        if t_live and time.time() - t_live > seconds:
            await ws.close()
            break
    if st:
        st.cancel()
    out["close_code"] = ws.close_code
    out["sent_frames"] = n
    return out


async def run_case(name, procs_spec, space_env, mode, wait_s=1.5):
    procs = []
    logs = []
    try:
        for cmd, env in procs_spec:
            lf = tempfile.NamedTemporaryFile("w+", prefix=f"xcheck-{name}-", suffix=".log", delete=False)
            logs.append(lf.name)
            procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT))
        await asyncio.sleep(wait_s)
        T.P_SPACE = 28860
        T.P_LB, T.P_Q = 28079, 28078            # Env's own fakes go to unused ports; C's fake is used instead
        e = T.Env(mode, space_env=space_env)
        async with e:
            st, js = await e.create()
            if not T.check(f"{name}: POST /api/session", st == 200, js):
                return
            sid = js["sid"]
            s, seen = await e.wait(sid, ("ready", "failed"), timeout=40)
            T.check(f"{name}: waking -> ready", s["state"] == "ready", (seen, s))
            if s["state"] != "ready":
                return
            out = await opus_call(e, sid)
            types = sorted({ev.get("type") for ev in out["events"]})
            T.check(f"{name}: handshake + 0x01 + 0x07 session relayed", out["handshake"] and out["kinds"].get(b"\x01")
                    and "session" in types, (out["kinds"], types))
            await asyncio.sleep(1.0)
            s = await e.state(sid)
            T.check(f"{name}: ended client_closed", s["state"] == "ended" and s.get("end_reason") == "client_closed", s)
            print(f"    {name}: worker_id={s.get('worker_id')} kinds={ {k.hex(): v for k, v in out['kinds'].items()} } "
                  f"events={types} sent_opus={out['sent_frames']}", flush=True)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        bad = [l for l in logs if "Traceback" in Path(l).read_text()]
        if bad:
            print(f"    {name}: tracebacks in {bad}", flush=True)


async def main(sel):
    cenv = dict(os.environ, S2S_SESSION_SECRET=T.SECRET, S2S_MODE="lb", S2S_AUDIENCE=T.AUD)
    fake_lb = [PY, str(R / "tests/fake_runpod.py"), "--lb-port", "28080", "--workers", "http://127.0.0.1:28000",
               "--worker-ids", "fw-0", "--hold-s", "0.5"]
    lb_space = {"RUNPOD_LB_URL": "http://127.0.0.1:28080"}
    if "A" in sel:
        await run_case("A(C-stub)", [
            ([PY, str(R / "tests/stub_worker.py"), "--port", "28000", "--internal-port", "28999", "--load-s", "2",
              "--prompt-s", "0.5", "--worker-id", "stub-0"], cenv), (fake_lb, cenv)], lb_space, "lb")
    if "B" in sel:
        wenv = w_env("lb", 28000)
        await run_case("B(W-mock,lb)", [
            ([wenv.get("S2S_PY_PP", PY), "-u", str(R / "worker/server/worker_server.py"), "--mock", "--port", "28000",
              "--internal-port", "28001", "--mock-prompt-s", "0.5", "--mock-load-s", "2"], wenv), (fake_lb, cenv)],
            lb_space, "lb")
    if "C" in sel:
        wenv = w_env("queue", 28100)
        qenv = dict(wenv)                       # the handler runs inside fake_runpod's process (C's design)
        await run_case("C(W-mock,queue)", [
            ([wenv.get("S2S_PY_PP", PY), "-u", str(R / "worker/server/worker_server.py"), "--mock", "--port", "28100",
              "--internal-port", "28101", "--mock-prompt-s", "0.5", "--mock-load-s", "2"], wenv),
            ([PY, str(R / "tests/fake_runpod.py"), "--workers", "", "--lb-port", "28082", "--queue-port", "28081",
              "--handler", str(R / "worker/rp_handler.py"), "--worker-port", "28100", "--endpoint-id", T.AUD], qenv)],
            {"RUNPOD_API_URL": f"http://127.0.0.1:28081/v2/{T.AUD}"}, "queue")
    n_fail = sum(1 for _, ok, _ in T.RESULTS if not ok)
    print(f"{len(T.RESULTS) - n_fail}/{len(T.RESULTS)} checks passed", flush=True)
    return n_fail


if __name__ == "__main__":
    sel = "".join(sys.argv[1:]) or "ABC"
    sys.exit(1 if asyncio.run(main(sel)) else 0)
