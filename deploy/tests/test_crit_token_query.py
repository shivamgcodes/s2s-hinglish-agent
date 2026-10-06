"""S2S serverless: CRIT-3 regression (O-W1). A fake LB proxy that DROPS the X-S2S-Token header on every request
(tests/fake_runpod.py --strip-headers X-S2S-Token) in front of the REAL worker (--mock) and the REAL Space.

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/test_crit_token_query.py [--base 56000]

Checks:
  1. S2S_TOKEN_IN_QUERY unset (default): the session ends failed, and the detail names S2S_TOKEN_IN_QUERY (the hint
     does not send the user to the secret).
  2. header dropped on the websocket upgrade only (--strip-ws-only), switch off: the claim passes, the call ends
     relay_error and the detail names S2S_TOKEN_IN_QUERY.
  3. S2S_TOKEN_IN_QUERY=1 (header dropped everywhere): a full call through the Space works (waking -> ready -> in_call -> ended), the claim, the
     websocket upgrade and the release all authenticate with ?token=.
Reuses the harness of tests/test_mock_e2e.py (same port layout around --base).
"""
import argparse
import asyncio
import os
import sys
import tempfile

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_mock_e2e as m  # noqa: E402


async def one_session_fails(space_url):
    async with aiohttp.ClientSession() as cs:
        async with cs.post(space_url + "/api/session", json={"record_id": "food_07", "pairing": "g1",
                                                             "seed": 1001}) as r:
            sid = (await r.json()).get("sid")
        s = {}
        for _ in range(120):
            async with cs.get(f"{space_url}/api/session/{sid}") as r:
                s = await r.json()
            if s.get("state") in ("failed", "ended", "ready"):
                break
            await asyncio.sleep(0.5)
        async with cs.delete(f"{space_url}/api/session/{sid}"):
            pass
        return s


async def amain(a):
    b = a.base
    procs = m.Procs(tempfile.mkdtemp(prefix="s2s_crit_tq_"))
    run_local = os.path.join(m.REPO, "worker", "local", "run_local.sh")
    space_dir = os.path.join(m.REPO, "space")
    space_url = f"http://127.0.0.1:{b - 140}"
    try:
        procs.start("worker", ["bash", run_local, "mock"], m.worker_env(b, "lb"))
        procs.start("fake", [m.PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(b + 80), "--workers",
                             f"http://127.0.0.1:{b}", "--worker-ids", "fw-0", "--hold-s", "2",
                             "--strip-headers", "X-S2S-Token"], {})
        ok = await m.wait_status(f"http://127.0.0.1:{b}/ping", 200, 120)
        m.check("worker (mock) + stripping proxy up", ok)
        if not ok:
            return
        # 1. switch off
        procs.start("space0", [a.space_python, "-u", "-m", "app.main"], m.space_env(b, "lb"), cwd=space_dir)
        m.check("Space (S2S_TOKEN_IN_QUERY unset) up", await m.wait_status(space_url + "/healthz"))
        s = await one_session_fails(space_url)
        det = str(s.get("detail", "")) + " " + str(s.get("end_reason", ""))
        m.check("switch off: header stripped -> session failed", s.get("state") == "failed", s)
        m.check("switch off: failure detail names S2S_TOKEN_IN_QUERY", "S2S_TOKEN_IN_QUERY" in det, det)
        procs.kill("space0")
        await asyncio.sleep(1)
        # 2. header dropped on the upgrade only, switch off
        procs.kill("fake")
        procs.start("fake_ws", [m.PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(b + 80),
                                "--workers", f"http://127.0.0.1:{b}", "--worker-ids", "fw-0", "--hold-s", "2",
                                "--strip-headers", "X-S2S-Token", "--strip-ws-only"], {})
        procs.start("space0b", [a.space_python, "-u", "-m", "app.main"], m.space_env(b, "lb"), cwd=space_dir)
        m.check("ws-only stripping proxy + Space up", await m.wait_status(space_url + "/healthz"))
        await asyncio.sleep(1)
        st, states, final, _ = await m.space_call(space_url, 4, "crit-tq-ws")
        det = str(final.get("detail", "")) + " " + str(final.get("end_reason", ""))
        m.check("ws-only strip, switch off: claim ok, call ends relay_error", "ready" in states and
                final.get("end_reason") == "relay_error", (states, final))
        m.check("ws-only strip, switch off: detail names S2S_TOKEN_IN_QUERY", "S2S_TOKEN_IN_QUERY" in det, det)
        procs.kill("space0b")
        procs.kill("fake_ws")
        procs.start("fake", [m.PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(b + 80), "--workers",
                             f"http://127.0.0.1:{b}", "--worker-ids", "fw-0", "--hold-s", "2",
                             "--strip-headers", "X-S2S-Token"], {})
        await asyncio.sleep(1.5)
        # the worker still holds the claim of the failed call until CLAIM_TTL_S; the relay released it (REVIEW-space-2)
        # 3. switch on
        env = dict(m.space_env(b, "lb"), S2S_TOKEN_IN_QUERY=1)
        procs.start("space1", [a.space_python, "-u", "-m", "app.main"], env, cwd=space_dir)
        m.check("Space (S2S_TOKEN_IN_QUERY=1) up", await m.wait_status(space_url + "/healthz"))
        st, states, final, _ = await m.space_call(space_url, 10, "crit-tq")
        want = ["waking", "ready", "in_call", "ended"]
        m.check("switch on: handshake + model audio through the stripping proxy",
                st.get("t_handshake") and st["model_samples"] > 24000, st)
        m.check("switch on: states waking -> ready -> in_call -> ended",
                m.dedup([x for x in states if x in want]) == want, states)
        m.check("switch on: Space session ended", final.get("state") == "ended", final)
    finally:
        procs.stop_all()
    print("ALL OK" if not m.FAILS else "FAILED: " + "; ".join(m.FAILS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=56000)
    ap.add_argument("--space-python", default=m.SPACE_PY_DEFAULT)
    a = ap.parse_args()
    asyncio.run(amain(a))
    sys.exit(1 if m.FAILS else 0)


if __name__ == "__main__":
    main()
