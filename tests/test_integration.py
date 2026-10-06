"""S2S serverless, Integrate stage: CPU end-to-end checks that tests/test_mock_e2e.py does not assert (DESIGN 8).

Same harness as test_mock_e2e.py: the REAL worker (worker/local/run_local.sh mock -> stack.sh -> worker_server.py
--mock), tests/fake_runpod.py (LB proxy / queue API) and the REAL Space backend (space/app), driven by a scripted
websocket client (tests/rt_common.py, Opus via sphn, the stock PersonaPlex client protocol). No GPU.

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/test_integration.py [--base 45000]
        [--phases lb,queue,router] [--space-python ...] [--json results/integ_cpu.json]

Ports (base B): worker B, internal B+999, asr B+996, router B+995, LB proxy B+80, queue API B+81, Space B-140.
Phases:
  lb     : call through the Space; during it: Space keepalive fired (strict-pinned, 200s; fake proxy request count),
           a direct second claim -> 409 and a direct second websocket -> 409 (one call per worker), the Space
           refuses a second socket for the same sid; after it: clean end (client_closed), worker /status idle and
           sessions_served +1, Space session ended.  Token checks: replayed token on the ws upgrade -> 401 replayed
           (after a fresh claim for the same sid), no / garbage / wrong-mode token -> 401, bad token through the fake
           LB proxy -> 401 passed through, no Bearer -> 401 at the proxy.
  queue  : call through the Space (/run -> progress -> ws://ip:port relay); during it: public ws with no / expired /
           lb-mode token -> 401, fresh queue token for another sid -> 409, public /status hides the sid; after it:
           worker idle, sessions_served +1, the handler's claim state 'ended' with a summary.
  router : (slow, ~90 s) mock engine + faster-whisper-small on CPU + Needle router, 62 s of the food_07_g1 input
           through the Space relay: router 'action' events (trigger -> asr -> needle -> ...) reach the client.
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
import urllib.parse

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import rt_common as rc  # noqa: E402
import s2s_token  # noqa: E402
from test_mock_e2e import Procs, wait_status, args_ns, dedup, SECRET  # noqa: E402

PY = sys.executable
SPACE_PY_DEFAULT = os.path.join(REPO, ".venv-space", "bin", "python")
FOOD07_WAV = "/workspace/hinglish/tests/inputs/V3/food_07_g1.wav"
FAILS, RESULTS = [], []
CFG = dict(record_id="food_07", pairing="g1", seed=1001)


def check(name, cond, info=""):
    RESULTS.append({"check": name, "ok": bool(cond), "info": None if cond else str(info)[:500]})
    print(("ok   " if cond else "FAIL ") + name + (f"   [{str(info)[:500]}]" if info and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def worker_env(b, mode, extra=None):
    e = {"PORT": b, "S2S_INTERNAL_PORT": b + 999, "S2S_ASR_PORT": b + 996, "S2S_ROUTER_PORT": b + 995,
         "S2S_TMP": f"/tmp/s2s-integ-{b}", "S2S_LOGS": f"/tmp/s2s-integ-{b}/logs", "S2S_SESSION_SECRET": SECRET,
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


def tok(mode, sid=None, now=None, **kw):
    c = dict(CFG, **kw)
    return s2s_token.mint(SECRET, sid=sid or s2s_token.new_sid(), mode=mode, aud="ep-test", record_id=c["record_id"],
                          pairing=c["pairing"], seed=c["seed"], max_s=300, now=now)


def chat_q(events=True):
    q = {"record_id": CFG["record_id"], "pairing": CFG["pairing"], "seed": str(CFG["seed"])}
    if events:
        q["events"] = "1"
    return "/api/chat?" + urllib.parse.urlencode(q)


async def ws_status(cs, url, headers):
    """Try a websocket upgrade; return the HTTP status of a refusal, or 101 (then close at once)."""
    try:
        ws = await cs.ws_connect(url, headers=headers, timeout=10)
    except aiohttp.WSServerHandshakeError as e:
        return e.status
    await ws.close()
    return 101


async def get_json(cs, url, headers=None):
    async with cs.get(url, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return r.status, await r.json(content_type=None)


async def wait_idle(cs, W, timeout=20):
    t_end = time.time() + timeout
    j = {}
    while time.time() < t_end:
        try:
            _, j = await get_json(cs, W + "/status")
            if j.get("state") == "idle":
                return j
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return j


async def space_call(space_url, pcm, tag, during=None, on_event=None, tick_s=3.0):
    """One call through the Space like the browser does. Returns (stream stats, states seen, final, during-result)."""
    seen, stop = [], asyncio.Event()
    async with aiohttp.ClientSession() as cs:
        tgt = rc.SpaceTarget(args_ns(space=space_url), cs)
        await tgt.prepare()

        async def poll():
            while not stop.is_set():
                s = await tgt.state()
                st_ = s.get("state")
                if st_ and (not seen or seen[-1]["state"] != st_ or s.get("keepalive") != seen[-1].get("keepalive")):
                    seen.append({"state": st_, "keepalive": s.get("keepalive"), "end_reason": s.get("end_reason")})
                await asyncio.sleep(0.3)
        poller = asyncio.create_task(poll())
        url, hdrs = tgt.ws_args()
        side = {}

        async def tick(st):
            if during and "task" not in side:
                side["task"] = asyncio.create_task(during(tgt, st))

        st = await rc.stream_call(cs, url, hdrs, pcm, tag, on_tick=tick, tick_s=tick_s, on_event=on_event)
        if "task" in side:
            side["result"] = await side["task"]
        await asyncio.sleep(1.5)
        final = await tgt.state()
        stop.set()
        await poller
        await tgt.finish()
    pre = [ph["phase"].split(":")[0] for ph in tgt.phases.items]
    return st, pre, seen, final, side.get("result"), tgt.sid


# ------------------------------------------------------------------------------------------------------------ LB
async def lb_phase(a, b, space_url):
    W, P = f"http://127.0.0.1:{b}", f"http://127.0.0.1:{b + 80}"
    async with aiohttp.ClientSession() as cs:
        _, s0 = await get_json(cs, W + "/status")
        _, f0 = await get_json(cs, P + "/_fake/state")

    async def during(tgt, st):
        out = {}
        async with aiohttp.ClientSession() as cs:
            async with cs.post(W + "/session/claim", headers={"X-S2S-Token": tok("lb")}) as r:
                out["claim2"] = (r.status, (await r.json(content_type=None)).get("error"))
            t2 = tok("lb")
            out["ws2"] = await ws_status(cs, W + chat_q(), {"X-S2S-Token": t2})
            # the Space refuses a second socket for the same sid (one browser socket per session)
            out["space_ws2"] = await ws_status(cs, rc.ws_from_http(space_url) + chat_q() + "&sid=" + tgt.sid, {})
            _, out["status_mid"] = await get_json(cs, W + "/status")
            async with cs.get(W + "/ping") as r:
                out["ping_mid"] = r.status
            await asyncio.sleep(7)                     # let >= 2 keepalives (KEEPALIVE_S=3) happen
        return out

    pcm = rc.tone_track(14)
    st, pre, seen, final, mid, sid = await space_call(space_url, pcm, "integ-lb", during=during)
    check("lb: call through the Space: handshake, model audio > 1 s, text tokens",
          st.get("t_handshake") and st["model_samples"] > 24000 and st["n_text"] > 0, st)
    states = dedup(pre + [s["state"] for s in seen])
    check("lb: Space states waking -> ready -> in_call -> ended",
          [s for s in states if s in ("waking", "ready", "in_call", "ended")] == ["waking", "ready", "in_call",
                                                                                    "ended"], states)
    check("lb: second claim at the worker during the call -> 409 busy", mid and mid["claim2"][0] == 409, mid)
    check("lb: second websocket at the worker during the call -> 409", mid and mid["ws2"] == 409, mid)
    check("lb: second socket for the same sid at the Space -> refused (not 101)", mid and mid["space_ws2"] != 101, mid)
    check("lb: worker /status mid-call is busy", mid and mid["status_mid"].get("state") == "busy",
          mid and mid["status_mid"])
    check("lb: worker /ping mid-call -> 200 (stays in the LB routing pool during a call)",
          mid and mid.get("ping_mid") == 200, mid)
    ka = [s["keepalive"] for s in seen if s.get("keepalive")]
    last_ka = ka[-1] if ka else {}
    check("lb: Space keepalive fired >= 3 times during the 14 s call, all 200",
          last_ka.get("sent", 0) >= 3 and set((last_ka.get("codes") or {}).keys()) <= {"200", 200}, last_ka)
    se = (st.get("session_end") or {}).get("reason")
    check("lb: clean end of call (client closed -> no session_end error, Space end_reason client_closed)",
          final.get("state") == "ended" and final.get("end_reason") == "client_closed", (se, final))
    async with aiohttp.ClientSession() as cs:
        s1 = await wait_idle(cs, W)
        _, f1 = await get_json(cs, P + "/_fake/state")
    check("lb: worker /status idle after the call", s1.get("state") == "idle", s1)
    check("lb: worker sessions_served +1", s1.get("sessions_served") == s0.get("sessions_served", 0) + 1,
          (s0.get("sessions_served"), s1.get("sessions_served")))
    nreq = f1["workers"][0]["n_req"] - f0["workers"][0]["n_req"]
    check("lb: fake proxy saw the keepalive /status requests (>= 3 extra HTTP requests on fw-0)", nreq >= 3 + 2,
          (nreq, f1))
    RESULTS.append({"info": "lb call", "stream": {k: st.get(k) for k in ("model_s", "user_s", "n_text", "events",
                                                                          "close_code", "session_end")},
                    "keepalive": last_ka, "proxy_requests_during": nreq})
    return st


async def lb_tokens(b):
    W, P = f"http://127.0.0.1:{b}", f"http://127.0.0.1:{b + 80}"
    async with aiohttp.ClientSession() as cs:
        await wait_idle(cs, W)
        sid = s2s_token.new_sid()
        t1 = tok("lb", sid=sid)
        async with cs.post(W + "/session/claim", headers={"X-S2S-Token": t1}) as r:
            c1 = r.status
        ws = await cs.ws_connect(W + chat_q(), headers={"X-S2S-Token": t1}, timeout=10)
        await asyncio.sleep(2.5)
        await ws.close()
        await wait_idle(cs, W)
        t2 = tok("lb", sid=sid)                                  # fresh token, same sid: the claim is allowed ...
        async with cs.post(W + "/session/claim", headers={"X-S2S-Token": t2}) as r:
            c2 = r.status
        rep_body = None
        try:
            w2 = await cs.ws_connect(W + chat_q(), headers={"X-S2S-Token": t1}, timeout=10)
            await w2.close()
            rep = 101
        except aiohttp.WSServerHandshakeError as e:
            rep = e.status
        try:
            w3 = await cs.ws_connect(W + chat_q(), headers={"X-S2S-Token": t2}, timeout=10)
            await w3.close()
            rep2 = 101
        except aiohttp.WSServerHandshakeError as e:
            rep2 = e.status
        async with cs.post(W + "/session/release", headers={"X-S2S-Token": t2}) as r:
            rel = (r.status, await r.json(content_type=None))
        check("lb: replay: first use of token T accepted (claim 200)", c1 == 200, c1)
        check("lb: replay: a fresh claim for the used sid is still allowed (replay is checked on the ws only)",
              c2 == 200, c2)
        check("lb: replay: same token T on a 2nd ws upgrade (after a fresh claim) -> 401", rep == 401, rep)
        check("lb: replay: a fresh token for an already-used sid -> 401 on the upgrade", rep2 == 401, rep2)
        check("lb: release of the dangling re-claim -> released", rel[0] == 200 and rel[1].get("released"), rel)
        # refusals of bad tokens on the ws upgrade (pre-upgrade HTTP errors)
        for name, h, want in (("no token", {}, 401), ("garbage token", {"X-S2S-Token": "v1.x.y"}, 401),
                              ("queue-mode token at an LB worker", {"X-S2S-Token": tok("queue")}, 401),
                              ("expired token", {"X-S2S-Token": tok("lb", now=time.time() - 1000)}, 401),
                              ("token cfg != query cfg", {"X-S2S-Token": tok("lb", seed=7)}, 401)):
            s = await ws_status(cs, W + chat_q(), h)
            check(f"lb: worker ws upgrade with {name} -> {want}", s == want, s)
        # through the fake LB proxy (Bearer ok): a bad token's 401 passes through; no Bearer -> proxy 401
        async with cs.post(P + "/session/claim", headers={"Authorization": "Bearer test-key",
                                                          "X-S2S-Token": "v1.bad.sig"}) as r:
            check("lb: bad token through the LB proxy -> 401 from the worker", r.status == 401, r.status)
        async with cs.post(P + "/session/claim", headers={"X-S2S-Token": tok("lb")}) as r:
            check("lb: no Bearer at the LB proxy -> 401", r.status == 401, r.status)
        s = await wait_idle(cs, W)
        check("lb: worker idle after the token checks", s.get("state") == "idle", s)


# --------------------------------------------------------------------------------------------------------- queue
async def queue_phase(a, b, space_url):
    W, I = f"http://127.0.0.1:{b}", f"http://127.0.0.1:{b + 999}"
    async with aiohttp.ClientSession() as cs:
        _, s0 = await get_json(cs, W + "/status")

    async def during(tgt, st):
        out = {}
        async with aiohttp.ClientSession() as cs:
            for name, h in (("none", {}), ("expired", {"X-S2S-Token": tok("queue", now=time.time() - 1000)}),
                            ("lb-mode", {"X-S2S-Token": tok("lb")}), ("bad-sig", {"X-S2S-Token": tok("queue")[:-3]
                                                                                   + "AAA"})):
                out[name] = await ws_status(cs, W + chat_q(), h)
            out["fresh_other_sid"] = await ws_status(cs, W + chat_q(), {"X-S2S-Token": tok("queue")})
            _, out["status_mid"] = await get_json(cs, W + "/status")
            async with cs.get(W + "/ping") as r:
                out["ping_mid"] = r.status
        return out

    st, pre, seen, final, mid, sid = await space_call(space_url, rc.tone_track(10), "integ-queue", during=during)
    check("queue: call through the Space via /run + ws://ip:port relay: audio + text",
          st.get("t_handshake") and st["model_samples"] > 24000 and st["n_text"] > 0, st)
    states = dedup(pre + [s["state"] for s in seen])
    check("queue: Space states waking -> ready -> in_call -> ended",
          [s for s in states if s in ("waking", "ready", "in_call", "ended")] == ["waking", "ready", "in_call",
                                                                                    "ended"], states)
    for k in ("none", "expired", "lb-mode", "bad-sig"):
        check(f"queue: public ws:// port, {k} token during the call -> 401", mid and mid[k] == 401, mid)
    check("queue: public ws:// port, valid token for another sid during the call -> 409",
          mid and mid["fresh_other_sid"] == 409, mid)
    check("queue: worker /ping mid-call -> 200", mid and mid.get("ping_mid") == 200, mid)
    am = (mid or {}).get("status_mid", {}).get("active") or {}
    check("queue: public /status mid-call is busy and hides the sid (REVIEW-worker-5)",
          mid and mid["status_mid"].get("state") == "busy" and not am.get("sid"), mid and mid["status_mid"])
    check("queue: clean end (Space end_reason client_closed)",
          final.get("state") == "ended" and final.get("end_reason") == "client_closed", final)
    async with aiohttp.ClientSession() as cs:
        s1 = await wait_idle(cs, W)
        _, cl = await get_json(cs, f"{I}/internal/claim/{sid}")
    check("queue: worker idle after the call, sessions_served +1",
          s1.get("state") == "idle" and s1.get("sessions_served") == s0.get("sessions_served", 0) + 1, s1)
    check("queue: handler's claim state 'ended' with a summary (frames > 0)",
          cl.get("phase") == "ended" and (cl.get("summary") or {}).get("frames", 0) > 0, cl)
    RESULTS.append({"info": "queue call", "stream": {k: st.get(k) for k in ("model_s", "user_s", "n_text", "events",
                                                                             "close_code", "session_end")},
                    "claim_state": {k: cl.get(k) for k in ("phase", "end_reason")}})


# -------------------------------------------------------------------------------------------------------- router
async def router_phase(a, b, space_url):
    evs = []
    pcm = rc.load_wav(FOOD07_WAV, seconds=62)
    st, pre, seen, final, _, sid = await space_call(space_url, pcm, "integ-router",
                                                    on_event=lambda ev, t: evs.append(ev))
    stages = [e.get("stage") for e in evs if e.get("type") == "action"]
    check("router: call ran (audio + text)", st.get("t_handshake") and st["model_samples"] > 24000, st)
    check("router: >= 1 router trigger reached the client through the Space relay", "trigger" in stages, stages)
    check("router: trigger -> asr -> needle stages relayed", all(s in stages for s in ("trigger", "asr", "needle")),
          stages)
    RESULTS.append({"info": "router call", "stages": stages,
                    "actions": [{k: e.get(k) for k in ("stage", "trigger_id", "latency_ms", "payload")}
                                for e in evs if e.get("type") == "action"][:20]})


async def amain(a):
    b = a.base
    logdir = tempfile.mkdtemp(prefix="s2s_integ_")
    procs = Procs(logdir)
    run_local = os.path.join(REPO, "worker", "local", "run_local.sh")
    space_dir = os.path.join(REPO, "space")
    space_url = f"http://127.0.0.1:{b - 140}"
    try:
        for ph in a.phases.split(","):
            mode = "queue" if ph == "queue" else "lb"
            extra = {}
            if ph == "router":
                extra = {"ASR_BACKEND": "fw-small", "ASR_DEVICE": "cpu", "ROUTER": "on"}
            wenv = worker_env(b, mode, extra)
            procs.start(f"worker_{ph}", ["bash", run_local, "mock"], wenv)
            if mode == "lb":
                procs.start(f"fake_{ph}", [PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--lb-port", str(b + 80),
                                           "--workers", f"http://127.0.0.1:{b}", "--worker-ids", "fw-0",
                                           "--hold-s", "2"], {})
            else:
                fenv = {k: wenv[k] for k in ("PORT", "S2S_INTERNAL_PORT", "S2S_SESSION_SECRET", "S2S_AUDIENCE",
                                             "S2S_MODE", "CLAIM_TTL_S", "CALL_MAX_S", "S2S_TMP", "S2S_LOGS")}
                procs.start(f"fake_{ph}", [PY, "-u", os.path.join(HERE, "fake_runpod.py"), "--workers", "",
                                           "--queue-port", str(b + 81), "--handler",
                                           os.path.join(REPO, "worker", "rp_handler.py"), "--worker-port", str(b)],
                            fenv)
            procs.start(f"space_{ph}", [a.space_python, "-u", "-m", "app.main"], space_env(b, mode), cwd=space_dir)
            ok = await wait_status(space_url + "/healthz") and \
                await wait_status(f"http://127.0.0.1:{b}/ping", 200, 300 if ph == "router" else 120)
            check(f"{ph}: worker (mock) + fake RunPod + Space up", ok)
            if ok:
                if ph == "lb":
                    await lb_phase(a, b, space_url)
                    await lb_tokens(b)
                elif ph == "queue":
                    await queue_phase(a, b, space_url)
                elif ph == "router":
                    await router_phase(a, b, space_url)
            for n in (f"space_{ph}", f"fake_{ph}", f"worker_{ph}"):
                procs.kill(n)
            await asyncio.sleep(1.5)
    finally:
        procs.stop_all()
    print(f"\nlogs: {logdir}")
    n_ok = sum(1 for r in RESULTS if r.get("ok"))
    n = sum(1 for r in RESULTS if "ok" in r)
    print(f"{n_ok}/{n} checks passed")
    print("ALL OK" if not FAILS else "FAILED: " + "; ".join(FAILS))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"t": time.strftime("%FT%TZ", time.gmtime()), "base": b, "phases": a.phases, "logs": logdir,
                       "passed": n_ok, "total": n, "results": RESULTS}, f, indent=1, default=str)
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=45000)
    ap.add_argument("--phases", default="lb,queue")
    ap.add_argument("--space-python", default=SPACE_PY_DEFAULT)
    ap.add_argument("--json", default="")
    sys.exit(asyncio.run(amain(ap.parse_args())))


if __name__ == "__main__":
    main()
