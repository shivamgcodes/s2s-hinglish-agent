#!/usr/bin/env python
"""Space backend flows against fake_runpod_lite (CPU only, no RunPod, no GPU). Space builder's test (DESIGN 8).

  /root/deploy_serverless/.venv-space/bin/python space/tests/test_space_flows.py [-k name]

Ports 27xxx (the repo-wide tests/ use 17860/18000/18080, so both can run at once).
Checks: LB cold wake (503s -> 204 -> ready), claim with the session token, strict pinning + Bearer on the websocket,
the proxy header winning over the claim body worker_id, the relay passing frames both ways, metrics peek, the
worker's session_end (time_limit) passed through, busy (409) -> ready, capacity / rate / passcode / config-mismatch
refusals, a second socket on the same sid, worker killed mid-call -> synthetic worker_lost + close 1011, the Space
call cap, claim-TTL expiry -> release, DELETE while waking, a wrong secret -> failed, keepalive codes, /api/diag
gating, /ws-echo, static client, records API; queue mode: /run -> loading -> ready -> ws://ip:port relay (token, no
Bearer) -> ended, and cancel.
"""
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
REPO = SPACE.parent
os.environ.setdefault("S2S_COMMON", str(REPO / "common") if (REPO / "common" / "s2s_token.py").is_file()
                      else str(SPACE / "common"))
sys.path.insert(0, str(SPACE))
sys.path.insert(0, str(HERE))

import aiohttp  # noqa: E402
from aiohttp import web, WSMsgType  # noqa: E402

from app.main import SpaceApp, STATIC  # noqa: E402
from app.sessions import Config  # noqa: E402
import s2s_token  # noqa: E402
from fake_runpod_lite import FakeLB, FakeQueue, FakeWorker  # noqa: E402

SECRET = "test-secret-0123456789-abcdefghijklmnop"
AUD = "ep-test"
KEY = "test-key"
P_SPACE, P_LB, P_Q, P_W = 27860, 27080, 27081, 27000
REC = "food_01"
BASE_ENV = {
    "S2S_MODE": "lb", "RUNPOD_ENDPOINT_ID": AUD, "RUNPOD_API_KEY": KEY, "S2S_SESSION_SECRET": SECRET,
    "RUNPOD_LB_URL": f"http://127.0.0.1:{P_LB}", "RUNPOD_API_URL": f"http://127.0.0.1:{P_Q}",
    "WAKE_POLL_S": "0.2", "WAKE_TIMEOUT_S": "20", "WAKE_HTTP_TIMEOUT_S": "5", "QUEUE_POLL_S": "0.2",
    "KEEPALIVE_S": "0.5", "KEEPALIVE_TIMEOUT_S": "2", "CLAIM_TTL_S": "30", "UPSTREAM_OPEN_TIMEOUT_S": "10",
    "MAX_CONCURRENT_CALLS": "2", "RATE_PER_IP_PER_HOUR": "1000", "MAX_CALLS_PER_DAY": "1000", "CALL_MAX_S": "300",
    "S2S_PASSCODE": "", "S2S_ALLOW_FREE_PROMPT": "0", "S2S_SELFTEST": "0",
}
RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond), info))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + str(info)) if info and not cond else ''}", flush=True)
    return bool(cond)


class Env:
    """Space + fakes for one scenario."""

    def __init__(self, mode="lb", space_env=None, workers=None):
        self.mode, self.space_env, self.workers = mode, dict(space_env or {}), workers or []
        self.front = None
        self.runner = None
        self.sa = None
        self.http = None
        self.base = f"http://127.0.0.1:{P_SPACE}"

    async def __aenter__(self):
        self.base = f"http://127.0.0.1:{P_SPACE}"
        for w in self.workers:
            await w.start()
        if not self.workers or self.mode == "local":
            self.front = None                     # an external fake (xcheck_repo_fakes.py) plays RunPod; local: none
        elif self.mode == "lb":
            self.front = await FakeLB(P_LB, self.workers, key=KEY).start()
        else:
            self.front = await FakeQueue(P_Q, self.workers[0], key=KEY).start()
        env = dict(BASE_ENV, S2S_MODE=self.mode)
        env.update(self.space_env)
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            self.sa = SpaceApp(Config.from_env())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.runner = web.AppRunner(self.sa.make_app(), handle_signals=False)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", P_SPACE).start()
        self.http = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *a):
        await self.http.close()
        await self.runner.cleanup()
        if self.front:
            await self.front.stop()
        for w in self.workers:
            await w.stop()
        await asyncio.sleep(0.05)

    async def create(self, headers=None, **body):
        body.setdefault("record_id", REC)
        body.setdefault("pairing", "g1")
        body.setdefault("seed", 1001)
        async with self.http.post(self.base + "/api/session", json=body, headers=headers or {}) as r:
            return r.status, await r.json()

    async def state(self, sid):
        async with self.http.get(f"{self.base}/api/session/{sid}") as r:
            return await r.json()

    async def wait(self, sid, want, timeout=15.0):
        seen = []
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = await self.state(sid)
            if not seen or seen[-1] != s["state"]:
                seen.append(s["state"])
            if s["state"] in want:
                return s, seen
            await asyncio.sleep(0.05)
        return s, seen

    def chat_url(self, sid, record_id=REC, pairing="g1", seed=1001):
        # the same query the client's buildURL sends (Conversation.tsx), plus sid
        q = {"text_temperature": "0.7", "text_topk": "25", "audio_temperature": "0.8", "audio_topk": "250",
             "pad_mult": "0", "text_seed": "123", "audio_seed": "456", "repetition_penalty_context": "64",
             "repetition_penalty": "1", "text_prompt": "", "voice_prompt": "", "seed": str(seed),
             "record_id": record_id, "pairing": pairing, "agent_type": "food_delivery_support", "events": "1",
             "sid": sid}
        return self.base.replace("http://", "ws://") + "/api/chat?" + "&".join(f"{k}={v}" for k, v in q.items())

    async def call(self, sid, n_audio=5, until_end=False, close_after=None, on_live=None, **kw):
        """Open the browser socket, wait for the handshake, send n_audio 0x01 frames, collect messages."""
        out = {"handshake": False, "kinds": {}, "events": [], "close_code": None, "status": None}
        try:
            ws = await self.http.ws_connect(self.chat_url(sid, **kw), max_msg_size=0)
        except aiohttp.WSServerHandshakeError as e:
            out["status"] = e.status
            return out
        t0 = time.time()
        sent = 0
        async for m in ws:
            if m.type != WSMsgType.BINARY:
                break
            k = m.data[:1]
            out["kinds"][k] = out["kinds"].get(k, 0) + 1
            if k == b"\x00":
                out["handshake"] = True
                if on_live:
                    await on_live()
            if k == b"\x07":
                out["events"].append(json.loads(m.data[1:]))
            if out["handshake"] and sent < n_audio:
                await ws.send_bytes(b"\x01OggS-browser")
                sent += 1
            if not until_end and close_after is not None and time.time() - t0 > close_after:
                await ws.close()
                break
        out["close_code"] = ws.close_code
        return out


def ends(out):
    return [e for e in out["events"] if e.get("type") == "session_end"]


# ------------------------------------------------------------------------------------------------ scenarios
async def t_lb_happy():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, load_s=1.5, body_worker_id="body-id-differs")
    async with Env("lb", workers=[w]) as e:
        st, js = await e.create()
        check("lb: POST /api/session -> 200 waking", st == 200 and js["state"] == "waking", js)
        sid = js["sid"]
        s, seen = await e.wait(sid, ("ready", "failed"))
        check("lb: cold wake waking -> ready", s["state"] == "ready" and seen[0] == "waking", seen)
        check("lb: LB answered 'no workers' while loading", e.front.log["no_worker"] >= 1, e.front.log["no_worker"])
        check("lb: proxy header worker id wins over the claim body", s.get("worker_id") == "wkr-A", s.get("worker_id"))
        check("lb: worker got exactly one claim", w.log["claims"] == 1, w.log["claims"])
        out = await e.call(sid, n_audio=5, close_after=1.5)
        check("lb: handshake 0x00 relayed", out["handshake"], out)
        check("lb: 0x01 + 0x02 + 0x07 relayed", all(out["kinds"].get(k) for k in (b"\x01", b"\x02", b"\x07")),
              out["kinds"])
        await asyncio.sleep(0.3)
        check("lb: browser audio reached the worker", w.log["audio_in"] >= 5, w.log["audio_in"])
        h = e.front.log["ws_headers"] or {}
        check("lb: ws upstream had Bearer", h.get("Authorization") == f"Bearer {KEY}", h.get("Authorization"))
        check("lb: ws upstream strict-pinned", h.get("X-Runpod-Worker-Id") == "strict wkr-A", h.get("X-Runpod-Worker-Id"))
        wh = w.log["ws_headers"] or {}
        tok = wh.get("X-S2S-Token", "")
        try:
            p = s2s_token.verify(SECRET, tok, mode="lb", aud=AUD)
            ok = p["sid"] == sid and p["record_id"] == REC and p["pairing"] == "g1" and p["seed"] == 1001
        except Exception as ex:
            ok, p = False, repr(ex)
        check("lb: worker got a valid token bound to sid + config", ok, p)
        q = w.log["ws_query"] or {}
        check("lb: sid/token not in the upstream query; DEP1 params kept",
              "sid" not in q and "token" not in q and q.get("events") == "1" and q.get("text_prompt") == "", q)
        s = await e.state(sid)
        check("lb: state ended client_closed after the browser closed",
              s["state"] == "ended" and s.get("end_reason") == "client_closed", s)
        async with e.http.get(f"{e.base}/metrics?sid={sid}") as r:
            m = await r.json()
        check("lb: /metrics?sid= returns the last 0x07 metrics", m.get("type") == "metrics" and "step_ms" in m, m)
        check("lb: worker recorded client_closed", w.log["sessions"] and w.log["sessions"][-1]["end_reason"] == "client_closed",
              w.log["sessions"])
        out2 = await e.call(sid)
        check("lb: reconnect on an ended sid -> 409 before upgrade", out2["status"] == 409, out2)
        async with e.http.get(f"{e.base}/api/diag?sid={sid}") as r:
            check("lb: /api/diag on an ended sid -> 409", r.status == 409, r.status)


async def t_lb_time_limit_and_keepalive():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w], space_env={"CALL_MAX_S": "2"}) as e:
        st, js = await e.create()
        sid = js["sid"]
        s, _ = await e.wait(sid, ("ready", "failed"))
        diag = {}

        async def on_live():
            async with e.http.get(f"{e.base}/api/diag?sid={sid}") as r:
                diag["a"] = (r.status, await r.json())
            async with e.http.get(f"{e.base}/api/diag?sid={sid}") as r:
                diag["b"] = r.status
        out = await e.call(sid, until_end=True, on_live=on_live)
        ev = ends(out)
        check("lb: worker session_end time_limit passed through", ev and ev[-1]["reason"] == "time_limit", ev)
        check("lb: browser socket closed normally (1000)", out["close_code"] == 1000, out["close_code"])
        s = await e.state(sid)
        check("lb: state ended time_limit", s["state"] == "ended" and s.get("end_reason") == "time_limit", s)
        check("lb: keepalive sent strict-pinned GET /status during the call", w.log["status_pinned"] >= 2,
              w.log["status_pinned"])
        a = diag.get("a")
        check("lb: /api/diag during the call -> 200 with RTT", a and a[0] == 200 and "space_to_worker_ms" in a[1], a)
        check("lb: /api/diag rate-limited (2nd within 10 s -> 429)", diag.get("b") == 429, diag.get("b"))


async def t_lb_busy_then_ready():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, n_409=4)
    async with Env("lb", workers=[w]) as e:
        st, js = await e.create()
        s, seen = await e.wait(js["sid"], ("ready", "failed"))
        check("lb: claim 409s -> busy -> ready", "busy" in seen and s["state"] == "ready", seen)


async def t_lb_limits():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, load_s=100)
    async with Env("lb", workers=[w], space_env={"MAX_CONCURRENT_CALLS": "1", "S2S_PASSCODE": "open-sesame"}) as e:
        st, js = await e.create()
        check("passcode missing -> 403", st == 403 and js.get("error") == "passcode", (st, js))
        st, js = await e.create(passcode="open-sesame")
        check("passcode ok -> 200", st == 200, (st, js))
        sid = js["sid"]
        st2, js2 = await e.create(passcode="open-sesame")
        check("capacity: 2nd session with MAX_CONCURRENT_CALLS=1 -> 429 capacity",
              st2 == 429 and js2.get("error") == "capacity", (st2, js2))
        st3, js3 = await e.create(passcode="open-sesame", record_id="no_such_record")
        check("unknown record -> 400", st3 == 400, (st3, js3))
        st4, js4 = await e.create(passcode="open-sesame", record_id="", pairing="g1")
        check("no record with free prompts off -> 400", st4 == 400, (st4, js4))
        out = await e.call(sid)
        check("ws on a waking sid -> 409", out["status"] == 409, out)
        await asyncio.sleep(0.5)
        task = e.sa.table.get(sid).wake_task
        async with e.http.delete(f"{e.base}/api/session/{sid}") as r:
            d = await r.json()
        await asyncio.sleep(0.1)
        check("DELETE while waking -> ended cancelled, wake task cancelled",
              d["state"] == "ended" and d.get("end_reason") == "cancelled" and task.done(), (d, task.done()))
        st5, _ = await e.create(passcode="open-sesame")
        check("slot freed after DELETE -> 200", st5 == 200, st5)
        e.sa.table.cfg.max_concurrent = 10
        e.sa.table.cfg.rate_per_ip_per_hour = 2
        st6, js6 = await e.create(passcode="open-sesame", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})
        st7, js7 = await e.create(passcode="open-sesame", headers={"X-Forwarded-For": "9.9.9.9"})
        st8, js8 = await e.create(passcode="open-sesame", headers={"X-Forwarded-For": "9.9.9.9"})
        check("per-IP rate: 3rd creation from one XFF IP in an hour -> 429 rate_limited",
              st6 == 200 and st7 == 200 and st8 == 429 and js8.get("error") == "rate_limited", (st6, st7, st8, js8))


async def t_lb_mismatch_and_double():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w]) as e:
        st, js = await e.create(pairing="g3", seed=-1)
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        out = await e.call(sid, pairing="g2", seed=-1)
        check("ws with a different pairing -> 400 before upgrade", out["status"] == 400, out)
        out = await e.call(sid, pairing="g3", seed=5)
        check("ws with a different seed -> 400 before upgrade", out["status"] == 400, out)
        res = {}

        async def second():
            o = await e.call(sid, pairing="g3", seed=-1)
            res["second"] = o["status"]
        out = await e.call(sid, pairing="g3", seed=-1, close_after=0.6, on_live=second)
        check("seed -1 session connects (token seed -1 == query seed -1)", out["handshake"], out)
        check("second socket on an in_call sid -> 409", res.get("second") == 409, res)


async def t_lb_worker_lost():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w]) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))

        async def killer():
            await asyncio.sleep(0.8)
            await w.kill()

        async def on_live():
            asyncio.ensure_future(killer())
        out = await e.call(sid, until_end=True, on_live=on_live)
        ev = ends(out)
        check("killed worker -> synthetic session_end worker_lost", ev and ev[-1]["reason"] == "worker_lost"
              and ev[-1].get("source") == "space", ev)
        check("killed worker -> browser close code 1011", out["close_code"] == 1011, out["close_code"])
        s = await e.state(sid)
        check("killed worker -> state ended worker_lost", s.get("end_reason") == "worker_lost", s)


async def t_lb_close_4409():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, close_after_upgrade=4409)
    async with Env("lb", workers=[w]) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        out = await e.call(sid, until_end=True)
        ev = ends(out)
        check("worker close 4409 after upgrade -> relay_error (not worker_lost) + 1011",
              ev and ev[-1]["reason"] == "relay_error" and "busy" in ev[-1].get("detail", "") and out["close_code"] == 1011,
              (ev, out["close_code"]))


async def t_lb_space_cap():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, ignore_cap=True)
    async with Env("lb", workers=[w], space_env={"CALL_MAX_S": "1"}) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        t0 = time.time()
        out = await e.call(sid, until_end=True)
        dt = time.time() - t0
        ev = ends(out)
        check("Space cap (CALL_MAX_S+15) closes a call the worker does not end",
              ev and ev[-1]["reason"] == "time_limit" and 15 < dt < 20, (ev, round(dt, 1)))


async def t_lb_claim_ttl_and_bad_secret():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w], space_env={"CLAIM_TTL_S": "1.5"}) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        s, _ = await e.wait(sid, ("ended",), timeout=6)
        await asyncio.sleep(0.3)
        check("ready but never connected -> ended claim_expired + release sent",
              s.get("end_reason") == "claim_expired" and w.log["releases"] == 1 and w.claim is None,
              (s, w.log["releases"], w.claim))
    w2 = FakeWorker(P_W, "wkr-B", "another-secret-0123456789-abcdefghijk", AUD)
    async with Env("lb", workers=[w2]) as e:
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("ready", "failed"))
        check("secret mismatch -> claim 401 -> failed with a hint", s["state"] == "failed" and "S2S_SESSION_SECRET" in s["detail"], s)
    w3 = FakeWorker(P_W, "wkr-C", SECRET, AUD)
    async with Env("lb", workers=[w3], space_env={"RUNPOD_API_KEY": "wrong-key"}) as e:
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("ready", "failed"))
        check("wrong RunPod key -> failed with a hint", s["state"] == "failed" and "RUNPOD_API_KEY" in s["detail"], s)


async def t_lb_wake_timeout():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, load_s=100)
    async with Env("lb", workers=[w], space_env={"WAKE_TIMEOUT_S": "1.5"}) as e:
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("failed",), timeout=6)
        check("no worker within WAKE_TIMEOUT_S -> failed", s["state"] == "failed", s)


async def t_queue():
    w = FakeWorker(P_W, "pod-q1", SECRET, AUD, mode="queue", load_s=1.0)
    async with Env("queue", workers=[w]) as e:
        st, js = await e.create(pairing="g2")
        sid = js["sid"]
        s, seen = await e.wait(sid, ("ready", "failed"))
        check("queue: /run -> waking -> ready", s["state"] == "ready", (seen, s))
        run = e.front.log["run"][-1]
        check("queue: /run input has sid+cfg and no token; policy set",
              run["input"] == {"sid": sid, "record_id": REC, "pairing": "g2", "seed": 1001}
              and run["policy"]["executionTimeout"] == 900000, run)
        check("queue: detail showed initializing/queued", any(t[0] == "waking" for t in s["timeline"]), s["timeline"])
        out = await e.call(sid, pairing="g2", close_after=1.0)
        check("queue: relayed call (handshake + audio)", out["handshake"] and out["kinds"].get(b"\x01"), out["kinds"])
        wh = w.log["ws_headers"] or {}
        try:
            p = s2s_token.verify(SECRET, wh.get("X-S2S-Token", ""), mode="queue", aud=AUD)
            ok = p["sid"] == sid
        except Exception as ex:
            ok, p = False, repr(ex)
        check("queue: ws has a queue-mode token and no Authorization", ok and "Authorization" not in wh, (p, list(wh)))
        await asyncio.sleep(0.6)
        job = e.front.jobs[list(e.front.jobs)[-1]]
        check("queue: job COMPLETED after the call", job["status"] == "COMPLETED", job["status"])
        s = await e.state(sid)
        check("queue: state ended client_closed", s["state"] == "ended" and s.get("end_reason") == "client_closed", s)
    w2 = FakeWorker(P_W, "pod-q2", SECRET, AUD, mode="queue", load_s=100)
    async with Env("queue", workers=[w2]) as e:
        st, js = await e.create()
        sid = js["sid"]
        await asyncio.sleep(1.0)
        async with e.http.delete(f"{e.base}/api/session/{sid}") as r:
            d = await r.json()
        await asyncio.sleep(0.3)
        jid = list(e.front.jobs)[-1]
        check("queue: DELETE while waking -> /cancel/{job} and job CANCELLED",
              d["state"] == "ended" and jid in e.front.log["cancel"] and e.front.jobs[jid]["status"] == "CANCELLED",
              (d, e.front.log["cancel"], e.front.jobs[jid]["status"]))
    w3 = FakeWorker(P_W, "pod-q3", SECRET, AUD, mode="queue", ready_without_port=True)
    async with Env("queue", workers=[w3]) as e:
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("failed", "ready"))
        await asyncio.sleep(0.3)
        jid = list(e.front.jobs)[-1]
        check("queue: ready without tcp_port -> failed AND the job is cancelled (no billed leak)",
              s["state"] == "failed" and jid in e.front.log["cancel"], (s, e.front.log["cancel"]))


async def t_misc():
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w]) as e:
        async with e.http.ws_connect(e.base.replace("http", "ws") + "/ws-echo") as ws:
            await ws.send_str("ping-text")
            a = await ws.receive()
            await ws.send_bytes(b"\x01\x02\x03")
            b = await ws.receive()
        check("/ws-echo echoes text and binary", a.data == "ping-text" and b.data == b"\x01\x02\x03", (a.data, b.data))
        async with e.http.get(e.base + "/api/config") as r:
            c = await r.json()
        check("/api/config", c.get("mode") == "lb" and c.get("passcode_required") is False and c.get("ok") is True
              and c.get("allow_free_prompt") is False, c)
        async with e.http.get(e.base + "/api/records") as r:
            recs = await r.json()
        async with e.http.get(e.base + f"/api/records/{REC}") as r:
            rec = await r.json()
        check("/api/records + /api/records/{id} with session_configs",
              len(recs) > 10 and set(rec.get("session_configs", {})) == {"g1", "g2", "g3", "g4"}, len(recs))
        async with e.http.get(e.base + "/api/samples") as r:
            check("/api/samples -> list", isinstance(await r.json(), list), r.status)
        async with e.http.get(e.base + "/healthz") as r:
            check("/healthz", r.status == 200, r.status)
        async with e.http.get(e.base + "/api/selftest/outbound?host=127.0.0.1&port=22") as r:
            check("selftest route off by default -> 404", r.status == 404, r.status)
        async with e.http.get(e.base + "/") as r:
            html = await r.text()
        if (STATIC / "index.html").is_file():
            check("GET / serves the built client", r.status == 200 and "<div id=\"root\"" in html, html[:200])
            assets = sorted((STATIC / "assets").glob("index-*.js"))
            if assets:
                async with e.http.get(e.base + "/assets/" + assets[0].name) as r2:
                    check("a hashed client asset -> 200", r2.status == 200, r2.status)
        else:
            check("GET / placeholder (client not built)", r.status == 200, r.status)


async def t_review_space():
    """Regressions for the review fixes REVIEW-space-1..3 (DECISIONS.md)."""
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w], space_env={"MAX_CONCURRENT_CALLS": "1"}) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        async with e.http.get(e.chat_url(sid).replace("ws://", "http://")) as r:
            code, body = r.status, await r.json(content_type=None)
        s = await e.state(sid)
        check("R1: plain GET /api/chat (no upgrade) -> 400 websocket_required, session stays ready",
              code == 400 and body.get("error") == "websocket_required" and s["state"] == "ready", (code, body, s))
        out = await e.call(sid, n_audio=3, close_after=0.8)
        check("R1: a real websocket on the same sid still works afterwards", out["handshake"], out)
        s = await e.state(sid)
        check("R1: ... and ends client_closed (slot freed)", s["state"] == "ended", s)
        st, js = await e.create()
        check("R1: capacity slot is free again", st == 200, (st, js))
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        orig = e.sa._relay

        async def boom(*a, **k):
            raise RuntimeError("relay crash (test)")
        e.sa._relay = boom
        rel0 = w.log["releases"]
        out = await e.call(sid, n_audio=0, close_after=0.2)
        e.sa._relay = orig
        await asyncio.sleep(0.3)
        s = await e.state(sid)
        check("R1: a crash inside the relay ends the session (relay_error) and releases the worker",
              s["state"] == "ended" and s.get("end_reason") == "relay_error" and w.log["releases"] > rel0,
              (s, w.log["releases"], rel0))
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        orig_open = e.sa.up.open_ws

        async def slow_open(sess, q):
            await asyncio.sleep(1.0)
            return await orig_open(sess, q)
        e.sa.up.open_ws = slow_open
        res = {}

        async def deleter():
            await asyncio.sleep(0.4)
            async with e.http.delete(f"{e.base}/api/session/{sid}") as r:
                res["del"] = r.status
        dt = asyncio.ensure_future(deleter())
        out = await e.call(sid, n_audio=0, until_end=True)
        await dt
        e.sa.up.open_ws = orig_open
        s = await e.state(sid)
        check("R1: DELETE during the upstream open ends the call when the open returns",
              res.get("del") == 200 and s["state"] == "ended" and not out["handshake"], (res, s, out["kinds"]))
    w2 = FakeWorker(P_W, "wkr-B", SECRET, AUD, close_after_upgrade=4409)
    async with Env("lb", workers=[w2]) as e:
        st, js = await e.create()
        sid = js["sid"]
        await e.wait(sid, ("ready", "failed"))
        await e.call(sid, until_end=True)
        await asyncio.sleep(0.3)
        check("R2: relay_error after the upgrade -> claim released on the worker", w2.log["releases"] >= 1,
              w2.log["releases"])
    w3 = FakeWorker(P_W, "wkr-C", SECRET, AUD, load_s=100)
    async with Env("lb", workers=[w3], space_env={"ABANDON_S": "1"}) as e:
        st, js = await e.create()
        sid = js["sid"]
        task = e.sa.table.get(sid).wake_task
        await asyncio.sleep(3.5)                       # nobody polls GET /api/session
        s = e.sa.table.get(sid)
        check("R3: a waking session nobody polls is cancelled (abandoned) and its wake stops",
              s.state == "ended" and s.end_reason == "abandoned" and task.done(), (s.state, s.end_reason))
        st, js = await e.create()
        sid = js["sid"]
        for _ in range(8):                             # a browser that keeps polling is never abandoned
            await asyncio.sleep(0.4)
            await e.state(sid)
        s = await e.state(sid)
        check("R3: a polled waking session is kept", s["state"] == "waking", s)


async def t_filler_toggle():
    """2026-10-06 D-TOGGLE port: /api/filler on the Space; the mode reaches the worker as the query param turn_fill at
    call start and as an in-band 0x08 control frame during a call; a browser-supplied turn_fill is not forwarded."""
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD)
    async with Env("lb", workers=[w]) as e:
        async with e.http.get(e.base + "/api/filler") as r:
            check("filler: GET default -> ticker", (await r.json()).get("mode") == "ticker", r.status)
        async with e.http.post(e.base + "/api/filler?mode=loud") as r:
            check("filler: POST bad mode -> 400", r.status == 400, r.status)
        async with e.http.post(e.base + "/api/filler?mode=off") as r:
            j = await r.json()
            check("filler: POST off -> off, 0 live calls", j.get("mode") == "off" and j.get("live_calls_updated") == 0, j)
        async with e.http.get(e.base + "/api/config") as r:
            check("filler: /api/config shows turn_fill", (await r.json()).get("turn_fill") == "off", r.status)
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("ready", "failed"))
        toggled = {}

        async def on_live():
            async with e.http.post(e.base + "/api/filler?mode=ticker") as r2:
                toggled.update(await r2.json())
        out = await e.call(js["sid"], close_after=1.2, on_live=on_live)
        await asyncio.sleep(0.3)
        check("filler: call start carries turn_fill=off upstream", (w.log["ws_query"] or {}).get("turn_fill") == "off",
              w.log["ws_query"])
        check("filler: mid-call POST ticker -> 1 live call updated", toggled.get("live_calls_updated") == 1, toggled)
        check("filler: worker got the 0x08 control frame", w.log["controls"] == [{"type": "turn_fill", "mode": "ticker"}],
              w.log["controls"])
        check("filler: the call itself was unaffected", out["handshake"] and w.log["audio_in"] >= 1, out)
        # a browser-supplied turn_fill is dropped; the Space's mode (now ticker) is sent
        st, js = await e.create()
        await e.wait(js["sid"], ("ready", "failed"))
        url = e.chat_url(js["sid"]) + "&turn_fill=off"
        ws = await e.http.ws_connect(url, max_msg_size=0)
        async for m in ws:
            if m.data[:1] == b"\x00":
                break
        await ws.close()
        await asyncio.sleep(0.3)
        check("filler: browser turn_fill ignored, Space mode forwarded", w.log["ws_query"].get("turn_fill") == "ticker",
              w.log["ws_query"])
    async with Env("lb", workers=[FakeWorker(P_W, "wkr-A", SECRET, AUD)],
                   space_env={"S2S_TURN_FILL_DEFAULT": "off"}) as e:
        async with e.http.get(e.base + "/api/filler") as r:
            check("filler: S2S_TURN_FILL_DEFAULT=off -> GET off", (await r.json()).get("mode") == "off", r.status)
    async with Env("lb", workers=[FakeWorker(P_W, "wkr-A", SECRET, AUD)],
                   space_env={"S2S_TURN_FILL_DEFAULT": "hmm"}) as e:
        check("filler: bad S2S_TURN_FILL_DEFAULT is a config problem", any("TURN_FILL" in p for p in e.sa.problems),
              e.sa.problems)


async def t_endpoint_health():
    """2026-10-06: GET /api/endpoint_health?passcode= passes RunPod's /v2/{id}/health through (no worker wake)."""
    w = FakeWorker(P_W, "wkr-A", SECRET, AUD, mode="queue")
    async with Env("queue", workers=[w], space_env={"S2S_PASSCODE": "pc-123"}) as e:
        async with e.http.get(e.base + "/api/endpoint_health?passcode=nope") as r:
            check("endpoint_health: wrong passcode -> 403", r.status == 403, r.status)
        async with e.http.get(e.base + "/api/endpoint_health?passcode=pc-123") as r:
            j = await r.json()
            check("endpoint_health: passcode -> RunPod /health body passed through",
                  r.status == 200 and j.get("runpod_health_status") == 200
                  and j["runpod_health"]["workers"]["idle"] == 1 and j.get("turn_fill") == "ticker", j)
        async with e.http.get(e.base + "/api/endpoint_health?passcode=pc-123") as r:
            check("endpoint_health: rate limited (5 s)", r.status == 429, r.status)
        check("endpoint_health: no RunPod key in the answer", KEY not in json.dumps(j))
    async with Env("lb", workers=[FakeWorker(P_W, "wkr-A", SECRET, AUD)]) as e:
        async with e.http.get(e.base + "/api/endpoint_health") as r:
            check("endpoint_health: off without S2S_PASSCODE -> 404", r.status == 404, r.status)


async def t_local():
    """D-LOCAL (2026-10-06): S2S_MODE=local drives the worker directly at S2S_WORKER_URL (no RunPod key/endpoint, no
    Bearer, no worker pin, no keepalive); the token is still an LB-mode token (the worker checks mode=lb), aud 'local'."""
    w = FakeWorker(P_W, "local-gpu", SECRET, "local", load_s=0.8)
    env = {"RUNPOD_ENDPOINT_ID": "", "RUNPOD_API_KEY": "", "S2S_WORKER_URL": f"http://127.0.0.1:{P_W}",
           "RATE_PER_IP_PER_HOUR": "", "MAX_CALLS_PER_DAY": "", "KEEPALIVE_S": "0.3"}
    async with Env("local", workers=[w], space_env=env) as e:
        check("local: no config problems without RunPod key / endpoint", e.sa.problems == [], e.sa.problems)
        check("local: audience defaults to 'local'", e.sa.cfg.audience == "local", e.sa.cfg.audience)
        check("local: per-IP / per-day brakes relaxed by default",
              e.sa.cfg.rate_per_ip_per_hour >= 1000 and e.sa.cfg.max_calls_per_day >= 1000,
              (e.sa.cfg.rate_per_ip_per_hour, e.sa.cfg.max_calls_per_day))
        async with e.http.get(e.base + "/api/config") as r:
            c = await r.json()
        check("local: /api/config mode local, ok", c.get("mode") == "local" and c.get("ok") is True, c)
        st, js = await e.create()
        s, seen = await e.wait(js["sid"], ("ready", "failed"))
        check("local: waking -> ready (claim straight to the worker)", s["state"] == "ready" and w.log["claims"] == 1,
              (seen, s))
        out = await e.call(js["sid"], n_audio=4, close_after=1.2)
        await asyncio.sleep(0.3)
        check("local: relayed call (handshake + 0x07 + audio in)", out["handshake"] and out["kinds"].get(b"\x07")
              and w.log["audio_in"] >= 4, (out["kinds"], w.log["audio_in"]))
        wh = w.log["ws_headers"] or {}
        try:
            p = s2s_token.verify(SECRET, wh.get("X-S2S-Token", ""), mode="lb", aud="local")
            tok_ok = p["sid"] == js["sid"] and p["record_id"] == REC
        except Exception as ex:
            tok_ok = repr(ex)
        check("local: worker got an LB-mode token for this sid (aud local)", tok_ok is True, tok_ok)
        check("local: no Authorization / worker pin sent to the worker",
              "Authorization" not in wh and "X-Runpod-Worker-Id" not in wh, list(wh))
        check("local: no keepalive polls during the call", w.log["status_pinned"] == 0, w.log["status_pinned"])
        s = await e.state(js["sid"])
        check("local: ended client_closed", s["state"] == "ended" and s.get("end_reason") == "client_closed", s)
    async with Env("local", workers=[FakeWorker(P_W, "local-gpu", SECRET, "local")],
                   space_env=dict(env, S2S_SESSION_SECRET="short")) as e:
        check("local: a short secret is still a config problem", any("SESSION_SECRET" in p for p in e.sa.problems),
              e.sa.problems)
    async with Env("local", workers=[], space_env=dict(env, S2S_WORKER_URL=f"http://127.0.0.1:{P_W}",
                                                       WAKE_TIMEOUT_S="1.5")) as e:
        st, js = await e.create()
        s, _ = await e.wait(js["sid"], ("failed",), timeout=6)
        check("local: no worker running -> failed with a local hint", s["state"] == "failed"
              and "local worker" in s.get("detail", ""), s)


async def t_script():
    """D-SCRIPT-PANEL (2026-10-06): GET /api/script/{record_id}?pairing=gN from common/data/scripts_v4.json."""
    async with Env("lb", workers=[FakeWorker(P_W, "wkr-A", SECRET, AUD)]) as e:
        async with e.http.get(e.base + "/api/script/food_08?pairing=g1") as r:
            j = await r.json()
        turns = j.get("turns") or []
        check("script: food_08 g1 -> 200, available, split train, agent Neha",
              r.status == 200 and j.get("available") is True and j.get("split") == "train"
              and j.get("agent_name") == "Neha" and j.get("call_id") == "food_08_g1", {k: j.get(k) for k in ("available", "split", "agent_name")})
        check("script: turns have speaker/text/tags; greeting first, customer lines present",
              turns and turns[0]["speaker"] == "agent" and "greeting" in turns[0]["tags"]
              and any(t["speaker"] == "customer" for t in turns) and all(set(t) == {"speaker", "text", "tags"} for t in turns),
              turns[:2])
        check("script: check_line + confirm_write tags and the expected write",
              any("check_line" in t["tags"] for t in turns) and any("confirm_write" in t["tags"] for t in turns)
              and j.get("writes") and j["writes"][0]["tool"] == "change_delivery_address", j.get("writes"))
        async with e.http.get(e.base + "/api/script/food_08") as r:
            check("script: pairing defaults to g1", (await r.json()).get("pairing") == "g1", r.status)
        async with e.http.get(e.base + "/api/script/food_07?pairing=g3") as r:
            check("script: food_07 g3 is the held-out test call", (await r.json()).get("split") == "test", r.status)
        async with e.http.get(e.base + "/api/script/food_07?pairing=g1") as r:
            check("script: food_07 g1 -> test_scenario (never trained on)", (await r.json()).get("split") == "test_scenario",
                  r.status)
        async with e.http.get(e.base + "/api/script/food_18?pairing=g1") as r:
            j = await r.json()
            check("script: dropped call food_18 g1 -> 200 available=false, no turns",
                  r.status == 200 and j.get("available") is False and j.get("turns") == [] and "no script" in j.get("detail", ""), j)
        async with e.http.get(e.base + "/api/script/food_01?pairing=g9") as r:
            check("script: bad pairing -> 400", r.status == 400, r.status)
        async with e.http.get(e.base + "/api/script/bank_01?pairing=g1") as r:
            check("script: non-demo record (bank) -> 404", r.status == 404, r.status)
        async with e.http.get(e.base + "/api/script/nope?pairing=g1") as r:
            check("script: unknown record -> 404", r.status == 404, r.status)
        async with e.http.get(e.base + "/api/records") as r:
            recs = await r.json()
        import app.main as am
        data = am.load_scripts()
        miss = [f"{x['record_id']}_{g}" for x in recs for g in ("g1", "g2", "g3", "g4")
                if f"{x['record_id']}_{g}" not in data["scripts"]]
        check("script: every demo record x pairing has a script except the 3 dropped calls",
              sorted(miss) == sorted(data["missing"]) == ["food_18_g1", "sub_12_g2", "sub_12_g3"], miss)


ALL = [t_lb_happy, t_lb_time_limit_and_keepalive, t_lb_busy_then_ready, t_lb_limits, t_lb_mismatch_and_double,
       t_lb_worker_lost, t_lb_close_4409, t_lb_space_cap, t_lb_claim_ttl_and_bad_secret, t_lb_wake_timeout, t_queue, t_misc,
       t_review_space, t_filler_toggle, t_endpoint_health, t_local, t_script]


async def main(sel):
    for t in ALL:
        if sel and sel not in t.__name__:
            continue
        print(f"--- {t.__name__}", flush=True)
        try:
            await asyncio.wait_for(t(), 60)
        except Exception as ex:
            traceback.print_exc()
            check(f"{t.__name__} raised", False, repr(ex))
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    return n_fail


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    sys.exit(1 if asyncio.run(main(sel)) else 0)
