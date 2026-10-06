#!/usr/bin/env python
"""Space builder's own stand-in for RunPod + a GPU worker (test only; no GPU, no RunPod, no model).

The repo-wide fake (tests/fake_runpod.py, builder C) drives the REAL worker_server --mock. This file is smaller and
self-contained so the Space backend can be tested on its own:

  FakeWorker  the worker's public contract (DESIGN 3.1): /ping 204->200, /status, POST /session/claim|release
              (X-S2S-Token verified with common/s2s_token.py), POST /internal/claim (queue), ws /api/chat with the
              DEP1 protocol (prompt phase -> 0x00 -> 0x07 session -> 0x01/0x02/0x07 metrics ... -> 0x07 session_end).
              Close codes / HTTP refusals as in 3.1. kill() drops the socket without session_end (worker_lost).
  FakeLB      the LB proxy (DESIGN 7): requires Authorization: Bearer <key>; routes only to workers whose /ping is 200;
              "no workers available" 503 after a short hold; sets X-Runpod-Worker-Id; honours strict <id> (404
              affinity_worker_gone); proxies websockets.
  FakeQueue   /run, /status/{id}, /cancel/{id}: runs a handler thread-free coroutine that waits for the worker,
              claims via /internal/claim and publishes progress {state, public_ip, tcp_port, worker_id, sid}.

CLI (for the browser e2e; with --opus it sends real Opus pages via sphn, run it with a venv that has sphn):
  python fake_runpod_lite.py --mode lb --lb-port 27080 --worker-port 27000 --secret ... --aud ep-test [--opus]
"""
import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web, WSMsgType

HERE = Path(__file__).resolve().parent
for c in (os.environ.get("S2S_COMMON"), HERE.parent / "common", HERE.parent.parent / "common"):
    if c and (Path(c) / "s2s_token.py").is_file():
        sys.path.insert(0, str(c))
        break
import s2s_token  # noqa: E402

log = logging.getLogger("fake")
RESET = (ConnectionResetError, getattr(aiohttp, "ClientConnectionResetError", ConnectionResetError))
HOP = {"host", "authorization", "content-length", "transfer-encoding", "connection", "upgrade",
       "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions", "sec-websocket-protocol"}


def ev(obj) -> bytes:
    return b"\x07" + json.dumps(obj).encode()


class OpusSource:
    """Real Ogg/Opus pages (a quiet 220 Hz tone) if sphn is importable, else fake bytes (fine for python tests)."""

    def __init__(self, real: bool):
        self.w = None
        self.n = 0
        if real:
            import numpy as np
            import sphn
            self.np = np
            self.w = sphn.OpusStreamWriter(24000)

    def frame(self) -> bytes:
        self.n += 1
        if self.w is None:
            return b"OggS-fake-%d" % self.n
        t = (self.n * 1920 + self.np.arange(1920)) / 24000.0
        self.w.append_pcm((0.05 * self.np.sin(2 * math.pi * 220 * t)).astype(self.np.float32))
        return self.w.read_bytes()


class FakeWorker:
    def __init__(self, port, worker_id, secret, aud, mode="lb", load_s=0.0, prompt_s=0.3, claim_ttl_s=90.0,
                 call_max_s=300, ignore_cap=False, n_409=0, body_worker_id=None, opus=False, close_after_upgrade=None,
                 ready_without_port=False):
        self.port, self.worker_id, self.secret, self.aud, self.mode = port, worker_id, secret, aud, mode
        self.t_start = time.time()
        self.load_s, self.prompt_s, self.claim_ttl_s, self.call_max_s = load_s, prompt_s, claim_ttl_s, call_max_s
        self.ignore_cap, self.n_409, self.body_worker_id, self.opus = ignore_cap, n_409, body_worker_id, opus
        self.close_after_upgrade = close_after_upgrade      # e.g. 4409: post-upgrade refusal (DESIGN 3.1)
        self.ready_without_port = ready_without_port        # queue: publish ready with no tcp_port
        self.claim = None              # (sid, until, cfg)
        self.busy_sid = None
        self.replay = set()
        self.ws = None
        self.up = True                 # False after kill(): the LB treats the worker as gone
        self.log = {"claims": 0, "releases": 0, "status": 0, "status_pinned": 0, "ws_headers": None,
                    "ws_query": None, "audio_in": 0, "sessions": [], "refusals": [], "controls": []}
        self.runner = None
        self.ended = asyncio.Event()

    @property
    def loading(self):
        return time.time() - self.t_start < self.load_s

    def state(self):
        if self.loading:
            return "loading"
        if self.busy_sid:
            return "busy"
        if self.claim and time.time() > self.claim[1]:
            self.claim = None
        return "claimed" if self.claim else "idle"

    def _verify(self, request):
        tok = request.headers.get("X-S2S-Token") or request.query.get("token")
        if not tok:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "bad_token"}), content_type="application/json")
        try:
            return s2s_token.verify(self.secret, tok, mode=self.mode, aud=self.aud)
        except s2s_token.TokenError as e:
            code = "expired" if e.code == "expired" else "bad_token"
            raise web.HTTPUnauthorized(text=json.dumps({"error": code}), content_type="application/json")

    async def ping(self, request):
        return web.Response(status=204) if self.loading else web.json_response({"status": "ok", "state": self.state()})

    async def status(self, request):
        self.log["status"] += 1
        if request.headers.get("X-Runpod-Worker-Id", "").startswith("strict "):
            self.log["status_pinned"] += 1
        return web.json_response({"state": self.state(), "worker_id": self.worker_id, "gpu": "FAKE",
                                  "active": {"sid": self.busy_sid} if self.busy_sid else None, "version": "fake"})

    def _do_claim(self, sid, cfg):
        st = self.state()
        if st == "loading":
            return web.json_response({"error": "loading"}, status=503)
        if self.n_409 > 0:
            self.n_409 -= 1
            return web.json_response({"error": "busy"}, status=409)
        if st == "busy" or (st == "claimed" and self.claim[0] != sid):
            return web.json_response({"error": "busy"}, status=409)
        self.claim = (sid, time.time() + self.claim_ttl_s, cfg)
        self.log["claims"] += 1
        return web.json_response({"worker_id": self.body_worker_id or self.worker_id, "sid": sid,
                                  "claim_ttl_s": self.claim_ttl_s})

    async def session_claim(self, request):
        p = self._verify(request)
        return self._do_claim(p["sid"], {k: p[k] for k in ("record_id", "pairing", "seed")})

    async def internal_claim(self, request):
        b = await request.json()
        return self._do_claim(b["sid"], {k: b.get(k) for k in ("record_id", "pairing", "seed")})

    async def session_release(self, request):
        p = self._verify(request)
        self.log["releases"] += 1
        rel = bool(self.claim and self.claim[0] == p["sid"] and not self.busy_sid)
        if rel:
            self.claim = None
        return web.json_response({"released": rel})

    async def chat(self, request):
        p = self._verify(request)
        q = request.query
        self.log["ws_headers"] = dict(request.headers)
        self.log["ws_query"] = dict(q)
        if self.loading:
            return web.json_response({"error": "loading"}, status=503)
        if p["sid"] in self.replay:
            self.log["refusals"].append("replayed")
            return web.json_response({"error": "replayed"}, status=401)
        if self.busy_sid or not self.claim or self.claim[0] != p["sid"]:
            self.log["refusals"].append("busy")
            return web.json_response({"error": "busy"}, status=409)
        seed = q.get("seed")
        if (q.get("record_id") or None, q.get("pairing") or None, int(seed) if seed else None) != \
                (p["record_id"], p["pairing"], p["seed"]):
            self.log["refusals"].append("cfg_mismatch")
            return web.json_response({"error": "cfg_mismatch"}, status=401)
        self.replay.add(p["sid"])
        self.busy_sid, self.claim = p["sid"], None
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        self.ws, self.req = ws, request
        if self.close_after_upgrade:
            await ws.close(code=self.close_after_upgrade, message=b"refused")
            self.busy_sid, self.ws = None, None
            self.log["refusals"].append(f"close {self.close_after_upgrade}")
            return ws
        cap = min(p["max_s"], self.call_max_s)
        reason = "client_closed"
        frames = 0
        src = OpusSource(self.opus)
        closed = asyncio.Event()

        async def recv():
            async for m in ws:
                if m.type == WSMsgType.BINARY and m.data[:1] == b"\x01":
                    self.log["audio_in"] += 1
                elif m.type == WSMsgType.BINARY and m.data[:1] == b"\x08":     # 2026-10-06: in-band control
                    self.log["controls"].append(json.loads(m.data[1:]))
                elif m.type not in (WSMsgType.BINARY, WSMsgType.TEXT):
                    break
            closed.set()

        rt = asyncio.ensure_future(recv())
        try:
            await asyncio.sleep(self.prompt_s)
            await ws.send_bytes(b"\x00")
            await ws.send_bytes(ev({"type": "session", "session_id": p["sid"][:8], "record_id": p["record_id"],
                                    "pairing": p["pairing"], "seed": p["seed"], "context_frames_left": 2770}))
            t0 = time.time()
            while not closed.is_set():
                frames += 1
                await ws.send_bytes(b"\x01" + src.frame())
                if frames % 6 == 0:
                    await ws.send_bytes(b"\x02" + b" hello")
                    await ws.send_bytes(ev({"type": "text", "frame": frames, "t": frames / 12.5, "token": 1,
                                            "piece": " hello", "forced": False}))
                if frames % 6 == 0:
                    await ws.send_bytes(ev({"type": "metrics", "step_ms": {"last": 50.0, "p50": 50.0, "p95": 60.0,
                                                                           "max": 70.0, "n": frames},
                                            "session": {"active": True, "frame": frames}}))
                if not self.ignore_cap and time.time() - t0 > cap:
                    reason = "time_limit"
                    break
                await asyncio.sleep(0.08)
            if ws.closed or closed.is_set():
                reason = "client_closed"
            else:
                await ws.send_bytes(ev({"type": "session_end", "reason": reason, "frames": frames}))
                await ws.close()
        except RESET:
            reason = "client_closed"
        except asyncio.CancelledError:
            reason = "killed"
            raise
        finally:
            rt.cancel()
            self.log["sessions"].append({"sid": p["sid"], "end_reason": reason, "frames": frames})
            self.busy_sid = None
            self.ws = None
            self.ended.set()
        return ws

    async def kill(self):
        """Simulate a worker scale-down mid-call: the socket drops with no session_end, the worker vanishes."""
        self.up = False
        req = getattr(self, "req", None)
        if self.ws is not None and req is not None and req.transport is not None:
            req.transport.abort()

    def app(self):
        a = web.Application()
        a.router.add_get("/ping", self.ping)
        a.router.add_get("/status", self.status)
        a.router.add_post("/session/claim", self.session_claim)
        a.router.add_post("/session/release", self.session_release)
        a.router.add_post("/internal/claim", self.internal_claim)
        a.router.add_get("/api/chat", self.chat)
        return a

    async def start(self):
        self.runner = web.AppRunner(self.app(), handle_signals=False)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        return self

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()


class FakeLB:
    def __init__(self, port, workers, key="test-key", hold_s=0.3):
        self.port, self.workers, self.key, self.hold_s = port, list(workers), key, hold_s
        self.rr = itertools.count()
        self.log = {"unauth": 0, "no_worker": 0, "gone": 0, "ws_headers": None, "requests": []}
        self.http = None
        self.runner = None

    async def _healthy(self, w):
        if not w.up:
            return False
        try:
            async with self.http.get(f"http://127.0.0.1:{w.port}/ping", timeout=aiohttp.ClientTimeout(total=2)) as r:
                return r.status == 200
        except Exception:
            return False

    async def _pick(self, request):
        pin = request.headers.get("X-Runpod-Worker-Id", "")
        if pin.startswith("strict "):
            wid = pin.split(" ", 1)[1].strip()
            for w in self.workers:
                if w.worker_id == wid and await self._healthy(w):
                    return w
            self.log["gone"] += 1
            raise web.HTTPNotFound(text=json.dumps({"error": "affinity_worker_gone"}), content_type="application/json")
        ready = [w for w in self.workers if await self._healthy(w)]
        if not ready:
            self.log["no_worker"] += 1
            await asyncio.sleep(self.hold_s)        # the real proxy holds up to 2 min
            raise web.HTTPServiceUnavailable(text=json.dumps({"error": "no workers available"}),
                                             content_type="application/json")
        return ready[next(self.rr) % len(ready)]

    async def handle(self, request):
        self.log["requests"].append((request.method, request.path, request.headers.get("X-Runpod-Worker-Id")))
        if request.headers.get("Authorization") != f"Bearer {self.key}":
            self.log["unauth"] += 1
            return web.json_response({"error": "unauthorized"}, status=401)
        w = await self._pick(request)
        fwd = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        url = f"http://127.0.0.1:{w.port}{request.path_qs}"
        if request.headers.get("Upgrade", "").lower() == "websocket":
            self.log["ws_headers"] = dict(request.headers)
            try:
                up = await self.http.ws_connect(url, headers=fwd, max_msg_size=0)
            except aiohttp.WSServerHandshakeError as e:
                return web.json_response({"error": "upstream refused"}, status=e.status,
                                         headers={"X-Runpod-Worker-Id": w.worker_id})
            down = web.WebSocketResponse(max_msg_size=0)
            down.headers["X-Runpod-Worker-Id"] = w.worker_id
            await down.prepare(request)

            async def a2b(src, dst):
                async for m in src:
                    if m.type == WSMsgType.BINARY:
                        await dst.send_bytes(m.data)
                    elif m.type == WSMsgType.TEXT:
                        await dst.send_str(m.data)
                    else:
                        break
            t1 = asyncio.ensure_future(a2b(up, down))
            t2 = asyncio.ensure_future(a2b(down, up))
            await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in (t1, t2):
                t.cancel()
            if t1.done() and not w.up:
                # worker vanished: drop the client socket abruptly (no clean close frame), like a lost worker
                if request.transport is not None:
                    request.transport.abort()
                return down
            if t1.done():        # worker side ended first: propagate its close code (e.g. 4409) to the client
                await down.close(code=up.close_code or 1000)
            await up.close(code=down.close_code or 1000) if t2.done() else await up.close()
            await down.close()
            return down
        body = await request.read()
        async with self.http.request(request.method, url, headers=fwd, data=body) as r:
            data = await r.read()
            return web.Response(status=r.status, body=data, content_type=r.content_type,
                                headers={"X-Runpod-Worker-Id": w.worker_id})

    async def start(self):
        self.http = aiohttp.ClientSession()
        a = web.Application()
        a.router.add_route("*", "/{tail:.*}", self.handle)
        self.runner = web.AppRunner(a, handle_signals=False)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        return self

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
        if self.http:
            await self.http.close()


class FakeQueue:
    """api.runpod.ai/v2/ID stand-in. One worker; the 'handler' is a coroutine (DESIGN 2.2 steps a-e)."""

    def __init__(self, port, worker, key="test-key", queued_s=0.5):
        self.port, self.worker, self.key, self.queued_s = port, worker, key, queued_s
        self.jobs = {}
        self.ids = itertools.count(1)
        self.log = {"run": [], "cancel": [], "unauth": 0}
        self.http = None
        self.runner = None

    def _auth(self, request):
        if request.headers.get("Authorization") != f"Bearer {self.key}":
            self.log["unauth"] += 1
            raise web.HTTPUnauthorized(text='{"error":"unauthorized"}', content_type="application/json")

    async def health(self, request):
        self._auth(request)                                # raises 401 on a bad key
        return web.json_response({"jobs": {"inQueue": 0}, "workers": {"idle": 1, "initializing": 0, "running": 0}})

    async def run(self, request):
        self._auth(request)
        b = await request.json()
        jid = f"job-{next(self.ids)}"
        job = {"id": jid, "status": "IN_QUEUE", "output": None, "input": b["input"], "policy": b.get("policy")}
        self.jobs[jid] = job
        self.log["run"].append(b)
        job["task"] = asyncio.ensure_future(self.handler(job))
        return web.json_response({"id": jid, "status": "IN_QUEUE"})

    async def handler(self, job):
        w = self.worker
        try:
            await asyncio.sleep(self.queued_s)
            job["status"] = "IN_PROGRESS"
            job["output"] = {"state": "loading", "sid": job["input"]["sid"]}
            while w.loading:
                await asyncio.sleep(0.1)
            inp = job["input"]
            async with self.http.post(f"http://127.0.0.1:{w.port}/internal/claim",
                                      json={k: inp.get(k) for k in ("sid", "record_id", "pairing", "seed")}) as r:
                if r.status != 200:
                    job["status"], job["error"] = "FAILED", f"claim {r.status}"
                    return
            w.ended.clear()
            job["output"] = {"state": "ready", "public_ip": "127.0.0.1",
                             "tcp_port": None if w.ready_without_port else w.port,
                             "worker_id": w.worker_id, "sid": inp["sid"]}
            await w.ended.wait()
            job["status"] = "COMPLETED"
            job["output"] = {"summary": w.log["sessions"][-1], "end_reason": w.log["sessions"][-1]["end_reason"]}
        except asyncio.CancelledError:
            job["status"] = "CANCELLED"

    async def status(self, request):
        self._auth(request)
        job = self.jobs.get(request.match_info["jid"])
        if not job:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({k: job.get(k) for k in ("id", "status", "output", "error") if k in job})

    async def cancel(self, request):
        self._auth(request)
        jid = request.match_info["jid"]
        self.log["cancel"].append(jid)
        job = self.jobs.get(jid)
        if job and job["status"] in ("IN_QUEUE", "IN_PROGRESS"):
            job["task"].cancel()
            job["status"] = "CANCELLED"
            if self.worker.claim and self.worker.claim[0] == job["input"]["sid"]:
                self.worker.claim = None
        return web.json_response({"id": jid, "status": job["status"] if job else "?"})

    async def start(self):
        self.http = aiohttp.ClientSession()
        a = web.Application()
        a.router.add_post("/run", self.run)
        a.router.add_get("/health", self.health)          # 2026-10-06: /api/endpoint_health passthrough
        a.router.add_get("/status/{jid}", self.status)
        a.router.add_post("/cancel/{jid}", self.cancel)
        self.runner = web.AppRunner(a, handle_signals=False)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        return self

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
        if self.http:
            await self.http.close()


async def _serve(args):
    w = await FakeWorker(args.worker_port, args.worker_id, args.secret, args.aud, mode=args.mode, load_s=args.load_s,
                         prompt_s=args.prompt_s, opus=args.opus).start()
    if args.mode == "lb":
        front = await FakeLB(args.lb_port, [w], key=args.key).start()
        log.info("fake LB on 127.0.0.1:%d -> worker %s on :%d", args.lb_port, args.worker_id, args.worker_port)
    else:
        front = await FakeQueue(args.lb_port, w, key=args.key).start()
        log.info("fake queue API on 127.0.0.1:%d -> worker %s on :%d", args.lb_port, args.worker_id, args.worker_port)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await front.stop()
        await w.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lb", "queue"), default="lb")
    ap.add_argument("--lb-port", type=int, default=27080)
    ap.add_argument("--worker-port", type=int, default=27000)
    ap.add_argument("--worker-id", default="fakeworker1")
    ap.add_argument("--secret", required=True)
    ap.add_argument("--aud", default="ep-test")
    ap.add_argument("--key", default="test-key")
    ap.add_argument("--load-s", type=float, default=3.0)
    ap.add_argument("--prompt-s", type=float, default=2.0)
    ap.add_argument("--opus", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_serve(a))
