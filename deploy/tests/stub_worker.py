"""S2S serverless: tiny CPU test double of the worker's PUBLIC contract (DESIGN 3.1), for testing fake_runpod.py and the
REAL-test scripts (ws_hold_test / cold_start_timer / latency_probe) without W's worker_server.py. Not shipped in the image.

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/stub_worker.py [--port 18000] [--internal-port 18999] \
      [--load-s 3] [--prompt-s 1] [--context-s 221] [--mode lb] [--aud ep-test] [--worker-id stub-0]
  env S2S_SESSION_SECRET (required, >= 32 chars), CALL_MAX_S (default 300), CLAIM_TTL_S (default 90)

Behaviour (subset of 3.1): /ping 204 for --load-s then 200 {"status":"ok","state"} (also while busy); /status;
/session/claim + /session/release (X-S2S-Token); ws /api/chat (X-S2S-Token header or ?token=, replay set, cfg match,
claimed sid only, 409 busy) -> prompt sleep -> 0x00 -> 0x07 session -> echoes every 0x01 page back unchanged (so the
client hears its own audio: a transport round trip with no model) -> 0x07 metrics every 2 s -> session_end
time_limit / context_full / client_closed / worker_shutdown (SIGTERM). Internal POST /internal/claim on --internal-port.
"""
import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
from collections import OrderedDict

from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import s2s_token  # noqa: E402


def log(*a):
    print(f"[stub_worker {time.strftime('%H:%M:%S')}]", *a, flush=True)


class Stub:
    def __init__(self, a):
        self.a = a
        self.secret = os.environ["S2S_SESSION_SECRET"]
        self.call_max_s = int(os.environ.get("CALL_MAX_S", "300"))
        self.claim_ttl_s = int(os.environ.get("CLAIM_TTL_S", "90"))
        self.t0 = time.time()
        self.state = "idle"            # idle | claimed | busy
        self.sid, self.cfg, self.until = None, None, 0.0
        self.replay = OrderedDict()
        self.served = 0
        self.ws = None
        self.end_reason = None
        self.lock = asyncio.Lock()

    def ready(self):
        return time.time() - self.t0 >= self.a.load_s

    def _expire(self):
        if self.state == "claimed" and time.time() > self.until:
            log(f"claim {self.sid} expired unused")
            self.state, self.sid, self.cfg = "idle", None, None

    def _tok(self, request):
        tok = request.headers.get("X-S2S-Token") or request.query.get("token", "")
        return s2s_token.verify(self.secret, tok, mode=self.a.mode, aud=self.a.aud)

    def _claim(self, sid, cfg):
        self._expire()
        if self.state == "idle" or (self.state == "claimed" and self.sid == sid):
            self.state, self.sid, self.cfg, self.until = "claimed", sid, cfg, time.time() + self.claim_ttl_s
            return web.json_response({"worker_id": self.a.worker_id, "sid": sid, "claim_ttl_s": self.claim_ttl_s})
        return web.json_response({"error": "busy"}, status=409)

    async def ping(self, request):
        if not self.ready():
            return web.Response(status=204)
        self._expire()
        return web.json_response({"status": "ok", "state": self.state})

    async def status(self, request):
        self._expire()
        st = self.state if self.ready() else "loading"
        act = {"sid": self.sid, "conv_s": None, "context_frames_left": None} if self.state == "busy" else None
        return web.json_response({"state": st, "worker_id": self.a.worker_id, "gpu": "stub-cpu", "vram_total_mib": 0,
                                  "load_s": self.a.load_s, "sessions_served": self.served, "active": act,
                                  "step_ms_p95_last": 1.0, "asr": True, "router": True, "version": "stub"})

    async def claim(self, request):
        if not self.ready():
            return web.json_response({"error": "loading"}, status=503)
        try:
            p = self._tok(request)
        except s2s_token.TokenError as e:
            return web.json_response({"error": "expired" if e.code == "expired" else "bad_token"}, status=401)
        return self._claim(p["sid"], {k: p[k] for k in ("record_id", "pairing", "seed", "max_s")})

    async def internal_claim(self, request):
        b = await request.json()
        if not self.ready():
            return web.json_response({"error": "loading"}, status=503)
        return self._claim(b["sid"], {"record_id": b.get("record_id"), "pairing": b.get("pairing"),
                                      "seed": b.get("seed"), "max_s": self.call_max_s})

    async def release(self, request):
        try:
            p = self._tok(request)
        except s2s_token.TokenError:
            return web.json_response({"error": "bad_token"}, status=401)
        rel = False
        if self.sid == p["sid"]:
            if self.state == "claimed":
                self.state, self.sid, self.cfg, rel = "idle", None, None, True
            elif self.state == "busy" and self.ws is not None:
                self.end_reason = "client_closed"
                await self.ws.close()
                rel = True
        return web.json_response({"released": rel})

    async def metrics(self, request):
        try:
            p = self._tok(request)
        except s2s_token.TokenError:
            return web.json_response({"error": "bad_token"}, status=401)
        if p["sid"] != self.sid:
            return web.json_response({"error": "not_active"}, status=409)
        return web.json_response(self._metrics())

    def _metrics(self):
        return {"t_wall": time.time(), "session": {"active": self.state == "busy", "session_id": self.sid},
                "step_ms": {"last": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0, "n": 1}, "rtf": 0.01, "backlog_frames": 0}

    async def chat(self, request):
        if not self.ready():
            return web.json_response({"error": "loading"}, status=503)
        try:
            p = self._tok(request)
        except s2s_token.TokenError as e:
            return web.json_response({"error": "expired" if e.code == "expired" else "bad_token"}, status=401)
        if p["sid"] in self.replay:
            return web.json_response({"error": "replayed"}, status=401)
        q = request.query
        for k in ("record_id", "pairing", "seed"):
            qv = q.get(k)
            if qv is not None and p[k] is not None and str(qv) != str(p[k]):
                return web.json_response({"error": "cfg_mismatch", "field": k}, status=401)
        async with self.lock:
            self._expire()
            if self.state != "claimed" or self.sid != p["sid"]:
                return web.json_response({"error": "busy"}, status=409)
            self.state = "busy"
            self.replay[p["sid"]] = p["exp"]
            while len(self.replay) > 1024:
                self.replay.popitem(last=False)
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        self.ws, self.end_reason = ws, None
        max_s = min(int(p["max_s"]), self.call_max_s)
        sid = p["sid"]
        log(f"session {sid} start (max_s {max_s}, context_s {self.a.context_s})")
        t_up = time.time()

        def ev(o):
            o.update({"session_id": sid, "t_wall": time.time()})
            return b"\x07" + json.dumps(o).encode()

        frames = 0
        try:
            await asyncio.sleep(self.a.prompt_s)
            await ws.send_bytes(b"\x00")
            t_hs = time.time()
            await ws.send_bytes(ev({"type": "session", "record_id": p["record_id"], "pairing": p["pairing"],
                                    "context_frames_left": int(self.a.context_s * 12.5)}))

            async def metrics_loop():
                while not ws.closed:
                    await asyncio.sleep(2)
                    m = self._metrics()
                    m["type"] = "metrics"
                    await ws.send_bytes(ev(m))

            async def watchdog():
                while not ws.closed:
                    await asyncio.sleep(0.2)
                    if time.time() - t_up >= max_s:
                        self.end_reason = "time_limit"
                    elif time.time() - t_hs >= self.a.context_s:
                        self.end_reason = "context_full"
                    if self.end_reason:
                        await ws.send_bytes(ev({"type": "session_end", "reason": self.end_reason, "frames": frames}))
                        await ws.close()
                        return

            tasks = [asyncio.create_task(metrics_loop()), asyncio.create_task(watchdog())]
            async for m in ws:
                if m.type.name == "BINARY" and m.data[:1] == b"\x01":
                    frames += 1
                    await ws.send_bytes(m.data)          # echo the Ogg/Opus page back
            for t in tasks:
                t.cancel()
        except Exception as e:
            log(f"session {sid} error {e!r}")
        reason = self.end_reason or "client_closed"
        self.served += 1
        self.state, self.sid, self.cfg, self.ws = "idle", None, None, None
        log(json.dumps({"summary": True, "sid": sid, "end_reason": reason, "frames_in": frames,
                        "worker_id": self.a.worker_id}))
        return ws

    async def shutdown(self):
        if self.ws is not None and not self.ws.closed:
            self.end_reason = "worker_shutdown"
            try:
                await self.ws.send_bytes(b"\x07" + json.dumps({"type": "session_end", "reason": "worker_shutdown",
                                                               "frames": None, "session_id": self.sid}).encode())
                await self.ws.close()
            except Exception:
                pass


async def amain(a):
    s = Stub(a)
    app = web.Application()
    app.router.add_get("/ping", s.ping)
    app.router.add_get("/status", s.status)
    app.router.add_post("/session/claim", s.claim)
    app.router.add_post("/session/release", s.release)
    app.router.add_get("/metrics", s.metrics)
    app.router.add_get("/api/chat", s.chat)
    iapp = web.Application()
    iapp.router.add_post("/internal/claim", s.internal_claim)
    r1, r2 = web.AppRunner(app, access_log=None), web.AppRunner(iapp, access_log=None)
    await r1.setup()
    await r2.setup()
    await web.TCPSite(r1, "0.0.0.0" if a.public else "127.0.0.1", a.port).start()
    await web.TCPSite(r2, "127.0.0.1", a.internal_port).start()
    log(f"listening :{a.port} (internal :{a.internal_port}); ready after {a.load_s}s; worker_id {a.worker_id}")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()
    await s.shutdown()
    await asyncio.sleep(0.5)
    log("SIGTERM: exit")
    sys.stdout.flush()
    os._exit(0)          # do not wait for in-flight handlers (the real worker exits after a 2 s flush too)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "18000")))
    ap.add_argument("--internal-port", type=int, default=18999)
    ap.add_argument("--public", action="store_true", help="bind 0.0.0.0 instead of 127.0.0.1")
    ap.add_argument("--load-s", type=float, default=3.0)
    ap.add_argument("--prompt-s", type=float, default=1.0)
    ap.add_argument("--context-s", type=float, default=221.0)
    ap.add_argument("--mode", default=os.environ.get("S2S_MODE", "lb"))
    ap.add_argument("--aud", default=os.environ.get("S2S_AUDIENCE", "ep-test"))
    ap.add_argument("--worker-id", default=os.environ.get("RUNPOD_POD_ID") or socket.gethostname())
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
