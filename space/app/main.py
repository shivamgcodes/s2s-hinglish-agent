"""S2S serverless: Hugging Face Space backend (DESIGN.md section 5). One plain aiohttp app on 0.0.0.0:7860.

  python -u -m app.main            (Docker-SDK Space, launcher a; also any container host, launcher c)
  python app.py                    (Gradio-SDK shim, launcher b)

- static client (space/static = client/dist), /api/records*, /api/samples, /samples/* (computed locally with
  common/session.py, same JSON as DEP1 INTERFACE section 5)
- POST/GET/DELETE /api/session: wake the RunPod endpoint, claim a worker, report waking/busy/ready/in_call/ended/failed
- ws /api/chat?sid=…: relay to the worker (LB: wss + Bearer + strict worker pin; queue: ws://ip:port), frames passed
  through untouched; 0x07 metrics/session_end are peeked; synthetic session_end on upstream loss
- /metrics?sid=, /ws-echo, /api/diag?sid=, /healthz, /api/config, /api/selftest/outbound (S2S_SELFTEST=1 only)
- GET /api/script/{record_id}?pairing=gN (D-SCRIPT-PANEL, 2026-10-06): the expected conversation for that record +
  pairing from the V4 synthetic calls (common/data/scripts_v4.json, built by ops/build_scripts.py), for the client's
  "Script: what to say" panel. A pairing whose call was dropped at generation answers 200 with available=false.
- S2S_MODE=local (D-LOCAL, 2026-10-06): no RunPod; the worker runs on the same machine (or LAN) at S2S_WORKER_URL and
  is driven with the LB worker protocol (claim, websocket, release) without the RunPod Bearer / worker pin.
- GET/POST /api/filler[?mode=ticker|off] (2026-10-06, port of the pod1 demo's D-TOGGLE): the "Turn filler" toggle.
  The mode is one value per Space process (initial S2S_TURN_FILL_DEFAULT, default ticker), as the demo's
  turn_fill.json was one value per server. A new call gets it as the worker query param turn_fill=; a change during
  a call is sent to every live call as an in-band control frame b"\x08"+{"type":"turn_fill","mode":...} on the relay's
  upstream websocket (no extra RunPod request). With MAX_CONCURRENT_CALLS=1 this is the one live call.

The browser only ever talks to this app (DESIGN section 0.4): the RunPod key and the session tokens stay here.
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # space/app
SPACE = HERE.parent                             # space/


def _find_common() -> Path:
    """common/ (s2s_token.py, session.py, data/records_v4.json): S2S_COMMON, else space/common (copied in before
    the Space push, space/stage_common.sh), else the repo's common/ next to space/."""
    for c in (os.environ.get("S2S_COMMON"), SPACE / "common", SPACE.parent / "common"):
        if c and (Path(c) / "s2s_token.py").is_file():
            return Path(c)
    raise SystemExit("common/ not found (need s2s_token.py + session.py): set S2S_COMMON or run space/stage_common.sh")


COMMON = _find_common()
sys.path.insert(0, str(COMMON))
if (COMMON / "data" / "records_v4.json").is_file():
    os.environ.setdefault("S2S_RECORDS", str(COMMON / "data" / "records_v4.json"))

import aiohttp  # noqa: E402
from aiohttp import web, WSMsgType  # noqa: E402

import s2s_token  # noqa: E402
import session as sessmod  # noqa: E402

if __package__ in (None, ""):                   # python app/main.py
    sys.path.insert(0, str(SPACE))
    from app.sessions import Config, Session, SessionTable  # noqa: E402
    from app.upstream import WakeFailed, make_upstream  # noqa: E402
else:
    from .sessions import Config, Session, SessionTable  # noqa: E402
    from .upstream import WakeFailed, make_upstream  # noqa: E402

log = logging.getLogger("space")
STATIC = Path(os.environ.get("S2S_STATIC", str(SPACE / "static")))
SAMPLES = Path(os.environ.get("S2S_SAMPLES", str(SPACE / "samples")))
PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8"><title>S2S Space</title></head>
<body style="font-family:sans-serif;max-width:40em;margin:2em auto"><h1>S2S Space backend</h1>
<p>The web client is not built into <code>space/static</code> yet (ops/build_client.sh).</p>
<p><a href="/api/config">/api/config</a> · <a href="/api/records">/api/records</a> · <a href="/healthz">/healthz</a></p>
</body></html>"""
CHAT_DROP_PARAMS = ("sid", "token", "turn_fill")
TURN_FILL_MODES = ("ticker", "off")
KIND_CONTROL = b"\x08"
WORKER_REFUSALS = {4401: "auth", 4409: "busy", 4503: "loading"}   # DESIGN 3.1 close codes             # never forwarded upstream (the token goes in a header)


SCRIPTS_FILE = Path(os.environ.get("S2S_SCRIPTS", str(COMMON / "data" / "scripts_v4.json")))
_SCRIPTS = None


def load_scripts() -> dict:
    """common/data/scripts_v4.json, read once ({} if absent: the panel then says no script)."""
    global _SCRIPTS
    if _SCRIPTS is None:
        try:
            _SCRIPTS = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logging.getLogger("space").warning("%s missing: /api/script answers available=false", SCRIPTS_FILE)
            _SCRIPTS = {}
    return _SCRIPTS


def ev_frame(obj: dict) -> bytes:
    """A DEP1 kind-0x07 JSON event (INTERFACE section 4)."""
    return b"\x07" + json.dumps(obj).encode("utf-8")


def client_ip(request) -> str:
    """First hop of X-Forwarded-For (the HF proxy sets it), else the socket peer (DESIGN 5.1)."""
    xff = request.headers.get("X-Forwarded-For", "")
    return xff.split(",")[0].strip() or (request.remote or "?")


class BadCfg(Exception):
    pass


def parse_cfg(record_id, pairing, seed, allow_free: bool):
    """Normalise a session config the same way for POST /api/session and the relay's /api/chat check.
    record_id: str (must be a demo record) or None (stock session, only with S2S_ALLOW_FREE_PROMPT=1).
    pairing: g1..g4, default g1. seed: int; absent -> 1001 with a record (DEP1 INTERFACE section 3), None without."""
    rid = (str(record_id).strip() if record_id is not None else "") or None
    pair = (str(pairing).strip() if pairing is not None else "") or "g1"
    if pair not in sessmod.PAIRINGS:
        raise BadCfg(f"pairing must be one of {list(sessmod.PAIRINGS)}")
    if rid is None:
        if not allow_free:
            raise BadCfg("record_id is required")
    elif rid not in sessmod.demo_records():
        raise BadCfg(f"unknown record_id {rid!r}")
    s = str(seed).strip() if seed is not None else ""
    if s == "":
        sd = sessmod.DEFAULT_SEED if rid else None
    else:
        try:
            sd = int(s)
        except ValueError:
            raise BadCfg("seed must be an integer")
    return rid, pair, sd


class SpaceApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.table = SessionTable(cfg)
        self.http = None
        self.up = None
        self.bg = set()
        self.problems = cfg.problems()
        self.turn_fill = cfg.turn_fill_default if cfg.turn_fill_default in TURN_FILL_MODES else "ticker"

    # ------------------------------------------------------------------------------------------- lifecycle
    async def on_startup(self, app):
        # No total timeout on the shared client session: the relay websocket lives for minutes. Every plain HTTP
        # call passes its own ClientTimeout; the websocket open is bounded by UPSTREAM_OPEN_TIMEOUT_S.
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30))
        self.up = make_upstream(self.cfg, self.http)
        self.spawn(self.janitor())
        log.info("space up: mode=%s endpoint=%s max_concurrent=%d call_max_s=%d passcode=%s static=%s common=%s",
                 self.cfg.mode, self.cfg.endpoint_id or "-", self.cfg.max_concurrent, self.cfg.call_max_s,
                 "on" if self.cfg.passcode else "off", STATIC if (STATIC / "index.html").is_file() else "MISSING",
                 COMMON)
        for p in self.problems:
            log.error("config: %s (sessions will fail until fixed)", p)

    async def on_cleanup(self, app):
        for s in list(self.table.sessions.values()):
            if s.active:
                try:
                    await asyncio.wait_for(self.up.release(s), 5)
                except Exception:
                    pass
        for t in list(self.bg):
            t.cancel()
        await self.http.close()

    def spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.bg.add(t)
        t.add_done_callback(self.bg.discard)
        return t

    async def janitor(self):
        """Expire unused claims (ready longer than CLAIM_TTL_S -> release/cancel), cancel abandoned wakes, end leaked
        in_call sessions, and GC old sessions. One bad pass is logged and never stops the janitor."""
        while True:
            await asyncio.sleep(2.0)
            try:
                for s in self.table.expired_ready():
                    log.info("sid %s: claim not used within %.0f s; releasing", s.sid, self.cfg.claim_ttl_s)
                    s.end_reason = "claim_expired"
                    s.set("ended", "the call was not started in time")
                    self.spawn(self.up.release(s))
                for s in self.table.abandoned():          # REVIEW-space-3
                    log.info("sid %s: no status poll for %.0f s while %s; cancelling the wake", s.sid,
                             self.cfg.abandon_s, s.state)
                    await self.cancel(s, "abandoned")
                for s in self.table.stuck_in_call():      # REVIEW-space-1 backstop
                    log.error("sid %s: in_call for %.0f s, past every cap; ending it", s.sid,
                              time.time() - s.call_started)
                    if s.relay_close:
                        await s.relay_close()
                    s.end_reason = s.end_reason or "relay_error"
                    s.set("ended", "the Space ended a call that outlived every cap")
                    self.spawn(self.up.release(s))
                self.table.gc()
            except Exception:
                log.exception("janitor pass failed")

    # -------------------------------------------------------------------------------------------- session
    async def handle_create(self, request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if not self.table.check_passcode(body.get("passcode")):
            return web.json_response({"error": "passcode"}, status=403)
        try:
            rid, pairing, seed = parse_cfg(body.get("record_id"), body.get("pairing"), body.get("seed"),
                                           self.cfg.allow_free_prompt)
        except BadCfg as e:
            return web.json_response({"error": "bad_config", "detail": str(e)}, status=400)
        if self.problems:
            return web.json_response({"error": "space_misconfigured", "detail": "; ".join(self.problems)}, status=503)
        ip = client_ip(request)
        refused = self.table.admit(ip)
        if refused:
            return web.json_response(refused[1], status=refused[0])
        sess = Session(sid=s2s_token.new_sid(), record_id=rid, pairing=pairing, seed=seed, ip=ip)
        sess.set("waking", "starting")
        self.table.add(sess)
        sess.wake_task = self.spawn(self.wake(sess))
        log.info("sid %s: created (record=%s pairing=%s seed=%s ip=%s) active=%d", sess.sid, rid, pairing, seed, ip,
                 self.table.n_active())
        return web.json_response({"sid": sess.sid, "state": sess.state})

    async def wake(self, sess):
        def on_state(state, detail=""):
            if sess.state in ("ended", "failed", "in_call"):
                return
            if state != sess.state or detail != sess.detail:
                if state != sess.state:
                    log.info("sid %s: %s -> %s (%s) at %.1f s", sess.sid, sess.state, state, detail,
                             time.time() - sess.created)
                sess.set(state, detail)

        try:
            await self.up.wake_and_claim(sess, on_state)
        except asyncio.CancelledError:
            raise
        except WakeFailed as e:
            log.warning("sid %s: wake failed: %s", sess.sid, e)
            if sess.state not in ("ended",):
                sess.set("failed", str(e))
            self.spawn(self.up.release(sess))      # queue: cancel a job that may hold a billed worker; LB: free a claim
        except Exception as e:
            log.exception("sid %s: wake crashed", sess.sid)
            sess.set("failed", f"internal error: {type(e).__name__}")
            self.spawn(self.up.release(sess))

    async def handle_get(self, request):
        sess = self.table.get(request.match_info["sid"])
        if not sess:
            return web.json_response({"error": "unknown sid"}, status=404)
        sess.last_seen = time.time()            # the browser is still waiting (REVIEW-space-3)
        return web.json_response(sess.public())

    async def handle_delete(self, request):
        sess = self.table.get(request.match_info["sid"])
        if not sess:
            return web.json_response({"error": "unknown sid"}, status=404)
        await self.cancel(sess, "cancelled")
        return web.json_response(sess.public())

    async def cancel(self, sess, why: str):
        prev = sess.state
        if prev in ("waking", "busy") and sess.wake_task and not sess.wake_task.done():
            sess.wake_task.cancel()
        if prev == "in_call" and sess.relay_close:
            await sess.relay_close()            # the relay marks the session ended (client_closed)
        elif prev == "in_call":
            sess.cancel_pending = True          # the relay is still opening upstream; it ends the call when the open returns
        elif prev in ("waking", "busy", "ready"):
            sess.end_reason = why
            sess.set("ended", why)
            self.spawn(self.up.release(sess))   # LB: release a claim that may have just succeeded; queue: cancel
        if prev != sess.state:
            log.info("sid %s: %s while %s", sess.sid, why, prev)

    # -------------------------------------------------------------------------------------------- relay
    async def handle_chat(self, request):
        q = request.query
        sess = self.table.get(q.get("sid"))
        if not sess:
            return web.json_response({"error": "unknown sid"}, status=404)
        if sess.state != "ready":
            return web.json_response({"error": "not_ready", "state": sess.state}, status=409)
        try:
            rid, pairing, seed = parse_cfg(q.get("record_id"), q.get("pairing"), q.get("seed"),
                                           self.cfg.allow_free_prompt)
        except BadCfg as e:
            return web.json_response({"error": "bad_config", "detail": str(e)}, status=400)
        if (rid, pairing, seed) != (sess.record_id, sess.pairing, sess.seed):
            return web.json_response({"error": "cfg_mismatch",
                                      "detail": "record_id/pairing/seed differ from POST /api/session"}, status=400)
        if not self.cfg.allow_free_prompt and (q.get("text_prompt") or q.get("voice_prompt")):
            return web.json_response({"error": "free_prompt_disabled"}, status=400)
        down = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
        if not down.can_prepare(request).ok:   # REVIEW-space-1: a plain GET must not consume the claim
            return web.json_response({"error": "websocket_required"}, status=400)
        sess.set("in_call", "connecting to the worker")      # synchronous: a second socket now gets 409
        sess.cancel_pending = False
        upq = {k: v for k, v in q.items() if k not in CHAT_DROP_PARAMS}
        upq["pairing"] = pairing                             # canonical values == the ones bound into the token
        if rid is not None:
            upq["record_id"] = rid
        if seed is not None:
            upq["seed"] = str(seed)
        sess.turn_fill = self.turn_fill
        upq["turn_fill"] = sess.turn_fill                    # D-TOGGLE port: the worker's filler mode for this call

        # REVIEW-space-1: from here on every exit (prepare failure, open failure, a crash, cancellation) ends the
        # session, and every end the worker did not see cleanly also releases the claim / cancels the queue job.
        clean = False
        try:
            await down.prepare(request)
            log.info("sid %s: browser connected; opening upstream (%s)", sess.sid, self.cfg.mode)
            t0 = time.time()
            try:
                up = await self.up.open_ws(sess, upq)
            except Exception as e:
                if isinstance(e, aiohttp.WSServerHandshakeError):
                    why = f"worker refused the call (HTTP {e.status})"
                    if e.status == 401 and self.cfg.mode == "lb" and not self.cfg.token_in_query:
                        why += ("; if every call fails like this, the RunPod proxy may drop the X-S2S-Token header "
                                "on the websocket upgrade: set S2S_TOKEN_IN_QUERY=1 (CRIT-3)")
                elif isinstance(e, asyncio.TimeoutError):
                    why = f"worker did not answer within {self.cfg.open_timeout_s:.0f} s"
                else:
                    why = f"cannot reach the worker ({type(e).__name__})"
                log.warning("sid %s: upstream open failed: %s", sess.sid, why)
                await self._end_down(down, "relay_error", why, code=1011)
                sess.end_reason = "relay_error"
                sess.set("ended", why)
                return down
            if sess.cancel_pending:                          # DELETE arrived during the open
                log.info("sid %s: cancelled while the upstream was opening", sess.sid)
                await up.close()
                sess.end_reason = "client_closed"
                sess.set("ended", "cancelled by the user")
                await down.close(code=1000)
                return down
            log.info("sid %s: upstream open in %.0f ms", sess.sid, 1000 * (time.time() - t0))
            sess.set("in_call", "relaying")
            clean = await self._relay(sess, down, up, t0)
            return down
        finally:
            if sess.state != "ended":
                sess.end_reason = sess.end_reason or "relay_error"
                sess.set("ended", "relay error")
            if not clean:
                self.spawn(self.up.release(sess))

    async def _relay(self, sess, down, up, t0) -> bool:
        """Pump frames both ways until either side ends or the Space cap fires. Returns True when the worker ended
        the call itself (session_end) or saw the browser leave (client_closed), i.e. no release is needed."""
        state = {"end": None}

        async def up2down():
            async for m in up:
                if m.type == WSMsgType.BINARY:
                    d = m.data
                    if d[:1] == b"\x07":
                        self._peek(sess, d, state)
                    await down.send_bytes(d)
                elif m.type == WSMsgType.TEXT:
                    await down.send_str(m.data)
                else:
                    break

        up_lock = asyncio.Lock()                # the browser pump and the toggle both write to `up`

        async def down2up():
            async for m in down:
                if m.type == WSMsgType.BINARY:
                    async with up_lock:
                        await up.send_bytes(m.data)
                elif m.type == WSMsgType.TEXT:
                    async with up_lock:
                        await up.send_str(m.data)
                else:
                    break

        async def send_control(data: bytes):
            if up.closed:
                return
            async with up_lock:
                await up.send_bytes(data)
        sess.send_control = send_control

        t_up = asyncio.ensure_future(up2down())
        t_down = asyncio.ensure_future(down2up())
        t_ka = self.spawn(self.up.keepalive_loop(sess))
        closing = asyncio.Event()

        async def relay_close():
            closing.set()
        sess.relay_close = relay_close
        t_close = asyncio.ensure_future(closing.wait())
        cap = self.cfg.call_max_s + 15
        try:
            done, _ = await asyncio.wait({t_up, t_down, t_close}, timeout=cap, return_when=asyncio.FIRST_COMPLETED)
            if t_up in done:
                if state["end"]:
                    reason, detail, code = state["end"], "the worker ended the call", 1000
                elif down.closed:                  # the send to the browser failed: the browser left
                    reason, detail, code = "client_closed", "the browser closed the call", 1000
                elif up.close_code in WORKER_REFUSALS:   # DESIGN 3.1 post-upgrade refusal, not a lost worker
                    reason, code = "relay_error", 1011
                    detail = f"the worker refused the call after the upgrade ({WORKER_REFUSALS[up.close_code]}, close {up.close_code})"
                    log.warning("sid %s: %s", sess.sid, detail)
                else:
                    reason, detail, code = "worker_lost", "the worker connection dropped without session_end", 1011
                    log.warning("sid %s: upstream closed without session_end (close code %s, exc %r)",
                                sess.sid, up.close_code, t_up.exception() if not t_up.cancelled() else None)
            elif t_down in done or t_close in done:
                reason = state["end"] or "client_closed"
                detail, code = ("cancelled by the user" if t_close in done else "the browser closed the call"), 1000
            else:
                reason, detail, code = state["end"] or "time_limit", f"Space call cap {cap} s reached", 1000
                log.warning("sid %s: Space call cap %d s reached; closing", sess.sid, cap)
            sess.end_reason = reason
            sess.set("ended", detail)
            for t in (t_up, t_down, t_close, t_ka):
                t.cancel()
            if not up.closed:
                await up.close()
            if not down.closed:
                if not state["end"] and reason != "client_closed":
                    await self._end_down(down, reason, detail, code=code)
                else:
                    await down.close(code=code)
        finally:
            for t in (t_up, t_down, t_close, t_ka):
                t.cancel()
            sess.relay_close = None
            sess.send_control = None
            if not up.closed:
                await up.close()
            if sess.state != "ended":
                sess.end_reason = sess.end_reason or "error"
                sess.set("ended", "relay error")
            log.info("sid %s: call ended (%s) after %.1f s; keepalive %s", sess.sid, sess.end_reason,
                     time.time() - (sess.call_started or t0), dict(sess.keepalive["codes"]))
        return bool(state["end"]) or sess.end_reason == "client_closed"

    @staticmethod
    def _peek(sess, d: bytes, state: dict):
        try:
            ev = json.loads(d[1:].decode("utf-8"))
        except Exception:
            return
        if not isinstance(ev, dict):
            return
        t = ev.get("type")
        if t == "metrics":
            sess.last_metrics = ev
        elif t == "session_end":
            state["end"] = str(ev.get("reason") or "unknown")

    @staticmethod
    async def _end_down(down, reason: str, detail: str, code: int):
        """Synthetic session_end (DESIGN 2.1 step 9) so the client shows why; then close."""
        try:
            await down.send_bytes(ev_frame({"type": "session_end", "reason": reason, "frames": None,
                                            "detail": detail, "t_wall": time.time(), "source": "space"}))
        except Exception:
            pass
        try:
            await down.close(code=code, message=reason.encode()[:120])
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------ misc routes
    async def handle_config(self, request):
        return web.json_response({"mode": self.cfg.mode, "passcode_required": bool(self.cfg.passcode),
                                  "max_call_s": self.cfg.call_max_s, "max_concurrent": self.cfg.max_concurrent,
                                  "build": self.cfg.build, "allow_free_prompt": self.cfg.allow_free_prompt,
                                  "claim_ttl_s": self.cfg.claim_ttl_s, "turn_fill": self.turn_fill,
                                  "ok": not self.problems})

    async def handle_metrics(self, request):
        sess = self.table.get(request.query.get("sid"))
        return web.json_response(sess.last_metrics if sess and sess.last_metrics else {})

    async def handle_diag(self, request):
        sess = self.table.get(request.query.get("sid"))
        if not sess or sess.state not in ("ready", "in_call"):
            return web.json_response({"error": "no active call for this sid"}, status=409)
        now = time.time()
        if now - sess.last_diag < 10:
            return web.json_response({"error": "rate_limited", "retry_after_s": round(10 - (now - sess.last_diag), 1)},
                                     status=429)
        sess.last_diag = now
        return web.json_response(await self.up.diag(sess))

    async def handle_echo(self, request):
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
        await ws.prepare(request)
        async for m in ws:
            if m.type == WSMsgType.TEXT:
                await ws.send_str(m.data)
            elif m.type == WSMsgType.BINARY:
                await ws.send_bytes(m.data)
            else:
                break
        return ws

    async def handle_selftest(self, request):
        if not self.cfg.selftest:
            return web.json_response({"error": "disabled (S2S_SELFTEST=1 enables it)"}, status=404)
        if not self.cfg.passcode or not self.table.check_passcode(request.query.get("passcode")):
            return web.json_response({"error": "passcode"}, status=403)
        host = request.query.get("host", "")
        try:
            port = int(request.query.get("port", ""))
        except ValueError:
            return web.json_response({"error": "port"}, status=400)
        t0 = time.time()
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(host, port), 10)
            w.close()
            out = {"ok": True, "ms": round(1000 * (time.time() - t0), 1), "error": None}
        except Exception as e:
            out = {"ok": False, "ms": round(1000 * (time.time() - t0), 1), "error": f"{type(e).__name__}: {e}"}
        log.info("selftest outbound %s:%d -> %s", host, port, out)
        return web.json_response(out)

    async def handle_filler(self, request):
        """GET -> {"mode"}; POST ?mode=ticker|off -> sets the Space-wide mode, applies it to live calls at once (in-band
        control frame), and later calls start with it. Same paths and JSON as the pod1 demo (server.py D-TOGGLE), so
        the demo's filler-toggle.js works unchanged."""
        if request.method == "POST":
            mode = request.query.get("mode", "")
            if mode not in TURN_FILL_MODES:
                return web.json_response({"error": "mode must be ticker or off"}, status=400)
            self.turn_fill = mode
            sent = 0
            frame = KIND_CONTROL + json.dumps({"type": "turn_fill", "mode": mode}).encode()
            for s in list(self.table.sessions.values()):
                if s.state == "in_call" and s.send_control:
                    try:
                        await s.send_control(frame)
                        s.turn_fill = mode
                        sent += 1
                    except Exception as e:
                        log.warning("sid %s: turn_fill control not sent: %r", s.sid, e)
            log.info("turn filler set to %s (sent to %d live call(s))", mode, sent)
            return web.json_response({"mode": self.turn_fill, "live_calls_updated": sent})
        return web.json_response({"mode": self.turn_fill})

    async def handle_endpoint_health(self, request):
        """2026-10-06: GET /api/endpoint_health?passcode=… (only when S2S_PASSCODE is set). Passes through RunPod's
        GET https://api.runpod.ai/v2/{id}/health (worker counts: idle / initializing / running / throttled ...), which
        does NOT wake a worker, so cold starts can be debugged with the Space URL + passcode alone (nobody but the
        Space holds the RunPod key). Unverified on LB endpoints: the raw status code and body are returned as is."""
        if not self.cfg.passcode:
            return web.json_response({"error": "disabled (needs S2S_PASSCODE)"}, status=404)
        if not self.table.check_passcode(request.query.get("passcode")):
            return web.json_response({"error": "passcode"}, status=403)
        now = time.time()
        if now - getattr(self, "_last_ep_health", 0.0) < 5:
            return web.json_response({"error": "rate_limited", "retry_after_s": 5}, status=429)
        self._last_ep_health = now
        url = self.cfg.api_url + "/health"
        try:
            async with self.http.get(url, headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                                     timeout=aiohttp.ClientTimeout(total=10)) as r:
                txt = await r.text()
                try:
                    body = json.loads(txt)
                except ValueError:
                    body = txt[:500]
                code = r.status
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            code, body = None, f"{type(e).__name__}"
        active = [s.public() for s in self.table.sessions.values() if s.active]
        return web.json_response({"runpod_health_status": code, "runpod_health": body,
                                  "space_sessions_active": active, "turn_fill": self.turn_fill, "t": now})

    async def handle_healthz(self, request):
        return web.Response(text="ok")

    # records / samples: same JSON as DEP1 server.py (INTERFACE section 5), computed locally
    async def handle_records(self, request):
        recs = sessmod.demo_records()
        keys = ("agent_type", "brand", "agent_name_f", "agent_name_m", "information", "primary_id", "secondary_id")
        return web.json_response([dict({"record_id": k}, **{x: r.get(x) for x in keys}) for k, r in recs.items()])

    async def handle_record(self, request):
        rid = request.match_info["record_id"]
        recs = sessmod.load_records()
        if rid not in recs:
            return web.json_response({"error": f"unknown record_id {rid!r}"}, status=404)
        r = dict(recs[rid])
        r["session_configs"] = {g: sessmod.session_config(rid, g) for g in sessmod.PAIRINGS}
        return web.json_response(r)

    async def handle_script(self, request):
        """D-SCRIPT-PANEL: GET /api/script/{record_id}?pairing=gN -> the expected V4 conversation for that record +
        pairing. 404 unknown record, 400 bad pairing, 200 + available=false when that pairing's call does not exist."""
        rid = request.match_info["record_id"]
        pairing = request.query.get("pairing", "g1")
        if rid not in sessmod.demo_records():
            return web.json_response({"error": "unknown_record", "detail": f"unknown record_id {rid!r}"}, status=404)
        if pairing not in sessmod.PAIRINGS:
            return web.json_response({"error": "bad_pairing", "detail": f"pairing must be one of {list(sessmod.PAIRINGS)}"},
                                     status=400)
        data = load_scripts()
        sc = (data.get("scripts") or {}).get(f"{rid}_{pairing}")
        base = {"record_id": rid, "pairing": pairing, "call_id": f"{rid}_{pairing}"}
        if not sc:
            return web.json_response(dict(base, available=False, turns=[], writes=[],
                                          detail="no script for this pairing (the synthetic call was not generated)"))
        split = sc.get("split")
        return web.json_response(dict(base, available=True, split=split,
                                      split_note=(data.get("splits") or {}).get(split, ""),
                                      agent_name=sc.get("agent_name"), turns=sc.get("turns", []),
                                      writes=sc.get("writes", [])))

    async def handle_samples(self, request):
        notes = {}
        if (SAMPLES / "notes.json").exists():
            try:
                notes = json.loads((SAMPLES / "notes.json").read_text())
            except Exception:
                notes = {}
        out = [{"name": p.stem, "url": f"/samples/{p.name}", "note": notes.get(p.stem, notes.get(p.name, ""))}
               for p in sorted(SAMPLES.glob("*.wav"))] if SAMPLES.is_dir() else []
        return web.json_response(out)

    async def handle_root(self, request):
        idx = STATIC / "index.html"
        if idx.is_file():
            return web.FileResponse(idx, headers={"Cache-Control": "no-cache"})
        return web.Response(text=PLACEHOLDER, content_type="text/html")

    def make_app(self) -> web.Application:
        app = web.Application()
        r = app.router
        r.add_get("/", self.handle_root)
        r.add_get("/healthz", self.handle_healthz)
        r.add_get("/api/config", self.handle_config)
        r.add_get("/api/records", self.handle_records)
        r.add_get("/api/records/{record_id}", self.handle_record)
        r.add_get("/api/samples", self.handle_samples)
        r.add_get("/api/script/{record_id}", self.handle_script)
        r.add_post("/api/session", self.handle_create)
        r.add_get("/api/session/{sid}", self.handle_get)
        r.add_delete("/api/session/{sid}", self.handle_delete)
        r.add_get("/api/chat", self.handle_chat)
        r.add_get("/metrics", self.handle_metrics)
        r.add_get("/api/diag", self.handle_diag)
        r.add_get("/ws-echo", self.handle_echo)
        r.add_get("/api/selftest/outbound", self.handle_selftest)
        r.add_get("/api/filler", self.handle_filler)
        r.add_get("/api/endpoint_health", self.handle_endpoint_health)
        r.add_post("/api/filler", self.handle_filler)
        if SAMPLES.is_dir():
            r.add_static("/samples/", str(SAMPLES), follow_symlinks=False)
        if STATIC.is_dir():
            r.add_static("/", str(STATIC), follow_symlinks=False, name="static")
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


def run(host: str = None, port: int = None):
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    cfg = Config.from_env()
    if host:
        cfg.host = host
    if port:
        cfg.port = port
    web.run_app(SpaceApp(cfg).make_app(), host=cfg.host, port=cfg.port, access_log=None,
                handle_signals=True, print=None)


if __name__ == "__main__":
    run()
