"""S2S serverless worker: fork of DEP1 server/server.py (itself a fork of moshi/server.py). DESIGN.md section 3.1.

  <venv-pp>/bin/python worker/server/worker_server.py [--port $PORT] [--internal-port 8999] [--mock ...]
  (started by worker/stack.sh; worker/local/run_local.sh on runpod2)

- 0.0.0.0:$PORT http (no TLS; the RunPod LB proxy terminates TLS, queue mode is plain TCP):
    GET  /ping              204 loading | 200 {"status":"ok","state":idle|claimed|busy} (ALSO while busy) | 500 failed
    GET  /status            worker state for the Space's wake/keepalive (no lock, no GPU call, no secrets)
    POST /session/claim     X-S2S-Token (or ?token=, CRIT-3) -> 200 {worker_id,sid,claim_ttl_s} | 401 | 409 busy | 503 loading
    POST /session/release   X-S2S-Token -> 200 {released}
    GET  /api/chat (ws)     X-S2S-Token header (or ?token=); DEP1 query params + protocol (kinds 0x00/0x01/0x02/0x07)
    GET  /metrics           X-S2S-Token of the active sid -> DEP1 metrics object
- 127.0.0.1:$S2S_INTERNAL_PORT http: the DEP1 internal API unchanged (/internal/{events,session,ring,action,inject,
  mute,play}, /metrics) + S2S (queue handler): POST /internal/claim, GET /internal/claim/{sid},
  POST /internal/claim_release, GET /internal/status

Changes from DEP1 server.py (everything else, incl. the chat handler body, Hub and internal API, is kept):
  bind first, load later (load runs as a task: resolve_models -> Engine; /ping 204 until engine + ASR + router are
  healthy); single-session state machine idle -> claimed(sid) -> busy(sid) -> idle (a second socket gets 409, no
  waiting on the lock); HMAC session token (common/s2s_token.py) on every public token route, replay set on the
  websocket upgrade only; CALL_MAX_S watchdog (end reason time_limit, also during the prompt phase); SIGTERM ends
  the call with worker_shutdown; public static/records/samples routes dropped (the Space serves them).
The same file runs the CPU mock (--mock uses mock_server.MockEngine), as DEP1's mock_server did.
"""
import argparse
import asyncio
import collections
import concurrent.futures
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

paths.ensure_common_on_path()

import aiohttp  # noqa: E402
import numpy as np  # noqa: E402
import sphn  # noqa: E402
from aiohttp import web  # noqa: E402

import s2s_token  # noqa: E402
import session as sessmod  # noqa: E402
from core import FRAME_RATE, FRAME_SIZE, SAMPLE_RATE  # noqa: E402

log = logging.getLogger("s2s.worker")
VERSION = "s2s-worker/2026-10-06"
ACTION_STAGES = ("trigger", "asr", "needle", "resolved", "executed", "unbound", "needs_clarification", "error")   # +ask (D-ROUTER-V2)
CLOSE_AUTH, CLOSE_BUSY, CLOSE_LOADING = 4401, 4409, 4503      # DESIGN 3.1: post-upgrade refusals
TOKEN_HEADER = "X-S2S-Token"
TURN_FILL_MODES = ("ticker", "off")
KIND_CONTROL = 8      # S2S 2026-10-06: client(Space)->worker JSON control frame b"\x08"+{"type":"turn_fill","mode":...}
REPLAY_MAX = 1024


def _env_int(name, default):
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return int(default)


class BadConfig(Exception):
    pass


def resolve_session_config(q) -> dict:
    """DEP1 INTERFACE.md section 3 (unchanged). With record_id: session.session_config (seed default 1001, explicit
    text_prompt / voice_prompt win). Without: stock session (seed absent or -1 = no reseed)."""
    record_id = (q.get("record_id") or "").strip()
    pairing = (q.get("pairing") or "g1").strip()
    text_prompt = q.get("text_prompt", "") or ""
    voice_prompt = q.get("voice_prompt", "") or ""
    seed_s = (q.get("seed") or "").strip()
    try:
        seed = int(seed_s) if seed_s else None
    except ValueError:
        raise BadConfig(f"bad seed {seed_s!r}")
    if record_id:
        if pairing not in sessmod.PAIRINGS:
            raise BadConfig(f"pairing must be one of {sessmod.PAIRINGS}")
        try:
            cfg = sessmod.session_config(record_id, pairing, sessmod.DEFAULT_SEED if seed is None else seed)
        except KeyError:
            raise BadConfig(f"unknown record_id {record_id!r}")
        at = (q.get("agent_type") or "").strip()
        if at and at != cfg["agent_type"]:
            raise BadConfig(f"agent_type {at!r} != record agent_type {cfg['agent_type']!r}")
        if text_prompt.strip():
            cfg["role_prompt"] = text_prompt
        if voice_prompt.strip():
            cfg["voice"] = voice_prompt
    else:
        cfg = {"type": "session_config", "agent_type": None, "record_id": None, "pairing": None,
               "agent_gender": None, "voice": voice_prompt or "NATF2.pt", "role_prompt": text_prompt,
               "seed": seed, "router_supported": False}
    if cfg["seed"] is not None and int(cfg["seed"]) == -1:
        cfg["seed"] = None
    if "/" in cfg["voice"] or "\\" in cfg["voice"]:
        raise BadConfig("voice_prompt must be a file name")
    # S2S 2026-10-06 (D-TOGGLE port): per-call turn filler, set by the Space from its toggle. Not part of the token's
    # (record_id, pairing, seed) triple. Absent -> env/JSON default (S2S_TURN_FILL, default ticker).
    tf = (q.get("turn_fill") or "").strip()
    if tf and tf not in TURN_FILL_MODES:
        raise BadConfig(f"turn_fill must be one of {TURN_FILL_MODES}")
    cfg["turn_fill"] = tf or None
    return cfg


def norm_cfg(record_id, pairing, seed):
    """The (record_id, pairing, seed) triple bound into the token, after DEP1 defaults (DECISIONS D-W-1):
    with a record_id: pairing None/'' -> 'g1', seed None/'' -> 1001; without: pairing None, seed None/-1 -> None.
    Token fields and websocket query params are both normalised with this before comparing."""
    rid = str(record_id).strip() if record_id is not None else ""
    rid = rid or None
    if isinstance(seed, str):
        seed = int(seed) if seed.strip() else None
    if rid is not None:
        pairing = (pairing or "").strip() or "g1"
        seed = sessmod.DEFAULT_SEED if seed is None else int(seed)
    else:
        pairing = None
        seed = None if seed is None or int(seed) == -1 else int(seed)
    return rid, pairing, seed


class Hub:
    """Event fan-out: /internal/events subscribers (drop-on-full queues) + the chat socket's out queue."""

    def __init__(self):
        self.subs = set()
        self.chat_q = None          # asyncio.Queue of bytes for the active chat socket when events=1
        self.session_id = None

    def publish(self, ev: dict):
        ev.setdefault("session_id", self.session_id)
        ev.setdefault("t_wall", time.time())
        s = json.dumps(ev, ensure_ascii=False)
        for q in list(self.subs):
            try:
                q.put_nowait(s)
            except asyncio.QueueFull:
                pass
        if self.chat_q is not None:
            self.chat_q.put_nowait(b"\x07" + s.encode("utf-8"))


class Slot:
    """Single-session guard (DESIGN 3.1): idle -> claimed(sid, until) -> busy(sid) -> idle."""

    def __init__(self):
        self.state = "idle"
        self.sid = None
        self.cfg = None             # normalised (record_id, pairing, seed) the claim was made for
        self.max_s = None
        self.until = 0.0
        self.via = None             # "token" | "internal"
        self.t_busy = None

    def to_idle(self):
        self.__init__()


def gpu_info():
    """nvidia-smi name/driver/memory (None fields when there is no GPU, e.g. the CPU mock on a laptop)."""
    info = {"gpu": None, "driver": None, "cuda": None, "vram_total_mib": None, "vram_free_mib": None}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                                       "--format=csv,noheader,nounits"], timeout=10, text=True)
        name, drv, tot, free = [x.strip() for x in out.splitlines()[0].split(",")]
        info.update(gpu=name, driver=drv, vram_total_mib=int(tot), vram_free_mib=int(free))
        head = subprocess.check_output(["nvidia-smi"], timeout=10, text=True)
        if "CUDA Version:" in head:
            info["cuda"] = head.split("CUDA Version:")[1].split()[0]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


class Server:
    def __init__(self, args, engine_factory):
        self.args = args
        self.gpu = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.engine_factory = engine_factory
        self.eng = None
        self.hub = Hub()
        self.lock = asyncio.Lock()
        self.loop = None
        self.backlog = {"last_frames": 0, "max_frames": 0}
        # ---- S2S serverless state
        self.mode = os.environ.get("S2S_MODE", "lb")
        self.secret = os.environ.get("S2S_SESSION_SECRET", "")
        self.aud = os.environ.get("S2S_AUDIENCE") or os.environ.get("RUNPOD_ENDPOINT_ID") or ""
        self.call_max_s = _env_int("CALL_MAX_S", 300)
        self.claim_ttl_s = _env_int("CLAIM_TTL_S", 90)
        self.allow_free_prompt = os.environ.get("S2S_ALLOW_FREE_PROMPT", "0") == "1"
        self.asr_backend = os.environ.get("ASR_BACKEND", "trelis")
        self.router_mode = os.environ.get("ROUTER", "on")
        self.worker_id = os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
        self.slot = Slot()
        self.replay = collections.OrderedDict()     # sid -> forget_at (token exp + 60 s)
        self.outcomes = collections.OrderedDict()   # sid -> {"phase": ended|expired, "end_reason", "summary"}
        self.engine_ok = False
        self.asr_ok = self.asr_backend == "off"
        self.router_ok = self.router_mode == "off"
        self.ready = False
        self.load_error = None
        self.load_s = None
        self.t_start = time.time()
        self.sessions_served = 0
        self.last_metrics = None
        self.last_step_p95 = None
        self.active_stop = None                     # callable(reason) for the running call
        self.shutting_down = False
        self.exit_event = asyncio.Event()
        self.health_kick = asyncio.Event()          # set when the engine finishes loading: check readiness at once
        self.gpuinfo = {}
        self.health_detail = {}

    # ---------------------------------------------------------------------------------------- lifecycle
    def config_error(self):
        if self.mode not in ("lb", "queue"):
            return f"S2S_MODE must be lb or queue, got {self.mode!r}"
        if len(self.secret) < s2s_token.MIN_SECRET_LEN:
            return f"S2S_SESSION_SECRET missing or shorter than {s2s_token.MIN_SECRET_LEN} chars"
        if not self.aud:
            return "S2S_AUDIENCE (or RUNPOD_ENDPOINT_ID) is not set"
        return None

    async def load(self):
        self.loop = asyncio.get_running_loop()
        t0 = time.time()
        self.eng = await self.loop.run_in_executor(self.gpu, self.engine_factory)
        self.eng.extra_metrics["backlog_frames"] = self.backlog
        self.eng.listeners.append(lambda ev: self.loop.call_soon_threadsafe(self.hub.publish, dict(ev)))
        self.load_s = round(time.time() - t0, 1)
        self.engine_ok = True
        self.health_kick.set()
        log.info("engine ready in %.1f s (%s)", time.time() - t0, type(self.eng).__name__)

    async def load_task(self):
        err = self.config_error()
        if err is not None:
            self.load_error = err
            log.error("CONFIG ERROR: %s (/ping answers 500)", err)
            return
        try:
            await self.load()
        except Exception as e:  # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"
            log.error("LOAD FAILED: %s (/ping answers 500)\n%s", self.load_error, traceback.format_exc())

    async def health_loop(self):
        """ready = engine loaded AND ASR /health AND router /health (each unless switched off). Every 2 s until
        ready, then every 10 s; after ready a failure is logged and shown in /status but does NOT flip /ping."""
        checks = []
        if self.asr_backend != "off":
            checks.append(("asr", f"http://127.0.0.1:{paths.ASR_PORT}/health"))
        if self.router_mode != "off":
            checks.append(("router", f"http://127.0.0.1:{paths.ROUTER_PORT}/health"))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as cs:
            while True:
                for name, url in checks:
                    ok = False
                    try:
                        async with cs.get(url) as r:
                            ok = r.status == 200
                            if ok:
                                d = await r.json(content_type=None)
                                self.health_detail[name] = {k: d.get(k) for k in ("backend", "device", "load_s",
                                                                                   "router_active", "router") if k in d}
                    except Exception:  # noqa: BLE001
                        ok = False
                    prev = getattr(self, f"{name}_ok")
                    if prev and not ok:
                        log.warning("%s health check failed (%s)", name, url)
                    setattr(self, f"{name}_ok", ok)
                if not self.ready and self.engine_ok and self.asr_ok and self.router_ok and not self.load_error:
                    self.ready = True
                    log.info("worker READY in %.1f s since start (engine %.1f s); /ping -> 200",
                             time.time() - self.t_start, self.load_s or 0)
                    print("READY", flush=True)
                try:
                    await asyncio.wait_for(self.health_kick.wait(), timeout=10.0 if self.ready else 2.0)
                except asyncio.TimeoutError:
                    pass
                self.health_kick.clear()

    async def tick_loop(self):
        while True:
            self._expire()
            await asyncio.sleep(1.0)

    def _expire(self):
        s = self.slot
        if s.state == "claimed" and time.time() > s.until:
            log.info("claim %s expired unused after %d s; worker idle again", s.sid, self.claim_ttl_s)
            self._outcome(s.sid, "expired", None, None)
            s.to_idle()
        now = time.time()
        while self.replay and next(iter(self.replay.values())) < now:
            self.replay.popitem(last=False)

    def _outcome(self, sid, phase, end_reason, summary):
        self.outcomes[sid] = {"phase": phase, "end_reason": end_reason, "summary": summary}
        while len(self.outcomes) > 64:
            self.outcomes.popitem(last=False)

    def phase(self):
        if self.load_error:
            return "failed"
        if not self.ready:
            return "loading"
        return self.slot.state

    def on_signal(self, signum):
        if self.shutting_down:
            return
        self.shutting_down = True
        log.warning("signal %s: shutting down (active call: %s)", signum, self.slot.sid if self.slot.state == "busy" else None)
        asyncio.ensure_future(self._shutdown())

    async def _shutdown(self):
        if self.active_stop is not None:
            self.active_stop("worker_shutdown")
            t_end = time.time() + 2.5           # DESIGN 3.1: flush for up to 2 s
            while self.slot.state == "busy" and time.time() < t_end:
                await asyncio.sleep(0.05)
        self.exit_event.set()

    async def gpu_call(self, fn, *a):
        return await self.loop.run_in_executor(self.gpu, fn, *a)

    # ------------------------------------------------------------------------------------------ token
    def _verify(self, token):
        """-> (payload, None) | (None, json_response 401)."""
        if not token:
            return None, web.json_response({"error": "bad_token", "detail": "missing"}, status=401)
        try:
            return s2s_token.verify(self.secret, token, mode=self.mode, aud=self.aud), None
        except s2s_token.TokenError as e:
            err = "expired" if e.code == "expired" else "bad_token"
            log.info("token refused: %s", e.code)
            return None, web.json_response({"error": err, "detail": e.code}, status=401)
        except ValueError as e:    # secret misconfigured (config_error already reports it)
            return None, web.json_response({"error": "bad_token", "detail": str(e)}, status=401)

    # NOTE: aiohttp responses are MutableMappings and evaluate False: always compare them with `is not None`.
    def _not_ready(self):
        ph = self.phase()
        if ph in ("loading", "failed"):
            return web.json_response({"error": "loading" if ph == "loading" else "failed"}, status=503)
        return None

    def _claim(self, sid, cfg, max_s, via):
        """-> (status, body). Accepted only in idle or as a re-claim of the same sid (idempotent)."""
        self._expire()
        s = self.slot
        if s.state == "claimed" and s.sid == sid:
            if s.cfg != cfg:
                return 401, {"error": "cfg_mismatch"}
            s.until = time.time() + self.claim_ttl_s
            return 200, {"worker_id": self.worker_id, "sid": sid, "claim_ttl_s": self.claim_ttl_s}
        if s.state != "idle" or self.shutting_down:
            return 409, {"error": "busy"}
        s.state, s.sid, s.cfg, s.max_s, s.via = "claimed", sid, cfg, max_s, via
        s.until = time.time() + self.claim_ttl_s
        log.info("claimed by sid %s via %s (record=%s pairing=%s seed=%s), ttl %d s", sid, via, *cfg, self.claim_ttl_s)
        return 200, {"worker_id": self.worker_id, "sid": sid, "claim_ttl_s": self.claim_ttl_s}

    # ---------------------------------------------------------------------------------- public routes
    async def handle_ping(self, request):
        ph = self.phase()
        if ph == "failed":
            return web.json_response({"status": "error", "error": self.load_error}, status=500)
        if ph == "loading":
            return web.Response(status=204)
        return web.json_response({"status": "ok", "state": ph})

    async def handle_status(self, request):
        """No lock, no GPU call: only values cached by the metrics loop / session end (keepalive hits this mid-call)."""
        return self._status(redact_sid=False)

    async def handle_public_status(self, request):
        """REVIEW-worker-5: in queue mode the public port is reachable by anyone (no RunPod Bearer in front), and the
        Space treats the sid as the session handle (DELETE /api/session/{sid}, /metrics?sid=). So the public /status
        hides active.sid there; LB mode (behind the RunPod key) and /internal/status keep it."""
        return self._status(redact_sid=self.mode == "queue")

    def _status(self, redact_sid):
        self._expire()
        s = self.slot
        active = None
        if s.state == "busy":
            sess = (self.last_metrics or {}).get("session") or {}
            e = self.eng
            conv_s = round(e.frame / FRAME_RATE, 2) if e is not None and e.active else None   # plain int, no GPU
            active = {"sid": None if redact_sid else s.sid, "conv_s": conv_s, "context_frames_left": sess.get("context_frames_left"),
                      "elapsed_s": round(time.time() - s.t_busy, 1) if s.t_busy else None}
        detail = self.load_error or ("" if self.ready else
                                     "loading: engine=%s asr=%s router=%s" % (self.engine_ok, self.asr_ok, self.router_ok))
        return web.json_response({
            "state": self.phase(), "detail": detail, "worker_id": self.worker_id, "mode": self.mode,
            "gpu": self.gpuinfo.get("gpu"), "vram_total_mib": self.gpuinfo.get("vram_total_mib"),
            "load_s": self.load_s, "uptime_s": round(time.time() - self.t_start, 1),
            "sessions_served": self.sessions_served, "active": active,
            "claim_ttl_left_s": round(s.until - time.time(), 1) if s.state == "claimed" else None,
            "step_ms_p95_last": self.last_step_p95, "engine": self.engine_ok,
            "asr": self.asr_ok, "router": self.router_ok, "asr_backend": self.asr_backend, "router_mode": self.router_mode,
            "health": self.health_detail, "version": VERSION})

    async def handle_claim(self, request):
        p, err = self._verify(request.headers.get(TOKEN_HEADER) or request.query.get("token"))  # CRIT-3
        if err is not None:
            return err
        nr = self._not_ready()
        if nr is not None:
            return nr
        cfg = norm_cfg(p["record_id"], p["pairing"], p["seed"])
        st, body = self._claim(p["sid"], cfg, int(p["max_s"]), "token")
        if st != 200:
            log.info("claim sid %s refused: %s %s", p["sid"], st, body)
        return web.json_response(body, status=st)

    async def handle_release(self, request):
        p, err = self._verify(request.headers.get(TOKEN_HEADER) or request.query.get("token"))  # CRIT-3
        if err is not None:
            return err
        return web.json_response({"released": self._release(p["sid"])})

    def _release(self, sid):
        s = self.slot
        if s.sid != sid:
            return False
        if s.state == "claimed":
            log.info("claim %s released before the websocket", sid)
            self._outcome(sid, "released", None, None)
            s.to_idle()
            return True
        if s.state == "busy" and self.active_stop is not None:
            log.info("release of sid %s during the call: ending it", sid)
            self.active_stop("client_closed")
            return True
        return False

    async def handle_public_metrics(self, request):
        p, err = self._verify(request.headers.get(TOKEN_HEADER) or request.query.get("token"))  # CRIT-3
        if err is not None:
            return err
        if self.slot.sid != p["sid"] or self.slot.state not in ("claimed", "busy") or self.eng is None:
            return web.json_response({"error": "not_active"}, status=403)
        return web.json_response(self.eng.metrics())

    # ---------------------------------------------------------------------------------------------- chat
    async def handle_chat(self, request):
        """Pre-upgrade checks (DESIGN 3.1), then the DEP1 chat handler. Refusals are HTTP errors with a JSON body."""
        nr = self._not_ready()
        if nr is not None:
            return nr
        p, err = self._verify(request.headers.get(TOKEN_HEADER) or request.query.get("token"))
        if err is not None:
            return err
        try:
            cfg = resolve_session_config(request.query)
        except BadConfig as e:
            return web.json_response({"error": str(e)}, status=400)
        q = request.query
        if not self.allow_free_prompt and ((q.get("text_prompt") or "").strip() or (q.get("voice_prompt") or "").strip()
                                           or not (q.get("record_id") or "").strip()):
            return web.json_response({"error": "free_prompt_disabled",
                                      "detail": "a record_id is required; text_prompt/voice_prompt are not allowed"},
                                     status=400)
        sid = p["sid"]
        tok_cfg = norm_cfg(p["record_id"], p["pairing"], p["seed"])
        try:
            q_cfg = norm_cfg(q.get("record_id"), q.get("pairing"), q.get("seed"))
        except ValueError:
            return web.json_response({"error": "bad seed"}, status=400)
        if tok_cfg != q_cfg:
            log.info("ws sid %s refused: cfg_mismatch token=%s query=%s", sid, tok_cfg, q_cfg)
            return web.json_response({"error": "cfg_mismatch"}, status=401)
        self._expire()
        if sid in self.replay:
            log.info("ws sid %s refused: replayed", sid)
            return web.json_response({"error": "replayed"}, status=401)
        s = self.slot
        if s.state == "busy" or (s.state == "claimed" and s.sid != sid) or self.shutting_down:
            log.info("ws sid %s refused: busy (slot %s %s)", sid, s.state, s.sid)
            return web.json_response({"error": "busy"}, status=409)
        if s.state != "claimed":
            return web.json_response({"error": "no_claim"}, status=409)
        if s.cfg != tok_cfg:
            return web.json_response({"error": "cfg_mismatch"}, status=401)
        # accept: no await between the checks above and this transition (single event loop => no race)
        max_s = max(1, min(int(p["max_s"]), self.call_max_s, s.max_s or self.call_max_s))
        s.state, s.t_busy = "busy", time.time()
        self.last_metrics = None
        self.replay[sid] = max(int(p["exp"]), int(time.time())) + 60
        while len(self.replay) > REPLAY_MAX:
            self.replay.popitem(last=False)
        holder = {}
        try:
            return await self._chat(request, cfg, sid, max_s, holder)
        finally:
            summ = holder.get("summary")
            self._outcome(sid, "ended", holder.get("end_reason", "error"), summ)
            self.sessions_served += 1
            self.active_stop = None
            s.to_idle()
            log.info("sid %s done (%s); worker idle (%d served)", sid, holder.get("end_reason"), self.sessions_served)

    async def _chat(self, request, cfg, sid_s2s, max_s, holder):
        """DEP1 handle_chat body. S2S changes are marked 'S2S'."""
        want_events = request.query.get("events") == "1"
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        peer = request.remote
        log.info("incoming connection from %s sid %s cfg record=%s pairing=%s voice=%s seed=%s events=%s max_s=%d",
                 peer, sid_s2s, cfg["record_id"], cfg["pairing"], cfg["voice"], cfg["seed"], want_events, max_s)
        close = False
        end_reason = "client_closed"
        holder["end_reason"] = end_reason

        def stop(reason, force=False):          # S2S: time cap / SIGTERM / release end the call from outside
            nonlocal close, end_reason
            if force or not close:
                end_reason = reason
                holder["end_reason"] = reason
                close = True

        self.active_stop = stop

        async def watchdog():                   # S2S: CALL_MAX_S from the upgrade, including the prompt phase
            await asyncio.sleep(max_s)
            if not close:
                log.info("sid %s: call cap %d s reached", sid_s2s, max_s)
                stop("time_limit")

        wd_task = asyncio.create_task(watchdog())
        try:
            return await self._chat_body(ws, cfg, sid_s2s, want_events, holder, lambda: close,
                                         lambda: end_reason, stop)
        finally:
            wd_task.cancel()

    async def _chat_body(self, ws, cfg, sid_s2s, want_events, holder, is_closed, get_reason, stop):
        async with self.lock:
            if ws.closed:
                return ws
            eng = self.eng
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
            out_q = asyncio.Queue()
            handshake_sent = False

            async def recv_loop():
                try:
                    async for message in ws:
                        if message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED,
                                            aiohttp.WSMsgType.CLOSE):
                            break
                        if message.type != aiohttp.WSMsgType.BINARY:
                            continue
                        data = message.data
                        if not data:
                            continue
                        if data[0] == 1:
                            if handshake_sent:          # stock drops client messages during the prompt phase
                                opus_reader.append_bytes(data[1:])
                        elif data[0] == KIND_CONTROL:    # S2S: in-band control from the Space (turn-filler toggle)
                            self._control(eng, data[1:], sid_s2s)
                        else:
                            log.warning("unknown message kind %d", data[0])
                finally:
                    stop("client_closed")

            recv_task = asyncio.create_task(recv_loop())
            try:
                sid = await self.gpu_call(lambda: eng.start_session(cfg, should_abort=is_closed))
            except Exception as e:
                if not is_closed():
                    stop("error")                       # S2S: a prompt-phase failure is not a client close
                recv_task.cancel()
                log.warning("prompt phase ended: %r (reason %s)", e, get_reason())
                if get_reason() in ("time_limit", "worker_shutdown") and want_events and not ws.closed:
                    # S2S: tell the client why, even though the conversation never started
                    await ws.send_bytes(b"\x07" + json.dumps({"type": "session_end", "reason": get_reason(),
                                                              "frames": 0, "t_wall": time.time()}).encode())
                holder["summary"] = {"end_reason": get_reason(), "frames": 0, "aborted_in": "prompt_phase"}
                if not ws.closed:
                    await ws.close()
                return ws
            self.hub.session_id = sid
            if want_events:
                self.hub.chat_q = out_q
            log.info("session %s (sid %s): prompt phase %.2f s, %d KV positions, %d frames left",
                     sid, sid_s2s, eng.prompt_phase_s, eng.prompt_positions, eng.context_frames_left())
            if is_closed() or ws.closed:
                self.hub.chat_q = None
                holder["summary"] = eng.end_session()
                holder["summary"]["end_reason"] = get_reason()
                return ws
            await ws.send_bytes(b"\x00")
            handshake_sent = True
            sess_ev = dict(cfg)
            sess_ev.update({"type": "session", "session_id": sid,
                            "context_frames_left": eng.context_frames_left(),
                            "prompt_positions": eng.prompt_positions, "prompt_phase_s": eng.prompt_phase_s})
            self.hub.publish(sess_ev)

            async def process_loop():
                buf = np.zeros(0, np.float32)
                while not is_closed():
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] == 0:
                        await asyncio.sleep(0.002)
                        continue
                    buf = np.concatenate([buf, pcm.astype(np.float32).reshape(-1)])
                    while len(buf) >= FRAME_SIZE and not is_closed():
                        chunk, buf = buf[:FRAME_SIZE], buf[FRAME_SIZE:]
                        out = await self.gpu_call(eng.step_frame, chunk)
                        opus_writer.append_pcm(out.pcm)
                        if out.piece is not None:
                            out_q.put_nowait(b"\x02" + out.piece.encode("utf-8"))
                            self.hub.publish({"type": "text", "frame": out.frame, "t": round(out.frame / FRAME_RATE, 3),
                                              "token": out.token, "piece": out.piece, "forced": out.forced})
                        bl = len(buf) // FRAME_SIZE
                        self.backlog["last_frames"] = bl
                        self.backlog["max_frames"] = max(self.backlog["max_frames"], bl)
                        if eng.context_full():
                            stop("context_full")
                            return

            async def send_loop():
                while True:
                    msg = opus_writer.read_bytes()
                    if len(msg) > 0:
                        await ws.send_bytes(b"\x01" + msg)
                    while not out_q.empty():
                        item = out_q.get_nowait()
                        if item is None:        # sentinel queued after session_end: everything is flushed
                            return
                        await ws.send_bytes(item)
                    await asyncio.sleep(0.001)

            async def metrics_loop():
                while not is_closed():
                    await asyncio.sleep(2.0)
                    m = eng.metrics()
                    self.last_metrics = m                       # S2S: /status reads this cache
                    self.last_step_p95 = (m.get("step_ms") or {}).get("p95")
                    m = dict(m)
                    m["type"] = "metrics"
                    self.hub.publish(m)

            self.backlog["last_frames"] = 0
            self.backlog["max_frames"] = 0
            tasks = [asyncio.create_task(process_loop()), asyncio.create_task(send_loop()),
                     asyncio.create_task(metrics_loop())]
            try:
                done, _ = await asyncio.wait([recv_task, tasks[0]], return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is not None:
                        stop("error", force=True)
                        log.error("session task failed: %r", t.exception())
                if not is_closed():
                    stop("client_closed")
            finally:
                stop("client_closed")                # no-op if a reason is already set
                end_reason = get_reason()
                summary = await self.gpu_call(eng.end_session)
                self.hub.publish({"type": "session_end", "reason": end_reason, "frames": summary["frames"]})
                out_q.put_nowait(None)
                try:
                    await asyncio.wait_for(tasks[1], timeout=2.0)   # flush audio/text/events
                except Exception:
                    pass
                for t in tasks + [recv_task]:
                    t.cancel()
                self.hub.chat_q = None
                if not ws.closed:
                    await ws.close()
                summary["end_reason"] = end_reason
                summary["max_backlog_frames"] = self.backlog["max_frames"]
                summary["cfg"] = {k: cfg.get(k) for k in ("record_id", "pairing", "voice", "seed", "turn_fill")}
                summary.update({"sid": sid_s2s, "worker_id": self.worker_id, "gpu": self.gpuinfo.get("gpu"),
                                "mode": self.mode})             # S2S (never the token)
                st = summary.get("step_ms_excl_first25") or {}
                if st.get("p95") is not None:
                    self.last_step_p95 = st["p95"]
                holder["summary"] = summary
                holder["end_reason"] = end_reason
                log.info("session %s ended (%s): %s", sid, end_reason, json.dumps(summary))
                print("SESSION_SUMMARY " + json.dumps(summary), flush=True)
                if self.args.session_log:
                    try:
                        os.makedirs(os.path.dirname(self.args.session_log), exist_ok=True)
                        with open(self.args.session_log, "a") as f:
                            f.write(json.dumps(summary) + "\n")
                    except OSError as e:
                        log.warning("session log: %r", e)
        return ws

    def _control(self, eng, payload, sid_s2s):
        """S2S 2026-10-06: kind 0x08 JSON control. Only {"type":"turn_fill","mode":"ticker"|"off"} is known; it is
        queued on the filler and applied by the GPU thread at the next frame (TurnFiller.request_mode)."""
        try:
            msg = json.loads(bytes(payload).decode("utf-8"))
        except Exception:
            log.warning("sid %s: bad control frame", sid_s2s)
            return
        if not isinstance(msg, dict) or msg.get("type") != "turn_fill" or msg.get("mode") not in TURN_FILL_MODES:
            log.warning("sid %s: unknown control %r", sid_s2s, str(msg)[:120])
            return
        f = getattr(eng, "filler", None)
        if f is None:
            return
        f.request_mode(msg["mode"])
        log.info("sid %s: turn filler -> %s (mid-call toggle)", sid_s2s, msg["mode"])
        self.hub.publish({"type": "turn_fill", "mode": msg["mode"], "t_wall": time.time()})

    # ------------------------------------------------------------------------------------- internal HTTP
    def _need_eng(self):
        if self.eng is None:
            return web.json_response({"error": "loading"}, status=503)
        return None

    async def handle_metrics(self, request):
        r = self._need_eng()
        return r if r is not None else web.json_response(self.eng.metrics())

    async def handle_internal_claim(self, request):
        """S2S queue mode: rp_handler claims the job's sid in process (no token; the job came through /run, which
        needs the RunPod key). Same semantics as /session/claim (DESIGN 2.2 step 2b)."""
        nr = self._not_ready()
        if nr is not None:
            return nr
        try:
            b = await request.json()
            sid = str(b["sid"])
            cfg = norm_cfg(b.get("record_id"), b.get("pairing"), b.get("seed"))
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"need JSON {{sid, record_id, pairing, seed}}: {e!r}"}, status=400)
        st, body = self._claim(sid, cfg, self.call_max_s, "internal")
        return web.json_response(body, status=st)

    async def handle_internal_release(self, request):
        """S2S queue mode: POST /internal/claim_release {"sid"} (handler gives a claim back, e.g. no public port)."""
        try:
            sid = str((await request.json())["sid"])
        except Exception:  # noqa: BLE001
            return web.json_response({"error": 'need {"sid"}'}, status=400)
        return web.json_response({"released": self._release(sid)})

    async def handle_internal_claim_state(self, request):
        """S2S queue mode: GET /internal/claim/{sid} -> {sid, phase: claimed|busy|ended|expired|released|unknown,
        end_reason, summary}; the handler blocks on this until the call is over."""
        self._expire()
        sid = request.match_info["sid"]
        s = self.slot
        if s.sid == sid and s.state in ("claimed", "busy"):
            return web.json_response({"sid": sid, "phase": s.state, "end_reason": None, "summary": None})
        o = self.outcomes.get(sid)
        if o:
            return web.json_response(dict(o, sid=sid))
        return web.json_response({"sid": sid, "phase": "unknown", "end_reason": None, "summary": None})

    async def handle_events(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        q = asyncio.Queue(maxsize=2000)
        self.hub.subs.add(q)
        try:
            async def pump():
                while True:
                    await ws.send_str(await q.get())
            pt = asyncio.create_task(pump())
            async for _ in ws:      # ignore anything the subscriber sends; ends on close
                pass
            pt.cancel()
        finally:
            self.hub.subs.discard(q)
        return ws

    async def handle_session(self, request):
        e = self.eng
        if e is None or not e.active:
            return web.json_response({"active": False})
        d = dict(e.cfg)
        d.update({"active": True, "session_id": e.session_id, "frame": e.frame,
                  "context_frames_left": e.context_frames_left()})
        return web.json_response(d)

    async def handle_ring(self, request):
        if self.eng is None:
            return self._need_eng()
        try:
            secs = float(request.query.get("seconds", "30"))
        except ValueError:
            return web.json_response({"error": "bad seconds"}, status=400)
        secs = max(0.0, min(30.0, secs))
        frame = self.eng.frame
        pcm, end = self.eng.ring.get_last_with_end(secs)
        return web.Response(body=pcm.astype("<f4").tobytes(), content_type="application/octet-stream",
                            headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-End-Sample": str(end),
                                     "X-End-Frame": str(frame)})

    async def handle_action(self, request):
        if self.eng is None:
            return self._need_eng()
        try:
            ev = await request.json()
            assert isinstance(ev, dict) and ev.get("stage") in ACTION_STAGES
        except Exception:
            return web.json_response({"ok": False, "error": f"need JSON object with stage in {ACTION_STAGES}"},
                                     status=400)
        ev["type"] = "action"
        ev.pop("session_id", None)
        ev.pop("t_wall", None)
        ev.setdefault("frame", self.eng.frame)
        self.eng.note_action(ev["stage"])
        self.hub.publish(ev)
        return web.json_response({"ok": True})

    async def handle_inject(self, request):
        if self.eng is None or not self.eng.active:
            return web.json_response({"error": "no active session"}, status=409)
        try:
            body = await request.json()
            plan = self.eng.inject(body["words"], int(body.get("start_frame", 0)))
        except Exception as e:
            return web.json_response({"error": repr(e)}, status=400)
        return web.json_response(plan)

    async def handle_mute(self, request):
        if self.eng is None:
            return self._need_eng()
        try:
            on = bool((await request.json())["on"])
        except Exception:
            return web.json_response({"ok": False, "error": 'need {"on": bool}'}, status=400)
        self.eng.mute(on)
        return web.json_response({"ok": True, "mute": self.eng.muted})

    async def handle_play(self, request):
        if self.eng is None:
            return self._need_eng()
        body = await request.read()
        if len(body) % 4:
            return web.json_response({"ok": False, "error": "body must be float32 LE"}, status=400)
        q = self.eng.play(np.frombuffer(body, dtype="<f4"))
        return web.json_response({"ok": True, "queued_s": q})

    # ------------------------------------------------------------------------------------------- apps
    def public_app(self):
        app = web.Application(client_max_size=1 * 2**20)
        app.router.add_get("/ping", self.handle_ping)
        hc = os.environ.get("HEALTH_CHECK_PATH", "/ping")
        if hc and hc != "/ping":                      # [LB-OV] health path override, if the endpoint sets one
            app.router.add_get(hc, self.handle_ping)
        app.router.add_get("/status", self.handle_public_status)
        app.router.add_post("/session/claim", self.handle_claim)
        app.router.add_post("/session/release", self.handle_release)
        app.router.add_get("/api/chat", self.handle_chat)
        app.router.add_get("/metrics", self.handle_public_metrics)
        return app

    def internal_app(self):
        app = web.Application(client_max_size=256 * 2**20)
        app.router.add_get("/internal/events", self.handle_events)
        app.router.add_get("/internal/session", self.handle_session)
        app.router.add_get("/internal/ring", self.handle_ring)
        app.router.add_post("/internal/action", self.handle_action)
        app.router.add_post("/internal/inject", self.handle_inject)
        app.router.add_post("/internal/mute", self.handle_mute)
        app.router.add_post("/internal/play", self.handle_play)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_post("/internal/claim", self.handle_internal_claim)            # S2S
        app.router.add_get("/internal/claim/{sid}", self.handle_internal_claim_state)  # S2S
        app.router.add_post("/internal/claim_release", self.handle_internal_release)   # S2S
        app.router.add_get("/internal/status", self.handle_status)                    # S2S
        return app


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=paths.PORT)
    ap.add_argument("--internal-port", type=int, default=paths.INTERNAL_PORT)
    ap.add_argument("--adapter", default=paths.ADAPTER, help="'' = base model; 'premerged' = no merge")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--session-log", default=str(paths.LOGS / "server_sessions.jsonl"))
    ap.add_argument("--mock", action="store_true", help="CPU MockEngine (DEP1 mock_server) instead of PersonaPlex")
    ap.add_argument("--mock-call", default="food_07_g1")
    ap.add_argument("--mock-prompt-s", type=float, default=2.0)
    ap.add_argument("--mock-load-s", type=float, default=0.0, help="simulated load time (tests: /ping 204 -> 200)")
    ap.add_argument("--mock-context", type=int, default=3000, help="KV positions; lower to test context_full")
    ap.add_argument("--mock-ring", choices=["input", "client"], default="input")
    ap.add_argument("--mock-fail", default="", help="tests: make the load fail with this message (/ping 500)")
    return ap


async def serve(args, engine_factory):
    srv = Server(args, engine_factory)
    srv.loop = asyncio.get_running_loop()
    srv.gpuinfo = gpu_info()
    g = srv.gpuinfo
    log.info("GPU %s driver %s CUDA %s VRAM %s MiB (free %s); worker_id %s mode %s",
             g.get("gpu"), g.get("driver"), g.get("cuda"), g.get("vram_total_mib"), g.get("vram_free_mib"),
             srv.worker_id, srv.mode)
    slow = [x.strip() for x in os.environ.get("S2S_SLOW_GPUS", "L4,A4000,A4500,RTX 4000,A2000").split(",") if x.strip()]
    name = (g.get("gpu") or "").upper()
    hit = [x for x in slow if re.search(r"(?<![A-Z0-9])" + re.escape(x.upper()) + r"(?![A-Z0-9])", name)]
    if args.mock is False and hit:
        log.warning("WARNING: GPU %r matches S2S_SLOW_GPUS %s: the 80 ms frame budget is likely missed (DESIGN R4)",
                    g.get("gpu"), hit)
    r1 = web.AppRunner(srv.public_app(), access_log=None)
    r2 = web.AppRunner(srv.internal_app(), access_log=None)
    await r1.setup()
    await r2.setup()
    await web.TCPSite(r1, args.host, args.port).start()          # S2S: bind BEFORE the load (/ping 204 meanwhile)
    await web.TCPSite(r2, "127.0.0.1", args.internal_port).start()
    log.info("public http://%s:%d  internal http://127.0.0.1:%d  (asr :%d router :%d)", args.host, args.port,
             args.internal_port, paths.ASR_PORT, paths.ROUTER_PORT)
    print("LISTENING", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        srv.loop.add_signal_handler(sig, srv.on_signal, sig)
    tasks = [asyncio.create_task(srv.load_task()), asyncio.create_task(srv.health_loop()),
             asyncio.create_task(srv.tick_loop())]
    await srv.exit_event.wait()
    for t in tasks:
        t.cancel()
    try:
        await asyncio.wait_for(r1.cleanup(), timeout=2.0)
    except Exception:  # noqa: BLE001
        pass
    log.info("worker_server exit")


def fetch_assets():
    """Run worker/fetch_assets.py with the venv-asr interpreter (its huggingface_hub + hf-xet are current; venv-pp's
    hub is 0.24). Raises with the script's last lines on failure, so /ping answers 500 with the reason."""
    if os.environ.get("S2S_SKIP_FETCH_ASSETS") == "1":
        return
    py = os.environ.get("S2S_PY_ASR") or sys.executable
    if not os.path.exists(py):
        py = sys.executable
    t0 = time.time()
    r = subprocess.run([py, "-u", str(HERE.parent / "fetch_assets.py")], capture_output=True, text=True,
                       timeout=int(os.environ.get("S2S_FETCH_TIMEOUT_S", "900")))
    for line in (r.stdout + r.stderr).splitlines()[-20:]:
        log.info("%s", line)
    if r.returncode != 0:
        tail = " | ".join((r.stdout + r.stderr).strip().splitlines()[-3:])
        raise RuntimeError(f"fetch_assets failed (exit {r.returncode}): {tail[:400]}")
    log.info("assets ready in %.1f s", time.time() - t0)


def make_engine_factory(args):
    if args.mock_fail:
        def fail():
            time.sleep(args.mock_load_s)
            raise RuntimeError(args.mock_fail)
        return fail
    if args.mock:
        from mock_server import MockEngine
        return lambda: MockEngine(args.mock_call, prompt_s=args.mock_prompt_s, ring=args.mock_ring,
                                  context=args.mock_context, load_s=args.mock_load_s)

    def real():
        fetch_assets()                          # 2026-10-06: V4 adapter + Needle weights (not baked; fetch_assets.py)
        import resolve_models                   # worker/resolve_models.py (DESIGN 6.3), run inside the load task
        r = resolve_models.resolve()
        from engine import Engine, pp_files_from_dir
        return Engine(adapter=args.adapter, device=args.device, pp_files=pp_files_from_dir(r["pp_dir"]),
                      voice_prompt_dir=r["voices"])
    return real


def main(argv=None, engine_factory=None, parser=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = (parser or build_parser()).parse_args(argv)
    if engine_factory is None:
        engine_factory = make_engine_factory(args)
    asyncio.run(serve(args, engine_factory))
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)     # the GPU thread may still be inside a load; do not join it


if __name__ == "__main__":
    main()
