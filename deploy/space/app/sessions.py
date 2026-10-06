"""Space session table, state machine and cost brakes (DESIGN.md section 5.1).

Everything is in memory: a Space restart drops all sessions.

States (GET /api/session/{sid}):
  waking   -> the wake task is polling the endpoint (LB: GET /status; queue: /run + /status/{job})
  busy     -> LB claim kept getting 409 (every worker is in a call); still retrying
  ready    -> a worker is claimed for this sid; the browser must open /api/chat within CLAIM_TTL_S
  in_call  -> the relay is up
  ended    -> the call is over (end_reason says why) or the session was cancelled / expired
  failed   -> the wake did not succeed (detail says why)
"""
import collections
import hmac
import os
import time
from dataclasses import dataclass, field
from typing import Optional

ACTIVE = ("waking", "busy", "ready", "in_call")
FINAL = ("ended", "failed")
GC_AFTER_S = 600          # drop ended/failed sessions 10 min after they ended


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


@dataclass
class Config:
    """All Space settings, from env (ops/ENV.md and space/README.md list them). Secrets are never logged."""
    mode: str = "lb"                         # S2S_MODE: lb | queue (must match the worker) | local (D-LOCAL: no RunPod)
    endpoint_id: str = ""                    # RUNPOD_ENDPOINT_ID
    api_key: str = ""                        # RUNPOD_API_KEY (secret)
    lb_url: str = ""                         # RUNPOD_LB_URL, default https://{id}.api.runpod.ai
    api_url: str = ""                        # RUNPOD_API_URL, default https://api.runpod.ai/v2/{id}
    secret: str = ""                         # S2S_SESSION_SECRET (secret, same value on the worker)
    audience: str = ""                       # S2S_AUDIENCE, default the endpoint id
    token_ttl_s: int = 120                   # S2S_TOKEN_TTL_S
    call_max_s: int = 300                    # CALL_MAX_S (the worker's value is authoritative)
    max_concurrent: int = 1                  # MAX_CONCURRENT_CALLS (= endpoint max workers)
    rate_per_ip_per_hour: int = 6            # RATE_PER_IP_PER_HOUR
    max_calls_per_day: int = 100             # MAX_CALLS_PER_DAY (all IPs, UTC day)
    passcode: str = ""                       # S2S_PASSCODE (secret; empty = no passcode)
    wake_poll_s: float = 3.0                 # WAKE_POLL_S
    wake_timeout_s: float = 600.0            # WAKE_TIMEOUT_S
    wake_http_timeout_s: float = 130.0       # WAKE_HTTP_TIMEOUT_S (LB holds a request up to 2 min with no worker)
    claim_ttl_s: float = 90.0                # CLAIM_TTL_S (same as the worker's)
    abandon_s: float = 120.0                 # ABANDON_S: cancel a waking/busy session nobody polls (0 = off; REVIEW-space-3)
    open_timeout_s: float = 60.0             # UPSTREAM_OPEN_TIMEOUT_S
    keepalive_s: float = 20.0                # KEEPALIVE_S (0 = off; hold test #3)
    keepalive_path: str = "/status"          # KEEPALIVE_PATH (/status or /ping; hold test #3)
    keepalive_timeout_s: float = 10.0        # KEEPALIVE_TIMEOUT_S (non-fatal)
    queue_exec_timeout_ms: int = 900000      # QUEUE_EXEC_TIMEOUT_MS (job policy executionTimeout; REVIEW-ops-3)
    queue_ttl_ms: int = 1200000              # QUEUE_TTL_MS (job policy ttl)
    queue_poll_s: float = 2.0                # QUEUE_POLL_S
    allow_free_prompt: bool = False          # S2S_ALLOW_FREE_PROMPT (must match the worker; default off)
    token_in_query: bool = False             # S2S_TOKEN_IN_QUERY: LB also sends the token as ?token= (CRIT-3, O-W1)
    selftest: bool = False                   # S2S_SELFTEST (outbound-port probe route; off by default)
    host: str = "0.0.0.0"                    # HOST
    port: int = 7860                         # PORT
    build: str = ""                          # S2S_BUILD (free text shown in /api/config)
    turn_fill_default: str = "ticker"        # S2S_TURN_FILL_DEFAULT: ticker | off (D-TOGGLE port, 2026-10-06)
    worker_url: str = ""                     # S2S_WORKER_URL (local mode only): the worker's public port, default http://127.0.0.1:8000

    @classmethod
    def from_env(cls) -> "Config":
        ep = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
        mode = os.environ.get("S2S_MODE", "lb").strip().lower() or "lb"
        local = mode == "local"                  # D-LOCAL: one user on their own GPU; no per-IP / per-day brakes by default
        c = cls(
            mode=mode,
            endpoint_id=ep,
            api_key=os.environ.get("RUNPOD_API_KEY", "").strip(),
            lb_url=(os.environ.get("RUNPOD_LB_URL", "").strip() or f"https://{ep}.api.runpod.ai").rstrip("/"),
            api_url=(os.environ.get("RUNPOD_API_URL", "").strip() or f"https://api.runpod.ai/v2/{ep}").rstrip("/"),
            secret=os.environ.get("S2S_SESSION_SECRET", ""),
            audience=os.environ.get("S2S_AUDIENCE", "").strip() or ("local" if local else ep),
            token_ttl_s=_env_int("S2S_TOKEN_TTL_S", 120),
            call_max_s=_env_int("CALL_MAX_S", 300),
            max_concurrent=_env_int("MAX_CONCURRENT_CALLS", 1),
            rate_per_ip_per_hour=_env_int("RATE_PER_IP_PER_HOUR", 100000 if local else 6),
            max_calls_per_day=_env_int("MAX_CALLS_PER_DAY", 100000 if local else 100),
            passcode=os.environ.get("S2S_PASSCODE", ""),
            wake_poll_s=_env_float("WAKE_POLL_S", 3.0),
            wake_timeout_s=_env_float("WAKE_TIMEOUT_S", 600.0),
            wake_http_timeout_s=_env_float("WAKE_HTTP_TIMEOUT_S", 130.0),
            claim_ttl_s=_env_float("CLAIM_TTL_S", 90.0),
            abandon_s=_env_float("ABANDON_S", 120.0),
            open_timeout_s=_env_float("UPSTREAM_OPEN_TIMEOUT_S", 60.0),
            keepalive_s=_env_float("KEEPALIVE_S", 20.0),
            keepalive_path=os.environ.get("KEEPALIVE_PATH", "/status").strip() or "/status",
            keepalive_timeout_s=_env_float("KEEPALIVE_TIMEOUT_S", 10.0),
            queue_exec_timeout_ms=_env_int("QUEUE_EXEC_TIMEOUT_MS", 900000),
            queue_ttl_ms=_env_int("QUEUE_TTL_MS", 1200000),
            queue_poll_s=_env_float("QUEUE_POLL_S", 2.0),
            allow_free_prompt=os.environ.get("S2S_ALLOW_FREE_PROMPT", "0").strip() == "1",
            token_in_query=os.environ.get("S2S_TOKEN_IN_QUERY", "0").strip() == "1",
            selftest=os.environ.get("S2S_SELFTEST", "0").strip() == "1",
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_env_int("PORT", 7860),
            build=os.environ.get("S2S_BUILD", ""),
            turn_fill_default=os.environ.get("S2S_TURN_FILL_DEFAULT", "ticker").strip().lower() or "ticker",
            worker_url=(os.environ.get("S2S_WORKER_URL", "").strip() or "http://127.0.0.1:8000").rstrip("/"),
        )
        return c

    def problems(self) -> list:
        """Config errors that make the GPU path unusable (the Space still serves the client and records)."""
        out = []
        if self.mode not in ("lb", "queue", "local"):
            out.append(f"S2S_MODE must be lb, queue or local, got {self.mode!r}")
        if self.turn_fill_default not in ("ticker", "off"):
            out.append(f"S2S_TURN_FILL_DEFAULT must be ticker or off, got {self.turn_fill_default!r}")
        if self.mode == "local":                 # D-LOCAL: no RunPod key / endpoint; the worker shares the secret
            if not self.worker_url.startswith(("http://", "https://")):
                out.append(f"S2S_WORKER_URL must be an http(s) URL, got {self.worker_url!r}")
            if len(self.secret) < 32:
                out.append("S2S_SESSION_SECRET is not set or shorter than 32 chars (use the worker's value)")
            return sorted(set(out))
        url_var = "RUNPOD_LB_URL" if self.mode == "lb" else "RUNPOD_API_URL"
        if not self.endpoint_id and not os.environ.get(url_var, "").strip():
            out.append("RUNPOD_ENDPOINT_ID is not set")
        if not self.api_key:
            out.append("RUNPOD_API_KEY is not set")
        if len(self.secret) < 32:
            out.append("S2S_SESSION_SECRET is not set or shorter than 32 chars")
        if not self.audience:
            out.append("S2S_AUDIENCE / RUNPOD_ENDPOINT_ID is not set")
        return sorted(set(out))


@dataclass
class Session:
    sid: str
    record_id: Optional[str]
    pairing: Optional[str]
    seed: Optional[int]
    ip: str
    created: float = field(default_factory=time.time)
    state: str = "waking"
    detail: str = "starting"
    t_state: float = field(default_factory=time.time)
    worker_id: Optional[str] = None          # LB: the proxy's X-Runpod-Worker-Id (or the claim body); queue: RUNPOD_POD_ID
    ws_url: Optional[str] = None             # queue: ws://public_ip:tcp_port
    job_id: Optional[str] = None             # queue
    end_reason: Optional[str] = None
    ready_at: Optional[float] = None
    call_started: Optional[float] = None
    ended_at: Optional[float] = None
    last_metrics: Optional[dict] = None      # last 0x07 metrics event seen by the relay
    wake_task: object = None                 # asyncio.Task
    relay_close: object = None               # coroutine fn: closes the relay (DELETE during in_call)
    released: bool = False
    last_diag: float = 0.0
    last_seen: float = field(default_factory=time.time)   # last POST/GET /api/session by the browser (REVIEW-space-3)
    cancel_pending: bool = False             # DELETE arrived while the relay was still opening upstream (REVIEW-space-1)
    keepalive: dict = field(default_factory=lambda: {"sent": 0, "codes": collections.Counter(), "last": None})
    timeline: list = field(default_factory=list)   # [(state, t_since_created_s, detail)] for cold_start_timer
    send_control: object = None              # coroutine fn(bytes): in-band control frame to the worker (in_call only)
    turn_fill: Optional[str] = None          # turn-filler mode sent to the worker for this call

    def __post_init__(self):
        self.timeline.append((self.state, 0.0, self.detail))

    def set(self, state: str, detail: str = ""):
        changed = state != self.state
        self.state = state
        self.detail = detail
        if changed:
            self.t_state = time.time()
            self.timeline.append((state, round(self.t_state - self.created, 2), detail))
            if state == "ready":
                self.ready_at = self.t_state
            elif state == "in_call":
                self.call_started = self.t_state
            elif state in FINAL:
                self.ended_at = self.t_state

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def public(self) -> dict:
        d = {"sid": self.sid, "state": self.state, "detail": self.detail,
             "elapsed_s": round(time.time() - self.created, 1),
             "state_elapsed_s": round(time.time() - self.t_state, 1),
             "timeline": self.timeline}
        if self.worker_id:
            d["worker_id"] = self.worker_id
        if self.end_reason:
            d["end_reason"] = self.end_reason
        if self.state == "in_call" and self.keepalive["sent"]:
            d["keepalive"] = {"sent": self.keepalive["sent"], "last": self.keepalive["last"],
                              "codes": dict(self.keepalive["codes"])}
        return d


class SessionTable:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sessions: dict = {}
        self.per_ip: dict = collections.defaultdict(collections.deque)   # ip -> creation times (last hour)
        self.day = None
        self.day_count = 0

    def get(self, sid: Optional[str]) -> Optional[Session]:
        return self.sessions.get(sid) if sid else None

    def n_active(self) -> int:
        return sum(1 for s in self.sessions.values() if s.active)

    def check_passcode(self, given: Optional[str]) -> bool:
        if not self.cfg.passcode:
            return True
        return hmac.compare_digest((given or "").encode(), self.cfg.passcode.encode())

    def admit(self, ip: str, now: Optional[float] = None):
        """Cost brakes. Returns None if admitted, else (http_status, body)."""
        now = time.time() if now is None else now
        if self.n_active() >= self.cfg.max_concurrent:
            return 429, {"error": "capacity", "retry_after_s": 30}
        q = self.per_ip[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.cfg.rate_per_ip_per_hour:
            return 429, {"error": "rate_limited", "retry_after_s": int(3600 - (now - q[0])) + 1}
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            self.day, self.day_count = day, 0
        if self.day_count >= self.cfg.max_calls_per_day:
            return 429, {"error": "rate_limited", "retry_after_s": int(86400 - now % 86400) + 1}
        return None

    def add(self, sess: Session, now: Optional[float] = None):
        now = time.time() if now is None else now
        self.per_ip[sess.ip].append(now)
        self.day_count += 1
        self.sessions[sess.sid] = sess

    def expired_ready(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        return [s for s in self.sessions.values()
                if s.state == "ready" and s.ready_at and now - s.ready_at > self.cfg.claim_ttl_s]

    def abandoned(self, now: Optional[float] = None) -> list:
        """waking/busy sessions whose browser stopped polling GET /api/session (tab closed and the pagehide DELETE
        was lost): cancelling them stops a wake that would otherwise claim (and keep billing) a worker for nobody."""
        now = time.time() if now is None else now
        if self.cfg.abandon_s <= 0:
            return []
        return [s for s in self.sessions.values()
                if s.state in ("waking", "busy") and now - s.last_seen > self.cfg.abandon_s]

    def stuck_in_call(self, now: Optional[float] = None) -> list:
        """Backstop: in_call far past every cap (relay cap = CALL_MAX_S + 15 s after the open) means a leaked
        session; without this it would hold a MAX_CONCURRENT_CALLS slot until the Space restarts."""
        now = time.time() if now is None else now
        lim = self.cfg.call_max_s + self.cfg.open_timeout_s + 60
        return [s for s in self.sessions.values()
                if s.state == "in_call" and s.call_started and now - s.call_started > lim]

    def gc(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        dead = [k for k, s in self.sessions.items() if s.state in FINAL and s.ended_at and now - s.ended_at > GC_AFTER_S]
        for k in dead:
            del self.sessions[k]
        for ip in [ip for ip, q in self.per_ip.items() if not q or now - q[-1] > 3600]:
            del self.per_ip[ip]
        return len(dead)
