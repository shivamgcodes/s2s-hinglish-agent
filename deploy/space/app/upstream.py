"""RunPod access for the Space (DESIGN.md section 5.3). The relay and session code use only this interface.

  LBUpstream     load-balancing endpoint: https://ENDPOINT_ID.api.runpod.ai/<worker route>, Authorization: Bearer
                 wake = poll GET /status (NOT the health path /ping, DESIGN 2.1 step 3), then POST /session/claim;
                 websocket and keepalive pinned with X-Runpod-Worker-Id: strict <id> ([LB-AFF]).
  QueueUpstream  queue endpoint: POST /v2/ID/run wakes a worker, poll /status/{job} for the handler's
                 progress_update {state, public_ip, tcp_port, worker_id, sid}, then ws://ip:port (plaintext, R6).

Every upstream action carries a freshly minted session token (X-S2S-Token, DESIGN section 4) for the same sid.
The RunPod key and the token are never logged and never sent to the browser.
"""
import asyncio
import logging
import time

import aiohttp

import s2s_token  # common/ (sys.path set by app.main)

log = logging.getLogger("space.upstream")

READY_STATES = ("idle", "claimed", "busy")      # worker /status "state" once it is ready (DESIGN 3.1)
QUEUE_DONE = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


class WakeFailed(Exception):
    """The wake did not succeed; str(e) is shown to the user as the session detail."""


def _short(body: str, n: int = 160) -> str:
    body = (body or "").strip().replace("\n", " ")
    return body if len(body) <= n else body[:n] + "…"


class Upstream:
    mode = "?"

    def __init__(self, cfg, http: aiohttp.ClientSession):
        self.cfg = cfg
        self.http = http

    def token(self, sess) -> str:
        return s2s_token.mint(self.cfg.secret, sid=sess.sid, mode=self.mode, aud=self.cfg.audience,
                              record_id=sess.record_id, pairing=sess.pairing, seed=sess.seed,
                              max_s=int(self.cfg.call_max_s), ttl_s=int(self.cfg.token_ttl_s))

    def bearer(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.api_key}"}

    async def wake_and_claim(self, sess, on_state) -> None:
        raise NotImplementedError

    async def open_ws(self, sess, query: dict) -> aiohttp.ClientWebSocketResponse:
        raise NotImplementedError

    async def keepalive_loop(self, sess) -> None:
        return None

    async def release(self, sess) -> None:
        return None

    async def diag(self, sess) -> dict:
        raise NotImplementedError

    async def _ws_connect(self, url: str, query: dict, headers: dict):
        # The ClientSession has no total timeout (the relay socket lives for minutes); the open is bounded here.
        return await asyncio.wait_for(
            self.http.ws_connect(url, params=query, headers=headers, max_msg_size=0, heartbeat=None,
                                 autoping=True, compress=0),
            timeout=self.cfg.open_timeout_s)


# ------------------------------------------------------------------------------------------------------ LB
class LBUpstream(Upstream):
    mode = "lb"
    proxied = True                              # behind the RunPod LB proxy (sets X-Runpod-Worker-Id)

    def __init__(self, cfg, http):
        super().__init__(cfg, http)
        self.base = cfg.lb_url
        self.ws_base = self.base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)

    def tq(self, tok: str) -> dict:
        """S2S_TOKEN_IN_QUERY=1 (CRIT-3, O-W1): also send the token as ?token=, in case the RunPod LB proxy drops the
        custom X-S2S-Token header. The worker reads the header first, then ?token=. Off by default."""
        return {"token": tok} if self.cfg.token_in_query else {}

    def pinned(self, sess) -> dict:
        h = self.bearer()
        if sess.worker_id:
            h["X-Runpod-Worker-Id"] = f"strict {sess.worker_id}"
        return h

    async def wake_and_claim(self, sess, on_state) -> None:
        cfg = self.cfg
        deadline = time.time() + cfg.wake_timeout_s
        to = aiohttp.ClientTimeout(total=cfg.wake_http_timeout_s)
        n_poll = 0
        while time.time() < deadline:
            n_poll += 1
            ready = False
            try:
                async with self.http.get(self.base + "/status", headers=self.bearer(), timeout=to) as r:
                    body = await r.text()
                    if r.status == 200:
                        try:
                            st = (await r.json(content_type=None)).get("state")
                        except Exception:
                            st = None
                        if st in READY_STATES:
                            ready = True
                        else:
                            on_state("waking", f"worker initializing ({st or 'no state'})")
                    elif r.status in (401, 403):
                        raise WakeFailed(f"RunPod rejected the API key (HTTP {r.status}); check RUNPOD_API_KEY")
                    elif r.status == 204:
                        on_state("waking", "worker initializing")
                    else:
                        on_state("waking", f"no worker yet (HTTP {r.status} {_short(body, 60)})")
            except WakeFailed:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                on_state("waking", f"no worker yet ({type(e).__name__})")
            if ready:
                if await self._claim(sess, on_state):
                    return
            await asyncio.sleep(cfg.wake_poll_s)
        raise WakeFailed(f"no worker became ready within {cfg.wake_timeout_s:.0f} s "
                         + (f"(is the local worker running at {cfg.worker_url}? see its log)" if cfg.mode == "local"
                            else "(endpoint throttled to 0 workers? GPU type unavailable?)"))

    async def _claim(self, sess, on_state) -> bool:
        h = self.bearer()
        h["X-S2S-Token"] = self.token(sess)
        try:
            async with self.http.post(self.base + "/session/claim", headers=h, json={"sid": sess.sid},
                                      params=self.tq(h["X-S2S-Token"]), timeout=aiohttp.ClientTimeout(total=30)) as r:
                body = await r.text()
                if r.status == 200:
                    try:
                        js = await r.json(content_type=None)
                    except Exception:
                        js = {}
                    hdr = r.headers.get("X-Runpod-Worker-Id")
                    wid = hdr or js.get("worker_id")
                    if hdr and js.get("worker_id") and hdr != js.get("worker_id"):
                        log.info("sid %s: claim body worker_id %s != proxy header %s; using the header",
                                 sess.sid, js.get("worker_id"), hdr)
                    if not hdr and self.proxied:
                        log.warning("sid %s: no X-Runpod-Worker-Id on the claim response; pinning to body "
                                    "worker_id %s (R14)", sess.sid, wid)
                    sess.worker_id = wid
                    on_state("ready", f"worker {wid} claimed")
                    return True
                if r.status == 409:
                    on_state("busy", "all workers are in a call; retrying")
                    return False
                if r.status == 503:
                    on_state("waking", "worker initializing")
                    return False
                if r.status == 401:
                    hint = "check S2S_SESSION_SECRET / S2S_AUDIENCE / S2S_MODE on both sides"
                    if "missing" in body and not self.cfg.token_in_query:
                        hint = ("the token did not reach the worker: the RunPod proxy may drop the X-S2S-Token "
                                "header; set the Space variable S2S_TOKEN_IN_QUERY=1 (CRIT-3)")
                    raise WakeFailed(f"worker rejected the session token ({_short(body, 80)}); {hint}")
                on_state("waking", f"claim HTTP {r.status}")
                return False
        except WakeFailed:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            on_state("waking", f"claim failed ({type(e).__name__}); retrying")
            return False

    async def open_ws(self, sess, query: dict):
        h = self.pinned(sess)
        h["X-S2S-Token"] = self.token(sess)
        return await self._ws_connect(self.ws_base + "/api/chat", dict(query, **self.tq(h["X-S2S-Token"])), h)

    async def keepalive_loop(self, sess) -> None:
        """Strict-pinned GET every KEEPALIVE_S while the call is up (DESIGN 2.1 step 7). Never fatal."""
        cfg = self.cfg
        if cfg.keepalive_s <= 0:
            return
        to = aiohttp.ClientTimeout(total=cfg.keepalive_timeout_s)
        ka = sess.keepalive
        while True:
            await asyncio.sleep(cfg.keepalive_s)
            ka["sent"] += 1
            t0 = time.time()
            try:
                async with self.http.get(self.base + cfg.keepalive_path, headers=self.pinned(sess), timeout=to) as r:
                    await r.read()
                    code = str(r.status)
            except asyncio.TimeoutError:
                code = "timeout"
            except aiohttp.ClientError as e:
                code = type(e).__name__
            ka["codes"][code] += 1
            ka["last"] = {"code": code, "ms": round(1000 * (time.time() - t0)), "t": round(t0)}
            if code != "200":
                log.warning("sid %s: keepalive %s -> %s (%d ms)", sess.sid, cfg.keepalive_path, code,
                            ka["last"]["ms"])

    async def release(self, sess) -> None:
        if not sess.worker_id or sess.released:
            return
        sess.released = True
        h = self.pinned(sess)
        h["X-S2S-Token"] = self.token(sess)
        try:
            async with self.http.post(self.base + "/session/release", headers=h, json={"sid": sess.sid},
                                      params=self.tq(h["X-S2S-Token"]),
                                      timeout=aiohttp.ClientTimeout(total=self.cfg.keepalive_timeout_s)) as r:
                body = await r.text()
                log.info("sid %s: release -> HTTP %d %s", sess.sid, r.status, _short(body, 80))
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            log.warning("sid %s: release failed (%s); the worker frees the claim after CLAIM_TTL_S", sess.sid,
                        type(e).__name__)

    async def diag(self, sess) -> dict:
        t0 = time.time()
        try:
            async with self.http.get(self.base + "/status", headers=self.pinned(sess),
                                     timeout=aiohttp.ClientTimeout(total=self.cfg.keepalive_timeout_s)) as r:
                await r.read()
                code = r.status
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            code = type(e).__name__
        return {"space_to_worker_ms": round(1000 * (time.time() - t0), 1), "status": code, "mode": self.mode}


# --------------------------------------------------------------------------------------------------- queue
class QueueUpstream(Upstream):
    mode = "queue"

    def __init__(self, cfg, http):
        super().__init__(cfg, http)
        self.api = cfg.api_url

    async def wake_and_claim(self, sess, on_state) -> None:
        cfg = self.cfg
        body = {"input": {"sid": sess.sid, "record_id": sess.record_id, "pairing": sess.pairing, "seed": sess.seed},
                "policy": {"executionTimeout": int(cfg.queue_exec_timeout_ms), "ttl": int(cfg.queue_ttl_ms)}}
        try:
            async with self.http.post(self.api + "/run", headers=self.bearer(), json=body,
                                      timeout=aiohttp.ClientTimeout(total=30)) as r:
                txt = await r.text()
                if r.status in (401, 403):
                    raise WakeFailed(f"RunPod rejected the API key (HTTP {r.status}); check RUNPOD_API_KEY")
                if r.status != 200:
                    raise WakeFailed(f"/run failed: HTTP {r.status} {_short(txt, 80)}")
                js = await r.json(content_type=None)
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise WakeFailed(f"/run failed ({type(e).__name__})")
        sess.job_id = js.get("id")
        if not sess.job_id:
            raise WakeFailed(f"/run returned no job id: {_short(str(js), 80)}")
        log.info("sid %s: queue job %s submitted", sess.sid, sess.job_id)
        on_state("waking", "queued (waiting for a worker)")
        deadline = time.time() + cfg.wake_timeout_s
        to = aiohttp.ClientTimeout(total=30)
        while time.time() < deadline:
            await asyncio.sleep(cfg.queue_poll_s)
            try:
                async with self.http.get(f"{self.api}/status/{sess.job_id}", headers=self.bearer(), timeout=to) as r:
                    if r.status != 200:
                        on_state("waking", f"job status HTTP {r.status}")
                        continue
                    js = await r.json(content_type=None)
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                on_state("waking", f"job status failed ({type(e).__name__})")
                continue
            st = js.get("status")
            if st in QUEUE_DONE:
                raise WakeFailed(f"job {st} before the worker was ready: {_short(str(js.get('error') or js.get('output')), 120)}")
            if st == "IN_QUEUE":
                on_state("waking", "queued (waiting for a worker)")
                continue
            prog = js.get("output")
            if isinstance(prog, list):          # tolerate a list of progress updates: take the last dict
                prog = next((p for p in reversed(prog) if isinstance(p, dict)), None)
            if not isinstance(prog, dict):
                on_state("waking", "worker starting")
                continue
            if prog.get("state") == "ready":
                if prog.get("sid") not in (None, sess.sid):     # the job id already ties it to us; a different sid is a bug
                    raise WakeFailed("worker progress is for another sid")
                ip, port = prog.get("public_ip"), prog.get("tcp_port")
                if not ip or not port:
                    raise WakeFailed("worker is ready but has no public IP / TCP port (expose the TCP port on the endpoint)")
                sess.ws_url = f"ws://{ip}:{int(port)}"
                sess.worker_id = prog.get("worker_id")
                on_state("ready", f"worker {sess.worker_id} at {ip}:{port}")
                return
            on_state("waking", "worker initializing" if prog.get("state") == "loading" else f"worker {prog.get('state')}")
        await self.release(sess)
        raise WakeFailed(f"no worker became ready within {cfg.wake_timeout_s:.0f} s")

    async def open_ws(self, sess, query: dict):
        return await self._ws_connect(sess.ws_url + "/api/chat", query, {"X-S2S-Token": self.token(sess)})

    async def release(self, sess) -> None:
        if not sess.job_id or sess.released:
            return
        sess.released = True
        try:
            async with self.http.post(f"{self.api}/cancel/{sess.job_id}", headers=self.bearer(),
                                      timeout=aiohttp.ClientTimeout(total=self.cfg.keepalive_timeout_s)) as r:
                body = await r.text()
                log.info("sid %s: cancel job %s -> HTTP %d %s", sess.sid, sess.job_id, r.status, _short(body, 80))
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            log.warning("sid %s: cancel failed (%s)", sess.sid, type(e).__name__)

    async def diag(self, sess) -> dict:
        t0 = time.time()
        url = sess.ws_url.replace("ws://", "http://", 1) + "/status"
        try:
            async with self.http.get(url, timeout=aiohttp.ClientTimeout(total=self.cfg.keepalive_timeout_s)) as r:
                await r.read()
                code = r.status
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            code = type(e).__name__
        return {"space_to_worker_ms": round(1000 * (time.time() - t0), 1), "status": code, "mode": self.mode}


# --------------------------------------------------------------------------------------------------- local
class LocalUpstream(LBUpstream):
    """D-LOCAL (2026-10-06): the worker runs on this machine (or the LAN) in S2S_MODE=lb, without RunPod. Same claim /
    websocket / release flow and session token as LB (the token says mode "lb", which the worker checks), but straight
    to S2S_WORKER_URL: no RunPod Bearer, no X-Runpod-Worker-Id pin, no keepalive (nothing scales the worker down)."""
    proxied = False

    def __init__(self, cfg, http):
        super().__init__(cfg, http)
        self.base = cfg.worker_url
        self.ws_base = self.base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)

    def bearer(self) -> dict:
        return {}

    def pinned(self, sess) -> dict:
        return {}

    async def keepalive_loop(self, sess) -> None:
        return None


def make_upstream(cfg, http) -> Upstream:
    if cfg.mode == "queue":
        return QueueUpstream(cfg, http)
    if cfg.mode == "local":
        return LocalUpstream(cfg, http)
    return LBUpstream(cfg, http)
