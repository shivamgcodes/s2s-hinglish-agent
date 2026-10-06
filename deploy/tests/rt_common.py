"""S2S serverless: shared code for the REAL-endpoint test scripts (ws_hold_test.py, cold_start_timer.py, latency_probe.py).

Three ways to reach a worker (pick one per run):
  --space URL          through the deployed HF Space, exactly like a browser (no secrets needed; --passcode if set)
  --direct lb          straight to the RunPod LB endpoint, doing the Space's job here: GET /status wake, token claim,
                       strict-pinned websocket with Bearer, optional keepalive.  needs env RUNPOD_API_KEY,
                       S2S_SESSION_SECRET, RUNPOD_ENDPOINT_ID (or --lb-url), S2S_AUDIENCE (default = endpoint id)
  --direct queue       straight to a queue endpoint: /run, poll /status, ws://ip:port with a token. Same env, or --api-url.
Nothing is sent anywhere unless one of these is given; without it the scripts print their plan and exit 0 (dry run).
--dry-run prints the plan even when a target is given. Secrets are read from env only and are never printed.

Needs: python >= 3.10 with aiohttp, numpy, sphn==0.1.12  (on runpod2: /root/deploy/venv-pp/bin/python;
laptop: python -m venv v && v/bin/pip install aiohttp numpy sphn==0.1.12). The token module comes from ../common.
"""
import asyncio
import json
import os
import ssl
import sys
import time
import urllib.parse

import aiohttp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import s2s_token  # noqa: E402

SR, FS = 24000, 1920
READY_STATES = ("idle", "claimed", "busy")


def now():
    return time.time()


def log(tag, *a):
    print(f"[{tag} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def mask(s):
    return "<unset>" if not s else f"<set, {len(s)} chars>"


def add_target_args(ap):
    g = ap.add_argument_group("target (give exactly one; none = dry run)")
    g.add_argument("--space", help="Space base URL, e.g. https://OWNER-SPACE.hf.space (use the direct *.hf.space URL)")
    g.add_argument("--direct", choices=["lb", "queue"], help="bypass the Space; needs env secrets (see module doc)")
    g.add_argument("--lb-url", default=None, help="default https://$RUNPOD_ENDPOINT_ID.api.runpod.ai")
    g.add_argument("--api-url", default=None, help="default https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID")
    g.add_argument("--passcode", default=os.environ.get("S2S_PASSCODE"), help="Space passcode (or env S2S_PASSCODE)")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed test hosts only)")
    c = ap.add_argument_group("call config")
    c.add_argument("--record-id", default="food_01")
    c.add_argument("--pairing", default="g1")
    c.add_argument("--seed", type=int, default=1001)
    c.add_argument("--wake-timeout-s", type=float, default=600.0)
    d = ap.add_argument_group("direct-lb keepalive (Space runs do their own keepalive from Space vars)")
    d.add_argument("--keepalive-path", default="/status", choices=["/status", "/ping"])
    d.add_argument("--keepalive-s", type=float, default=20.0, help="0 = off (hold test run A)")
    d.add_argument("--keepalive-timeout-s", type=float, default=10.0)
    d.add_argument("--no-strict", action="store_true", help="direct lb: do not pin keepalive/ws with X-Runpod-Worker-Id")


def direct_env(a):
    ep = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    env = {"key": os.environ.get("RUNPOD_API_KEY", ""), "secret": os.environ.get("S2S_SESSION_SECRET", ""),
           "ep": ep, "aud": os.environ.get("S2S_AUDIENCE") or ep,
           "lb_url": (a.lb_url or (f"https://{ep}.api.runpod.ai" if ep else "")).rstrip("/"),
           "api_url": (a.api_url or (f"https://api.runpod.ai/v2/{ep}" if ep else "")).rstrip("/"),
           "max_s": int(os.environ.get("CALL_MAX_S", "300"))}
    return env


def plan_or_exit(a, tag, what):
    """Dry-run gate. Returns normally only when a real target is configured and --dry-run is not set."""
    lines = [f"[{tag}] plan: {what}"]
    missing = []
    if a.space and a.direct:
        print(f"[{tag}] give --space OR --direct, not both")
        sys.exit(2)
    if a.space:
        lines.append(f"  target : Space {a.space}  (POST /api/session -> poll -> wss /api/chat?sid=...)")
        lines.append(f"  passcode: {mask(a.passcode)}")
    elif a.direct:
        e = direct_env(a)
        url = e["lb_url"] if a.direct == "lb" else e["api_url"]
        lines.append(f"  target : direct {a.direct} {url or '<no URL>'}")
        lines.append(f"  RUNPOD_API_KEY {mask(e['key'])}  S2S_SESSION_SECRET {mask(e['secret'])}  aud {e['aud'] or '<unset>'}")
        if a.direct == "lb":
            lines.append(f"  keepalive: {'off' if a.keepalive_s <= 0 else f'GET {a.keepalive_path} every {a.keepalive_s:g}s'}"
                         f"{'' if a.no_strict else ' (strict-pinned)'}")
        for k, v in (("RUNPOD_API_KEY", e["key"]), ("S2S_SESSION_SECRET", e["secret"]), ("endpoint URL", url),
                     ("S2S_AUDIENCE/RUNPOD_ENDPOINT_ID", e["aud"])):
            if not v:
                missing.append(k)
    else:
        lines.append("  target : NONE given -> dry run (add --space URL or --direct lb|queue)")
    lines.append(f"  call   : record_id={a.record_id} pairing={a.pairing} seed={a.seed}")
    print("\n".join(lines), flush=True)
    if missing:
        print(f"[{tag}] missing: {', '.join(missing)}; nothing sent")
        sys.exit(2)
    if a.dry_run or not (a.space or a.direct):
        print(f"[{tag}] dry run: nothing sent")
        sys.exit(0)


def ssl_ctx(a):
    if getattr(a, "insecure", False):
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c
    return None


def ws_from_http(url):
    return "wss://" + url[8:] if url.startswith("https://") else "ws://" + url[7:] if url.startswith("http://") else url


class Phases:
    """Timestamped phase log: phases.mark('waking:no worker yet') only records changes."""

    def __init__(self):
        self.t0 = now()
        self.items = []

    def mark(self, name, **kw):
        if self.items and self.items[-1]["phase"] == name:
            return
        self.items.append({"phase": name, "t": round(now() - self.t0, 3), **kw})
        log("phase", f"+{now() - self.t0:7.1f}s  {name}" + (f"  {kw}" if kw else ""))

    def durations(self):
        out = []
        for i, it in enumerate(self.items):
            t_end = self.items[i + 1]["t"] if i + 1 < len(self.items) else None
            out.append({"phase": it["phase"], "start_s": it["t"],
                        "dur_s": None if t_end is None else round(t_end - it["t"], 3)})
        return out


# ------------------------------------------------------------------------------------------------ targets


class SpaceTarget:
    kind = "space"

    def __init__(self, a, cs):
        self.a, self.cs = a, cs
        self.base = a.space.rstrip("/")
        self.sid = None
        self.worker_id = None
        self.timeline = None
        self.phases = Phases()
        self.ssl = ssl_ctx(a)

    async def prepare(self):
        body = {"record_id": self.a.record_id, "pairing": self.a.pairing, "seed": self.a.seed}
        if self.a.passcode:
            body["passcode"] = self.a.passcode
        self.phases.mark("post_session")
        async with self.cs.post(self.base + "/api/session", json=body, ssl=self.ssl) as r:
            j = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"POST /api/session -> {r.status} {j}")
        self.sid = j["sid"]
        t_end = now() + self.a.wake_timeout_s + 30
        while now() < t_end:
            st = await self.state()
            s = st.get("state")
            self.phases.mark(f"{s}:{st.get('detail') or ''}".rstrip(":"))
            if s == "ready":
                self.worker_id = st.get("worker_id")
                self.timeline = st.get("timeline")      # Space-side [[state, t_s, detail], ...] (Amendment S1.3)
                return st
            if s in ("failed", "ended"):
                raise RuntimeError(f"session {s}: {st}")
            await asyncio.sleep(0.5)
        raise RuntimeError("timed out waiting for ready")

    async def state(self):
        try:
            async with self.cs.get(f"{self.base}/api/session/{self.sid}", ssl=self.ssl,
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                return await r.json(content_type=None)
        except Exception as e:
            return {"state": "poll_error", "detail": repr(e)}

    def ws_args(self):
        q = {"sid": self.sid, "events": "1", "record_id": self.a.record_id, "pairing": self.a.pairing,
             "seed": str(self.a.seed)}
        return ws_from_http(self.base) + "/api/chat?" + urllib.parse.urlencode(q), {}

    async def keepalive_loop(self, rec):
        return                       # the Space does it (its KEEPALIVE_S / KEEPALIVE_PATH vars)

    async def diag(self):
        async with self.cs.get(f"{self.base}/api/diag", params={"sid": self.sid}, ssl=self.ssl) as r:
            return r.status, await r.json(content_type=None)

    async def finish(self):
        if self.sid:
            try:
                async with self.cs.delete(f"{self.base}/api/session/{self.sid}", ssl=self.ssl,
                                          timeout=aiohttp.ClientTimeout(total=15)) as r:
                    return r.status
            except Exception as e:
                return repr(e)


class LBDirectTarget:
    kind = "lb"

    def __init__(self, a, cs):
        self.a, self.cs = a, cs
        self.e = direct_env(a)
        self.sid = s2s_token.new_sid()
        self.worker_id = None
        self.phases = Phases()

    def _tok(self):
        a = self.a
        return s2s_token.mint(self.e["secret"], sid=self.sid, mode="lb", aud=self.e["aud"], record_id=a.record_id,
                              pairing=a.pairing, seed=a.seed, max_s=self.e["max_s"])

    def _h(self, pinned=True, token=False):
        h = {"Authorization": f"Bearer {self.e['key']}"}
        if pinned and self.worker_id and not self.a.no_strict:
            h["X-Runpod-Worker-Id"] = f"strict {self.worker_id}"
        if token:
            h["X-S2S-Token"] = self._tok()
        return h

    async def prepare(self):
        t_end = now() + self.a.wake_timeout_s
        self.phases.mark("wake:GET /status")
        while now() < t_end:
            try:
                async with self.cs.get(self.e["lb_url"] + "/status", headers=self._h(pinned=False),
                                       timeout=aiohttp.ClientTimeout(total=130)) as r:
                    txt = await r.text()
                    j = json.loads(txt) if txt.startswith("{") else {}
                    if r.status == 200 and j.get("state") in READY_STATES:
                        self.phases.mark("worker_ready", worker_state=j.get("state"), gpu=j.get("gpu"))
                        break
                    self.phases.mark(f"waking:{r.status}")
            except asyncio.TimeoutError:
                self.phases.mark("waking:timeout")
            except aiohttp.ClientError as e:
                self.phases.mark(f"waking:{type(e).__name__}")
            await asyncio.sleep(3)
        else:
            raise RuntimeError("wake timed out")
        while now() < t_end:
            async with self.cs.post(self.e["lb_url"] + "/session/claim", headers=self._h(pinned=False, token=True),
                                    timeout=aiohttp.ClientTimeout(total=130)) as r:
                j = await r.json(content_type=None)
                if r.status == 200:
                    hdr = r.headers.get("X-Runpod-Worker-Id")
                    self.worker_id = hdr or j.get("worker_id")
                    self.phases.mark("claimed", worker_id=self.worker_id, body_worker_id=j.get("worker_id"),
                                     header_worker_id=hdr)
                    return j
                self.phases.mark(f"claim:{r.status}:{j.get('error')}")
                if r.status not in (409, 503, 502, 504):
                    raise RuntimeError(f"claim -> {r.status} {j}")
            await asyncio.sleep(3)
        raise RuntimeError("claim timed out")

    def ws_args(self):
        q = {"events": "1", "record_id": self.a.record_id, "pairing": self.a.pairing, "seed": str(self.a.seed)}
        return ws_from_http(self.e["lb_url"]) + "/api/chat?" + urllib.parse.urlencode(q), self._h(token=True)

    async def keepalive_loop(self, rec):
        """rec: list to append {"t","status","ms"} to. Strict-pinned unless --no-strict."""
        if self.a.keepalive_s <= 0:
            return
        while True:
            await asyncio.sleep(self.a.keepalive_s)
            t = now()
            try:
                async with self.cs.get(self.e["lb_url"] + self.a.keepalive_path, headers=self._h(),
                                       timeout=aiohttp.ClientTimeout(total=self.a.keepalive_timeout_s)) as r:
                    await r.read()
                    rec.append({"t": round(t, 3), "status": r.status, "ms": round((now() - t) * 1000, 1)})
            except asyncio.TimeoutError:
                rec.append({"t": round(t, 3), "status": "timeout", "ms": None})
            except Exception as e:
                rec.append({"t": round(t, 3), "status": type(e).__name__, "ms": None})

    async def diag(self):
        t = now()
        async with self.cs.get(self.e["lb_url"] + "/status", headers=self._h(),
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            j = await r.json(content_type=None)
        return 200, {"space_to_worker_ms": round((now() - t) * 1000, 1), "note": "measured from this machine", **{
            k: j.get(k) for k in ("gpu", "step_ms_p95_last", "state")}}

    async def finish(self):
        try:
            async with self.cs.post(self.e["lb_url"] + "/session/release", headers=self._h(token=True),
                                    timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status
        except Exception as e:
            return repr(e)


class QueueDirectTarget:
    kind = "queue"

    def __init__(self, a, cs):
        self.a, self.cs = a, cs
        self.e = direct_env(a)
        self.sid = s2s_token.new_sid()
        self.job = None
        self.ws_base = None
        self.worker_id = None
        self.phases = Phases()

    def _h(self):
        return {"Authorization": f"Bearer {self.e['key']}"}

    async def prepare(self):
        a = self.a
        body = {"input": {"sid": self.sid, "record_id": a.record_id, "pairing": a.pairing, "seed": a.seed},
                "policy": {"executionTimeout": int(os.environ.get("QUEUE_EXEC_TIMEOUT_MS", "900000"))}}
        self.phases.mark("run")
        async with self.cs.post(self.e["api_url"] + "/run", json=body, headers=self._h()) as r:
            j = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"/run -> {r.status} {j}")
        self.job = j["id"]
        t_end = now() + a.wake_timeout_s
        while now() < t_end:
            async with self.cs.get(f"{self.e['api_url']}/status/{self.job}", headers=self._h()) as r:
                j = await r.json(content_type=None)
            st, out = j.get("status"), j.get("output")
            pstate = out.get("state") if isinstance(out, dict) else None
            self.phases.mark(f"{st}:{pstate or ''}".rstrip(":"))
            if pstate == "ready":
                self.ws_base = f"ws://{out['public_ip']}:{out['tcp_port']}"
                self.worker_id = out.get("worker_id")
                return out
            if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"job ended before ready: {j}")
            await asyncio.sleep(2)
        raise RuntimeError("queue wake timed out")

    def ws_args(self):
        a = self.a
        tok = s2s_token.mint(self.e["secret"], sid=self.sid, mode="queue", aud=self.e["aud"], record_id=a.record_id,
                             pairing=a.pairing, seed=a.seed, max_s=self.e["max_s"])
        q = {"events": "1", "record_id": a.record_id, "pairing": a.pairing, "seed": str(a.seed)}
        return self.ws_base + "/api/chat?" + urllib.parse.urlencode(q), {"X-S2S-Token": tok}

    async def keepalive_loop(self, rec):
        return

    async def diag(self):
        t = now()
        async with self.cs.get(self.ws_base.replace("ws://", "http://") + "/status",
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            j = await r.json(content_type=None)
        return 200, {"space_to_worker_ms": round((now() - t) * 1000, 1), "note": "measured from this machine",
                     "gpu": j.get("gpu"), "step_ms_p95_last": j.get("step_ms_p95_last")}

    async def finish(self):
        if not self.job:
            return None
        try:
            async with self.cs.get(f"{self.e['api_url']}/status/{self.job}", headers=self._h()) as r:
                j = await r.json(content_type=None)
            if j.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
                async with self.cs.post(f"{self.e['api_url']}/cancel/{self.job}", headers=self._h()) as r:
                    return f"cancel {r.status}"
            return j.get("status")
        except Exception as e:
            return repr(e)


def make_target(a, cs):
    if a.space:
        return SpaceTarget(a, cs)
    return LBDirectTarget(a, cs) if a.direct == "lb" else QueueDirectTarget(a, cs)


# ------------------------------------------------------------------------------------------------ audio


def tone_track(seconds, every_s=5.0, tone_s=0.4, hz=440.0, amp=0.2):
    """Silence with a short tone burst every `every_s` seconds (DESIGN 9 #3: 'silence + periodic tones')."""
    n = int(round(seconds * SR / FS)) * FS
    x = np.zeros(n, np.float32)
    t = np.arange(int(tone_s * SR)) / SR
    burst = (amp * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))).astype(np.float32)
    for s0 in range(int(every_s * SR), n - len(burst), int(every_s * SR)):
        x[s0:s0 + len(burst)] = burst
    return x


def load_wav(path, seconds=None):
    import sphn
    a, sr = sphn.read(path)
    a = a.mean(axis=0) if a.shape[0] > 1 else a[0]
    if sr != SR:
        a = sphn.resample(a, sr, SR)
    a = a.astype(np.float32)
    if seconds:
        n = int(round(seconds * SR / FS)) * FS
        a = np.tile(a, int(np.ceil(n / max(1, len(a)))))[:n]
    return a


def pct(xs, p):
    xs = [x for x in xs if x is not None]
    return None if not xs else round(float(np.percentile(xs, p)), 1)


async def stream_call(cs, url, headers, pcm, tag, ssl_=None, open_timeout=60.0, on_tick=None, tick_s=10.0,
                      on_event=None):
    """Open the websocket, stream `pcm` in real time (80 ms frames, Opus via sphn), record everything.

    Returns a dict: open/handshake times, frames sent/received, receive gaps, per-frame pipeline lag
    (send time of user frame k -> arrival of the k-th model frame's worth of decoded audio), 0x07 events
    (session_end, metrics), close code. The stream stops early on session_end or socket close.
    """
    import sphn
    st = {"url": url.split("?")[0], "t_open_start": now(), "t_open": None, "t_handshake": None, "frames_sent": 0,
          "model_samples": 0, "bytes_in": 0, "bytes_out": 0, "n_text": 0, "session_end": None, "close_code": None,
          "refused": None, "metrics_last": None, "step_p95_seen": [], "events": {}, "send_late_max_ms": 0.0}
    gaps, lag = [], []
    send_t = []
    writer, reader = sphn.OpusStreamWriter(SR), sphn.OpusStreamReader(SR)
    try:
        ws = await cs.ws_connect(url, headers=headers, max_msg_size=0, ssl=ssl_, timeout=open_timeout,
                                 heartbeat=None, autoping=True)
    except aiohttp.WSServerHandshakeError as e:
        st["refused"] = {"status": e.status, "message": e.message}
        return st
    st["t_open"] = now()
    hs = asyncio.get_running_loop().create_future()
    last_audio = [None]

    async def recv():
        k_done = 0
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            d = msg.data
            st["bytes_in"] += len(d)
            k = d[0]
            t = now()
            if k == 0:
                st["t_handshake"] = t
                if not hs.done():
                    hs.set_result(True)
            elif k == 1:
                if st.get("t_first_audio") is None:
                    st["t_first_audio"] = t
                if last_audio[0] is not None:
                    gaps.append((t - last_audio[0]) * 1000)
                last_audio[0] = t
                reader.append_bytes(d[1:])
                p = reader.read_pcm()
                st["model_samples"] += int(p.shape[-1])
                while (k_done + 1) * FS <= st["model_samples"]:
                    if k_done < len(send_t):
                        lag.append((t - send_t[k_done]) * 1000)
                    k_done += 1
            elif k == 2:
                st["n_text"] += 1
            elif k == 7:
                try:
                    ev = json.loads(d[1:].decode("utf-8"))
                except Exception:
                    continue
                ty = ev.get("type")
                st["events"][ty] = st["events"].get(ty, 0) + 1
                if on_event is not None:            # integration tests: every 0x07 event object (additive, INTEG)
                    on_event(ev, t)
                if ty == "metrics":
                    st["metrics_last"] = ev
                    p95 = (ev.get("step_ms") or {}).get("p95")
                    if p95 is not None:
                        st["step_p95_seen"].append(p95)
                elif ty == "session_end":
                    st["session_end"] = {"reason": ev.get("reason"), "frames": ev.get("frames"), "t": round(t, 3)}
        st["close_code"] = ws.close_code
        if not hs.done():
            hs.set_result(False)

    rt = asyncio.create_task(recv())
    try:
        ok = await asyncio.wait_for(hs, timeout=180)
    except asyncio.TimeoutError:
        ok = False
    if not ok:
        await ws.close()
        await asyncio.gather(rt, return_exceptions=True)
        st["error"] = "no handshake"
        return st
    log(tag, f"handshake {st['t_handshake'] - st['t_open']:.2f} s after open; streaming {len(pcm) / SR:.0f} s")
    t0 = time.perf_counter()
    t_tick = now()
    for i in range(len(pcm) // FS):
        due = t0 + i * FS / SR
        lagn = time.perf_counter()
        if due > lagn:
            await asyncio.sleep(due - lagn)
        else:
            st["send_late_max_ms"] = max(st["send_late_max_ms"], (lagn - due) * 1000)
        if st["session_end"] is not None or ws.closed:
            break
        writer.append_pcm(pcm[i * FS:(i + 1) * FS])
        b = writer.read_bytes()
        send_t.append(now())
        if len(b):
            try:
                await ws.send_bytes(b"\x01" + b)
            except Exception as e:
                st["send_error"] = repr(e)
                break
            st["bytes_out"] += len(b) + 1
        st["frames_sent"] = i + 1
        if on_tick is not None and now() - t_tick >= tick_s:
            t_tick = now()
            await on_tick(st)
    st["t_stream_end"] = now()
    if not ws.closed and st["session_end"] is None:
        await asyncio.sleep(1.0)
        await ws.close()
    try:
        await asyncio.wait_for(rt, timeout=5)
    except Exception:
        pass
    st["close_code"] = ws.close_code
    st["recv_gap_ms"] = {"p50": pct(gaps, 50), "p95": pct(gaps, 95), "p99": pct(gaps, 99),
                         "max": round(max(gaps), 1) if gaps else None, "n_over_500": sum(g > 500 for g in gaps),
                         "n_over_1000": sum(g > 1000 for g in gaps)}
    st["pipeline_lag_ms"] = {"p50": pct(lag, 50), "p95": pct(lag, 95), "min": round(min(lag), 1) if lag else None,
                             "n": len(lag)}
    st["model_s"] = round(st["model_samples"] / SR, 2)
    st["user_s"] = round(st["frames_sent"] * FS / SR, 2)
    if st["step_p95_seen"]:
        st["step_p95_max"] = max(st["step_p95_seen"])
    st.pop("step_p95_seen")
    return st


def outcome(st, ran_full):
    """PASS = clean end (context_full / time_limit / client_closed after a full run). DROP otherwise."""
    if st.get("refused"):
        return "REFUSED"
    if st.get("error"):
        return "FAIL"
    se = (st.get("session_end") or {}).get("reason")
    if se in ("context_full", "time_limit"):
        return "PASS"
    if se is None and ran_full:
        return "PASS"
    if se in ("worker_lost", "worker_shutdown", "error"):
        return "DROP"
    if se is None:
        return "DROP"          # socket closed early without a session_end
    return "PASS" if ran_full else "DROP"
