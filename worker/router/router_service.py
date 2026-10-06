"""S2S serverless worker: copy of DEP1 router/router_service.py (path edits only: common dir, log dir and default
ports from paths.py). Router service (venv-needle, CPU). DEP1 INTERFACE.md section 9.

  <venv-needle>/bin/python router_service.py [--internal http://127.0.0.1:8999] [--asr-url http://127.0.0.1:8996]
                                             [--window ring30|since_prev] [--unbound strict|none_only] [--port 8995]

- Subscribes to ws://<internal>/internal/events (reconnects forever).
- "session" event with a router-supported record_id -> RouterCore for that record (V4 records.json via
  common/session.py, the live demo record set), Needle agent pre-warmed in the worker thread.
- "text" events (forced=false only; injected text never triggers) -> trigger detector.
- On a trigger: the ring is fetched IMMEDIATELY (GET /internal/ring?seconds=N, N=30 or since the previous
  trigger), then the trigger is queued to one worker thread that runs ASR (POST <asr-url>/asr?lang=hi),
  Needle, resolver, guard (v2: ask -> needs_clarification; N1: unbound), stub tools. Every stage is POSTed to /internal/action (the server relays it to the
  client as kind 0x07 and on /internal/events) and appended to logs/router_events.jsonl.
- "session_end" -> the session's core is dropped (its final record copy is logged).
- GET /health on 127.0.0.1:<port> -> {"ok", "session", "counts", "last_events", ...}.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402

paths.ensure_common_on_path()
import router_core  # noqa: E402
from router_core import RouterCore  # noqa: E402  (sets JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='')
import session as sessmod  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

LOG = os.path.join(str(paths.LOGS), "router_events.jsonl")


def http_json(url, body=None, timeout=300, raw=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/octet-stream" if isinstance(body, bytes)
                                                          else "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (r.read(), dict(r.headers)) if raw else json.loads(r.read())


class Service:
    def __init__(self, a):
        self.a = a
        self.internal = a.internal.rstrip("/")
        self.asr_url = a.asr_url.rstrip("/")
        self.core = None
        self.sess = None
        self.prev_end_sample = None
        self.q = queue.Queue()
        self.counts = {k: 0 for k in ("trigger", "asr", "needle", "resolved", "executed", "unbound",
                                      "needs_clarification", "error")}
        # D-ROUTER-V2: import the router now (v2 default | n1), so a broken install fails /health, not the first call
        self.router_kind = router_core.ROUTER
        router_core.load_router(self.router_kind)
        self.last = []
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ outputs
    def log(self, obj):
        with self.lock, open(LOG, "a") as f:
            f.write(json.dumps(dict(obj, t_log=time.time()), ensure_ascii=False) + "\n")

    def emit(self, ev, core=None):
        if core is not None and core is not self.core:
            # review fix D-T3-7: a trigger of an ended/replaced session finished late; the server would stamp
            # it with the CURRENT session_id and show it in the next call's actions panel. Log only.
            self.log({"kind": "stale_action", "session_id": getattr(core, "session_id", None), **ev})
            return
        with self.lock:
            self.counts[ev["stage"]] = self.counts.get(ev["stage"], 0) + 1
            self.last = (self.last + [ev])[-20:]
        self.log({"kind": "action", "session_id": (self.sess or {}).get("session_id"), **ev})
        try:
            http_json(self.internal + "/internal/action", ev, timeout=5)
        except Exception as e:  # noqa: BLE001
            self.log({"kind": "post_failed", "error": str(e), "stage": ev["stage"]})

    def asr(self, pcm):
        return http_json(self.asr_url + "/asr?lang=hi", np.asarray(pcm, "<f4").tobytes())

    # ------------------------------------------------------------ session lifecycle
    def on_session(self, ev):
        sid = ev.get("session_id")
        if sid is not None and sid == (self.sess or {}).get("session_id"):
            return   # already known (resync on connect + the ws "session" event)
        self.sess = ev
        self.core = None
        self.prev_end_sample = None
        rid = ev.get("record_id")
        if not rid or not ev.get("router_supported"):
            self.log({"kind": "session_idle", "session_id": ev.get("session_id"), "record_id": rid})
            return
        # integration (INTERFACE Amendment 1): --records selects the record set; default = V4 demo records.
        # The 12-call offline eval runs a router with --records <S2S_NEEDLE>/src/records.json (V3 records).
        recs = sessmod.load_records(self.a.records) if self.a.records else sessmod.load_records()
        if rid not in recs:
            self.log({"kind": "session_idle", "session_id": sid, "record_id": rid, "why": "record_id not in --records"})
            return
        rec = recs[rid]
        core = RouterCore(rec["agent_type"], rec, self.asr, None, unbound_rule=self.a.unbound)
        core.session_id = sid
        core.emit_fn = lambda e, c=core: self.emit(e, c)
        self.core = core
        self.q.put(("prewarm", core, None, None))
        self.log({"kind": "session_router", "session_id": ev.get("session_id"), "record_id": rid,
                  "agent_type": rec["agent_type"], "records": self.a.records or str(sessmod.RECORDS_V4),
                  "primary_id": rec.get("primary_id")})

    def on_text(self, ev):
        core = self.core
        if core is None or ev.get("forced"):
            return
        tr = core.on_text(int(ev["frame"]), ev["piece"])
        if not tr:
            return
        secs = 30.0
        if self.a.window == "since_prev" and self.prev_end_sample is not None:
            secs = None
        try:
            if secs is None:
                body, hdr = http_json(self.internal + "/internal/ring?seconds=30", raw=True, timeout=5)
                end = int(hdr.get("X-End-Sample", 0))
                keep = max(0, end - self.prev_end_sample)
                pcm = np.frombuffer(body, "<f4")
                pcm = pcm[max(0, len(pcm) - keep):] if keep < len(pcm) else pcm
            else:
                body, hdr = http_json(self.internal + "/internal/ring?seconds=30", raw=True, timeout=5)
                end = int(hdr.get("X-End-Sample", 0))
                pcm = np.frombuffer(body, "<f4")
            self.prev_end_sample = end
        except Exception as e:  # noqa: BLE001
            self.emit({"trigger_id": None, "stage": "error", "payload": {"where": "ring", "message": str(e)},
                       "latency_ms": 0.0, "since_trigger_ms": 0.0, "frame": tr["frame"]})
            return
        self.q.put(("trigger", core, tr, pcm.copy()))

    def on_end(self, ev):
        if self.core is not None:
            self.log({"kind": "session_end", "session_id": ev.get("session_id"), "reason": ev.get("reason"),
                      "record_final": {k: self.core.tools.record.get(k) for k in ("entity_updates", "executed_calls")}})
        self.core = None

    def _worker(self):
        while True:
            kind, core, tr, pcm = self.q.get()
            if core is not self.core:   # review fix D-T3-7: queued work of an ended/replaced session
                self.log({"kind": "stale_skipped", "what": kind, "session_id": getattr(core, "session_id", None),
                          "frame": (tr or {}).get("frame")})
                continue
            try:
                if kind == "prewarm":
                    ms = core.prewarm()
                    self.log({"kind": "prewarm", "ms": ms, "agent_type": core.agent_type})
                elif kind == "trigger":
                    s = core.handle_trigger(tr, pcm)
                    self.log({"kind": "trigger_summary", **{k: v for k, v in s.items() if k != "events"}})
            except Exception as e:  # noqa: BLE001
                self.log({"kind": "worker_error", "error": f"{type(e).__name__}: {e}"})

    # ------------------------------------------------------------ main loop
    def run(self):
        url = self.internal.replace("http://", "ws://").replace("https://", "wss://") + "/internal/events"
        while True:
            try:
                with connect(url, max_size=None, open_timeout=5) as ws:
                    self.log({"kind": "connected", "url": url})
                    # review fix D-T3-7: /internal/events never replays the "session" event, so a router that
                    # (re)connects after the handshake would stay idle for the whole call. Resync here.
                    try:
                        cur = http_json(self.internal + "/internal/session", timeout=5)
                        if cur.get("active"):
                            self.on_session(cur)
                            self.log({"kind": "resync_session", "session_id": cur.get("session_id"),
                                      "frame": cur.get("frame")})
                    except Exception as e:  # noqa: BLE001
                        self.log({"kind": "resync_failed", "error": f"{type(e).__name__}: {e}"})
                    for msg in ws:
                        ev = json.loads(msg)
                        t = ev.get("type")
                        if t == "session":
                            self.on_session(ev)
                        elif t == "text":
                            self.on_text(ev)
                        elif t == "session_end":
                            self.on_end(ev)
            except Exception as e:  # noqa: BLE001
                self.log({"kind": "disconnected", "error": f"{type(e).__name__}: {e}"})
                time.sleep(1.0)

    def health(self):
        with self.lock:
            return {"ok": True, "session": {k: (self.sess or {}).get(k) for k in ("session_id", "record_id", "agent_type")},
                    "router_active": self.core is not None, "queue": self.q.qsize(), "counts": dict(self.counts),
                    "last_events": list(self.last), "window": self.a.window, "unbound_rule": self.a.unbound,
                    "router": self.router_kind,
                    "asr_url": self.asr_url, "internal": self.internal,
                    "records": self.a.records or str(sessmod.RECORDS_V4)}


def serve_health(svc, port):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] != "/health":
                self.send_response(404); self.end_headers(); return
            b = json.dumps(svc.health(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever, daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", default=f"http://127.0.0.1:{paths.INTERNAL_PORT}")
    ap.add_argument("--asr-url", default=f"http://127.0.0.1:{paths.ASR_PORT}")
    ap.add_argument("--window", default="ring30", choices=["ring30", "since_prev"])
    ap.add_argument("--unbound", default="strict", choices=["strict", "none_only"])
    ap.add_argument("--port", type=int, default=paths.ROUTER_PORT)
    ap.add_argument("--records", default=None, help="records.json for session records (default: V4 demo set)")
    a = ap.parse_args()
    svc = Service(a)
    serve_health(svc, a.port)
    print(f"router_service: events {a.internal}/internal/events, asr {a.asr_url}, health :{a.port}", flush=True)
    svc.run()
