"""S2S serverless: local stand-in for RunPod (DESIGN section 7, "fake_runpod.py"). CPU only, no network beyond 127.0.0.1.

  /root/deploy/venv-pp/bin/python /root/deploy_serverless/tests/fake_runpod.py \
      [--lb-port 18080] [--workers http://127.0.0.1:18000[,http://127.0.0.1:18001]] [--worker-ids fw-0,fw-1] \
      [--queue-port 18081] [--handler /root/deploy_serverless/worker/rp_handler.py] [--worker-port 18000] \
      [--api-key test-key] [--hold-s 3] [--endpoint-id ep-test]

LB proxy (https://ENDPOINT_ID.api.runpod.ai stand-in, plain http here) on --lb-port:
  - every request needs "Authorization: Bearer <api-key>" (401 otherwise); the header is NOT forwarded to the worker
  - health: polls each worker's GET /ping every 0.5 s; only workers answering 200 get traffic ([LB-OV]: 200 healthy,
    204 initializing, anything else unhealthy)
  - no healthy worker: holds the request up to --hold-s (the real proxy: 2 min), then 503 {"error":"no workers available"}
  - "X-Runpod-Worker-Id: strict <id>" (or "strict-resume <id>") routes only to that worker, 404
    {"error":"affinity_worker_gone"} if it is unknown or unhealthy; "<id>" alone is a soft preference ([LB-AFF])
  - unpinned requests go round-robin over healthy workers (no busy signal: a claim may land on a busy worker -> 409, R11)
  - every response (and the websocket 101) carries "X-Runpod-Worker-Id: <id>"
  - websockets are proxied frame by frame; close codes are propagated both ways; an upstream refusal is returned as
    the same HTTP status with {"error":"upstream_refused","status":N} (the real body is not recoverable through aiohttp)
  - GET /_fake/state (no auth) = worker table + counters, for tests
Queue API (https://api.runpod.ai/v2/ENDPOINT_ID stand-in) on --queue-port, only if --handler is given:
  - POST /v2/{ep}/run {"input":{...},"policy":{...}} -> {"id","status":"IN_QUEUE"}; runs rp_handler.handler(job) in a
    thread with RUNPOD_PUBLIC_IP=127.0.0.1 and RUNPOD_TCP_PORT_<worker-port>=<worker-port> (public port = local port)
  - GET /v2/{ep}/status/{id} -> {"id","status":IN_QUEUE|IN_PROGRESS|COMPLETED|FAILED|CANCELLED,"output":...};
    while IN_PROGRESS, output = the last progress_update payload ([EX-QWS] README steps 3-4)
  - POST /v2/{ep}/cancel/{id} -> {"id","status":"CANCELLED"}; the handler thread cannot be killed, it is only marked
  - runpod.serverless.progress_update / start are replaced by an in-process fake module before rp_handler is imported
Not emulated: the real 2-min hold, scaling, billing, the 330 s processing cap, idle timeout, TLS.
"""
import argparse
import asyncio
import importlib.util
import itertools
import os
import sys
import threading
import time
import types
import uuid

import aiohttp
from aiohttp import web

HOP = {"host", "connection", "keep-alive", "upgrade", "proxy-authorization", "proxy-authenticate", "te", "trailer",
       "transfer-encoding", "authorization", "x-runpod-worker-id", "content-length", "sec-websocket-key",
       "sec-websocket-version", "sec-websocket-extensions", "sec-websocket-accept", "sec-websocket-protocol"}
STRIP = set()   # --strip-headers (lower case), filled in main()
STRIP_WS_ONLY = False   # --strip-ws-only: drop them only on websocket upgrades


def log(*a):
    print(f"[fake_runpod {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _auth_ok(request, key):
    return request.headers.get("Authorization", "") == f"Bearer {key}"


class Worker:
    def __init__(self, wid, url):
        self.id, self.url = wid, url.rstrip("/")
        self.ping = None          # last /ping status (None = unreachable)
        self.healthy = False
        self.n_req = 0
        self.n_ws_open = 0


class LBProxy:
    def __init__(self, workers, api_key, hold_s):
        self.workers = workers
        self.by_id = {w.id: w for w in workers}
        self.key = api_key
        self.hold_s = hold_s
        self.rr = itertools.cycle(range(len(workers)))
        self.counts = {"requests": 0, "unauthorized": 0, "no_worker": 0, "affinity_gone": 0, "ws": 0}
        self.session = None

    async def health_loop(self):
        while True:
            for w in self.workers:
                try:
                    async with self.session.get(w.url + "/ping", timeout=aiohttp.ClientTimeout(total=2)) as r:
                        st = r.status
                except Exception:
                    st = None
                if st != w.ping:
                    log(f"worker {w.id} /ping {w.ping} -> {st}")
                w.ping, w.healthy = st, st == 200
            await asyncio.sleep(0.5)

    async def pick(self, request):
        """Returns (worker, None) or (None, error_response)."""
        aff = request.headers.get("X-Runpod-Worker-Id", "").strip()
        mode, _, wid = aff.partition(" ")
        if mode in ("strict", "strict-resume") and wid:
            w = self.by_id.get(wid.strip())
            if w is None or not w.healthy:
                self.counts["affinity_gone"] += 1
                return None, web.json_response({"error": "affinity_worker_gone", "worker_id": wid.strip()}, status=404)
            return w, None
        prefer = self.by_id.get(aff) if aff and not wid else None
        t_end = time.monotonic() + self.hold_s
        while True:
            if prefer is not None and prefer.healthy:
                return prefer, None
            for _ in range(len(self.workers)):
                w = self.workers[next(self.rr)]
                if w.healthy:
                    return w, None
            if time.monotonic() >= t_end:
                self.counts["no_worker"] += 1
                return None, web.json_response({"error": "no workers available"}, status=503)
            await asyncio.sleep(0.25)

    async def handle(self, request):
        if request.path == "/_fake/state":
            return web.json_response({"workers": [{"id": w.id, "url": w.url, "ping": w.ping, "healthy": w.healthy,
                                                   "n_req": w.n_req, "n_ws_open": w.n_ws_open} for w in self.workers],
                                      "counts": self.counts})
        self.counts["requests"] += 1
        if not _auth_ok(request, self.key):
            self.counts["unauthorized"] += 1
            return web.json_response({"error": "unauthorized"}, status=401)
        w, err = await self.pick(request)
        if err is not None:
            return err
        w.n_req += 1
        is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
        drop = STRIP if (is_ws or not STRIP_WS_ONLY) else set()
        fwd = {k: v for k, v in request.headers.items() if k.lower() not in HOP and k.lower() not in drop}
        url = w.url + request.path_qs
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.ws(request, w, url, fwd)
        body = await request.read()
        try:
            async with self.session.request(request.method, url, headers=fwd, data=body or None,
                                            timeout=aiohttp.ClientTimeout(total=330)) as r:
                data = await r.read()
                hdrs = {k: v for k, v in r.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}
                hdrs["X-Runpod-Worker-Id"] = w.id
                return web.Response(status=r.status, body=data, headers=hdrs)
        except Exception as e:
            return web.json_response({"error": "worker_unreachable", "detail": repr(e)}, status=502,
                                     headers={"X-Runpod-Worker-Id": w.id})

    async def ws(self, request, w, url, fwd):
        self.counts["ws"] += 1
        wsurl = "ws" + url[4:] if url.startswith("http") else url
        try:
            up = await self.session.ws_connect(wsurl, headers=fwd, max_msg_size=0, autoping=True, heartbeat=None,
                                               timeout=60)
        except aiohttp.WSServerHandshakeError as e:
            return web.json_response({"error": "upstream_refused", "status": e.status}, status=e.status,
                                     headers={"X-Runpod-Worker-Id": w.id})
        except Exception as e:
            return web.json_response({"error": "worker_unreachable", "detail": repr(e)}, status=502,
                                     headers={"X-Runpod-Worker-Id": w.id})
        down = web.WebSocketResponse(max_msg_size=0, autoping=True)
        down.headers["X-Runpod-Worker-Id"] = w.id
        await down.prepare(request)
        w.n_ws_open += 1
        log(f"ws open -> {w.id} {request.path_qs[:80]}")

        async def pump(src, dst):
            async for m in src:
                if m.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(m.data)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(m.data)
                else:
                    break

        t1 = asyncio.create_task(pump(up, down))
        t2 = asyncio.create_task(pump(down, up))
        try:
            await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (t1, t2):
                t.cancel()
            uc = up.close_code
            dc = down.close_code
            if not down.closed:
                await down.close(code=uc if uc else 1011, message=b"upstream closed")
            if not up.closed:
                await up.close(code=dc if dc else 1000)
            w.n_ws_open -= 1
            log(f"ws closed {w.id}: upstream code {uc}, downstream code {dc}")
        return down


# ---------------------------------------------------------------- queue API


def _install_fake_runpod_module(jobs):
    rp = types.ModuleType("runpod")
    sl = types.ModuleType("runpod.serverless")

    def progress_update(job, progress):
        j = jobs.get(job.get("id"))
        if j is not None:
            j["progress"] = progress
            j["progress_log"].append([round(time.time(), 3), progress])

    sl.progress_update = progress_update
    sl.start = lambda cfg: log("runpod.serverless.start() ignored under fake_runpod")
    rp.serverless = sl
    sys.modules["runpod"] = rp
    sys.modules["runpod.serverless"] = sl


class QueueAPI:
    def __init__(self, handler_path, api_key, worker_port):
        self.key = api_key
        self.jobs = {}
        os.environ.setdefault("RUNPOD_PUBLIC_IP", "127.0.0.1")
        os.environ.setdefault(f"RUNPOD_TCP_PORT_{worker_port}", str(worker_port))
        os.environ.setdefault("RUNPOD_POD_ID", "fake-queue-worker")
        _install_fake_runpod_module(self.jobs)
        spec = importlib.util.spec_from_file_location("rp_handler", handler_path)
        self.mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(handler_path)))
        spec.loader.exec_module(self.mod)
        log(f"queue API: handler {handler_path} loaded; RUNPOD_TCP_PORT_{worker_port}={worker_port}")

    def _run_job(self, jid):
        j = self.jobs[jid]
        j["status"] = "IN_PROGRESS"
        j["t_start"] = time.time()
        try:
            out = self.mod.handler({"id": jid, "input": j["input"]})
            if j["status"] != "CANCELLED":
                j["status"], j["output"] = "COMPLETED", out
        except Exception as e:
            if j["status"] != "CANCELLED":
                j["status"], j["error"] = "FAILED", repr(e)
        j["t_end"] = time.time()

    async def handle(self, request):
        if not _auth_ok(request, self.key):
            return web.json_response({"error": "unauthorized"}, status=401)
        parts = request.path.strip("/").split("/")     # v2 {ep} run | status {id} | cancel {id}
        if len(parts) < 3 or parts[0] != "v2":
            return web.json_response({"error": "not found"}, status=404)
        op = parts[2]
        if op == "run" and request.method == "POST":
            body = await request.json()
            jid = "fake-" + uuid.uuid4().hex[:12]
            self.jobs[jid] = {"id": jid, "status": "IN_QUEUE", "input": body.get("input", {}),
                              "policy": body.get("policy"), "progress": None, "progress_log": [], "output": None}
            threading.Thread(target=self._run_job, args=(jid,), daemon=True).start()
            return web.json_response({"id": jid, "status": "IN_QUEUE"})
        if op in ("status", "cancel") and len(parts) >= 4:
            j = self.jobs.get(parts[3])
            if j is None:
                return web.json_response({"error": "job not found"}, status=404)
            if op == "cancel":
                if j["status"] in ("IN_QUEUE", "IN_PROGRESS"):
                    j["status"] = "CANCELLED"
                return web.json_response({"id": j["id"], "status": j["status"]})
            res = {"id": j["id"], "status": j["status"]}
            if j["status"] == "IN_PROGRESS":
                res["output"] = j["progress"]
            elif j["status"] == "COMPLETED":
                res["output"] = j["output"]
            elif j["status"] == "FAILED":
                res["error"] = j.get("error")
            return web.json_response(res)
        return web.json_response({"error": "not found"}, status=404)


async def amain(a):
    runners = []
    if a.workers:
        urls = [u for u in a.workers.split(",") if u]
        ids = a.worker_ids.split(",") if a.worker_ids else [f"fw-{i}" for i in range(len(urls))]
        lb = LBProxy([Worker(i, u) for i, u in zip(ids, urls)], a.api_key, a.hold_s)
        lb.session = aiohttp.ClientSession(auto_decompress=False)
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", lb.handle)
        r = web.AppRunner(app, access_log=None)
        await r.setup()
        await web.TCPSite(r, a.host, a.lb_port).start()
        runners.append(r)
        asyncio.create_task(lb.health_loop())
        log(f"LB proxy on http://{a.host}:{a.lb_port} -> {[(w.id, w.url) for w in lb.workers]}  hold {a.hold_s}s")
    if a.handler:
        q = QueueAPI(a.handler, a.api_key, a.worker_port)
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", q.handle)
        r = web.AppRunner(app, access_log=None)
        await r.setup()
        await web.TCPSite(r, a.host, a.queue_port).start()
        runners.append(r)
        log(f"queue API on http://{a.host}:{a.queue_port}/v2/{a.endpoint_id}")
    if not runners:
        log("nothing to do: give --workers and/or --handler")
        return
    await asyncio.Event().wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lb-port", type=int, default=18080)
    ap.add_argument("--queue-port", type=int, default=18081)
    ap.add_argument("--workers", default="http://127.0.0.1:18000", help="comma list of worker base URLs ('' = no LB)")
    ap.add_argument("--worker-ids", default="", help="comma list of proxy worker ids (default fw-0,fw-1,...)")
    ap.add_argument("--handler", default="", help="path to worker/rp_handler.py to enable the queue API")
    ap.add_argument("--worker-port", type=int, default=18000, help="the worker $PORT (queue mode public port)")
    ap.add_argument("--api-key", default="test-key")
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--endpoint-id", default="ep-test")
    ap.add_argument("--strip-headers", default="", help="comma list of request headers the LB proxy drops "
                    "(CRIT-3: simulate a proxy that strips X-S2S-Token)")
    ap.add_argument("--strip-ws-only", action="store_true", help="apply --strip-headers to websocket upgrades only")
    a = ap.parse_args()
    STRIP.update(h.strip().lower() for h in a.strip_headers.split(",") if h.strip())
    global STRIP_WS_ONLY
    STRIP_WS_ONLY = a.strip_ws_only
    try:
        asyncio.run(amain(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
