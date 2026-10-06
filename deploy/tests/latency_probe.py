"""S2S serverless REAL test #4 (+ #5 GPU check), DESIGN 9: per-leg latency, run FROM INDIA (where the users are).

  python tests/latency_probe.py                                      # dry run: prints the plan, sends nothing
  python tests/latency_probe.py --space https://OWNER-SPACE.hf.space              # leg 1 only (no GPU woken, free)
  python tests/latency_probe.py --space https://OWNER-SPACE.hf.space --passcode ... --call     # legs 1-3 (wakes a GPU)
  python tests/latency_probe.py --direct lb --call                   # legs 2-3 measured from this machine (no Space)
  ... --baseline-url wss://localhost:8998 --insecure                 # also leg 3 against the DEP1 pod tunnel, to compare

Legs:
  1 browser->Space : DNS, TCP connect, TLS handshake, HTTP GET /healthz RTT (xN), wss /ws-echo RTT (xN binary pings)
  2 Space->worker  : GET /api/diag?sid= (one strict-pinned GET /status from the Space; rate-limited 1 per 10 s; only
                     for an active sid, so it needs --call). Direct mode: the same GET /status timed from here.
  3 full path      : stream --talk-s s of audio; "pipeline lag" = send time of user frame k -> arrival of the k-th
                     80 ms of model audio. It is transport + relay + GPU step + Opus buffering, NOT the agent's
                     reaction time (the model answers when it wants). Compare it with --baseline-url (DEP1 tunnel).
  5 GPU check      : step_ms p95 from the 0x07 metrics events (must stay < 80 ms) + /status gpu name via diag.
Output: one JSON object on stdout (+ --out).
"""
import argparse
import asyncio
import json
import socket
import ssl
import time
import urllib.parse

import aiohttp

import rt_common as rc


async def leg1(a, cs):
    """Each sub-measurement has its own try, so one failure does not discard the others."""
    u = urllib.parse.urlparse(a.space)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    r = {"host": host}
    ip = host
    try:
        t = time.perf_counter()
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        r["dns_ms"] = round((time.perf_counter() - t) * 1000, 1)
        ip = r["ip"] = infos[0][4][0]
    except Exception as e:
        r["dns_error"] = repr(e)
    tcp, tls = [], []
    try:
        for _ in range(a.n):
            if u.scheme == "https":
                # TCP connect alone, then a full TCP+TLS connect; TLS time = difference of the two (works on any 3.x)
                t = time.perf_counter()
                _, wr = await asyncio.wait_for(asyncio.open_connection(ip, port), 15)
                tcp.append((time.perf_counter() - t) * 1000)
                wr.close()
                ctx = rc.ssl_ctx(a) or ssl.create_default_context()
                t = time.perf_counter()
                _, wr = await asyncio.wait_for(asyncio.open_connection(ip, port, ssl=ctx, server_hostname=host), 15)
                tls.append((time.perf_counter() - t) * 1000 - tcp[-1])
                wr.close()
            else:
                t = time.perf_counter()
                _, wr = await asyncio.wait_for(asyncio.open_connection(ip, port), 15)
                tcp.append((time.perf_counter() - t) * 1000)
                wr.close()
        r["tcp_connect_ms"] = {"p50": rc.pct(tcp, 50), "min": round(min(tcp), 1)}
        if tls:
            r["tls_handshake_ms"] = {"p50": rc.pct(tls, 50), "min": round(min(tls), 1), "note": "TCP+TLS minus TCP"}
    except Exception as e:
        r["connect_error"] = repr(e)
    try:
        h = []
        for _ in range(a.n):
            t = time.perf_counter()
            async with cs.get(a.space.rstrip("/") + "/healthz", ssl=rc.ssl_ctx(a),
                              timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                r["healthz_status"] = resp.status
            h.append((time.perf_counter() - t) * 1000)
        r["http_rtt_ms"] = {"p50": rc.pct(h, 50), "min": round(min(h), 1), "max": round(max(h), 1)}
    except Exception as e:
        r["http_error"] = repr(e)
    try:
        e = []
        async with cs.ws_connect(rc.ws_from_http(a.space.rstrip("/")) + "/ws-echo", ssl=rc.ssl_ctx(a),
                                 timeout=30) as ws:
            for i in range(a.n * 4):
                t = time.perf_counter()
                await ws.send_bytes(i.to_bytes(4, "big") + b"\x00" * 200)
                m = await asyncio.wait_for(ws.receive(), 10)
                if m.type == aiohttp.WSMsgType.BINARY and m.data[:4] == i.to_bytes(4, "big"):
                    e.append((time.perf_counter() - t) * 1000)
        r["ws_echo_rtt_ms"] = {"p50": rc.pct(e, 50), "p95": rc.pct(e, 95), "min": round(min(e), 1) if e else None,
                               "n_ok": len(e), "n": a.n * 4}
    except aiohttp.WSServerHandshakeError as ex:
        r["ws_echo_error"] = f"handshake {ex.status} (R1: websocket blocked by the HF proxy?)"
    except Exception as ex:
        r["ws_echo_error"] = repr(ex)
    return r


async def call_legs(a, cs):
    r = {}
    tgt = rc.make_target(a, cs)
    try:
        await tgt.prepare()
    except Exception as e:
        r.update({"error": repr(e), "phases": tgt.phases.durations(), "finish": await tgt.finish()})
        return r
    r["worker_id"] = tgt.worker_id
    url, hdrs = tgt.ws_args()
    diags = []

    async def diag_loop():
        while True:
            try:
                st, j = await tgt.diag()
                diags.append({"status": st, **(j if isinstance(j, dict) else {})})
            except Exception as e:
                diags.append({"error": repr(e)})
            await asyncio.sleep(11)          # Space limit: 1 per 10 s per sid

    dt = asyncio.create_task(diag_loop())
    st = await rc.stream_call(cs, url, hdrs, rc.tone_track(a.talk_s), "probe", ssl_=rc.ssl_ctx(a))
    dt.cancel()
    r["finish"] = await tgt.finish()
    ms = [d.get("space_to_worker_ms") for d in diags if d.get("space_to_worker_ms") is not None]
    r["leg2_space_to_worker"] = {"p50": rc.pct(ms, 50), "min": min(ms) if ms else None, "samples": diags}
    r["leg3_full_path"] = {k: st.get(k) for k in ("pipeline_lag_ms", "recv_gap_ms", "refused", "session_end",
                                                  "close_code", "user_s", "model_s")}
    r["leg3_full_path"]["handshake_after_open_s"] = (round(st["t_handshake"] - st["t_open"], 2)
                                                      if st.get("t_handshake") else None)
    r["gpu_check"] = {"step_ms_p95_max": st.get("step_p95_max"),
                      "step_ms_last_metrics": (st.get("metrics_last") or {}).get("step_ms"),
                      "gpu": next((d.get("gpu") for d in diags if d.get("gpu")), None),
                      "verdict": None if st.get("step_p95_max") is None else
                      ("OK (< 80 ms)" if st["step_p95_max"] < 80 else "TOO SLOW (>= 80 ms frame budget, R4)")}
    return r


async def baseline(a, cs):
    q = {"events": "1", "record_id": a.record_id, "pairing": a.pairing, "seed": str(a.seed)}
    st = await rc.stream_call(cs, a.baseline_url.rstrip("/") + "/api/chat?" + urllib.parse.urlencode(q), {},
                              rc.tone_track(a.talk_s), "baseline", ssl_=rc.ssl_ctx(a))
    return {k: st.get(k) for k in ("pipeline_lag_ms", "recv_gap_ms", "refused", "step_p95_max")}


async def run(a):
    out = {"test": "latency_probe", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "target": a.space or f"direct-{a.direct}"}
    async with aiohttp.ClientSession() as cs:
        if a.space:
            try:
                out["leg1_browser_to_space"] = await leg1(a, cs)
            except Exception as e:
                out["leg1_browser_to_space"] = {"error": repr(e)}
        if a.call:
            out.update(await call_legs(a, cs))
        else:
            out["note"] = "legs 2-3 skipped: add --call (wakes a billed GPU worker)"
        if a.baseline_url:
            out["baseline_dep1"] = await baseline(a, cs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_target_args(ap)
    ap.add_argument("--call", action="store_true", help="also wake a worker and measure legs 2-3 (+ GPU check)")
    ap.add_argument("--n", type=int, default=5, help="samples per leg-1 measurement")
    ap.add_argument("--talk-s", type=float, default=30.0)
    ap.add_argument("--baseline-url", help="DEP1 pod server for comparison, e.g. wss://localhost:8998 (via ssh -L)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.direct and not a.call:
        a.call = True
    rc.plan_or_exit(a, "probe", "leg 1 (browser->Space)" + (" + legs 2-3 + GPU check (wakes a worker)" if a.call else ""))
    res = asyncio.run(run(a))
    s = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
