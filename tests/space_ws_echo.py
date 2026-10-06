"""S2S serverless REAL test #1 (DESIGN 9, risk R1/R17): does a websocket survive the HF Space proxy? Run this FIRST.

  python tests/space_ws_echo.py                                       # dry run: prints the plan, sends nothing
  python tests/space_ws_echo.py https://OWNER-SPACE.hf.space          # wss echo: text + binary, sizes 1 B..64 KB, 20 s hold
  python tests/space_ws_echo.py https://OWNER-SPACE.hf.space --outbound HOST:PORT --passcode ...
        # + one TCP connect FROM the Space to HOST:PORT (route /api/selftest/outbound, only when the Space has
        #   S2S_SELFTEST=1; turn it off again afterwards). Use a public non-443 port you control, e.g. a RunPod pod's
        #   exposed TCP port, to learn whether queue mode (ws://IP:<random port>) can work from the Space (R17).
No GPU is touched. Needs only aiohttp. Exit code 0 = echo OK.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp


async def run(a):
    base = a.url.rstrip("/")
    wsu = ("wss://" + base[8:]) if base.startswith("https://") else ("ws://" + base[7:])
    res = {"url": base, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    async with aiohttp.ClientSession() as cs:
        try:
            async with cs.get(base + "/healthz", timeout=aiohttp.ClientTimeout(total=90)) as r:
                res["healthz"] = r.status
            async with cs.get(base + "/api/config", timeout=aiohttp.ClientTimeout(total=30)) as r:
                res["config"] = await r.json(content_type=None) if r.status == 200 else r.status
        except Exception as e:
            res["http_error"] = repr(e)
        rtts, bad = [], []
        try:
            t = time.perf_counter()
            async with cs.ws_connect(wsu + "/ws-echo", timeout=30, heartbeat=None) as ws:
                res["ws_open_ms"] = round((time.perf_counter() - t) * 1000, 1)
                for size in (1, 100, 1000, 16000, 65536):
                    payload = os.urandom(size)
                    t = time.perf_counter()
                    await ws.send_bytes(payload)
                    m = await asyncio.wait_for(ws.receive(), 15)
                    rtts.append((time.perf_counter() - t) * 1000)
                    if m.type != aiohttp.WSMsgType.BINARY or m.data != payload:
                        bad.append(f"binary {size}: {m.type}")
                await ws.send_str("hello ✓ हिंदी")
                m = await asyncio.wait_for(ws.receive(), 15)
                if m.type != aiohttp.WSMsgType.TEXT or m.data != "hello ✓ हिंदी":
                    bad.append(f"text: {m.type}")
                # idle hold: does the proxy drop an idle-ish socket? one ping every 5 s
                t_end = time.time() + a.hold_s
                n = 0
                while time.time() < t_end:
                    await asyncio.sleep(5)
                    n += 1
                    await ws.send_bytes(n.to_bytes(4, "big"))
                    m = await asyncio.wait_for(ws.receive(), 15)
                    if m.type != aiohttp.WSMsgType.BINARY:
                        bad.append(f"hold ping {n}: {m.type}")
                        break
                res["hold_s"] = a.hold_s
            res["ws_echo"] = "OK" if not bad else "PARTIAL"
        except aiohttp.WSServerHandshakeError as e:
            res["ws_echo"] = f"FAIL handshake {e.status}"
            if e.status == 404:
                res["hint"] = ("R1: websocket 404 through the HF proxy. Check visibility is public/protected (not private), "
                               "that you used the direct *.hf.space URL, then fall back to launcher (c) on another host.")
        except Exception as e:
            res["ws_echo"] = f"FAIL {e!r}"
        res["rtt_ms"] = [round(x, 1) for x in rtts]
        res["problems"] = bad
        if a.outbound:
            h, _, p = a.outbound.rpartition(":")
            try:
                async with cs.get(base + "/api/selftest/outbound", params={"host": h, "port": p, "passcode": a.passcode or ""},
                                  timeout=aiohttp.ClientTimeout(total=30)) as r:
                    res["outbound"] = {"status": r.status, "body": await r.json(content_type=None)}
            except Exception as e:
                res["outbound"] = {"error": repr(e)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", nargs="?", help="https://OWNER-SPACE.hf.space (none = dry run)")
    ap.add_argument("--hold-s", type=float, default=20.0)
    ap.add_argument("--outbound", help="HOST:PORT for the Space outbound self-test (R17)")
    ap.add_argument("--passcode", default=os.environ.get("S2S_PASSCODE"))
    a = ap.parse_args()
    if not a.url:
        print("[ws_echo] plan: GET /healthz, /api/config; wss /ws-echo text+binary 1 B..64 KB; hold 20 s; "
              "optional --outbound HOST:PORT self-test.\n[ws_echo] dry run: no URL given, nothing sent")
        return 0
    res = asyncio.run(run(a))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res.get("ws_echo") == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
