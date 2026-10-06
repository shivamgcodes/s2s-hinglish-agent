#!/usr/bin/env python
"""Headless protocol check against a server on /api/chat (mock or real): counts message kinds, validates 0x07
event fields against INTERFACE.md section 4.1, and checks that events=0 yields no 0x07. Streams a customer wav
as Opus (like the browser). Usage: venv-pp python ws_check.py --url https://localhost:8998 --record food_01"""
import argparse
import asyncio
import json
import ssl
import time
from collections import Counter

import aiohttp
import numpy as np
import sphn

REQ = {"session": ["record_id", "agent_type", "pairing", "voice", "role_prompt", "seed", "context_frames_left"],
       "text": ["frame", "t", "token", "piece", "forced"],
       "action": ["trigger_id", "stage", "payload", "latency_ms", "since_trigger_ms", "frame"],
       "metrics": ["step_ms", "lm_step_ms", "rtf", "vram", "session", "router"],
       "session_end": ["reason", "frames"]}
COMMON = ["type", "session_id", "t_wall"]


async def run(url, params, wav, seconds):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    kinds, evtypes, problems, stages = Counter(), Counter(), [], Counter()
    t0 = time.time()
    t_hs = None
    pcm, _ = sphn.read(wav, sample_rate=24000)
    pcm = pcm[0].astype(np.float32)
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(url.replace("http", "ws", 1) + "/api/chat", params=params, ssl=ctx,
                                max_msg_size=0) as ws:
            w = sphn.OpusStreamWriter(24000)

            async def sender():
                while t_hs is None:
                    await asyncio.sleep(0.02)
                i = 0
                while not ws.closed and i * 1920 < min(len(pcm), seconds * 24000):
                    w.append_pcm(pcm[i * 1920:(i + 1) * 1920])
                    b = w.read_bytes()
                    if b:
                        await ws.send_bytes(b"\x01" + b)
                    i += 1
                    await asyncio.sleep(0.08)
                await ws.close()

            st = asyncio.create_task(sender())
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue
                k = msg.data[0]
                kinds[k] += 1
                if k == 0 and t_hs is None:
                    t_hs = time.time()
                if k == 7:
                    ev = json.loads(msg.data[1:].decode())
                    evtypes[ev.get("type")] += 1
                    for f in COMMON + REQ.get(ev.get("type"), []):
                        if f not in ev:
                            problems.append(f"{ev.get('type')} missing {f}")
                    if ev.get("type") == "action":
                        stages[ev["stage"]] += 1
            st.cancel()
    return {"params": params, "kinds": dict(kinds), "event_types": dict(evtypes), "action_stages": dict(stages),
            "handshake_after_s": round(t_hs - t0, 2) if t_hs else None, "wall_s": round(time.time() - t0, 1),
            "problems": sorted(set(problems))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://localhost:8998")
    ap.add_argument("--record", default="food_01")
    ap.add_argument("--pairing", default="g2")
    ap.add_argument("--wav", default="/root/deploy/assets/ws/hinglish/tests/inputs/V3/food_07_g1.wav")
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    base = {"text_prompt": "", "voice_prompt": "", "seed": "1001", "record_id": a.record, "pairing": a.pairing}
    r1 = asyncio.run(run(a.url, {**base, "events": "1"}, a.wav, a.seconds))
    print(json.dumps(r1))
    r2 = asyncio.run(run(a.url, base, a.wav, a.seconds))
    print(json.dumps(r2))
    ok = (r1["kinds"].get(7, 0) > 0 and r2["kinds"].get(7, 0) == 0 and not r1["problems"]
          and r1["kinds"].get(0) == 1 and r1["kinds"].get(1, 0) > 0 and r1["kinds"].get(2, 0) > 0)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
