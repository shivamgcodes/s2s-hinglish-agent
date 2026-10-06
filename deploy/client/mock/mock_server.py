#!/usr/bin/env python
"""DEP1 Track 2 test-only mock of the browser-facing server (no GPU, no model).

Implements the browser side of /root/deploy/INTERFACE.md on one https port:
  GET /                    built client (client/dist), SPA fallback
  GET /api/records         demo records (common/session.demo_records)
  GET /api/records/{id}    full record + session_configs g1..g4 (common/session.session_config)
  GET /api/samples         [{name,url,note}] from client/samples/*.wav (+ optional samples/notes.json {name: note})
  GET /samples/<name>.wav
  GET /metrics             section-7 shape, fake timing numbers, nulls when idle
  GET /api/chat            ws: query-param session config, simulated prompt phase, 0x00 handshake, then replays
                           a V3_A test output (tests/out/V3_A/V3/<call>_s1001.{wav,json}) as 0x01 Opus + 0x02 text
                           at 12.5 Hz real time, plus (events=1) kind-0x07 JSON events: session, text, action
                           (scripted router chains: executed / unbound / error), metrics every 2 s, session_end.
Incoming 0x01 Opus from the browser is decoded (sphn) and counted (ring.end_sample in /metrics).

Every scripted action payload is labelled "[mock]". This is NOT the Track 1 server (server/mock_server.py);
it exists only to test the client. Run with /root/deploy/venv-pp/bin/python (aiohttp + sphn).
"""
import argparse
import asyncio
import json
import logging
import os
import random
import re
import ssl
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sphn
from aiohttp import web, WSMsgType

sys.path.insert(0, "/root/deploy/common")
import session as S  # noqa: E402  (frozen, INTERFACE.md section 3)

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent
OUT_DIR = Path("/root/deploy/assets/ws/hinglish/tests/out/V3_A/V3")
SR = 24000
FRAME = 1920
HZ = 12.5
CONTEXT_FRAMES = 3000
TRIGGER_RE = re.compile(r"ek minute|rukiye|ek second|dekh (leta|leti)|update kar (deta|deti)|just a sec|let me check"
                        r"|checking|hold on|check (kar|kr)|one (second|minute)", re.I)
log = logging.getLogger("mock")

TYPE_PREFIX = {"food_delivery_support": "food", "ecommerce_support": "ecom", "cab_ride_support": "cab",
               "subscription_account_support": "sub", "airport_ticket_counter": "air"}
MOCK_CALL = {"food_delivery_support": "change_delivery_address", "ecommerce_support": "initiate_return",
             "cab_ride_support": "report_lost_item", "subscription_account_support": "cancel_subscription",
             "airport_ticket_counter": "change_seat"}


def pick_call(agent_type: str | None, forced: str | None) -> str:
    if forced:
        return forced
    pre = TYPE_PREFIX.get(agent_type or "", "food")
    calls = sorted(p.name[:-len("_s1001.json")] for p in OUT_DIR.glob(f"{pre}_*_s1001.json")
                   if not any(s in p.name for s in (".asr.", ".judge.", ".meta.")))
    return calls[0] if calls else "food_07_g1"


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.session_id = None
        self.cfg = None
        self.frame = 0
        self.steps = []
        self.lm_steps = []
        self.user_samples = 0
        self.counts = {k: 0 for k in ("trigger", "asr", "needle", "resolved", "executed", "unbound", "error")}
        self.last_router_t = None
        self.t0 = None

    def blk(self, xs):
        if not xs:
            return {"last": None, "p50": None, "p95": None, "max": None, "n": 0}
        a = np.array(xs[-500:])
        return {"last": round(float(a[-1]), 2), "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2), "n": int(len(a))}

    def metrics(self):
        act = self.active
        return {
            "t_wall": time.time(),
            "session": {"active": act, "session_id": self.session_id if act else None,
                        "record_id": (self.cfg or {}).get("record_id") if act else None,
                        "agent_type": (self.cfg or {}).get("agent_type") if act else None,
                        "frame": self.frame if act else None, "conv_s": round(self.frame / HZ, 2) if act else None,
                        "context_frames_left": (CONTEXT_FRAMES - 171 - self.frame) if act else None},
            "step_ms": self.blk(self.steps), "lm_step_ms": self.blk(self.lm_steps),
            "rtf": round(float(np.mean(self.steps[-500:])) / 80.0, 3) if self.steps else None,
            "vram": {"allocated_gib": 16.33 if act else None, "reserved_gib": 18.40 if act else None,
                     "max_allocated_gib": 17.93 if act else None, "nvidia_smi_used_mib": 19280 if act else None},
            "ring": {"seconds": round(min(30.0, self.user_samples / SR), 2), "end_sample": self.user_samples},
            "inject": {"active": False, "queued_frames": 0},
            "mute": False,
            "router": {"last_event_t_wall": self.last_router_t, "counts": dict(self.counts)},
            "mock": True,
        }


STATS = Stats()
LOCK = asyncio.Lock()


def load_spm():
    try:
        import sentencepiece as spm
        cands = list(Path("/root/hf").rglob("tokenizer_spm_32k_3.model"))
        if cands:
            return spm.SentencePieceProcessor(model_file=str(cands[0]))
    except Exception as e:  # pragma: no cover
        log.warning("no spm: %s", e)
    return None


SPM = None


def tok_id(piece: str) -> int:
    if SPM is None:
        return -1
    i = SPM.piece_to_id(piece.replace(" ", "▁"))
    return int(i)


def scripted_chain(kind: str, cfg: dict, rec: dict | None, seg: str, phrase: str, rule: str):
    """Returns list of (delay_s, stage, payload, latency_ms) for one scripted router chain."""
    at = cfg.get("agent_type", "")
    name = MOCK_CALL.get(at, "get_status")
    pid = (rec or {}).get("primary_id", "X0000")
    chain = [(0.0, "trigger", {"phrase": phrase, "rule": rule, "score": 1.0 if rule == "regex" else 0.71,
                               "segment": seg}, 0.0)]
    asr_ms = random.uniform(2400, 3200)
    chain.append((asr_ms / 1000, "asr", {"text": "[mock] haan ji mera order wala address change karna hai please",
                                         "backend": "trelis", "audio_s": 30.0}, asr_ms))
    if kind == "error":
        chain.append((0.3, "error", {"where": "needle", "message": "[mock] simulated failure"}, 300.0))
        return chain
    nd_ms = random.uniform(900, 1400)
    ref = "primary" if kind == "executed" else "the other one from last year"
    fc = [{"name": name, "arguments": {"order_ref": ref, "new_value": "[mock] Flat 9B"}}]
    chain.append((nd_ms / 1000, "needle", {"function_calls": fc, "model_segments": ["[mock] ji, ek minute", seg]}, nd_ms))
    if kind == "unbound":
        chain.append((0.01, "resolved", {"calls": [{"name": name, "arguments": fc[0]["arguments"], "order_ref": ref,
                                                    "resolved_id": None, "resolver_rule": "none"}]}, 4.0))
        chain.append((0.0, "unbound", {"name": name, "order_ref": ref, "reason": "[mock] order_ref not bound to the record"}, 0.0))
        return chain
    chain.append((0.01, "resolved", {"calls": [{"name": name, "arguments": fc[0]["arguments"], "order_ref": ref,
                                                "resolved_id": pid, "resolver_rule": "default_active"}]}, 3.0))
    chain.append((0.01, "executed", {"name": name, "resolved_id": pid, "result": {"ok": True},
                                     "record_diff": {"[mock] field": ["old", "new"]}}, 2.0))
    return chain


async def chat(request: web.Request):
    q = request.query
    for k in ("text_prompt", "voice_prompt"):
        if k not in q:  # the stock server indexes these keys
            return web.Response(status=400, text=f"missing query param {k}")
    events = q.get("events") == "1"
    cfg, rec = None, None
    if q.get("record_id"):
        try:
            seed = int(q.get("seed", S.DEFAULT_SEED))
            cfg = S.session_config(q["record_id"], q.get("pairing", "g1"), seed)
            rec = S.load_records()[q["record_id"]]
        except KeyError as e:
            return web.Response(status=400, text=f"unknown record/pairing {e}")
        if q.get("agent_type") and q["agent_type"] != cfg["agent_type"]:
            return web.Response(status=400, text="agent_type does not match record")
        text_prompt = q["text_prompt"] or cfg["role_prompt"]
        voice = q["voice_prompt"] or cfg["voice"]
    else:
        text_prompt, voice = q["text_prompt"], q["voice_prompt"]
    log.info("chat query keys=%s record=%s pairing=%s events=%s seed=%s voice=%s prompt_len=%d explicit_prompt=%s",
             sorted(q.keys()), q.get("record_id"), q.get("pairing"), events, q.get("seed"), voice, len(text_prompt),
             bool(q["text_prompt"]))
    QLOG.write(json.dumps({"t": time.time(), "query": dict(q)}) + "\n")
    QLOG.flush()

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async with LOCK:
        STATS.reset()
        sid = "s-mock-" + uuid.uuid4().hex[:8]
        STATS.session_id, STATS.cfg = sid, cfg
        call = pick_call((cfg or {}).get("agent_type"), ARGS.call)
        pcm, _ = sphn.read(str(OUT_DIR / f"{call}_s1001.wav"), sample_rate=SR)
        pcm = pcm[0].astype(np.float32)
        pieces = json.load(open(OUT_DIR / f"{call}_s1001.json"))
        n_frames = min(len(pieces), pcm.shape[0] // FRAME, ARGS.max_frames or 10 ** 9)
        log.info("session %s replaying %s (%d frames)", sid, call, n_frames)
        sent = {"audio": 0, "text": 0, "events": 0}
        closed = asyncio.Event()

        async def send_event(obj):
            if not events or ws.closed:
                return
            obj = {**obj, "session_id": sid, "t_wall": time.time()}
            await ws.send_bytes(b"\x07" + json.dumps(obj, ensure_ascii=False).encode())
            sent["events"] += 1

        # prompt phase: stock is_alive() drops client messages here
        t_end = time.time() + ARGS.prompt_delay
        while time.time() < t_end:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=max(0.01, t_end - time.time()))
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    log.info("client left during prompt phase")
                    return ws
            except asyncio.TimeoutError:
                pass
        await ws.send_bytes(b"\x00")
        STATS.active = True
        STATS.t0 = time.time()
        if cfg is not None:
            await send_event({**cfg, "type": "session", "context_frames_left": CONTEXT_FRAMES - 171,
                              "mock_replay": call})
        else:
            await send_event({"type": "session", "record_id": None, "agent_type": None, "router_supported": False,
                              "role_prompt": text_prompt, "voice": voice, "context_frames_left": CONTEXT_FRAMES - 171})

        opus_w = sphn.OpusStreamWriter(SR)
        opus_r = sphn.OpusStreamReader(SR)

        async def recv_loop():
            async for msg in ws:
                if msg.type == WSMsgType.BINARY and msg.data and msg.data[0] == 1:
                    opus_r.append_bytes(msg.data[1:])
                    while True:
                        x = opus_r.read_pcm()
                        if x is None or x.shape[-1] == 0:
                            break
                        STATS.user_samples += int(x.shape[-1])
            closed.set()

        async def metrics_loop():
            while not closed.is_set() and STATS.active:
                await asyncio.sleep(2.0)
                await send_event({"type": "metrics", **STATS.metrics()})

        chains = []  # pending (t_due, stage, payload, latency, trigger_id, t_trigger, frame)
        last_trig_t = -1e9
        kinds = ["executed", "unbound", "error"]
        cur_seg = []
        last_tok_frame = -100

        def on_trigger(frame, phrase, rule):
            nonlocal last_trig_t
            k = kinds[STATS.counts["trigger"] % 3] if cfg is not None else None
            if k is None:
                return
            seg = "".join(cur_seg).strip()
            tid = f"t{STATS.counts['trigger'] + 1:02d}-f{frame}"
            t = time.time()
            last_trig_t = t
            acc = 0.0
            for d, st, pl, lat in scripted_chain(k, cfg, rec, seg, phrase, rule):
                acc += d
                chains.append([t + acc, st, pl, lat, tid, t, frame])

        rt = asyncio.create_task(recv_loop())
        mt = asyncio.create_task(metrics_loop())
        reason = "context_full"
        t_start = time.time()
        try:
            for f in range(n_frames):
                if closed.is_set():
                    reason = "client_closed"
                    break
                piece = pieces[f]
                opus_w.append_pcm(pcm[f * FRAME:(f + 1) * FRAME])
                b = opus_w.read_bytes()
                if b:
                    await ws.send_bytes(b"\x01" + b)
                    sent["audio"] += 1
                if piece not in ("PAD", "EPAD"):
                    await ws.send_bytes(b"\x02" + piece.encode())
                    sent["text"] += 1
                    await send_event({"type": "text", "frame": f, "t": round(f / HZ, 2), "token": tok_id(piece),
                                      "piece": piece, "forced": False})
                    if f - last_tok_frame > 15 or re.search(r"[.?!]\s*$", "".join(cur_seg)):
                        cur_seg.clear()
                    cur_seg.append(piece)
                    last_tok_frame = f
                    seg = "".join(cur_seg)
                    m = TRIGGER_RE.search(seg[-40:])
                    if m and time.time() - last_trig_t > 3.0:
                        on_trigger(f, m.group(0), "regex")
                if f in FORCE_FRAMES and not chains and time.time() - last_trig_t > 3.0:
                    on_trigger(f, "[mock] forced trigger", "difflib")
                now = time.time()
                for c in [c for c in chains if c[0] <= now]:
                    chains.remove(c)
                    _, st, pl, lat, tid, t0, fr = c
                    STATS.counts[st] += 1
                    STATS.last_router_t = now
                    await send_event({"type": "action", "trigger_id": tid, "stage": st, "payload": pl,
                                      "latency_ms": round(lat, 1), "since_trigger_ms": round((now - t0) * 1000, 1),
                                      "frame": fr if st == "trigger" else f})
                STATS.frame = f + 1
                lm = random.gauss(38.3, 0.3)
                STATS.lm_steps.append(lm)
                STATS.steps.append(lm + random.gauss(16.2, 0.6))
                # real-time pacing
                target = t_start + (f + 1) / HZ / ARGS.speed
                await asyncio.sleep(max(0.0, target - time.time()))
            else:
                reason = "context_full"
        finally:
            STATS.active = False
            if not ws.closed:
                await send_event({"type": "session_end", "reason": reason, "frames": STATS.frame})
                await ws.close()
            rt.cancel()
            mt.cancel()
            log.info("session %s end reason=%s frames=%d sent=%s user_samples=%d", sid, reason, STATS.frame, sent,
                     STATS.user_samples)
            QLOG.write(json.dumps({"t": time.time(), "end": reason, "frames": STATS.frame, "sent": sent,
                                   "user_samples": STATS.user_samples}) + "\n")
            QLOG.flush()
    return ws


async def records(_):
    out = []
    for rid, r in S.demo_records().items():
        out.append({"record_id": rid, "agent_type": r["agent_type"], "brand": r["brand"],
                    "agent_name_f": r["agent_name_f"], "agent_name_m": r["agent_name_m"],
                    "information": r["information"], "primary_id": r["primary_id"], "secondary_id": r["secondary_id"]})
    return web.json_response(out)


async def record(request):
    rid = request.match_info["rid"]
    demo = S.demo_records()
    if rid not in demo:
        return web.json_response({"error": "unknown record"}, status=404)
    r = dict(demo[rid])
    r["session_configs"] = {g: S.session_config(rid, g) for g in S.PAIRINGS}
    return web.json_response(r)


async def samples(_):
    d = CLIENT / "samples"
    notes = {}
    if (d / "notes.json").exists():
        try:
            notes = json.loads((d / "notes.json").read_text())
        except Exception:
            notes = {}
    out = [{"name": p.stem, "url": f"/samples/{p.name}", "note": notes.get(p.stem, notes.get(p.name, ""))}
           for p in sorted(d.glob("*.wav"))]
    return web.json_response(out)


async def sample_file(request):
    name = request.match_info["name"]
    p = (CLIENT / "samples" / name).resolve()
    if p.parent != (CLIENT / "samples").resolve() or not p.exists() or p.suffix != ".wav":
        raise web.HTTPNotFound()
    return web.FileResponse(p)


async def metrics(_):
    return web.json_response(STATS.metrics())


async def index(_):
    return web.FileResponse(ARGS.dist / "index.html")


def ssl_ctx():
    cert, key = ARGS.cert, ARGS.key
    if not (cert.exists() and key.exists()):
        cert.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"], check=True, capture_output=True)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


def main():
    global ARGS, QLOG, SPM
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--dist", type=Path, default=CLIENT / "dist")
    ap.add_argument("--cert", type=Path, default=HERE / "certs" / "cert.pem")
    ap.add_argument("--key", type=Path, default=HERE / "certs" / "key.pem")
    ap.add_argument("--http", action="store_true", help="plain http (tests only)")
    ap.add_argument("--prompt-delay", type=float, default=3.0)
    ap.add_argument("--call", default=None, help="V3_A call to replay, e.g. food_07_g1 (default: by agent type)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--force-trigger-frames", default="150,400,650",
                    help="frames where a scripted trigger fires (rotation executed/unbound/error)")
    ap.add_argument("--log", type=Path, default=HERE / "logs" / "mock_queries.jsonl")
    ARGS = ap.parse_args()
    global FORCE_FRAMES
    FORCE_FRAMES = {int(x) for x in ARGS.force_trigger_frames.split(",") if x.strip()}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ARGS.log.parent.mkdir(parents=True, exist_ok=True)
    QLOG = open(ARGS.log, "a")
    SPM = load_spm()
    app = web.Application()
    app.router.add_get("/api/chat", chat)
    app.router.add_get("/api/records", records)
    app.router.add_get("/api/records/{rid}", record)
    app.router.add_get("/api/samples", samples)
    app.router.add_get("/samples/{name}", sample_file)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/", index)
    app.router.add_static("/", path=str(ARGS.dist), follow_symlinks=True)
    web.run_app(app, host=ARGS.host, port=ARGS.port, ssl_context=None if ARGS.http else ssl_ctx())


if __name__ == "__main__":
    main()
