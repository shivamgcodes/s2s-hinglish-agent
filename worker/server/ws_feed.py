"""S2S serverless: fork of DEP1 server/ws_feed.py (headless websocket client). Streams a wav in real time
(80 ms pacing) as Opus to /api/chat with events=1, records everything the server sends, and checks the ring buffer.

  <venv-pp>/bin/python worker/server/ws_feed.py WAV --record-id food_01 --pairing g1 \
      --out DIR [--seconds 180] [--seed 1001] [--url ws://127.0.0.1:18000] [--internal http://127.0.0.1:18999]
      [--token <X-S2S-Token>] [--sid SID] [--header "Authorization: Bearer ..."]...

S2S additions: --token sends X-S2S-Token (worker direct; also used for the token-gated /metrics), --sid adds
?sid= (through the Space relay), --header adds raw request headers (e.g. via a RunPod LB URL), and a
"ws_refused" summary when the upgrade is refused (HTTP status + JSON body).

--seconds N  : stream N s of user audio (the wav is looped if shorter); default = wav length.
--tail-s S   : silence streamed after the audio (default 1.0).
Outputs in DIR: stereo.wav (L/ch0 = model audio received, R/ch1 = user audio sent; both from the handshake),
model.wav, events.jsonl (every 0x07 event + t_recv), frames.txt (frame, t, piece, forced from text events),
text_02.txt (stock 0x02 pieces), ring.wav (+ ring check), metrics_end.json, summary.json.
CPU only (no torch).
"""
import argparse
import asyncio
import json
import ssl
import time
import urllib.parse
from pathlib import Path

import aiohttp
import numpy as np
import sphn

SR, FS = 24000, 1920


def load_wav(path):
    a, sr = sphn.read(path)
    a = a.mean(axis=0) if a.shape[0] > 1 else a[0]
    if sr != SR:
        a = sphn.resample(a, sr, SR)
    return a.astype(np.float32)


def ring_check(ring, sent, end_sample):
    """Align ring (post-Opus-decode, last len(ring) samples up to end_sample) with what was sent."""
    res = {"ring_samples": int(len(ring)), "end_sample": int(end_sample), "sent_samples": int(len(sent))}
    if len(ring) == 0:
        return res
    ref_end = min(end_sample, len(sent))
    ref = sent[max(0, ref_end - len(ring)):ref_end]
    n = min(len(ref), len(ring))
    ref, rg = ref[-n:], ring[-n:]
    # lag search +-2400 samples (Opus codec delay) on a 5 s window from the middle
    w0, w = max(0, n // 2 - 60000), min(n, 120000)
    a, b = ref[w0:w0 + w], rg[w0:w0 + w]
    best = (None, -1.0)
    for lag in range(-2400, 2401, 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        if len(x) < 1000:
            continue
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best[1]:
            best = (lag, c)
    lag = best[0]
    if lag >= 0:
        x, y = ref[:n - lag], rg[lag:]
    else:
        x, y = ref[-lag:], rg[:n + lag]
    g = float(np.dot(x, y) / (np.dot(x, x) + 1e-9))
    snr = 10 * np.log10(np.sum(x ** 2) / (np.sum((y - x) ** 2) + 1e-12))
    res.update({"lag_samples": lag, "corr": round(best[1], 4), "gain": round(g, 4), "snr_db": round(float(snr), 2),
                "ring_rms": round(float(np.sqrt(np.mean(rg ** 2))), 4), "ref_rms": round(float(np.sqrt(np.mean(ref ** 2))), 4)})
    return res


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = load_wav(a.wav)
    n_sec = a.seconds if a.seconds else len(src) / SR
    n_frames = int(round(n_sec * SR / FS))
    reps = int(np.ceil(n_frames * FS / len(src)))
    user = np.tile(src, reps)[:n_frames * FS]
    tail = np.zeros(int(round(a.tail_s * SR / FS)) * FS, np.float32)
    user = np.concatenate([user, tail])
    q = {"events": "1"}
    for k, v in (("sid", a.sid), ("record_id", a.record_id), ("pairing", a.pairing), ("seed", a.seed),
                 ("text_prompt", a.text_prompt), ("voice_prompt", a.voice_prompt), ("agent_type", a.agent_type)):
        if v is not None:
            q[k] = str(v)
    url = a.url.rstrip("/") + "/api/chat?" + urllib.parse.urlencode(q)
    sslctx = ssl.create_default_context()
    sslctx.check_hostname = False
    sslctx.verify_mode = ssl.CERT_NONE
    writer = sphn.OpusStreamWriter(SR)
    reader = sphn.OpusStreamReader(SR)
    events, texts, model_pcm = [], [], []
    st = {"t_connect": time.time(), "t_handshake": None, "frames_sent": 0, "send_late_max_ms": 0.0,
          "bytes_in": 0, "bytes_out": 0, "session_end": None}
    hdrs = {}
    for h in a.header or []:
        k, _, v = h.partition(":")
        hdrs[k.strip()] = v.strip()
    if a.token:
        hdrs["X-S2S-Token"] = a.token
    async with aiohttp.ClientSession(headers=hdrs) as cs:
        try:
            ws_cm = await cs.ws_connect(url, ssl=sslctx, max_msg_size=0, timeout=60.0)
        except aiohttp.WSServerHandshakeError as e:
            res = {"ws_refused": e.status, "message": e.message, "url": url}
            (out / "summary.json").write_text(json.dumps(res, indent=1))
            print("[ws_feed] refused", json.dumps(res), flush=True)
            return res
        async with ws_cm as ws:
            hs = asyncio.get_running_loop().create_future()

            async def recv():
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.BINARY:
                        continue
                    d = msg.data
                    st["bytes_in"] += len(d)
                    k = d[0]
                    if k == 0:
                        st["t_handshake"] = time.time()
                        if not hs.done():
                            hs.set_result(True)
                    elif k == 1:
                        reader.append_bytes(d[1:])
                        p = reader.read_pcm()
                        if p.shape[-1]:
                            model_pcm.append(p.reshape(-1).astype(np.float32))
                    elif k == 2:
                        texts.append(d[1:].decode("utf-8", "replace"))
                    elif k == 7:
                        ev = json.loads(d[1:].decode("utf-8"))
                        ev["t_recv"] = time.time()
                        events.append(ev)
                        if ev.get("type") == "session_end":
                            st["session_end"] = ev

            rt = asyncio.create_task(recv())
            await asyncio.wait_for(hs, timeout=120)
            print(f"[ws_feed] handshake after {st['t_handshake'] - st['t_connect']:.2f} s; streaming "
                  f"{len(user) / SR:.1f} s", flush=True)
            t0 = time.perf_counter()
            for i in range(len(user) // FS):
                due = t0 + i * FS / SR
                now = time.perf_counter()
                if due > now:
                    await asyncio.sleep(due - now)
                else:
                    st["send_late_max_ms"] = max(st["send_late_max_ms"], (now - due) * 1000)
                writer.append_pcm(user[i * FS:(i + 1) * FS])
                b = writer.read_bytes()
                if st["session_end"] is not None or ws.closed:
                    break
                if len(b):
                    try:
                        await ws.send_bytes(b"\x01" + b)
                    except (aiohttp.ClientConnectionResetError, ConnectionResetError):
                        st["send_error"] = "server closed the socket"
                        break
                    st["bytes_out"] += len(b) + 1
                st["frames_sent"] = i + 1
            await asyncio.sleep(0.6)   # let the last frames come back
            st["t_stream_end"] = time.time()
            # ring + metrics before closing (session still active)
            if a.internal:
                try:
                    async with cs.get(a.internal + "/internal/session") as r:
                        st["internal_session"] = await r.json()
                    async with cs.get(a.internal + "/internal/ring?seconds=30") as r:
                        ring = np.frombuffer(await r.read(), dtype="<f4").astype(np.float32)
                        st["ring_headers"] = {k: r.headers.get(k) for k in ("X-Sample-Rate", "X-End-Sample", "X-End-Frame")}
                    sphn.write_wav(str(out / "ring.wav"), ring, SR)
                    st["ring_check"] = ring_check(ring, user[:st["frames_sent"] * FS], int(st["ring_headers"]["X-End-Sample"]))
                except Exception as e:
                    st["internal_error"] = repr(e)
            try:
                async with cs.get(a.url.replace("wss://", "https://").replace("ws://", "http://").rstrip("/") + "/metrics",
                                  ssl=sslctx) as r:
                    m = await r.json()
            except Exception as e:
                m = {"error": repr(e)}
            (out / "metrics_end.json").write_text(json.dumps(m, indent=1))
            await ws.close()
            try:
                await asyncio.wait_for(rt, timeout=3)
            except Exception:
                pass
    sent = user[:st["frames_sent"] * FS]
    model = np.concatenate(model_pcm) if model_pcm else np.zeros(0, np.float32)
    n = max(len(model), len(sent))
    stereo = np.zeros((2, n), np.float32)
    stereo[0, :len(model)] = model
    stereo[1, :len(sent)] = sent
    sphn.write_wav(str(out / "stereo.wav"), stereo, SR)
    sphn.write_wav(str(out / "model.wav"), model, SR)
    with open(out / "events.jsonl", "w") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    tev = [e for e in events if e.get("type") == "text"]
    with open(out / "frames.txt", "w") as f:
        f.write("# frame\tt\ttoken\tpiece\tforced   (text events, non-PAD only)\n")
        for e in tev:
            f.write(f"{e['frame']}\t{e['t']:.2f}\t{e['token']}\t{e['piece']!r}\t{int(e['forced'])}\n")
    (out / "text_02.txt").write_text("".join(texts))
    summ = dict(st)
    summ.update({"url": url, "wav": a.wav, "user_s": round(len(sent) / SR, 2), "model_s": round(len(model) / SR, 2),
                 "n_text_02": len(texts), "n_text_events": len(tev),
                 "text_02_equals_events": "".join(texts) == "".join(e["piece"] for e in tev),
                 "event_counts": {t: sum(1 for e in events if e.get("type") == t) for t in {e.get("type") for e in events}},
                 "session_event": next((e for e in events if e.get("type") == "session"), None),
                 "text": "".join(texts).strip()[:2000],
                 "metrics_end": {k: m.get(k) for k in ("step_ms", "lm_step_ms", "rtf", "vram", "backlog_frames", "session")}})
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: summ[k] for k in ("frames_sent", "user_s", "model_s", "n_text_events", "text_02_equals_events",
                                           "event_counts", "send_late_max_ms")}, ensure_ascii=False), flush=True)
    print("[ws_feed] ring_check", json.dumps(summ.get("ring_check")), flush=True)
    print("[ws_feed] metrics", json.dumps(summ["metrics_end"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--record-id")
    ap.add_argument("--pairing", default="g1")
    ap.add_argument("--agent-type")
    ap.add_argument("--seed")
    ap.add_argument("--text-prompt")
    ap.add_argument("--voice-prompt")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--url", default="ws://127.0.0.1:18000")
    ap.add_argument("--internal", default="http://127.0.0.1:18999", help="'' to skip ring/session checks")
    ap.add_argument("--token", default=None, help="X-S2S-Token header (worker direct)")
    ap.add_argument("--sid", default=None, help="adds ?sid= (Space relay)")
    ap.add_argument("--header", action="append", help='extra request header "Name: value" (repeatable)')
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
