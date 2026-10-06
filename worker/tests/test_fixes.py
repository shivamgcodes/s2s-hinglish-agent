"""S2S serverless worker: CPU tests for the 2026-10-06 live-demo fixes ported from pod1 (D-AGC, D-GATE, D-FILL2,
D-TOGGLE; session recording off by default). No GPU.

  PYTHONDONTWRITEBYTECODE=1 <venv-pp>/bin/python worker/tests/test_fixes.py [--demo-core PATH] [--port-base 18400]

Part A (unit, in process): worker core.InputAGC / TurnFiller.
  - gate: frames below -55 dBFS -> exact zeros, 4-frame hangover; AGC brings -41 dBFS speech to about -21 dBFS
  - D-GATE2 adaptive gate: open threshold = clip(noise floor + 12, -55, -38); a loud room (-50 dBFS noise) is gated,
    speech passes; digital silence keeps -55; long speech is never gated (cap); --gate-wav (the live call_0851 user
    channel) -> % speech frames passed / noise frames gated, fixed vs adaptive
  - ticker buffer: RMS -34 dBFS, 1.0 s; fill after 400 ms of model quiet; check-line skip; user speech cancels
  - per-session mode (reset("off"/"ticker")), request_mode() applied at the next input() (GPU thread), env defaults
  - IDENTICAL to the pod1 demo: with --demo-core (the live demo's server/core.py, copied to a temp dir first, so
    nothing is written next to it) the same input frames give bit-identical AGC output and filler buffers/events.
Part B (worker_server --mock over the websocket, ports port-base..):
  - turn_fill=off|ticker query -> /metrics turn_fill.mode + source "session"; bad value -> 400
  - mid-call b"\\x08" control frame -> mode switches; fills counted only while ticker
  - input_agc in the metrics (gate on, -55 dBFS); recording OFF by default (no files), ON with S2S_RECORD_SESSIONS=1
"""
import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "server"))
sys.path.insert(0, str(W))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_worker_mock as twm  # noqa: E402  (helpers: Worker, tok, http, wait_ping, check)

check = twm.check
FS, SR = 1920, 24000


def load_core(path, name):
    d = tempfile.mkdtemp(prefix=f"s2s-{name}-")
    shutil.copy(path, Path(d) / "core.py")
    shutil.copy(W / "server" / "ring.py", Path(d) / "ring.py")
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(name, Path(d) / "core.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


def db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-12)


def frames_scenario(seed=3):
    """Room noise (-66 dBFS) / speech bursts (-41 dBFS) / silence: 80 ms frames."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(60):
        if 10 <= k < 25 or 40 <= k < 48:
            t = np.arange(FS) / SR
            x = np.sin(2 * np.pi * 180 * t) + 0.3 * rng.normal(0, 1, FS)
            out.append((x / np.sqrt(np.mean(x ** 2)) * 10 ** (-41 / 20)).astype(np.float32))
        else:
            out.append((rng.normal(0, 1, FS) * 10 ** (-66 / 20)).astype(np.float32))
    return out


def model_out_scenario():
    """Model talks frames 0-9 ('aapka order ...'), quiet afterwards; then talks 30-35 with a check-line."""
    t = np.arange(FS) / SR
    loud = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    quiet = np.zeros(FS, np.float32)
    pieces = {2: " aapka", 4: " order", 6: " ready", 31: " ek", 32: " minute", 33: " rukiye"}
    return [(pieces.get(k), loud if (k < 10 or 30 <= k < 36) else quiet) for k in range(60)]


def run_filler(core_mod, filler, user_frames):
    ev = []
    for k, (piece, pm) in enumerate(model_out_scenario()):
        u = user_frames[k] if user_frames is not None else np.zeros(FS, np.float32)
        x = filler.input(u)
        r = filler.after(piece, pm, u)
        ev.append((k, r, float(np.abs(x).max()), x))
    return ev


def noisy_scenario(noise_db, speech_db, n=150, seed=5):
    """Noise at noise_db with -speech_db bursts (frames 30-45, 80-100); returns (frames, is_speech)."""
    rng = np.random.default_rng(seed)
    out, sp = [], []
    for k in range(n):
        if 30 <= k < 46 or 80 <= k < 100:
            t = np.arange(FS) / SR
            x = np.sin(2 * np.pi * 160 * t) + 0.3 * rng.normal(0, 1, FS)
            out.append((x / np.sqrt(np.mean(x ** 2)) * 10 ** (speech_db / 20)).astype(np.float32)); sp.append(True)
        else:
            out.append((rng.normal(0, 1, FS) * 10 ** (noise_db / 20)).astype(np.float32)); sp.append(False)
    return out, np.array(sp)


def gate_stats(core, frames, adaptive, speech_mask):
    g = core.InputAGC(adaptive=adaptive)
    passed = np.array([bool(np.any(g(x))) for x in frames])
    return passed, (passed[speech_mask].mean() if speech_mask.any() else None), \
        ((~passed[~speech_mask]).mean() if (~speech_mask).any() else None), g


def part_a_gate2(core, gate_wav):
    print("---- D-GATE2 adaptive gate", flush=True)
    # loud room: -50 dBFS noise, -30 dBFS speech (the fixed -55 gate passes all noise)
    fr, sp = noisy_scenario(-50, -30)
    _, sp_f, nz_f, _ = gate_stats(core, fr, False, sp)
    p, sp_a, nz_a, g = gate_stats(core, fr, True, sp)
    tail = np.array([k >= 16 and not sp[k] and not any(sp[max(0, k - 5):k]) for k in range(len(sp))])
    check("GATE2: -50 dBFS room noise: fixed gate passes it (the live failure)", nz_f < 0.05, nz_f)
    check("GATE2: adaptive gates the noise after the 1 s warm-up + hangover (except word-tail hangover)",
          (~p[tail]).mean() == 1.0, (~p[tail]).mean())
    check("GATE2: adaptive passes every -30 dBFS speech frame", sp_a == 1.0, sp_a)
    m = g.metrics()
    check("GATE2: metrics floor ~-50, threshold ~-38..-39 (floor+12, capped -38)",
          m["gate_adaptive"] and abs(m["noise_floor_db"] - (-50)) < 1.5 and -39.5 <= m["gate_threshold_db"] <= -38, m)
    # quiet room (-66) keeps the D-GATE behaviour: threshold -55 (lower bound)
    g = core.InputAGC(adaptive=True)
    [g(x) for x in frames_scenario(seed=7)]
    check("GATE2: -66 dBFS room -> threshold stays at the -55 lower bound", g.metrics()["gate_threshold_db"] in (-55.0, -54.0)
          or abs(g.metrics()["gate_threshold_db"] + 54.5) < 1.0, g.metrics())
    # digital silence (scripted food_08 path / mock): -55, everything zero
    g = core.InputAGC(adaptive=True)
    outs = [g(np.zeros(FS, np.float32)) for _ in range(80)]
    check("GATE2: digital silence -> threshold exactly -55, output exact zeros",
          g.metrics()["gate_threshold_db"] == -55.0 and not any(np.any(o) for o in outs), g.metrics())
    # 8 s of continuous -32 dBFS speech: the window is all speech, the -38 cap keeps it open
    fr = [x for x, s in zip(*noisy_scenario(-32, -32, n=100)) ]
    g = core.InputAGC(adaptive=True)
    p = [bool(np.any(g(x))) for x in fr]
    check("GATE2: 8 s of continuous -32 dBFS speech never gated (cap -38)", all(p), sum(p))
    if gate_wav and Path(gate_wav).is_file():
        import wave
        w = wave.open(gate_wav)
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        a = np.frombuffer(w.readframes(n), "<i2").reshape(-1, ch).astype(np.float32) / 32768
        u = a[:, 1] if ch > 1 else a[:, 0]          # ch1 = the user's mic as the browser captured it
        assert sr == SR, sr
        fr = [u[i:i + FS] for i in range(0, len(u) - FS + 1, FS)]
        dbs = np.array([db(x) for x in fr])
        sp = dbs > -35                               # rough ground truth (orchestrator spec)
        pf, sp_f, nz_f, _ = gate_stats(core, fr, False, sp)
        pa, sp_a, nz_a, g = gate_stats(core, fr, True, sp)
        quiet = slice(int(70 * 12.5), int(120 * 12.5))   # customer silent 70-120 s
        print(f"   call_0851 ch1 ({len(fr)} frames, {sp.sum()} speech > -35 dBFS): fixed -55: speech passed "
              f"{sp_f * 100:.1f}%, noise gated {nz_f * 100:.1f}%, 70-120 s passed {pf[quiet].mean() * 100:.1f}% | "
              f"adaptive: speech passed {sp_a * 100:.1f}%, noise gated {nz_a * 100:.1f}%, 70-120 s passed "
              f"{pa[quiet].mean() * 100:.1f}% | metrics {g.metrics()}", flush=True)
        check("GATE2 call_0851: adaptive passes >= 98% of speech frames (> -35 dBFS)", sp_a >= 0.98, sp_a)
        check("GATE2 call_0851: adaptive gates >= 75% of noise frames (fixed: < 20%)", nz_a >= 0.75 and nz_f < 0.2,
              (nz_a, nz_f))
    else:
        print(f"(skip call_0851 replay: --gate-wav {gate_wav!r} not found)", flush=True)


def part_a(demo_core, gate_wav=""):
    os.environ.pop("S2S_TURN_FILL_JSON", None)
    core = load_core(W / "server" / "core.py", "core_worker")
    # --- gate + AGC
    agc = core.InputAGC()
    fr = frames_scenario()
    outs = [agc(x) for x in fr]
    check("gate: room noise (-66 dBFS) frames before speech -> exact zeros", all(not np.any(o) for o in outs[:10]))
    check("gate: 4-frame hangover after speech (frames 25-28 not zeroed, 29 zeroed)",
          all(np.any(outs[k]) for k in range(25, 29)) and not np.any(outs[29]),
          [bool(np.any(outs[k])) for k in range(24, 31)])
    check("AGC: -41 dBFS speech reaches about -21 dBFS (end of 1st burst)", abs(db(outs[24]) - (-21)) < 1.5,
          round(db(outs[24]), 1))
    m = agc.metrics()
    check("AGC metrics: gate_on, -55 dBFS, target -21", m["gate_on"] and m["gate_open_db"] == -55 and m["target_db"] == -21, m)
    # --- filler buffer + timing
    f = core.TurnFiller()
    f.reset()
    check("filler: env default mode ticker, 1.0 s, quiet 400 ms, -34 dB", (f.mode, f.seconds, f.quiet_frames * 80,
          f.level_db) == ("ticker", 1.0, 400, -34.0), (f.mode, f.seconds, f.quiet_frames, f.level_db))
    check("filler: ticker buffer RMS -34 dBFS, 24000 samples", f.buf_rms_db == -34.0 and len(f.buf) == 24000,
          (f.buf_rms_db, len(f.buf)))
    ev = run_filler(core, f, None)
    fills = [k for k, r, _, _ in ev if r == "fill"]
    skips = [k for k, r, _, _ in ev if r == "skip_check"]
    check("filler: fills after 400 ms (5 frames) of model quiet (frame 14)", fills == [14], fills)
    check("filler: check-line ('ek minute rukiye') -> skip, no fill", skips == [40], skips)
    fed = np.concatenate([ev[k][3] for k in range(15, 28)])
    check("filler: model input frames 15..27 = the 1.0 s ticker buffer, then zeros",
          np.array_equal(fed[:24000], f.buf) and not np.any(fed[24000:]) and not np.any(ev[14][3])
          and not np.any(ev[28][3]))
    f.reset()
    user = [np.zeros(FS, np.float32)] * 17 + [np.full(FS, 0.05, np.float32)] + [np.zeros(FS, np.float32)] * 42
    run_filler(core, f, user)
    check("filler: user speech cancels a running fill", f.n_cancel_user == 1, f.metrics())
    # --- per-session mode + mid-call request
    f.reset("off")
    check("filler: session mode off -> no buffer, source session", f.buf is None and f.mode == "off"
          and f.source == "session", f.metrics())
    ev = run_filler(core, f, None)
    check("filler: off -> no fills", not any(r == "fill" for _, r, _, _ in ev))
    f.reset("off")
    f.request_mode("ticker")
    check("filler: request_mode is deferred (still off before the next input)", f.mode == "off")
    f.input(np.zeros(FS, np.float32))
    check("filler: applied at next input() -> ticker -34", f.mode == "ticker" and f.level_db == -34.0, f.metrics())
    f.reset("ticker")
    run_filler(core, f, None)
    f.reset("ticker")
    for k, (piece, pm) in enumerate(model_out_scenario()[:16]):
        f.input(np.zeros(FS, np.float32)); f.after(piece, pm, np.zeros(FS, np.float32))
    filling = f.pos >= 0
    f.request_mode("off")
    y = f.input(np.zeros(FS, np.float32))
    check("filler: mid-fill switch to off stops at once (zero input)", filling and not np.any(y) and f.pos == -1)
    f.request_mode("hmm")
    check("filler: request_mode ignores modes other than ticker/off", f._pending is None)
    part_a_gate2(core, gate_wav)
    # --- identical to the demo
    if demo_core and Path(demo_core).is_file():
        demo = load_core(demo_core, "core_demo")
        a1, a2 = core.InputAGC(adaptive=False), demo.InputAGC()
        same = all(np.array_equal(a1(x), a2(x)) for x in frames_scenario(seed=11))
        check("IDENTICAL to demo: InputAGC (S2S_GATE_ADAPTIVE=0) output bit-exact on noise/speech frames", same)
        a1, a2 = core.InputAGC(adaptive=True), demo.InputAGC()
        same = all(np.array_equal(a1(x), a2(x)) for x in frames_scenario(seed=11))
        check("IDENTICAL to demo: adaptive InputAGC bit-exact on the pod1 case (-66 dBFS room, -41 dBFS speech)", same)
        d1, d2 = core.TurnFiller(), demo.TurnFiller()
        d1.reset(); d2.reset()       # demo: env defaults (no turn_fill.json read: the temp copy's FILL_JSON may exist)
        check("IDENTICAL to demo: ticker buffer bit-exact", d1.buf is not None and d2.buf is not None
              and np.array_equal(d1.buf, d2.buf), (d2.mode, d2.source))
        e1, e2 = run_filler(core, d1, None), run_filler(demo, d2, None)
        check("IDENTICAL to demo: filler decisions + model input frame by frame",
              all(a[:3] == b[:3] and np.array_equal(a[3], b[3]) for a, b in zip(e1, e2)))
        d1.reset("ticker")
        check("IDENTICAL to demo: toggle 'ticker' preset == demo turn_fill.json ticker",
              (d1.mode, d1.seconds, d1.quiet_frames, d1.level_db) == ("ticker", 1.0, 5, -34.0))
    else:
        print(f"(skip demo equivalence: --demo-core {demo_core} not found)")


async def mock_call(cs, w, sid, query, control_at=None, seconds=6.0):
    """Mock call; returns the list of 0x07 metrics events (dicts)."""
    import aiohttp
    import sphn
    t = twm.tok(sid)
    for _ in range(50):                      # the previous call may still be ending
        st, body, _ = await twm.http(cs, "GET", w.url + "/status")
        if st == 200 and body.get("state") == "idle":
            break
        await asyncio.sleep(0.2)
    st, body, _ = await twm.http(cs, "POST", w.url + "/session/claim", token=t)
    assert st == 200, (st, body)
    q = {"record_id": "food_07", "pairing": "g1", "seed": "1001", "events": "1"}
    q.update(query)
    ws = await cs.ws_connect(w.url + "/api/chat", params=q, headers={"X-S2S-Token": twm.tok(sid)}, max_msg_size=0)
    mets, started = [], asyncio.Event()

    async def recv():
        async for m in ws:
            if m.type != aiohttp.WSMsgType.BINARY:
                continue
            if m.data[0] == 0:
                started.set()
            if m.data[0] == 7:
                e = json.loads(m.data[1:])
                if e.get("type") == "metrics":
                    mets.append((time.time(), e))
                elif e.get("type") == "filler":          # 2026-10-06 ticker indicator events
                    FILLER_EVS.setdefault(sid, []).append((time.time(), e))
    rt = asyncio.create_task(recv())
    await asyncio.wait_for(started.wait(), 20)
    wr = sphn.OpusStreamWriter(SR)
    t0 = time.perf_counter()
    i, sent_ctl = 0, None
    while time.perf_counter() - t0 < seconds:
        due = t0 + i * FS / SR
        if due > time.perf_counter():
            await asyncio.sleep(due - time.perf_counter())
        if control_at is not None and sent_ctl is None and time.perf_counter() - t0 > control_at[0]:
            await ws.send_bytes(b"\x08" + json.dumps({"type": "turn_fill", "mode": control_at[1]}).encode())
            sent_ctl = time.time()
        wr.append_pcm(np.zeros(FS, np.float32))
        b = wr.read_bytes()
        if len(b):
            await ws.send_bytes(b"\x01" + b)
        i += 1
    await ws.close()
    try:
        await asyncio.wait_for(rt, 8)
    except asyncio.TimeoutError:
        pass
    return mets, sent_ctl


FILLER_EVS = {}   # sid -> [(t_recv, 0x07 filler event)]


def filler_pairs_ok(evs):
    """Every start is followed by exactly one stop/cancel before the next start (the last may be cut by the call end)."""
    open_ = False
    for _, e in evs:
        if e["state"] == "start":
            if open_:
                return False
            open_ = True
        elif e["state"] in ("stop", "cancel"):
            if not open_:
                return False
            open_ = False
    return True


async def part_b(pb):
    import aiohttp
    logs = tempfile.mkdtemp(prefix="s2s-fixes-logs-")
    w = twm.Worker(pb, extra_env={"S2S_LOGS": logs}, name="fixes")
    try:
        async with aiohttp.ClientSession() as cs:
            seen = await twm.wait_ping(cs, w, 200, 40)
            check("B: mock worker /ping 200", seen and seen[-1] == 200, seen)
            st = await twm.ws_refusal(cs, w.url, token=twm.tok("s-bad-tf"), query={"turn_fill": "loud"})
            check("B: turn_fill=loud -> 400 before the upgrade", st == 400, st)
            mets, _ = await mock_call(cs, w, "s-fix-off", {"turn_fill": "off"}, seconds=5)
            tf = mets[-1][1].get("turn_fill", {}) if mets else {}
            check("B: turn_fill=off -> metrics mode off, source session", tf.get("mode") == "off"
                  and tf.get("source") == "session", tf)
            ag = mets[-1][1].get("input_agc", {}) if mets else {}
            check("B: input_agc in metrics (gate on, -55 dBFS, AGC on)", ag.get("gate_on") is True
                  and ag.get("gate_open_db") == -55 and ag.get("on") is True, ag)
            check("B: input_agc metrics carry the D-GATE2 adaptive gate (adaptive, margin 12, threshold in [-55, -38])",
                  ag.get("gate_adaptive") is True and ag.get("gate_margin_db") == 12
                  and -55 <= (ag.get("gate_threshold_db") or 0) <= -38 and "noise_floor_db" in ag, ag)
            mets, t_ctl = await mock_call(cs, w, "s-fix-tog", {"turn_fill": "off"}, control_at=(5.0, "ticker"),
                                          seconds=16)
            before = [e["turn_fill"] for t, e in mets if t < t_ctl - 0.1]
            after = [e["turn_fill"] for t, e in mets if t > t_ctl + 0.5]
            check("B: before the control frame: off", before and all(x["mode"] == "off" for x in before),
                  [x["mode"] for x in before])
            check("B: 0x08 {turn_fill: ticker} mid-call -> ticker -34 dB", after and after[-1]["mode"] == "ticker"
                  and after[-1]["level_db"] == -34.0, after[-1] if after else None)
            check("B: fills happen only after the switch (mock replay model output)",
                  all(x["fills"] == 0 for x in before) and after[-1]["fills"] >= 1, (before[-1:], after[-1]))
            fe = FILLER_EVS.get("s-fix-tog", [])
            starts = [e for t, e in fe if e["state"] == "start"]
            check("B: ticker indicator: 0x07 filler events only after the switch, start count == metrics fills",
                  fe and all(t > t_ctl for t, _ in fe) and starts and len(starts) == after[-1]["fills"],
                  ([(round(t - t_ctl, 2), e["state"], e["frame"]) for t, e in fe], after[-1]["fills"]))
            check("B: filler event fields (type, state, reason, frame, t, mode ticker, seconds 1.0, session_id)",
                  all({"state", "reason", "frame", "t", "mode", "seconds", "session_id", "t_wall"} <= set(e)
                      and e["mode"] == "ticker" and e["seconds"] == 1.0 for _, e in fe), fe[:2])
            check("B: every filler start is closed by one stop|cancel (real timing: stop ~12-13 frames later)",
                  filler_pairs_ok(fe) and all(0 < b["frame"] - a["frame"] <= 13 for (_, a), (_, b) in zip(fe, fe[1:])
                                              if a["state"] == "start" and b["state"] in ("stop", "cancel")),
                  [(e["state"], e["frame"]) for _, e in fe])
            check("B: turn_fill=off call -> no filler events", not FILLER_EVS.get("s-fix-off"), FILLER_EVS.get("s-fix-off"))
            mets, _ = await mock_call(cs, w, "s-fix-def", {}, seconds=4)
            tf = mets[-1][1].get("turn_fill", {}) if mets else {}
            check("B: no turn_fill param -> env default ticker (source env)", tf.get("mode") == "ticker"
                  and tf.get("source") == "env", tf)
            check("B: recording off by default -> no session files",
                  not (Path(logs) / "sessions").exists() or not any((Path(logs) / "sessions").iterdir()))
            check("B: worker log shows the toggle", "turn filler -> ticker (mid-call toggle)" in w.log_text())
    finally:
        w.stop()
    logs2 = tempfile.mkdtemp(prefix="s2s-fixes-rec-")
    w = twm.Worker(pb + 10, extra_env={"S2S_LOGS": logs2, "S2S_RECORD_SESSIONS": "1"}, name="fixes-rec")
    try:
        async with aiohttp.ClientSession() as cs:
            await twm.wait_ping(cs, w, 200, 40)
            await mock_call(cs, w, "s-fix-rec", {"turn_fill": "ticker"}, seconds=4)
            await asyncio.sleep(1.0)
            files = sorted(p.name.split("_", 1)[-1] if "_" in p.name else "main.wav"
                           for p in (Path(logs2) / "sessions").glob("*")) if (Path(logs2) / "sessions").exists() else []
            check("B: S2S_RECORD_SESSIONS=1 -> <sid>.wav, _user_agc.wav, _model_in.wav, _fill.json",
                  len(files) == 4, files)
    finally:
        w.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-core", default=os.environ.get("S2S_DEMO_CORE", ""),
                    help="the live demo core.py (pod1: the deploy tree server/core.py) for the equivalence checks")
    ap.add_argument("--port-base", type=int, default=18400)
    ap.add_argument("--gate-wav", default=os.environ.get("S2S_GATE_WAV", ""),
                    help="live call recording, ch1 = user mic (D-GATE2 replay; skipped if missing)")
    ap.add_argument("--only", default="", help="a | b")
    a = ap.parse_args()
    if a.only in ("", "a"):
        print("==== part A (unit)", flush=True)
        part_a(a.demo_core, a.gate_wav)
    if a.only in ("", "b"):
        print(f"==== part B (worker_server --mock, ports {a.port_base}..)", flush=True)
        asyncio.run(part_b(a.port_base))
    n_fail = sum(1 for _, ok in twm.RESULTS if not ok)
    print(f"\n{len(twm.RESULTS) - n_fail}/{len(twm.RESULTS)} checks passed", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
