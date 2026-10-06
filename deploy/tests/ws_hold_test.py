"""S2S serverless REAL test #3 (DESIGN 9): hold one call open for --minutes and see whether RunPod keeps the worker.
(DESIGN 1 calls this file hold_test.py; renamed per the ops task, DESIGN Amendment 1.)

  # dry run (default: prints the plan, sends nothing)
  python tests/ws_hold_test.py
  # through the deployed Space (the Space does the keepalive; change its KEEPALIVE_S / KEEPALIVE_PATH vars per run)
  python tests/ws_hold_test.py --space https://OWNER-SPACE.hf.space --passcode ... --minutes 4
  # direct to the LB endpoint (this script claims + keeps alive itself; needs env RUNPOD_API_KEY,
  # S2S_SESSION_SECRET, RUNPOD_ENDPOINT_ID) -- the clean way to run the 3-run matrix:
  python tests/ws_hold_test.py --direct lb --keepalive-s 0                 # A: no keepalive (does RunPod count the ws?)
  python tests/ws_hold_test.py --direct lb                                 # B: GET /status every 20 s, strict-pinned
  python tests/ws_hold_test.py --direct lb --keepalive-path /ping --keepalive-s 30   # C: health path (the example's advice)

Audio: --wav FILE (looped) or, by default, silence + a 0.4 s tone every 5 s. Streamed in real time as Opus.
The context ends at about 221 s (DEP1 A1.4) and the worker caps a call at CALL_MAX_S=300 s, so a 4-minute hold
normally ends with session_end context_full BEFORE 4 min: that counts as PASS. DROP = the socket closed without
session_end, or reason worker_lost / worker_shutdown / error. Output: one JSON summary on stdout + --out FILE.
"""
import argparse
import asyncio
import json
import time

import aiohttp

import rt_common as rc


async def run(a):
    out = {"test": "ws_hold", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "minutes": a.minutes,
           "target": a.space or f"direct-{a.direct}",
           "keepalive": None if a.space else {"path": a.keepalive_path, "every_s": a.keepalive_s, "strict": not a.no_strict}}
    ka, space_polls = [], []
    async with aiohttp.ClientSession() as cs:
        tgt = rc.make_target(a, cs)
        try:
            await tgt.prepare()
        except Exception as e:
            out.update({"result": "WAKE_FAIL", "error": repr(e), "phases": tgt.phases.durations()})
            await tgt.finish()
            return out
        out["worker_id"] = tgt.worker_id
        pcm = rc.load_wav(a.wav, a.minutes * 60) if a.wav else rc.tone_track(a.minutes * 60)
        url, hdrs = tgt.ws_args()
        kt = asyncio.create_task(tgt.keepalive_loop(ka))

        async def tick(st):
            el = time.time() - st["t_handshake"]
            line = f"{el:6.0f}s sent {st['frames_sent']} frames, model {st['model_samples'] / rc.SR:.0f}s"
            if a.space:
                s = await tgt.state()
                space_polls.append({"t": round(el, 1), "state": s.get("state"), "detail": s.get("detail"),
                                    "keepalive": s.get("keepalive")})
                line += f", space {s.get('state')}:{s.get('detail')}"
                if s.get("keepalive"):
                    line += f", space keepalive {s['keepalive'].get('codes')}"
            if ka:
                line += f", keepalive last {ka[-1]['status']}"
            rc.log("hold", line)

        st = await rc.stream_call(cs, url, hdrs, pcm, "hold", ssl_=rc.ssl_ctx(a), on_tick=tick, tick_s=a.tick_s)
        kt.cancel()
        out["finish"] = await tgt.finish()
        if a.space:
            out["space_final"] = await tgt.state()
    held = (st.get("t_stream_end") or time.time()) - (st.get("t_handshake") or time.time())
    ran_full = st.get("frames_sent", 0) * rc.FS >= len(pcm) - rc.FS
    out.update({"result": rc.outcome(st, ran_full), "held_s": round(held, 1), "ran_full": ran_full,
                "phases": tgt.phases.durations(), "stream": st,
                "keepalive_log": ka, "keepalive_summary": {
                    "n": len(ka), "codes": {str(k): sum(1 for x in ka if x["status"] == k) for k in {x["status"] for x in ka}},
                    "ms_p50": rc.pct([x["ms"] for x in ka], 50),
                    "ms_max": max((x["ms"] for x in ka if x["ms"] is not None), default=None)},
                "space_polls": space_polls})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_target_args(ap)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--wav", help="user audio to loop (default: silence + periodic tones)")
    ap.add_argument("--tick-s", type=float, default=10.0, help="progress line / Space state poll interval")
    ap.add_argument("--out", help="write the JSON summary here too")
    a = ap.parse_args()
    rc.plan_or_exit(a, "hold", f"hold one call for {a.minutes:g} min, report drops + keepalive codes")
    res = asyncio.run(run(a))
    s = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(s)
    print(s)
    st = res.get("stream") or {}
    print(f"\nRESULT {res['result']}: held {res.get('held_s')} s, end {(st.get('session_end') or {}).get('reason')}, "
          f"close {st.get('close_code')}, recv gap p95 {(st.get('recv_gap_ms') or {}).get('p95')} ms, "
          f"keepalive {res.get('keepalive_summary', {}).get('codes')}")


if __name__ == "__main__":
    main()
