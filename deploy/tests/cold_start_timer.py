"""S2S serverless REAL test #2 (DESIGN 9): time from "user clicks Connect" to "agent talking", split into phases.

  python tests/cold_start_timer.py                                   # dry run: prints the plan, sends nothing
  python tests/cold_start_timer.py --space https://OWNER-SPACE.hf.space --passcode ... --runs 3 --gap-s 0
  python tests/cold_start_timer.py --direct lb --runs 1              # env RUNPOD_API_KEY, S2S_SESSION_SECRET, RUNPOD_ENDPOINT_ID

Per run:  POST /api/session (or direct wake) -> every state change with its time (waking:no worker yet,
waking:worker initializing, busy, ready) -> websocket open -> 0x00 handshake (prompt phase, about 9-10 s) ->
first model audio -> stream --talk-s seconds of silence+tones -> close (DELETE / release).
DESIGN 9 #2 wants three situations; you create them with timing, not flags:
  cold host        : first run after (re)deploying the image / changing GPU types   (expect 4-8 min)
  scale from zero  : wait > endpoint idle timeout (120 s) + a margin after the last call, then run  (expect 1-2 min)
  warm             : --runs 2 --gap-s 10  (second run inside the idle timeout; expect seconds)
--gap-s is the pause between runs (default 0). Output: JSON lines (one per run) on stdout and --out.
"""
import argparse
import asyncio
import json
import time

import aiohttp

import rt_common as rc


async def one(a, i):
    res = {"run": i, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "target": a.space or f"direct-{a.direct}"}
    t0 = time.time()
    async with aiohttp.ClientSession() as cs:
        tgt = rc.make_target(a, cs)
        try:
            await tgt.prepare()
        except Exception as e:
            res.update({"result": "WAKE_FAIL", "error": repr(e), "phases": tgt.phases.durations(),
                        "finish": await tgt.finish()})
            return res
        t_ready = time.time()
        res["worker_id"] = tgt.worker_id
        url, hdrs = tgt.ws_args()
        st = await rc.stream_call(cs, url, hdrs, rc.tone_track(a.talk_s), f"cold#{i}", ssl_=rc.ssl_ctx(a))
        res["finish"] = await tgt.finish()
    hs = st.get("t_handshake")
    res.update({
        "result": "OK" if hs else ("REFUSED" if st.get("refused") else "NO_HANDSHAKE"),
        "phases": tgt.phases.durations(),
        "space_timeline": getattr(tgt, "timeline", None),
        "to_ready_s": round(t_ready - t0, 2),
        "ws_open_s": round(st["t_open"] - t_ready, 2) if st.get("t_open") else None,
        "prompt_phase_s": round(hs - st["t_open"], 2) if hs and st.get("t_open") else None,
        "first_audio_after_handshake_s": round(st["t_first_audio"] - hs, 2) if hs and st.get("t_first_audio") else None,
        "total_to_handshake_s": round(hs - t0, 2) if hs else None,
        "refused": st.get("refused"), "session_end": st.get("session_end"),
        "step_p95_max": st.get("step_p95_max"), "gpu_metrics_last": (st.get("metrics_last") or {}).get("step_ms")})
    return res


async def run(a):
    outs = []
    for i in range(a.runs):
        if i and a.gap_s:
            rc.log("cold", f"sleeping {a.gap_s:g} s before run {i}")
            await asyncio.sleep(a.gap_s)
        r = await one(a, i)
        outs.append(r)
        line = json.dumps(r, default=str)
        print(line, flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(line + "\n")
        rc.log("cold", f"run {i}: {r['result']} ready {r.get('to_ready_s')} s, handshake total "
                       f"{r.get('total_to_handshake_s')} s; phases " +
               ", ".join(f"{p['phase']}={p['dur_s']}" for p in r.get("phases", [])))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_target_args(ap)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--gap-s", type=float, default=0.0)
    ap.add_argument("--talk-s", type=float, default=8.0, help="seconds streamed after the handshake")
    ap.add_argument("--out", help="append JSON lines here")
    a = ap.parse_args()
    rc.plan_or_exit(a, "cold", f"{a.runs} run(s), gap {a.gap_s:g} s, {a.talk_s:g} s of audio each")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
