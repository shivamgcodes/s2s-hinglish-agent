"""S2S serverless, Integrate stage GPU smoke: the browser's path against the REAL worker (tests/gpu_smoke.sh starts it).

POST /api/session on the Space right when the worker process starts (so the Space's wake loop sees the cold load
through the fake LB proxy, as a browser would), wait for ready, then stream a real Hinglish customer recording over
the Space relay with the stock client protocol (Opus, kind 0x01; 0x07 events on). Writes one JSON with the phases,
the stream stats, every 0x07 event, and the worker /status before/after.

  /root/deploy/venv-pp/bin/python tests/gpu_call.py --space http://127.0.0.1:17860 --worker http://127.0.0.1:18000 \
      --wav /workspace/hinglish/tests/inputs/V3/food_07_g1.wav --seconds 75 --out results/gpu_smoke_call.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rt_common as rc  # noqa: E402
from test_mock_e2e import args_ns  # noqa: E402


async def amain(a):
    out = {"t_start": time.time(), "args": vars(a)}
    evs = []
    async with aiohttp.ClientSession() as cs:
        tgt = rc.SpaceTarget(args_ns(space=a.space, record_id=a.record_id, pairing=a.pairing, seed=a.seed,
                                     wake_timeout_s=a.wake_timeout_s), cs)
        t0 = time.time()
        try:
            ready = await tgt.prepare()
        except Exception as e:
            out["error"] = f"prepare: {e!r}"
            out["phases"] = tgt.phases.durations()
            json.dump(out, open(a.out, "w"), indent=1, default=str)
            print(json.dumps(out, default=str)[:2000])
            return 1
        out["ready_after_s"] = round(time.time() - t0, 1)
        out["ready_state"] = ready
        async with cs.get(a.worker + "/status") as r:
            out["worker_status_before"] = await r.json(content_type=None)
        pcm = rc.load_wav(a.wav, seconds=a.seconds)
        url, hdrs = tgt.ws_args()
        ka = []

        async def tick(st):
            s = await tgt.state()
            ka.append({"t": round(time.time() - t0, 1), "keepalive": s.get("keepalive"), "state": s.get("state")})

        st = await rc.stream_call(cs, url, hdrs, pcm, "gpu", on_tick=tick, tick_s=10.0,
                                  on_event=lambda ev, t: evs.append(dict(ev, _t_recv=round(t, 3))))
        await asyncio.sleep(2.0)
        out["space_final"] = await tgt.state()
        out["space_ticks"] = ka
        await tgt.finish()
        await asyncio.sleep(1.0)
        async with cs.get(a.worker + "/status") as r:
            out["worker_status_after"] = await r.json(content_type=None)
    out["phases"] = tgt.phases.durations()
    out["stream"] = st
    out["events"] = evs
    acts = [e for e in evs if e.get("type") == "action"]
    out["action_stages"] = [e.get("stage") for e in acts]
    m = [e for e in evs if e.get("type") == "metrics"]
    out["metrics_last"] = m[-1] if m else None
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    summ = {k: out.get(k) for k in ("ready_after_s", "action_stages")}
    summ.update({k: st.get(k) for k in ("model_s", "user_s", "n_text", "events", "session_end", "close_code",
                                        "recv_gap_ms", "pipeline_lag_ms", "step_p95_max", "t_handshake", "t_open")})
    print(json.dumps(summ, default=str))
    ok = st.get("t_handshake") and st.get("model_samples", 0) > 24000 * 10 and st.get("n_text", 0) > 0
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="http://127.0.0.1:17860")
    ap.add_argument("--worker", default="http://127.0.0.1:18000")
    ap.add_argument("--wav", default="/workspace/hinglish/tests/inputs/V3/food_07_g1.wav")
    ap.add_argument("--seconds", type=float, default=75)
    ap.add_argument("--record-id", default="food_07")
    ap.add_argument("--pairing", default="g1")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--wake-timeout-s", type=float, default=900)
    ap.add_argument("--out", required=True)
    sys.exit(asyncio.run(amain(ap.parse_args())))


if __name__ == "__main__":
    main()
