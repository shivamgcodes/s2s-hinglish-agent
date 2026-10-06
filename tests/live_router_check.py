"""S2S serverless live check (D-ROUTER-V2, 2026-10-06): scripted calls through the PUBLIC Space, back to back on the
same worker, recording every router action event and the turn-filler (ticker indicator) events.

  python tests/live_router_check.py --space https://OWNER-SPACE.hf.space --passcode ... \
      --call food_08:g1:/path/food_08_g1_cust.wav --call food_21:g1:/path/food_21_g1_cust.wav [--tail-s 8] [--out f.json]

Each --call is record_id:pairing:wav (mono customer audio, 24 kHz; a stereo file is averaged, so pass the customer
channel only). The wav is streamed once in real time, then --tail-s of silence so the agent can finish its
check-line and the router can act. Without --space: dry run (prints the plan).
"""
import argparse
import asyncio
import copy
import json
import time

import aiohttp
import numpy as np

import rt_common as rc


async def one_call(a, cs, rid, pairing, wav):
    na = copy.copy(a)
    na.record_id, na.pairing = rid, pairing
    out = {"record_id": rid, "pairing": pairing, "wav": wav, "actions": [], "filler": [], "turn_fill": []}
    tgt = rc.make_target(na, cs)
    t0 = time.time()
    try:
        await tgt.prepare()
    except Exception as e:  # noqa: BLE001
        out.update(result="WAKE_FAIL", error=repr(e), phases=tgt.phases.durations())
        await tgt.finish()
        return out
    out["ready_s"] = round(time.time() - t0, 1)
    out["worker_id"] = tgt.worker_id
    pcm = rc.load_wav(wav)
    if a.max_s:
        pcm = pcm[:int(a.max_s * rc.SR)]
    pcm = np.concatenate([pcm, np.zeros(int(a.tail_s * rc.SR), np.float32)])
    pcm = pcm[:len(pcm) // rc.FS * rc.FS]
    url, hdrs = tgt.ws_args()

    def on_event(ev, t):
        ty = ev.get("type")
        if ty == "action":
            out["actions"].append({k: ev.get(k) for k in ("trigger_id", "stage", "payload", "latency_ms",
                                                           "since_trigger_ms", "frame")})
        elif ty == "filler":
            out["filler"].append({k: ev.get(k) for k in ("state", "reason", "frame", "t", "mode")})
        elif ty == "turn_fill":
            out["turn_fill"].append(ev.get("mode"))
        elif ty == "session":
            out["session_id"] = ev.get("session_id")

    st = await rc.stream_call(cs, url, hdrs, pcm, f"live-{rid}", ssl_=rc.ssl_ctx(a), on_event=on_event)
    out["finish"] = await tgt.finish()
    m = st.get("metrics_last") or {}
    out.update(result=rc.outcome(st, st.get("frames_sent", 0) * rc.FS >= len(pcm) - rc.FS),
               audio_s=round(len(pcm) / rc.SR, 1), session_end=st.get("session_end"),
               events=st.get("events"), n_text=st.get("n_text"),
               router_counts=(m.get("router") or {}).get("counts"), turn_fill_metrics=m.get("turn_fill"),
               input_agc=m.get("input_agc"),
               step_p95=(m.get("step_ms") or {}).get("p95"))
    return out


async def run(a):
    res = {"test": "live_router_check", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "target": a.space, "calls": []}
    async with aiohttp.ClientSession() as cs:
        for spec in a.call:
            rid, pairing, wav = spec.split(":", 2)
            r = await one_call(a, cs, rid, pairing, wav)
            res["calls"].append(r)
            rc.log("live", f"{rid}: {r.get('result')} actions " +
                   ", ".join(f"{x['stage']}" for x in r.get("actions", [])))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_target_args(ap)
    ap.add_argument("--call", action="append", required=True, help="record_id:pairing:wav (repeatable)")
    ap.add_argument("--tail-s", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=0.0, help="stream only the first N s of each wav (0 = all)")
    ap.add_argument("--out")
    a = ap.parse_args()
    rc.plan_or_exit(a, "live", f"{len(a.call)} scripted call(s) through the Space: {a.call}")
    res = asyncio.run(run(a))
    s = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    if a.out:
        open(a.out, "w").write(s)
    for c in res["calls"]:
        print(f"\n== {c['record_id']} {c.get('result')} session {c.get('session_id')} ready {c.get('ready_s')} s, "
              f"router counts {c.get('router_counts')}, filler events {len(c.get('filler', []))}\n   input_agc {c.get('input_agc')}")
        for x in c.get("actions", []):
            p = x["payload"] or {}
            if x["stage"] == "asr":
                print(f"   asr: {p.get('text')!r}  raw: {p.get('text_raw')!r}")
            elif x["stage"] == "needle":
                print(f"   needle ({p.get('router')}): {json.dumps(p.get('function_calls'), ensure_ascii=False)}")
            elif x["stage"] in ("executed", "unbound", "needs_clarification", "error"):
                print(f"   {x['stage']}: {json.dumps(p, ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
