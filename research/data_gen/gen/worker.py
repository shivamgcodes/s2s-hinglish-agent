"""Tuning worker: loads Gemma once, runs job files dropped in gen/queue/, reloading the generator modules each job.
Exits after IDLE seconds without a job, or when gen/queue/STOP exists. Launch under flock in tmux.
Job file gen/queue/<name>.json: {"jobs": ["V1:id1,id2:/out/dir", ...], "records": false, "only_records": [..]}
"""
import importlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
Q = os.path.join(HERE, "queue")
IDLE = int(os.environ.get("IDLE", "420"))


def main():
    os.makedirs(Q, exist_ok=True)
    import common
    llm = common.LLM()
    last = time.time()
    while True:
        if os.path.exists(os.path.join(Q, "STOP")):
            os.remove(os.path.join(Q, "STOP"))
            break
        jobs = sorted(f for f in os.listdir(Q) if f.endswith(".json"))
        if not jobs:
            if time.time() - last > IDLE:
                print("[worker] idle, exiting", flush=True)
                break
            time.sleep(3)
            continue
        jf = os.path.join(Q, jobs[0])
        spec = json.load(open(jf))
        os.rename(jf, jf + ".running")
        print(f"[worker] job {jobs[0]}: {spec}", flush=True)
        try:
            import hindi_share, validate, records, generate
            for m in (common, hindi_share, validate, records, generate):
                importlib.reload(m)
            llm.__class__ = common.LLM  # pick up reloaded chat()
            if spec.get("records") or spec.get("only_records"):
                records.make_records(llm, only=spec.get("only_records"))
            scen = common.load_scenarios()
            recs = json.load(open(f"{common.DATA}/records.json"))
            for j in spec.get("jobs", []):
                parts = j.split(":")
                v = parts[0]
                ids = parts[1].split(",") if len(parts) > 1 and parts[1] else (common.GATE1_CALLS if v == "GATE1" else common.all_call_ids(scen))
                out = parts[2] if len(parts) > 2 else f"{common.DATA}/{v}"
                generate.run_job(llm, v, ids, out, scen, recs, judge=not spec.get("no_judge"), seed_base=spec.get("seed_base", 0),
                                 samples=tuple(spec.get("samples", (3, 2, 2))))
            print(f"[worker] totals: gen {llm.gen_s:.0f}s, out tokens {llm.tok_out} ({llm.tok_out / max(llm.gen_s, 1):.0f} tok/s)", flush=True)
        except Exception:
            traceback.print_exc()
        os.rename(jf + ".running", jf + ".done")
        print(f"[worker] DONE {jobs[0]}", flush=True)
        last = time.time()


if __name__ == "__main__":
    main()
