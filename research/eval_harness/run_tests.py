"""Run PersonaPlex test runs for one model tag, sharded over K driver processes (each loads the model once).

  python run_tests.py --tag base [--adapter DIR | --moshi-weight F] [--variants V1,V2,V3] [--calls a,b]
                      [--seeds 1001,1002,1003] [--gate0/--no-gate0] [-K 1] [--dry]

Launch it through run_tests.sh, which takes the GPU lock once around all K processes.
Inputs: tests/inputs/<V>/<call>.wav + .meta.json (make_inputs.py). Outputs: tests/out/<tag>/<V>/<call>_s<seed>.*
and tests/out/<tag>/gate0/<clip>_s<seed>.*. Logs: tests/out/<tag>/logs/driver_k<i>.log, vram.csv (nvidia-smi per
pid every 5 s), timing.json. Input fed as ONE continuous stream (FixedFeeder): the test input already has the
customer turns at their scripted times with silent agent slots, so turn release is not needed.
Gate 0 clips "02"/"03": no numbered Hindi Gate 0 clips exist on the pod; substitutes = exp5 X4_hinglish_phrase
(sw1-sw4, turn-gated, Swiggy prompt, NATF2) as "02" and exp6 H3_template_greeting (sw1) as "03" (NOTES.md).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import INPUTS, OUT, SEEDS, TESTS, jdump, jload  # noqa: E402

G0 = Path("/workspace/personaplex-gate0")
GATE0 = [("02_X4_hinglish_phrase", G0 / "exp5/experiments.json", "X4_hinglish_phrase_s1"),
         ("03_H3_template_greeting", G0 / "exp6/experiments.json", "H3_template_greeting_s1")]
PY = "/workspace/venv-pp/bin/python"


def gate0_runs(seeds):
    runs = []
    for tag, spec, ename in GATE0:
        sp = jload(spec)
        e = next(x for x in sp["experiments"] if x["name"] == ename)
        base = spec.parent
        turns = []
        for t in e["turns"]:
            if isinstance(t, str):
                turns.append(str((base / t).resolve()))
            else:
                turns.append({"clip": str((base / t["clip"]).resolve()), "quiet_s": t["quiet_s"]})
        for s in seeds:
            runs.append({"name": f"gate0/{tag}_s{s}", "voice": e["voice"], "prompt": e["prompt"], "seed": s,
                         "turns": turns})
    return runs


def call_runs(variants, calls, seeds):
    runs = []
    for v in variants:
        d = INPUTS / v
        for mp in sorted(d.glob("*.meta.json")):
            m = jload(mp)
            if calls and m["call_id"] not in calls:
                continue
            for s in seeds:
                runs.append({"name": f"{v}/{m['call_id']}_s{s}", "voice": m["voice"], "prompt": m["role_prompt"],
                             "seed": s, "input_wav": m["input_wav"], "dur": m["duration"]})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--moshi-weight")
    ap.add_argument("--variants", default="")
    ap.add_argument("--calls", default="")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--gate0", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("-K", type=int, default=1)  # K>1 adds no throughput (NOTES tests)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--customer-mode", choices=["fixed", "vad"], default="fixed",
                    help="vad = reactive customer (driver.py / tests/VAD_GATING.md); fixed = unchanged default")
    ap.add_argument("--vad-params", help="JSON overrides of vad_gate.DEFAULTS (vad mode only)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    variants = [v for v in a.variants.split(",") if v] or sorted(p.name for p in INPUTS.iterdir()
                                                           if p.is_dir() and re.fullmatch(r"V\d+", p.name))
    calls = set(c for c in a.calls.split(",") if c)
    out = OUT / a.tag
    (out / "logs").mkdir(parents=True, exist_ok=True)
    runs = call_runs(variants, calls, seeds) + (gate0_runs(seeds) if a.gate0 else [])
    todo = [r for r in runs if not (out / f"{r['name']}.meta.json").exists()]
    # balance shards by expected duration (longest first, greedy)
    k = max(1, min(a.K, len(todo)))
    shards, load = [[] for _ in range(k)], [0.0] * k
    for r in sorted(todo, key=lambda r: -r.get("dur", 40.0)):
        i = load.index(min(load))
        shards[i].append(r)
        load[i] += r.get("dur", 40.0)
    print(f"[run_tests] tag={a.tag} variants={variants} runs={len(runs)} todo={len(todo)} K={k} "
          f"shard audio s={[round(x) for x in load]}", flush=True)
    if a.dry or not todo:
        return
    procs = []
    t0 = time.time()
    for i, sh in enumerate(shards):
        sp = out / "logs" / f"spec_k{i}.json"
        jdump(sp, {"runs": sh})
        cmd = [PY, "-u", str(Path(__file__).resolve().parent / "driver.py"), str(sp), str(out)]
        if a.adapter:
            cmd += ["--adapter", a.adapter]
        if a.moshi_weight:
            cmd += ["--moshi-weight", a.moshi_weight]
        if a.customer_mode != "fixed":
            cmd += ["--customer-mode", a.customer_mode]
            if a.vad_params:
                cmd += ["--vad-params", a.vad_params]
        log = open(out / "logs" / f"driver_k{i}.log", "a")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/workspace/personaplex"))
    vram = open(out / "logs" / "vram.csv", "a")
    vram.write("t_s,pid,used_mib,total_used_mib\n")
    peak_total, peak_pid = 0, {}
    while any(p.poll() is None for p in procs):
        try:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20).stdout
            tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            mine = {p.pid for p in procs}
            for line in q.strip().splitlines():
                pid, mib = [x.strip() for x in line.split(",")]
                if int(pid) in mine:
                    vram.write(f"{time.time() - t0:.0f},{pid},{mib},{tot}\n")
                    peak_pid[pid] = max(peak_pid.get(pid, 0), int(mib))
            peak_total = max(peak_total, int(tot or 0))
            vram.flush()
        except Exception as e:  # noqa: BLE001
            print(f"[run_tests] nvidia-smi failed: {e}", flush=True)
        time.sleep(5)
    wall = time.time() - t0
    codes = [p.returncode for p in procs]
    done = [r for r in todo if (out / f"{r['name']}.meta.json").exists()]
    audio_s = sum(jload(out / f"{r['name']}.meta.json")["duration_s"] for r in done)
    timing = {"tag": a.tag, "K": k, "runs_done": len(done), "runs_todo": len(todo), "exit_codes": codes,
              "wall_s": round(wall, 1), "audio_s": round(audio_s, 1),
              "audio_s_per_wall_s": round(audio_s / max(wall, 1e-6), 2),
              "peak_vram_mib_per_pid": peak_pid, "peak_total_gpu_mib": peak_total}
    hist = out / "logs" / "timing.jsonl"
    with open(hist, "a") as f:
        f.write(json.dumps(timing) + "\n")
    print(f"[run_tests] {json.dumps(timing)}", flush=True)
    sys.exit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
