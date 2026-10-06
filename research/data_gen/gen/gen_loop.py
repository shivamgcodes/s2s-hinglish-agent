"""One vLLM load: full generation of a variant, then --only-missing top-up rounds (new seed bases) until all
call_ids pass, MAX_ROUNDS top-ups were run, or TOPUP_MIN minutes of top-up wall time have elapsed (no new round is
started after that). Equivalent to `generate.py --variant V` followed by `generate.py --variant V --only-missing
--seed-base N` runs, without reloading Gemma.
  cd /workspace/hinglish/gen && flock ../gpu.lock /workspace/venv-vllm/bin/python gen_loop.py V1
Writes <out>/gen_rounds.json (per round: seed_base, ids submitted, passed, gen_log line range, summary).
D2 / V4: inputs per variant via generate.inputs_for (V4 -> scenario_creation_v2.json + data/V4/records.json);
optional env GEN_SCENARIOS / GEN_RECORDS / GEN_OUT override them (single-variant runs); unset = old behaviour."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate
from common import DATA, LLM

VARIANTS = sys.argv[1:]  # one or more variants, one Gemma load (V3 data task: "V3 CONTROL")
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "6"))
TOPUP_MIN = float(os.environ.get("TOPUP_MIN", "90"))
SEEDS = [0] + [101 * k for k in range(1, MAX_ROUNDS + 1)]
inputs = {V: generate.inputs_for(V, os.environ.get("GEN_SCENARIOS"), os.environ.get("GEN_RECORDS")) for V in VARIANTS}
llm = LLM(gpu_util=float(os.environ.get("GEN_GPU_UTIL", "0.94")))
for V in VARIANTS:
    scen, records = inputs[V]
    out = os.environ.get("GEN_OUT") or f"{DATA}/{V}"
    os.makedirs(out, exist_ok=True)
    ids_all = list(generate.default_ids(V, scen))  # V1: == all_call_ids(scen); V3: minus dropped ids; CONTROL: 100 held-out
    rounds = []

    def have():
        p = f"{out}/calls.jsonl"
        return {json.loads(x)["call_id"] for x in open(p, encoding="utf-8")} if os.path.exists(p) else set()

    def nlog():
        p = f"{out}/gen_log.jsonl"
        return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0

    t_top = None
    for r, sb in enumerate(SEEDS):
        h = have()
        todo = [c for c in ids_all if c not in h]
        if not todo:
            break
        if r > 0:
            if t_top is None:
                t_top = time.time()
            elif (time.time() - t_top) / 60 > TOPUP_MIN:
                print(f"[loop {V}] top-up time cap {TOPUP_MIN} min reached before round {r}", flush=True)
                break
        l0, t0 = nlog(), time.time()
        print(f"[loop {V}] round {r} seed_base {sb}: {len(todo)} calls", flush=True)
        summ = generate.run_job(llm, V, todo, out, scen, records, judge=True, seed_base=sb,
                                keep_existing=list(ids_all) if r > 0 or h else None)
        rounds.append({"round": r, "seed_base": sb, "submitted": len(todo), "passed": summ["final_pass"],
                       "log_lines": [l0, nlog()], "wall_min": round((time.time() - t0) / 60, 1), "summary": summ})
        json.dump(rounds, open(f"{out}/gen_rounds.json", "w"), indent=1)
    missing = [c for c in ids_all if c not in have()]
    json.dump(missing, open(f"{out}/gen_missing.json", "w"))
    print(f"[loop {V}] final: {len(have())}/{len(ids_all)} calls; missing {missing}; llm load {llm.load_s:.0f}s gen {llm.gen_s:.0f}s", flush=True)
print("LOOP DONE", flush=True)
