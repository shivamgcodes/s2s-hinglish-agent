import json, os, sys, time, needle
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir
w, n, mx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = [json.loads(l) for l in open(f"{N2_ROOT}/data/rows/val.jsonl")][n]
t = time.perf_counter(); a = needle.Needle(tools=r["tools"], system=r["system"], weights=w, auto_date=False); print("init", round(time.perf_counter()-t, 2), flush=True)
for k in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
    a.reset(); t = time.perf_counter(); resp = a.complete(r["query"], mx)
    print("infer", round(time.perf_counter()-t, 2), json.dumps(resp, ensure_ascii=False)[:800], flush=True)
a.close()
