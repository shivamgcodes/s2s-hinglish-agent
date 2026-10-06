"""Load a .cact with needle.Needle (system pinned, auto_date=False) and run the first N rows of a jsonl."""
import json, sys, time
import needle
w, data, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(data)][:n]
for r in rows:
    t = time.perf_counter()
    a = needle.Needle(tools=r["tools"], system=r.get("system"), weights=w, auto_date=False)
    init = time.perf_counter() - t; t = time.perf_counter()
    resp = a.complete(r["query"], 128)
    print(f"init {init:.2f}s infer {1000*(time.perf_counter()-t):.0f}ms | q={r['query'][:70]!r}\n  got={json.dumps(resp.get('function_calls'), ensure_ascii=False)[:200]}\n  ref={str(r.get('answers'))[:200]}")
    a.close()
