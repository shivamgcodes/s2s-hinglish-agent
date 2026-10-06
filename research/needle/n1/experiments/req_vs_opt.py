"""Quick check (decision D3): order_ref required vs optional on base Needle 3, pinned system text."""
import copy, json, sys, time
import os
from pathlib import Path  # monorepo shim: N1 modules from packages/needle_router/n1 ($N1_DIR)
REPO = Path(__file__).resolve().parents[4]
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (was the laptop needle/ dir)
sys.path.insert(0, N1_PKG)
import needle
from tools import tools_for
SYSTEM = ("date: 2026-10-04 Sun 10:00; user: orders: FD1565 Behrouz Biryani, Kashmiri Dum Biryani x1, "
          "Galouti Kebab x1, out for delivery; older order FD4065 Wow! Momo, delivered 15 October")
TURNS = [
    "Mera naya number 98000 11085 hai, ise update kar dijiye.",
    "Naya address hai Office 405, DLF Cyber City, Phase 2, Gurgaon.",
    "Mujhe ab ye order nahi chahiye. Please mera order FD1565 cancel kar do.",
    "Please Behrouz wala order cancel kar do.",
    "Rider ko bolo bell mat bajana, call kar lena.",
]
def variant(req):
    ts = copy.deepcopy(tools_for("food_delivery_support"))
    if req:
        for t in ts:
            t["parameters"]["required"] = ["order_ref"] + t["parameters"].get("required", [])
    return ts
out = []
for req in (True, False):
    ag = needle.Needle(tools=variant(req), system=SYSTEM, auto_date=False)
    for q in TURNS:
        ag.reset(); t0 = time.time()
        r = ag.complete(q)
        out.append({"order_ref_required": req, "q": q, "function_calls": r.get("function_calls"),
                    "suppressed_calls": r.get("suppressed_calls"), "reasoning": r.get("reasoning"),
                    "ms": round(1000*(time.time()-t0))})
        print(json.dumps(out[-1], ensure_ascii=False))
    ag.close() if hasattr(ag, "close") else None
json.dump(out, open(os.path.join(HERE, "experiments", "req_vs_opt.json"), "w"), indent=1, ensure_ascii=False)
