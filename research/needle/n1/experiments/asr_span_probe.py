"""Probe (assembly decision A2): does base Needle 3 suppress a call whose required arg is spoken as
number words (ASR rendering b) rather than digits? Same turns in (a) digits and (b) ASR form.
The engine's span check is native (not in the Python package), so this is measured, not read."""
import json, sys, time
import os
from pathlib import Path  # monorepo shim: N1 modules from packages/needle_router/n1 ($N1_DIR)
REPO = Path(__file__).resolve().parents[4]
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (was the laptop needle/ dir)
sys.path.insert(0, N1_PKG)
import needle
from tools import tools_for
SYSTEM = ("date: 2026-10-04 Sun 10:00; user: customer phone 98000 01026, Order FD1226 from Wow! Momo: "
          "Steamed Momo x2, Rs 620, status out for delivery, delivery to B-42, Sector 15, Noida")
PAIRS = [
    ("Mera naya number 98000 11085 hai, ise update kar dijiye.",
     "mera naya number nine eight zero zero zero one one zero eight five hai ise update kar dijiye"),
    ("Naya address hai Office 405, DLF Cyber City, Phase 2, Gurgaon.",
     "naya address hai office four hundred five DLF cyber city phase two gurgaon"),
    ("please cancel my order FD1904, mujhe ab nahi chahiye",
     "please cancel my order efdi one nine zero four mujhe ab nahi chahiye"),
    ("haan ji the new contact number is 98000 21171, please isko update karo.",
     "haan ji the new contact number is nine eight zero zero zero two one one seven one please isko update karo"),
    ("address flat 502, Tower C, Sector 45, Gurgaon hai, please change kar do",
     "address flat five hundred two tower C sector forty five gurgaon hai please change kar do"),
]
out = []
ag = needle.Needle(tools=tools_for("food_delivery_support"), system=SYSTEM, auto_date=False)
for a, b in PAIRS:
    for rend, q in (("a", a), ("b", b)):
        ag.reset(); t0 = time.time()
        r = ag.complete(q)
        out.append({"rendering": rend, "q": q, "function_calls": r.get("function_calls"),
                    "suppressed_calls": r.get("suppressed_calls"), "reasoning": r.get("reasoning"),
                    "ms": round(1000 * (time.time() - t0))})
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(HERE, "experiments", "asr_span_probe.json"), "w"),
          indent=1, ensure_ascii=False)
