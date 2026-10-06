"""Route one customer transcript to resolved write calls with the Needle v2 Hinglish router (CPU).

    pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router" huggingface_hub
    python example.py                       # the food_01 example (record + weights from the HF repo)
    python example.py my_record.json "mujhe order cancel karna hai"
    python example.py my_record.json "..." path/to/tuned_full.cact          # or NEEDLE_V2_WEIGHTS=path

D-LEAN-HF (2026-10-07): this is the one copy of the HF card's quickstart (was example.py in the HF repo); the router
code is the `needle_router` package (packages/needle_router), the weights + example record come from the HF repo
shivamgupta/needle-hinglish-router-v2 via hf_hub_download.

router_v2 pipeline: transcript -> numconv (spoken numbers -> digits) -> romanise (Devanagari -> Roman Hinglish)
-> Needle (tools of the record's agent type, system = pinned date + the record's information + customer phone)
-> function calls -> resolver_v2 (every reference resolved to an exact record value, or ask=True).
A call with ask=True must not be executed: the agent has to ask the customer (see ask_reasons).
"""
import json
import os
import sys

try:
    import needle_router  # pip-installed (see above)
except ImportError:       # running from a monorepo checkout without installing: use packages/ in place
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "packages"))
    import needle_router
from huggingface_hub import hf_hub_download  # noqa: E402

router_v2 = needle_router.router_v2
WEIGHTS = (sys.argv[3] if len(sys.argv) > 3 else
           os.environ.get("NEEDLE_V2_WEIGHTS") or hf_hub_download(needle_router.HF_REPO, needle_router.WEIGHTS_V2))

record_path = sys.argv[1] if len(sys.argv) > 1 else hf_hub_download(needle_router.HF_REPO, "examples/food_01.json")
transcript = sys.argv[2] if len(sys.argv) > 2 else \
    "mujhe order cancel karna hai. order cancel karna hai. I want to cancel the order"

record = json.load(open(record_path, encoding="utf-8"))
agent_type = record["agent_type"]          # one of needle_router.AGENT_TYPES

resp = router_v2.route_raw(agent_type, record, transcript, weights=WEIGHTS)
calls = router_v2.resolve_calls(resp.get("function_calls") or [], record, agent_type)

print("needle text:", router_v2.prepare_transcript(transcript))
print("model calls:", json.dumps(resp.get("function_calls"), ensure_ascii=False))
for c in calls:
    print(json.dumps({k: c.get(k) for k in ("name", "resolved_id", "server_args", "ask", "ask_reasons")},
                     ensure_ascii=False))
router_v2.close()
