"""needle_router: the Needle v2 Hinglish action router (one copy; D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07).

    pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router"

    import json, needle_router
    from huggingface_hub import hf_hub_download
    w = hf_hub_download("shivamgupta/needle-hinglish-router-v2", "tuned_full.cact")
    rec = json.load(open(hf_hub_download("shivamgupta/needle-hinglish-router-v2", "examples/food_01.json")))
    for c in needle_router.route(rec["agent_type"], rec, "mujhe order cancel karna hai", weights=w):
        print(c["name"], c["resolved_id"], c["server_args"], c["ask"])     # cancel_order FD4124 {...} False

A call with ask=True must not be executed (ask the customer instead). Needle N1 (the earlier router, file
n1/tuned_full.cact in the same HF repo) is needle_router.n1_router. Layout (flat modules, as in the code of record):
  v2/       router_v2 (route, route_raw, resolve_calls, prepare_transcript, close), resolver_v2, tools_v2, targets, n1path
  n1/       router (N1), resolver, romanise (+ romanise_lexicon.json), tools (+ tools_json/), build_data (system_text)
  numconv/  numconv (spoken numbers -> digits)
Importing this package puts those three folders on sys.path (bare imports between the modules), and sets N1_DIR.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _d in ("n1", "numconv", "v2"):          # v2 ends up first on sys.path
    _p = _os.path.join(_HERE, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.environ.setdefault("N1_DIR", _os.path.join(_HERE, "n1"))

import router_v2  # noqa: E402  (the engine itself is imported lazily, on the first route)
import tools_v2  # noqa: E402

route = router_v2.route
route_raw = router_v2.route_raw
resolve_calls = router_v2.resolve_calls
prepare_transcript = router_v2.prepare_transcript
close = router_v2.close
AGENT_TYPES = tools_v2.AGENT_TYPES
HF_REPO = "shivamgupta/needle-hinglish-router-v2"
WEIGHTS_V2 = "tuned_full.cact"          # hf_hub_download(HF_REPO, WEIGHTS_V2)
WEIGHTS_N1 = "n1/tuned_full.cact"


def n1_router():
    """The N1 router module (n1/router.py; route(agent_type, record, transcript, weights=...))."""
    import router
    return router
