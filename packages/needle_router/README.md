# needle_router (packages/needle_router)

The Needle v2 Hinglish action router: a customer-support record plus a Hinglish/English customer transcript
(raw ASR text is fine) go in, and resolved write calls come out (for example `cancel_order` on `FD4124`). When a
reference cannot be resolved, the call is marked `ask=True`. Weights are in the HF model repo
[`shivamgupta/needle-hinglish-router-v2`](https://huggingface.co/shivamgupta/needle-hinglish-router-v2):
`tuned_full.cact` is v2 and `n1/tuned_full.cact` is N1. Runs on CPU.

```bash
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router" huggingface_hub
```
```python
import json, needle_router
from huggingface_hub import hf_hub_download
w = hf_hub_download(needle_router.HF_REPO, needle_router.WEIGHTS_V2)
rec = json.load(open(hf_hub_download(needle_router.HF_REPO, "examples/food_01.json")))
for c in needle_router.route(rec["agent_type"], rec, "mujhe order cancel karna hai", weights=w):
    print(c["name"], c["resolved_id"], c["server_args"], c["ask"])
```
The full example is `research/inference/needle_router/example.py`. This folder is the only copy of the router code.
The live worker (`deploy/worker/router/router_core.py`) and `research/needle/` use the same files in place. The
modules are flat, as in the code of record. Importing `needle_router` puts `v2/`, `n1/` and `numconv/` on `sys.path`.
The cactus-needle package sends anonymous usage counts; set `NEEDLE_TELEMETRY=0` to turn that off.
