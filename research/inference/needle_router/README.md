# Needle v2 router: inference example (= the HF card's quickstart)

The router code is the `needle_router` package (`packages/needle_router`). The weights and the example record come
from the HF repo [`shivamgupta/needle-hinglish-router-v2`](https://huggingface.co/shivamgupta/needle-hinglish-router-v2).
Since D-LEAN-HF (2026-10-07), that repo holds only weights, examples and the model card.

```bash
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router" huggingface_hub
python example.py                                         # food_01 record + tuned_full.cact from the HF repo
python example.py my_record.json "mujhe order cancel karna hai"
python example.py my_record.json "..." /path/to/tuned_full.cact     # or NEEDLE_V2_WEIGHTS=/path
```
Expected output (CPU):
```
model calls: [{"name": "cancel_order", "arguments": {"order_ref": "FD4124"}}]
{"name": "cancel_order", "resolved_id": "FD4124", "server_args": {"order_id": "FD4124"}, "ask": false, "ask_reasons": []}
```
From a monorepo checkout, `example.py` also runs without installing: it falls back to `../../../packages`.
