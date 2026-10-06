# personaplex_lora (packages/personaplex_lora)

This is the only copy of the code that applies the V4 Hinglish LoRA
([`shivamgupta/personaplex-hinglish-v4-lora`](https://huggingface.co/shivamgupta/personaplex-hinglish-v4-lora)) to
PersonaPlex 7B ([`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1), gated). It also holds
the record/pairing → role prompt + voice format used in training and in the live demo.

```bash
# 1. PersonaPlex (torch + its `moshi` package) per github.com/NVIDIA/personaplex; 2. this package:
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora" safetensors huggingface_hub
```
```python
import os, personaplex_lora as ppl
from huggingface_hub import hf_hub_download
adapter_dir = os.path.dirname(hf_hub_download(ppl.HF_REPO, "config.json"))
hf_hub_download(ppl.HF_REPO, "lora.safetensors")
info = ppl.merge_adapter(lm, adapter_dir)        # lm = moshi.models.loaders.get_moshi_lm(<base model.safetensors>, device="cuda")
prompt, voice = ppl.build_role_prompt(record, "g1"), ppl.voice_for("g1")
```

The full offline run is `research/inference/personaplex_lora/run_offline.py`. It downloads the base files and the
adapter, merges them, and runs a customer wav with a role prompt and a voice prompt.

| file | what | used by |
|---|---|---|
| `role_prompt.py` | `ascii_prompt`, `build_role_prompt`, `voice_for`, `PAIRINGS`, `AGENT_GENDER`, `VOICE` | `deploy/common/session.py` (worker + Space), research eval harness |
| `infer/lora_merge.py` | `load_adapter`, `merge_into` | everything below |
| `trainer/merge_lora.py` + `trainer/voice_codes/` | `merge_into(lm, path)` hook, incl. the voice-embedding fix for `ft_embed` adapters | live worker (`paths.MERGE_LORA`), research eval harness |
