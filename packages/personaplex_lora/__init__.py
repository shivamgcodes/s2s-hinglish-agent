"""personaplex_lora: PersonaPlex V4 Hinglish LoRA merge + the record/pairing format (one copy; D-SINGLE-SOURCE /
D-LEAN-HF 2026-10-07).

    pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora"
    # torch + the PersonaPlex `moshi` package: install PersonaPlex first (github.com/NVIDIA/personaplex)

    import json, personaplex_lora as ppl
    from huggingface_hub import hf_hub_download
    cfg = hf_hub_download("shivamgupta/personaplex-hinglish-v4-lora", "config.json")
    hf_hub_download("shivamgupta/personaplex-hinglish-v4-lora", "lora.safetensors")
    info = ppl.merge_adapter(lm, os.path.dirname(cfg))          # lm = moshi loaders.get_moshi_lm(...)
    prompt = ppl.build_role_prompt(record, "g1"); voice = ppl.voice_for("g1")     # "NATF2.pt"

  role_prompt.py       ascii_prompt, build_role_prompt, voice_for, PAIRINGS, AGENT_GENDER, VOICE (stdlib only)
  infer/lora_merge.py  load_adapter, merge_into (W += s * B @ A, bf16, same ops as training)   (torch, safetensors)
  trainer/merge_lora.py merge_into(lm, path): the hook the live worker and the eval harness use (+ the voice-embedding
                       fix for ft_embed adapters, from trainer/voice_codes/)
The full offline example is research/inference/personaplex_lora/run_offline.py. torch is imported only when a merge
function is called.
"""
from .role_prompt import AGENT_GENDER, PAIRINGS, VOICE, ascii_prompt, build_role_prompt, voice_for  # noqa: F401

HF_REPO = "shivamgupta/personaplex-hinglish-v4-lora"
BASE_REPO = "nvidia/personaplex-7b-v1"
BASE_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"   # the base snapshot the adapter was trained on


def load_adapter(path):
    """(state_dict, config) from a dir with lora.safetensors + config.json (or the .safetensors path)."""
    from .infer import lora_merge
    return lora_merge.load_adapter(path)


def merge_adapter(lm, path):
    """Merge the adapter at `path` into a loaded PersonaPlex LMModel in memory (the live worker's hook). Returns info."""
    from .trainer import merge_lora
    return merge_lora.merge_into(lm, path)
