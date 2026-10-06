"""S2S serverless copy (path edits only) of the hinglish repo's trainer/merge_lora.py.
Hook used by the hinglish tests/driver.py (merge_into(lm, path)): merges an adapter written by
the patched moshi-finetune trainer (format personaplex_weight_lora_v1). Implementation: infer/lora_merge.py.
Config-B (ft_embed) adapters: the shipped voice .pt replay embeddings were made with the base tables, so
merge_into also wraps LMGen.load_voice_prompt_embeddings at runtime (no file under the personaplex tree
is changed): after the normal load, NATF2/NATM1 embeddings are recomputed from the recovered codes with
the merged tables (bit-exact with the .pt on the base model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infer"))   # <hinglish>/infer (was the pod1 path)
import lora_merge  # noqa: E402

voice_embeddings_from_codes = lora_merge.voice_embeddings_from_codes


def _patch_lmgen_voice_loader():
    from moshi.models.lm import LMGen
    if getattr(LMGen.load_voice_prompt_embeddings, "_pp_ft_embed_patch", False):
        return
    orig = LMGen.load_voice_prompt_embeddings

    def load_voice_prompt_embeddings(self, path):
        orig(self, path)
        if getattr(self.lm_model, "_pp_adapter_ft_embed", False) and Path(path).stem in ("NATF2", "NATM1"):
            self.voice_prompt_embeddings = voice_embeddings_from_codes(self.lm_model, Path(path).stem)
            print(f"[merge_lora] voice prompt {Path(path).stem}: embeddings recomputed with merged tables", flush=True)

    load_voice_prompt_embeddings._pp_ft_embed_patch = True
    LMGen.load_voice_prompt_embeddings = load_voice_prompt_embeddings


def merge_into(lm, path):
    sd, cfg = lora_merge.load_adapter(path)
    info = lora_merge.merge_into(lm, sd, cfg)
    lm._pp_adapter_ft_embed = info["ft_embed"]
    if info["ft_embed"]:
        _patch_lmgen_voice_loader()
    return info
