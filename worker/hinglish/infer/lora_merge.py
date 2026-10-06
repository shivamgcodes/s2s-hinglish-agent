"""S2S serverless copy (path edits only) of the hinglish repo's infer/lora_merge.py.
Merge a moshi-finetune PersonaPlex adapter (format personaplex_weight_lora_v1, written by the
patched trainer: finetune/pp_lora.py) into a PersonaPlex LMModel loaded with the stock PersonaPlex
loaders. Works in venv-pp (no moshi-finetune import). the personaplex tree is never modified.

Adapter keys: '<PP param name>.lora_A' [n, r, in], '<PP param name>.lora_B' [n, out, r]
(n = number of adapted per-step blocks of a stacked depformer weight, else 1), plus full embedding
tables under their PP names when the run used lora.ft_embed=true.
Merge: W[block i] += scaling * B[i] @ A[i], computed with the same bf16 ops as the training-time
parametrization, so the merged model equals the trained one.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

VOICE_CODES_DIR = str(Path(__file__).resolve().parent.parent / "trainer" / "voice_codes")   # was the pod1 path


def load_adapter(path):
    p = Path(path)
    f = p / "lora.safetensors" if p.is_dir() else p
    cfg = json.loads((f.parent / "config.json").read_text())
    assert cfg.get("format") == "personaplex_weight_lora_v1", cfg.get("format")
    return load_file(str(f)), cfg


@torch.no_grad()
def merge_into(lm, sd: dict, cfg: dict) -> dict:
    s = float(cfg["lora_scaling"])
    params = dict(lm.named_parameters())
    n_lora, n_emb = 0, 0
    lora_keys = [k for k in sd if k.endswith(".lora_A")]
    for k in sorted(lora_keys):
        base = k[: -len(".lora_A")]
        W = params[base]  # KeyError = adapter/model name mismatch
        A = sd[k].to(W.device, W.dtype)
        B = sd[base + ".lora_B"].to(W.device, W.dtype)
        na, out_f = A.shape[0], B.shape[1]
        assert W.shape[0] % out_f == 0 and A.shape[2] == W.shape[1], (base, W.shape, A.shape, B.shape)
        d = (torch.bmm(B, A) * s).to(W.dtype)
        Wv = W.data.view(-1, out_f, W.shape[1])
        Wv[:na] = Wv[:na] + d
        n_lora += 1
    for k in sd:
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        P = params[k]
        assert P.shape == sd[k].shape, (k, P.shape, sd[k].shape)
        P.data.copy_(sd[k].to(P.device, P.dtype))
        n_emb += 1
    assert n_lora == len(lora_keys) and 2 * n_lora + n_emb == len(sd), (n_lora, n_emb, len(sd))
    if "n_lora_weights" in cfg:
        assert n_lora == cfg["n_lora_weights"], (n_lora, cfg["n_lora_weights"])
    return {"n_lora": n_lora, "n_emb": n_emb, "ft_embed": bool(cfg.get("ft_embed")), "scaling": s}


@torch.no_grad()
def voice_embeddings_from_codes(lm, voice: str, voice_codes_dir: str = VOICE_CODES_DIR) -> torch.Tensor:
    """Recompute a voice prompt's replay embeddings [51, 1, 1, dim] from its recovered input columns
    with the model's current (possibly fine-tuned) embedding tables. On the base model this equals
    the shipped .pt embeddings bit for bit (checked in trainer/verify_sample.py)."""
    v = Path(voice).stem
    cols = torch.load(Path(voice_codes_dir) / f"{v}.columns.pt")["columns"].to(lm.device)  # [17, 51]
    emb = lm.embed_codes(cols[None])  # [1, 51, dim]
    return emb[0][:, None, None, :].contiguous()
