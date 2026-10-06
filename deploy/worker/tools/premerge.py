"""S2S serverless worker, DESIGN.md 6.4 option A: write PersonaPlex weights with the V3 LoRA already merged.

  <venv-pp>/bin/python worker/tools/premerge.py --out /root/premerged [--device cuda] [--pp-dir DIR] [--adapter DIR]

Loads model.safetensors exactly as engine.Engine does (moshi loaders.get_moshi_lm), merges the adapter with the same
merge_lora hook (bf16 ops; run on cuda so the arithmetic matches the live merge), and writes <out>/model.safetensors
plus copies of the Mimi, tokenizer and voices.tgz files and a MERGE_INFO.json. Upload <out> to a PRIVATE HF repo
(e.g. <user>/personaplex-7b-v3merged; check the NVIDIA PersonaPlex license before re-hosting merged weights), cache
THAT repo as the endpoint model, and run the worker with S2S_PP_REPO=<user>/personaplex-7b-v3merged S2S_ADAPTER=premerged.
Not run in the build stage (needs the GPU; about 16 GB VRAM + 32 GB RAM). Verify once in a GPU stage: a seeded mock
call (ws_feed, same wav, seed 1001) must give the same text tokens with the adapter and with the pre-merged weights.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W))
sys.path.insert(0, str(W / "server"))
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pp-dir", default=paths.PP_DIR or None, help="PersonaPlex snapshot dir (default S2S_PP_DIR)")
    ap.add_argument("--adapter", default=paths.ADAPTER)
    a = ap.parse_args()
    import torch
    from moshi.models import loaders
    from safetensors.torch import save_file
    from engine import pp_files_from_dir, resolve_adapter
    if not a.pp_dir:
        sys.exit("--pp-dir or S2S_PP_DIR is required")
    files = pp_files_from_dir(a.pp_dir)
    adapter = resolve_adapter(a.adapter)
    if not adapter:
        sys.exit("an adapter is required (S2S_ADAPTER / --adapter)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lm = loaders.get_moshi_lm(files["moshi"], device=a.device, cpu_offload=False)
    spec = importlib.util.spec_from_file_location("merge_lora", paths.MERGE_LORA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = mod.merge_into(lm, adapter)
    if info.get("ft_embed"):
        sys.exit("ft_embed adapters also patch the voice prompt loader at runtime; pre-merging them is not supported")
    # clone: the loader may share storage between copied depformer weights (safetensors refuses shared tensors)
    sd = {k: v.detach().to("cpu").contiguous().clone() for k, v in lm.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt", "s2s_premerged": "1"})
    for k in ("mimi", "tokenizer"):
        shutil.copy2(files[k], out / Path(files[k]).name)
    tgz = Path(a.pp_dir) / "voices.tgz"
    if tgz.exists():
        shutil.copy2(tgz, out / "voices.tgz")
    lora = Path(adapter) / "lora.safetensors" if Path(adapter).is_dir() else Path(adapter)
    meta = {"base": files["moshi"], "adapter": str(adapter), "merge": info, "device": a.device,
            "adapter_md5": hashlib.md5(lora.read_bytes()).hexdigest(), "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1), "n_tensors": len(sd)}
    (out / "MERGE_INFO.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
