"""S2S serverless worker: fetch the small private assets at worker start (DECISIONS 2026-10-06 "WEIGHTS").

The image is built by RunPod from a GitHub repo, which must stay code-only (GitHub rejects files > 100 MB; no LFS).
So the V4 LoRA adapter (370 MB), the Needle router weights (63 MB) and the Needle native lib are NOT baked; this
script puts them where paths.py expects them, before the engine loads:

  $S2S_ASSETS/v4_adapter/{config.json,lora.safetensors}     -> paths.ADAPTER (default)
  $S2S_ASSETS/tuned_full.cact                               -> paths.NEEDLE_WEIGHTS (Needle N1, S2S_ROUTER=n1)
  $S2S_ASSETS/needle_v2/tuned_full.cact                     -> paths.NEEDLE_V2_WEIGHTS (Needle v2, default router)
  ~/.cache/cactus-needle/v3/3.0.2/libneedle.so              -> the cactus-needle package's own cache path

Sources, in order (the first that gives every file with the right md5 wins; assets_manifest.json has the md5s):
  1. already in place (a previous start of this container, or a custom image that bakes them)
  2. RunPod network volume: $S2S_VOLUME_ASSETS (default /runpod-volume/s2s-assets), same layout as the HF repo
  3. private HF model repo $S2S_ASSETS_REPO (e.g. <hf_user>/s2s-v4-assets), revision $S2S_ASSETS_REVISION
     (default main), token $HF_TOKEN (the same read token that reads the gated PersonaPlex repo)
The Needle lib is optional here: if it is missing everywhere, the cactus-needle package downloads it itself at the
first route call (it did so on pod1), and the Docker build also pre-fetches it.

  <venv-asr>/bin/python worker/fetch_assets.py [--check]     (worker_server.py runs it in its load task, so a
                                                             failure shows as /ping 500 with the reason)
Exit 0 = all required assets in place; 2 = missing/bad (the reason is printed). Never prints the token.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

MANIFEST = json.loads((HERE / "assets_manifest.json").read_text())
VOLUME_ASSETS = Path(os.environ.get("S2S_VOLUME_ASSETS") or str(paths.RUNPOD_VOLUME / "s2s-assets"))
NEEDLE_LIB_DIR = Path(os.environ.get("S2S_NEEDLE_LIB_DIR")
                      or str(Path.home() / ".cache" / "cactus-needle" / "v3" / "3.0.2"))


def dest(entry) -> Path:
    """Where a manifest entry must end up in the container."""
    if entry["dest"].startswith("NEEDLE_LIB_DIR/"):
        return NEEDLE_LIB_DIR / entry["dest"].split("/", 1)[1]
    return paths.ASSETS / entry["dest"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ok(p: Path, entry) -> bool:
    return p.is_file() and p.stat().st_size == entry["size"] and md5(p) == entry["md5"]


def missing(entries):
    return [e for e in entries if not ok(dest(e), e)]


def from_volume(entries):
    got = 0
    for e in entries:
        src = VOLUME_ASSETS / e["repo_path"]
        if ok(src, e):
            d = dest(e)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d)
            got += 1
    return got


def from_hf(entries):
    repo = os.environ.get("S2S_ASSETS_REPO", "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not repo:
        raise RuntimeError("S2S_ASSETS_REPO is not set (and the assets are not on the network volume)")
    if not token:
        raise RuntimeError("HF_TOKEN is not set (needed to read the private assets repo " + repo + ")")
    os.environ["HF_HUB_OFFLINE"] = "0"           # the image sets offline mode for everything else
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download
    cache = paths.TMP / "hf-assets"
    rev = os.environ.get("S2S_ASSETS_REVISION", "").strip() or None
    for e in entries:
        t0 = time.time()
        p = Path(hf_hub_download(repo, e["repo_path"], revision=rev, token=token, cache_dir=str(cache)))
        d = dest(e)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, d)
        print(f"[assets] hf {repo}:{e['repo_path']} -> {d} ({e['size'] / 2**20:.0f} MiB, {time.time() - t0:.1f} s)",
              flush=True)
    shutil.rmtree(cache, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what is in place")
    a = ap.parse_args()
    t0 = time.time()
    req = [e for e in MANIFEST["files"] if e.get("required", True)]
    opt = [e for e in MANIFEST["files"] if not e.get("required", True)]
    todo = missing(req + opt)
    if a.check or not todo:
        bad = [e["repo_path"] for e in missing(req)]
        print(json.dumps({"assets": str(paths.ASSETS), "missing_required": bad,
                          "missing_optional": [e["repo_path"] for e in missing(opt)]}), flush=True)
        sys.exit(2 if bad else 0)
    src = "in_place"
    if VOLUME_ASSETS.is_dir() and from_volume(todo):
        src = "network_volume"
        todo = missing(req + opt)
    if any(e.get("required", True) for e in todo) or (todo and os.environ.get("S2S_ASSETS_REPO")):
        try:
            from_hf(todo)
            src = "hf_repo" if src == "in_place" else src + "+hf_repo"
        except Exception as ex:  # noqa: BLE001
            print(f"[assets] HF download failed: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    bad = missing(req)
    if bad:
        print("[assets] MISSING or md5 mismatch: " + ", ".join(e["repo_path"] for e in bad)
              + f" (looked in {paths.ASSETS}, {VOLUME_ASSETS}, HF repo {os.environ.get('S2S_ASSETS_REPO') or '-'})",
              flush=True)
        sys.exit(2)
    for e in missing(opt):
        print(f"[assets] optional {e['repo_path']} not available; {e.get('note', '')}", flush=True)
    print(f"[assets] ready ({src}) in {time.time() - t0:.1f} s: adapter {paths.ASSETS / 'v4_adapter'}, "
          f"cact n1 {paths.ASSETS / 'tuned_full.cact'}, v2 {paths.ASSETS / 'needle_v2' / 'tuned_full.cact'}", flush=True)


if __name__ == "__main__":
    main()
