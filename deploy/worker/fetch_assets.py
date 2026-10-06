"""S2S serverless worker: put the small weights where paths.py expects them (D-BAKE; D-ASSETS-PUBLIC 2026-10-07).

  $S2S_ASSETS/v4_adapter/{config.json,lora.safetensors}     -> paths.ADAPTER (default)
  $S2S_ASSETS/tuned_full.cact                               -> paths.NEEDLE_WEIGHTS (Needle N1, S2S_ROUTER=n1)
  $S2S_ASSETS/needle_v2/tuned_full.cact                     -> paths.NEEDLE_V2_WEIGHTS (Needle v2, default router)
  ~/.cache/cactus-needle/v3/3.0.2/libneedle.so              -> the cactus-needle package's own cache path

assets_manifest.json lists every file: source repo + pinned revision + path, size and md5. Sources, in order:
  1. already in place (the image bakes them: the Dockerfile runs this script with S2S_ASSETS_FROM_HF=1)
  2. RunPod network volume: $S2S_VOLUME_ASSETS (default /runpod-volume/s2s-assets), files at their volume_path
  3. Hugging Face, only when S2S_ASSETS_FROM_HF=1: each entry from its own repo at its pinned revision
     (shivamgupta/personaplex-hinglish-v4-lora, shivamgupta/needle-hinglish-router-v2, Cactus-Compute/needle3), with
     $HF_TOKEN if set (needed while the two model repos are private). ALLOWLIST: only the manifest's (repo, path)
     pairs are ever requested, a path under a forbidden prefix (V4_A/, V4_A2/, V3_A/, checkpoints/, ...) is refused,
     and after the downloads the HF cache is listed and the run fails if it holds any other file. The list of what
     was fetched (repo, revision, path, bytes) is printed and written to $S2S_ASSETS/fetched.json.
An entry with "extract" is a zip (the Needle engine wheel): its download_md5 is checked, then the member is extracted.

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
import zipfile
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
        src = VOLUME_ASSETS / e["volume_path"]
        if ok(src, e):
            d = dest(e)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d)
            got += 1
    return got


def name(e):
    return f"{e['repo']}:{e['path']}"


def check_allowed(e):
    allowed = {(f["repo"], f["path"]) for f in MANIFEST["files"]}
    bad_prefix = [p for p in MANIFEST.get("forbidden_prefixes", []) if e["path"].startswith(p)]
    if (e["repo"], e["path"]) not in allowed or bad_prefix or ".." in e["path"]:
        raise RuntimeError(f"refusing to fetch {name(e)}: not on the manifest allowlist / forbidden prefix {bad_prefix}")


def cache_listing(cache):
    """Every file the HF cache holds: (repo, path) from snapshots/<rev>/<path>, + total blob bytes."""
    files, total = [], 0
    for repo_dir in cache.glob("models--*"):
        repo = repo_dir.name[len("models--"):].replace("--", "/")
        for snap in (repo_dir / "snapshots").glob("*"):
            for f in snap.rglob("*"):
                if f.is_file() or f.is_symlink():
                    files.append((repo, f.relative_to(snap).as_posix()))
        total += sum(b.stat().st_size for b in (repo_dir / "blobs").glob("*") if b.is_file())
    return files, total


def from_hf(entries):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    os.environ["HF_HUB_OFFLINE"] = "0"           # the image sets offline mode for everything else
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download
    cache = paths.TMP / "hf-assets"
    shutil.rmtree(cache, ignore_errors=True)
    fetched = []
    for e in entries:
        check_allowed(e)
        t0 = time.time()
        p = Path(hf_hub_download(e["repo"], e["path"], revision=e["revision"], token=token, cache_dir=str(cache)))
        n = p.stat().st_size
        d = dest(e)
        d.parent.mkdir(parents=True, exist_ok=True)
        if e.get("extract"):
            if n != e["download_size"] or md5(p) != e["download_md5"]:
                raise RuntimeError(f"{name(e)}: download size/md5 mismatch")
            with zipfile.ZipFile(p) as z:
                d.write_bytes(z.read(e["extract"]))
        else:
            shutil.copyfile(p, d)
        fetched.append({"repo": e["repo"], "revision": e["revision"], "path": e["path"], "bytes": n})
        print(f"[assets] hf {name(e)}@{e['revision'][:8]} -> {d} ({n / 2**20:.1f} MiB, {time.time() - t0:.1f} s)", flush=True)
    files, total = cache_listing(cache)
    allowed = {(f["repo"], f["path"]) for f in MANIFEST["files"]}
    extra = sorted(f"{r}:{q}" for r, q in files if (r, q) not in allowed)
    shutil.rmtree(cache, ignore_errors=True)
    report = {"fetched": fetched, "files_in_hf_cache": sorted(f"{r}:{q}" for r, q in files), "total_bytes": total}
    paths.ASSETS.mkdir(parents=True, exist_ok=True)
    (paths.ASSETS / "fetched.json").write_text(json.dumps(report, indent=1) + "\n")
    print("[assets] FETCHED " + json.dumps(report), flush=True)
    print(f"[assets] total fetched: {total} bytes ({total / 2**20:.1f} MiB) in {len(files)} file(s)", flush=True)
    if extra:
        raise RuntimeError("ALLOWLIST VIOLATION: the HF cache holds files outside the manifest: " + ", ".join(extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what is in place")
    a = ap.parse_args()
    t0 = time.time()
    req = [e for e in MANIFEST["files"] if e.get("required", True)]
    opt = [e for e in MANIFEST["files"] if not e.get("required", True)]
    todo = missing(req + opt)
    if a.check or not todo:
        bad = [name(e) for e in missing(req)]
        print(json.dumps({"assets": str(paths.ASSETS), "missing_required": bad,
                          "missing_optional": [name(e) for e in missing(opt)]}), flush=True)
        sys.exit(2 if bad else 0)
    src = "in_place"
    if VOLUME_ASSETS.is_dir() and from_volume(todo):
        src = "network_volume"
        todo = missing(req + opt)
    if todo and os.environ.get("S2S_ASSETS_FROM_HF") == "1":
        try:
            from_hf(todo)
            src = "hf" if src == "in_place" else src + "+hf"
        except Exception as ex:  # noqa: BLE001
            print(f"[assets] HF download failed: {type(ex).__name__}: {str(ex)[:400]}", flush=True)
            sys.exit(2)
    bad = missing(req)
    if bad:
        print("[assets] MISSING or md5 mismatch: " + ", ".join(name(e) for e in bad)
              + f" (looked in {paths.ASSETS}, {VOLUME_ASSETS}; HF fetch {'on' if os.environ.get('S2S_ASSETS_FROM_HF') == '1' else 'off (S2S_ASSETS_FROM_HF=1)'})",
              flush=True)
        sys.exit(2)
    for e in missing(opt):
        print(f"[assets] optional {name(e)} not available; {e.get('note', '')}", flush=True)
    print(f"[assets] ready ({src}) in {time.time() - t0:.1f} s: adapter {paths.ASSETS / 'v4_adapter'}, "
          f"cact n1 {paths.ASSETS / 'tuned_full.cact'}, v2 {paths.ASSETS / 'needle_v2' / 'tuned_full.cact'}", flush=True)


if __name__ == "__main__":
    main()
