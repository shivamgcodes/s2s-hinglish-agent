"""S2S serverless worker: find the PersonaPlex files and the voice prompts (DESIGN.md section 6.3).

  <venv-pp>/bin/python worker/resolve_models.py        # prints the result JSON; exit 2 if nothing is found

Called in process by worker_server.py at the start of its load task (so a failure shows as /ping 500, DESIGN
Amendment W1), and runnable alone for diagnosis. Order:
  1. S2S_PP_DIR                       explicit dir (network volume, pre-merged snapshot, local snapshot on runpod2)
  2. RunPod cached model              $S2S_MODEL_CACHE/models--<org>--<name>/snapshots/<refs/main | newest>/  ([CACHE],
                                      default /runpod-volume/huggingface-cache/hub, [HF-MODELS] resolve_snapshot_path)
  3. network-volume HF cache          $S2S_VOLUME_HF/models--<org>--<name>/snapshots/...   ([VOLCACHE], /runpod-volume/hf/hub)
  4. download (debug only)            S2S_ALLOW_DOWNLOAD=1 + HF_TOKEN -> hf_hub_download into $S2S_DOWNLOAD_DIR (WARNING:
                                      16 GB per cold start, billed)
Repo = S2S_PP_REPO (default nvidia/personaplex-7b-v1; a private pre-merged repo for DESIGN 6.4 option A).
Voices: <dir>/voices/NATF2.pt if present, else voices.tgz is extracted to $S2S_TMP/voices (the cache mount is treated as
read-only; DEP1's _get_voice_prompt_dir extracted next to the blob). Writes S2S_PP_DIR / S2S_VOICES to $S2S_MODELS_ENV
and into os.environ. Never prints or logs a token.
"""
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

log = logging.getLogger("s2s.resolve")
NEEDED = ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model")
VOICE_CHECK = "NATF2.pt"


class NotFound(Exception):
    pass


def _complete(d):
    d = Path(d)
    if not d.is_dir():
        return False, [f"{d} is not a dir"]
    miss = [f for f in NEEDED if not (d / f).exists()]
    if not (d / "voices" / VOICE_CHECK).exists() and not (d / "voices.tgz").exists():
        miss.append("voices/ or voices.tgz")
    return not miss, miss


def snapshot_dir(hub_dir, repo):
    """<hub>/models--org--name/snapshots/<refs/main or newest> (the HF cache layout), or None."""
    base = Path(hub_dir) / ("models--" + repo.replace("/", "--"))
    snaps = base / "snapshots"
    if not snaps.is_dir():
        return None
    ref = base / "refs" / "main"
    if ref.is_file():
        d = snaps / ref.read_text().strip()
        if d.is_dir():
            return d
    cands = sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _download(repo):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if os.environ.get("S2S_ALLOW_DOWNLOAD") != "1" or not token:
        raise NotFound("download disabled (needs S2S_ALLOW_DOWNLOAD=1 and HF_TOKEN)")
    log.warning("WARNING: downloading %s at runtime into %s (slow, billed; debug only)", repo, paths.DOWNLOAD_DIR)
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import hf_hub_download
    got = None
    for f in NEEDED + ("voices.tgz",):
        got = hf_hub_download(repo, f, cache_dir=str(paths.DOWNLOAD_DIR), token=token)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return Path(got).parent


def find_pp_dir(repo=None):
    """-> (dir, source); raises NotFound with every reason tried."""
    repo = repo or paths.PP_REPO
    tried = []
    if paths.PP_DIR:
        ok, miss = _complete(paths.PP_DIR)
        if ok:
            return Path(paths.PP_DIR), "S2S_PP_DIR"
        tried.append(f"S2S_PP_DIR={paths.PP_DIR}: missing {miss}")
    for name, hub in (("runpod_model_cache", paths.MODEL_CACHE), ("network_volume", paths.VOLUME_HF)):
        d = snapshot_dir(hub, repo)
        if d is None:
            tried.append(f"{name}: no snapshot of {repo} under {hub}")
            continue
        ok, miss = _complete(d)
        if ok:
            return d, name
        tried.append(f"{name} {d}: missing {miss}")
    try:
        d = _download(repo)
        ok, miss = _complete(d)
        if ok:
            return d, "download"
        tried.append(f"download {d}: missing {miss}")
    except NotFound as e:
        tried.append(str(e))
    raise NotFound("PersonaPlex not found: " + " | ".join(tried))


def voices_dir(pp_dir):
    pp_dir = Path(pp_dir)
    if paths.VOICES and (Path(paths.VOICES) / VOICE_CHECK).exists():
        return Path(paths.VOICES), "S2S_VOICES"
    if (pp_dir / "voices" / VOICE_CHECK).exists():
        return pp_dir / "voices", "snapshot"
    out = paths.TMP / "voices"
    if (out / VOICE_CHECK).exists():
        return out, "extracted_earlier"
    tgz = pp_dir / "voices.tgz"
    if not tgz.exists():
        raise NotFound(f"no voices/ and no voices.tgz in {pp_dir}")
    t0 = time.time()
    paths.TMP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tar:
        try:
            tar.extractall(path=paths.TMP, filter="data")
        except TypeError:          # python without extraction filters
            tar.extractall(path=paths.TMP)
    if not (out / VOICE_CHECK).exists():
        raise NotFound(f"{tgz} did not contain voices/{VOICE_CHECK}")
    log.info("extracted %s -> %s in %.1f s", tgz, out, time.time() - t0)
    return out, "extracted"


def resolve(write_env=True):
    t0 = time.time()
    d, src = find_pp_dir()
    v, vsrc = voices_dir(d)
    res = {"pp_dir": str(d), "source": src, "voices": str(v), "voices_source": vsrc, "repo": paths.PP_REPO,
           "files": {f: round((d / f).stat().st_size / 2**20, 1) for f in NEEDED}, "resolve_s": round(time.time() - t0, 2)}
    os.environ["S2S_PP_DIR"] = res["pp_dir"]
    os.environ["S2S_VOICES"] = res["voices"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if write_env:
        paths.MODELS_ENV.parent.mkdir(parents=True, exist_ok=True)
        paths.MODELS_ENV.write_text(f"S2S_PP_DIR={res['pp_dir']}\nS2S_VOICES={res['voices']}\n")
    log.info("PersonaPlex: %s (%s), voices %s (%s), MiB %s", d, src, v, vsrc, res["files"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        print(json.dumps(resolve(write_env="--no-write" not in sys.argv), indent=1), flush=True)
    except NotFound as e:
        log.error("%s", e)
        print(json.dumps({"error": str(e)}), flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
