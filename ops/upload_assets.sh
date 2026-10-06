#!/bin/bash
# S2S serverless: create the PRIVATE Hugging Face model repo that the worker fetches its small assets from at start
# (worker/fetch_assets.py), and upload them. DRY RUN unless --yes. DECISIONS 2026-10-06 "WEIGHTS".
#
#   bash ops/upload_assets.sh                          dry run on the host that has the files (pod1): md5-check the
#                                                      sources against worker/assets_manifest.json, print the plan
#   bash ops/upload_assets.sh --yes                    ON POD1 (2026-10-06 default): token from /workspace/hf/token
#                                                      (read in place, never copied or printed); create
#                                                      shivamgupta/s2s-v4-assets (private) + upload + verify
#   bash ops/upload_assets.sh --remote runpod [--yes]  from the laptop: the token is read here and sent to the pod
#                                                      over ssh STDIN (never on a command line, never written there)
# Options: --repo OWNER/NAME (default shivamgupta/s2s-v4-assets)  --token-file FILE (default /workspace/hf/token if it
#          exists)  --secrets FILE (default ~/.config/s2s/secrets.env; used when there is no token file)
#          --python PY (a python with huggingface_hub >= 0.30; default: pod1's venv-asr, else python3)
#          --only REPO_PATH[,REPO_PATH] upload only these files (+ MANIFEST.json); all are still md5-checked and verified
#
# Token (a WRITE token), first found: env HF_TOKEN_WRITE / HF_TOKEN, --token-file (pod1: /workspace/hf/token),
# the secrets file (HF_TOKEN_WRITE, then HF_TOKEN). Never printed; passed to python on stdin. Files uploaded (repo path <- pod1 source; md5s in worker/assets_manifest.json):
#   v4_adapter/config.json, v4_adapter/lora.safetensors <- /workspace/runs/V4_A2/checkpoints/checkpoint_000600/consolidated
#   needle/tuned_full.cact                              <- /workspace/hinglish/needle/tuned_full.cact   (Needle N1)
#   needle_v2/tuned_full.cact                           <- /workspace/hinglish/needle_v2/finetune/tuned_full.cact (Needle v2,
#                                                          R_e15; D-ROUTER-V2 2026-10-06)
#   needle/libneedle.so                                 <- /root/.cache/cactus-needle/v3/3.0.2/libneedle.so
#   data/records_v4.json                                <- /workspace/hinglish/data/V4/records.json
#   MANIFEST.json                                       <- worker/assets_manifest.json
# After the upload every file's size and sha256 are compared with the repo's LFS metadata (no re-download).
# Network-volume alternative (no HF repo): copy the same layout to /runpod-volume/s2s-assets/ (ops/NETWORK_VOLUME.md).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
YES=0; ONLY=""; REMOTE=""; REPO="shivamgupta/s2s-v4-assets"; SECRETS="$HOME/.config/s2s/secrets.env"; PY=""
TOKEN_FILE=""; [ -r /workspace/hf/token ] && TOKEN_FILE=/workspace/hf/token
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) YES=1; shift ;;
    --remote) REMOTE=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --secrets) SECRETS=$2; shift 2 ;;
    --token-file) TOKEN_FILE=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
TOKEN="${HF_TOKEN_WRITE:-${HF_TOKEN:-}}"
if [ -z "$TOKEN" ] && [ -n "$TOKEN_FILE" ] && [ -r "$TOKEN_FILE" ]; then
  TOKEN=$(head -n1 "$TOKEN_FILE" | tr -d ' \r\n')
fi
if [ -z "$TOKEN" ] && [ -f "$SECRETS" ]; then
  [ "$(stat -c %a "$SECRETS")" = 600 ] || echo "WARNING: $SECRETS is not mode 600" >&2
  TOKEN=$(sed -n 's/^HF_TOKEN_WRITE=//p' "$SECRETS" | tail -1 | tr -d '"'"'"' \r')
  [ -n "$TOKEN" ] || TOKEN=$(sed -n 's/^HF_TOKEN=//p' "$SECRETS" | tail -1 | tr -d '"'"'"' \r')
fi
if [ -z "$TOKEN" ] && [ $YES = 1 ]; then echo "no token: env HF_TOKEN_WRITE/HF_TOKEN, --token-file, or $SECRETS" >&2; exit 1; fi
echo "token: $([ -n "$TOKEN" ] && echo "found (${#TOKEN} chars, not shown)" || echo "none (dry run only)"); repo: $REPO"
MANIFEST_JSON=$(cat "$ROOT/worker/assets_manifest.json")

read -r -d '' PYCODE <<'PYEOF' || true
import hashlib, json, os, sys
tok = sys.stdin.readline().strip()
cfg = json.loads(sys.stdin.read())
man, yes, repo = cfg["manifest"], cfg["yes"], cfg["repo"]
SRC = {"v4_adapter/config.json": "/workspace/runs/V4_A2/checkpoints/checkpoint_000600/consolidated/config.json",
       "v4_adapter/lora.safetensors": "/workspace/runs/V4_A2/checkpoints/checkpoint_000600/consolidated/lora.safetensors",
       "needle/tuned_full.cact": "/workspace/hinglish/needle/tuned_full.cact",
       "needle_v2/tuned_full.cact": "/workspace/hinglish/needle_v2/finetune/tuned_full.cact",
       "needle/libneedle.so": os.path.expanduser("~/.cache/cactus-needle/v3/3.0.2/libneedle.so"),
       "data/records_v4.json": "/workspace/hinglish/data/V4/records.json"}
for k in list(SRC):
    SRC[k] = os.environ.get("SRC_" + k.replace("/", "_").replace(".", "_").upper(), SRC[k])
def digest(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
entries = man["files"] + man.get("also_uploaded", [])
bad = 0
for e in entries:
    p = SRC[e["repo_path"]]
    if not os.path.isfile(p):
        print(f"  MISSING {e['repo_path']:30s} <- {p}"); bad += 1; continue
    m, sz = digest(p, "md5"), os.path.getsize(p)
    good = m == e["md5"] and sz == e["size"]
    bad += not good
    print(f"  {'ok ' if good else 'BAD'} {e['repo_path']:30s} {sz/2**20:8.1f} MiB md5 {m} <- {p}")
if bad:
    print(f"{bad} source file(s) missing or not matching the manifest: nothing uploaded"); sys.exit(1)
if not yes:
    print(f"DRY RUN: would create the PRIVATE model repo {repo or '<whoami>/s2s-v4-assets'} and upload the files above "
          "+ MANIFEST.json, then verify size + sha256. Add --yes."); sys.exit(0)
from huggingface_hub import HfApi
api = HfApi(token=tok)
if not repo:
    repo = api.whoami()["name"] + "/s2s-v4-assets"
url = api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.repo_info(repo, repo_type="model")
if not info.private:
    print(f"REFUSING: {repo} exists and is PUBLIC"); sys.exit(1)
print(f"repo {repo} (private) ready")
only = [x for x in cfg.get("only", "").split(",") if x]
unknown = [x for x in only if x not in [e["repo_path"] for e in entries]]
if unknown:
    print(f"--only names files not in the manifest: {unknown}"); sys.exit(1)
for e in entries:
    if only and e["repo_path"] not in only:
        continue
    api.upload_file(path_or_fileobj=SRC[e["repo_path"]], path_in_repo=e["repo_path"], repo_id=repo, repo_type="model",
                    commit_message=f"add {e['repo_path']} (md5 {e['md5']})")
    print(f"  uploaded {e['repo_path']}")
api.upload_file(path_or_fileobj=json.dumps(man, indent=1).encode(), path_in_repo="MANIFEST.json", repo_id=repo,
                repo_type="model", commit_message="add MANIFEST.json (sizes + md5s)")
bad = 0
for e in entries:
    (pi,) = api.get_paths_info(repo, [e["repo_path"]], repo_type="model", expand=True)
    sha_remote = pi.lfs.sha256 if getattr(pi, "lfs", None) else None
    sha_local = digest(SRC[e["repo_path"]], "sha256")
    good = pi.size == e["size"] and (sha_remote is None or sha_remote == sha_local)
    bad += not good
    print(f"  verify {e['repo_path']:30s} size {pi.size} sha256 {'(not LFS)' if sha_remote is None else sha_remote[:16]} "
          f"{'OK' if good else 'MISMATCH'}")
print(("ALL VERIFIED: " if not bad else f"{bad} MISMATCH(ES): ") + f"https://huggingface.co/{repo} (private)")
print(f"Worker env: S2S_ASSETS_REPO={repo}")
sys.exit(1 if bad else 0)
PYEOF

CFG=$(printf '{"manifest": %s, "yes": %s, "repo": "%s", "only": "%s"}' "$MANIFEST_JSON" "$([ $YES = 1 ] && echo true || echo false)" "$REPO" "$ONLY")
if [ -n "$REMOTE" ]; then
  RPY=${PY:-/root/deploy/venv-asr/bin/python}
  echo "running on $REMOTE with $RPY (token via stdin)"
  { printf '%s\n' "$TOKEN"; printf '%s' "$CFG"; } | ssh "$REMOTE" "$RPY -c $(printf '%q' "$PYCODE")"
else
  if [ -z "$PY" ]; then
    for c in /root/dep2/.venvs/needle/bin/python /root/deploy/venv-asr/bin/python /opt/venv-asr/bin/python python3; do
      command -v "$c" >/dev/null 2>&1 && "$c" -c "import huggingface_hub" 2>/dev/null && { PY=$c; break; }; done
  fi
  { printf '%s\n' "$TOKEN"; printf '%s' "$CFG"; } | "$PY" -c "$PYCODE"
fi
