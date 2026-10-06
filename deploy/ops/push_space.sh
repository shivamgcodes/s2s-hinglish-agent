#!/bin/bash
# S2S serverless: create the HF Docker Space and upload the Space tree with huggingface_hub (NOT git push:
# a web-created Space has its own initial commit, and HF rejects plain-git binaries such as .wasm/.png/.ico;
# upload_folder routes them through Xet and ignores history). DRY RUN unless --yes.
#
#   bash ops/push_space.sh [--dry-run] [--build-client] [--out DIR] [--space shivamgupta/hinglish-agent] [--yes]
#       default: assemble the tree from this checkout with ops/stage_space.sh (space/ + common/ + client/dist) into
#                a temp dir (or --out DIR, kept), then upload it. --build-client runs npm ci && npm run build first
#                (pod1: PATH=/root/node/bin:$PATH). D-MONOREPO 2026-10-06: this replaces the s2s-space repo.
#   bash ops/push_space.sh --dir <tree> [...]    upload an already assembled tree as is (e.g. an old s2s-space clone)
# Token (a WRITE token), first found: env HF_TOKEN_WRITE / HF_TOKEN, --token-file (default /workspace/hf/token on
# pod1, read in place), ~/.config/s2s/secrets.env HF_TOKEN_WRITE. Never printed; given to python on stdin.
# Creates the Space as PUBLIC, sdk=docker, cpu-basic (a private Space returns 404 to visitors and their websockets;
# switch it to "Protected" in Settings if wanted). Variables/secrets are NOT set here: set them in the Space settings
# BEFORE the upload (GO_LIVE.md section 4.2) or the first build starts without them (it then just logs config errors).
# upload_folder adds and replaces files but never deletes: old hashed client bundles stay in the Space (harmless;
# index.html names the current one).
set -euo pipefail
DIR=""; OUTDIR=""; BUILD=""; SPACE="shivamgupta/hinglish-agent"; YES=0; PY=""
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOKEN_FILE=""; [ -r /workspace/hf/token ] && TOKEN_FILE=/workspace/hf/token
SECRETS="$HOME/.config/s2s/secrets.env"
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR=$2; shift 2 ;;
    --space) SPACE=$2; shift 2 ;;
    --yes) YES=1; shift ;;
    --dry-run) YES=0; shift ;;
    --out) OUTDIR=$2; shift 2 ;;
    --build-client) BUILD=--build; shift ;;
    --token-file) TOKEN_FILE=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    -h|--help) sed -n 2,19p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
if [ -z "$DIR" ]; then
  if [ -n "$OUTDIR" ]; then DIR=$OUTDIR; else DIR=$(mktemp -d "${TMPDIR:-/tmp}/s2s-space.XXXXXX"); fi
  bash "$ROOT/ops/stage_space.sh" "$DIR" $BUILD
fi
[ -n "$DIR" ] && [ -f "$DIR/README.md" ] && [ -f "$DIR/Dockerfile" ] && [ -f "$DIR/static/index.html" ] && [ -f "$DIR/common/s2s_token.py" ] && [ -f "$DIR/common/role_prompt.py" ] && [ -f "$DIR/common/data/records_v4.json" ] && [ -f "$DIR/common/data/scripts_v4.json" ] \
  || { echo "$DIR is not an assembled Space tree (README.md, Dockerfile, static/, common/): see ops/stage_space.sh" >&2; exit 2; }
head -12 "$DIR/README.md" | grep -q '^sdk: docker' || { echo "$DIR/README.md has no 'sdk: docker' Space header" >&2; exit 2; }
bash "$ROOT/ops/scan_secrets.sh" "$DIR" | sed 's/^/[scan] /'; [ "${PIPESTATUS[0]}" = 0 ] || { echo "secret/size scan failed: not uploading" >&2; exit 1; }
TOKEN="${HF_TOKEN_WRITE:-${HF_TOKEN:-}}"
[ -z "$TOKEN" ] && [ -n "$TOKEN_FILE" ] && [ -r "$TOKEN_FILE" ] && TOKEN=$(head -n1 "$TOKEN_FILE" | tr -d ' \r\n')
[ -z "$TOKEN" ] && [ -f "$SECRETS" ] && TOKEN=$(sed -n 's/^HF_TOKEN_WRITE=//p' "$SECRETS" | tail -1 | tr -d '"'"'"' \r')
if [ -z "$PY" ]; then
  for c in /root/dep2/.venvs/needle/bin/python /root/deploy/venv-asr/bin/python python3; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c "import huggingface_hub" 2>/dev/null && { PY=$c; break; }
  done
fi
[ -n "$PY" ] || { echo "no python with huggingface_hub" >&2; exit 1; }
N=$(find "$DIR" -type f ! -path '*/.git/*' | wc -l)
echo "space: $SPACE  dir: $DIR ($N files, $(du -sh --exclude=.git "$DIR" | cut -f1))  token: $([ -n "$TOKEN" ] && echo found || echo none)"
if [ $YES = 0 ]; then
  echo "DRY RUN: would create_repo($SPACE, repo_type=space, space_sdk=docker, exist_ok) and upload_folder($DIR, ignore .git, tests).  Add --yes."
  exit 0
fi
[ -n "$TOKEN" ] || { echo "no HF write token" >&2; exit 1; }
printf '%s\n' "$TOKEN" | "$PY" -c '
import sys
from huggingface_hub import HfApi
tok = sys.stdin.readline().strip(); space, d = sys.argv[1], sys.argv[2]
api = HfApi(token=tok)
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
c = api.upload_folder(repo_id=space, repo_type="space", folder_path=d, ignore_patterns=[".git/*", "tests/*", "**/__pycache__/*"],
                      commit_message="S2S Space front: aiohttp relay + built client (from shivamgcodes/s2s-hinglish-agent space/ + client/)")
print("uploaded:", c.commit_url if hasattr(c, "commit_url") else c)
host = space.replace("/", "-").replace("_", "-").lower()
print(f"Space: https://huggingface.co/spaces/{space}   app: https://{host}.hf.space/")
' "$SPACE" "$DIR"
