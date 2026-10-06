#!/bin/bash
# Assemble the Hugging Face Space tree from this checkout (D-MONOREPO 2026-10-06; replaces the s2s-space repo).
# The Space builds from this tree alone (space/Dockerfile, context = the tree), so it gets copies of what it needs:
#   OUT/            = space/ minus tests/, static/, common/ (and caches)
#   OUT/common/     = common/{s2s_token.py,session.py,data/records_v4.json,data/scripts_v4.json} (space/stage_common.sh's list)
#   OUT/static/     = the built web client (client/dist)
#   OUT/.gitignore  = the same 3 lines the s2s-space repo had
#
#   bash ops/stage_space.sh OUT_DIR [--build]     --build: npm ci && npm run build in client/ first (needs Node >= 18;
#                                                 pod1: PATH=/root/node/bin:$PATH); default: use client/dist as is
# OUT_DIR is replaced (a .git inside it is kept). Nothing is uploaded: ops/push_space.sh does that.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:?usage: stage_space.sh OUT_DIR [--build]}; BUILD=0
[ "${2:-}" = --build ] && BUILD=1
case "$(cd "$(dirname "$OUT")" 2>/dev/null && pwd)/$(basename "$OUT")" in "$ROOT"|"$ROOT/space"|"$ROOT/client"*) echo "refusing OUT=$OUT" >&2; exit 2 ;; esac
if [ $BUILD = 1 ]; then
  (cd "$ROOT/client" && npm ci --no-audit --no-fund && npm run build)
fi
[ -f "$ROOT/client/dist/index.html" ] || { echo "client/dist missing: build the client first (--build, or cd client && npm ci && npm run build)" >&2; exit 1; }
mkdir -p "$OUT"
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude '.venv*' --exclude tests --exclude static --exclude common "$ROOT/space/" "$OUT/"
mkdir -p "$OUT/common/data"
for f in s2s_token.py session.py data/records_v4.json data/scripts_v4.json; do   # = space/stage_common.sh (D-SCRIPT-PANEL)
  cp "$ROOT/common/$f" "$OUT/common/$f"
done
mkdir -p "$OUT/static"; cp -a "$ROOT/client/dist/." "$OUT/static/"
printf '__pycache__/\n*.pyc\n.venv*/\n' > "$OUT/.gitignore"
echo "staged Space tree: $OUT ($(find "$OUT" -type f ! -path '*/.git/*' | wc -l) files, $(du -sh --exclude=.git "$OUT" | cut -f1)); client bundle: $(grep -o 'assets/index-[A-Za-z0-9_-]*\.js' "$OUT/static/index.html")"
