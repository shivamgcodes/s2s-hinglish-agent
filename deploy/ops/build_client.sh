#!/bin/bash
# S2S serverless: build the web client (client/, DESIGN 5.4) and copy it into the Space (space/static).
# runpod2 only (never on the laptop). npm is CPU-heavy: do not run it while a live DEP1 call is in progress (A1.3).
#
#   bash ops/build_client.sh               npm ci && npm run build (nice -n 19), then client/dist -> space/static
#   bash ops/build_client.sh --copy-only   skip npm; copy the existing client/dist
#   bash ops/build_client.sh --dry-run     print the steps only
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
C=$ROOT/client; S=$ROOT/space/static
MODE=build
case "${1:-}" in
  --copy-only) MODE=copy ;;
  --dry-run) MODE=dry ;;
  "") ;;
  *) echo "usage: build_client.sh [--copy-only|--dry-run]" >&2; exit 2 ;;
esac
echo "client: $C  ->  $S"
if [ $MODE = dry ]; then
  echo "  (cd $C && nice -n 19 npm ci && nice -n 19 npm run build)"
  echo "  rm -rf $S && mkdir -p $S && cp -a $C/dist/. $S/"
  echo "DRY RUN: nothing done"; exit 0
fi
if [ $MODE = build ]; then
  if tmux has-session -t dep1_gpu 2>/dev/null && curl -s 127.0.0.1:8999/internal/session 2>/dev/null | grep -q '"session_id"'; then
    echo "a live DEP1 call is in progress; npm would disturb it (A1.3). Try later or use --copy-only." >&2; exit 1
  fi
  (cd "$C" && nice -n 19 npm ci --no-audit --no-fund && nice -n 19 npm run build)
fi
[ -f "$C/dist/index.html" ] || { echo "no $C/dist/index.html (build first)" >&2; exit 1; }
rm -rf "$S"; mkdir -p "$S"; cp -a "$C/dist/." "$S/"
echo "space/static: $(find "$S" -type f | wc -l) files, $(du -sh "$S" | cut -f1)"
