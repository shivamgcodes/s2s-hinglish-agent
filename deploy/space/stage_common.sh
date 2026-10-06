#!/usr/bin/env bash
# Copy the shared modules the Space needs into a Space tree's common/ before pushing it as the HF Space repo (the Space
# builds from that tree alone, so ../common and ../../packages are outside its Docker context). The ONE list of what
# the Space gets (ops/stage_space.sh calls this too):
#   deploy/common/{s2s_token.py,session.py,data/records_v4.json,data/scripts_v4.json}
#   packages/personaplex_lora/role_prompt.py   (D-SINGLE-SOURCE 2026-10-07: session.py imports it from next to itself)
#   bash space/stage_common.sh [SRC_COMMON] [DST_COMMON]     defaults: deploy/common -> deploy/space/common
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/../common}"
DST="${2:-$HERE/common}"
FMT="$HERE/../../packages/personaplex_lora/role_prompt.py"
for f in "$SRC/s2s_token.py" "$SRC/session.py" "$SRC/data/records_v4.json" "$SRC/data/scripts_v4.json" "$FMT"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p "$DST/data"
cp "$SRC/s2s_token.py" "$SRC/session.py" "$FMT" "$DST/"
cp "$SRC/data/records_v4.json" "$SRC/data/scripts_v4.json" "$DST/data/"   # scripts: D-SCRIPT-PANEL
echo "staged common/ -> $DST: $(ls "$DST" | tr '\n' ' ')"
[ "$DST" != "$HERE/common" ] || [ -f "$HERE/static/index.html" ] || echo "WARNING: $HERE/static/index.html missing (run ops/build_client.sh)" >&2
