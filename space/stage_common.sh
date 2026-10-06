#!/usr/bin/env bash
# Copy the shared modules the Space needs from the repo's common/ into space/common/ before pushing space/ as the
# HF Space repo (the Space builds from space/ alone, so ../common is outside its Docker context).
#   bash space/stage_common.sh            (run from anywhere)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/../common}"
DST="$HERE/common"
for f in s2s_token.py session.py data/records_v4.json data/scripts_v4.json; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f" >&2; exit 1; }
done
mkdir -p "$DST/data"
cp "$SRC/s2s_token.py" "$SRC/session.py" "$DST/"
cp "$SRC/data/records_v4.json" "$SRC/data/scripts_v4.json" "$DST/data/"   # scripts: D-SCRIPT-PANEL
echo "staged common/ -> $DST: $(ls "$DST" | tr '\n' ' ')"
[ -f "$HERE/static/index.html" ] || echo "WARNING: $HERE/static/index.html missing (run ops/build_client.sh)" >&2
