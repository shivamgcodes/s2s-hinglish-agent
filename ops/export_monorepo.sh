#!/bin/bash
# Export this working folder (laptop hinglish/deploy_serverless) into a checkout of the monorepo
# github.com/shivamgcodes/s2s-hinglish-agent (D-MONOREPO 2026-10-06; replaces make_repos.sh, which split it into
# s2s-worker + s2s-space). Then runs the secret + size scan. Does NOT commit or push.
#
#   bash ops/export_monorepo.sh OUT_DIR       (OUT_DIR is replaced, its .git kept)
#
# Same layout as this folder, except:
#   docs/        <- DECISIONS.md, GO_LIVE.md, DESIGN.md, README.md (the older internal "start here", as docs/SERVERLESS_NOTES.md)
#   README.md    <- ops/monorepo/README.md;  .gitignore <- ops/monorepo/gitignore;  docs/README.md <- ops/monorepo/docs_README.md
# Left out: results/ (raw outputs), .local-run/, .test-out/, venvs, node_modules/, client/dist (built by
# ops/stage_space.sh --build / run_local.sh setup / CI), client/.env.local (client/.env.production has the same
# non-secret value), space/static + space/common (staging copies), space/tests/out_e2e, caches.
# In a clone of the monorepo this script is not needed (edit the clone directly).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:?usage: export_monorepo.sh OUT_DIR}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
[ "$OUT" != "$ROOT" ] || { echo "OUT must not be this folder" >&2; exit 2; }
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
RS=(rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude '.venv*' --exclude .venvs --exclude node_modules)
"${RS[@]}" --exclude build "$ROOT/worker/" "$OUT/worker/"
"${RS[@]}" --exclude static --exclude common --exclude out_e2e "$ROOT/space/" "$OUT/space/"
"${RS[@]}" --exclude dist --exclude .env.local --exclude '*.pem' "$ROOT/client/" "$OUT/client/"
"${RS[@]}" "$ROOT/common/" "$OUT/common/"
"${RS[@]}" --exclude monorepo "$ROOT/ops/" "$OUT/ops/"
"${RS[@]}" "$ROOT/tests/" "$OUT/tests/"
"${RS[@]}" "$ROOT/.github/" "$OUT/.github/"
cp -p "$ROOT/run_local.sh" "$ROOT/run_all_cpu_tests.sh" "$OUT/"
mkdir -p "$OUT/docs"
cp "$ROOT/DECISIONS.md" "$ROOT/GO_LIVE.md" "$ROOT/DESIGN.md" "$OUT/docs/"
cp "$ROOT/README.md" "$OUT/docs/SERVERLESS_NOTES.md"
cp "$ROOT/ops/monorepo/docs_README.md" "$OUT/docs/README.md"
cp "$ROOT/ops/monorepo/README.md" "$OUT/README.md"
cp "$ROOT/ops/monorepo/gitignore" "$OUT/.gitignore"
chmod u+w "$OUT/docs/DESIGN.md"
bash "$ROOT/ops/scan_secrets.sh" "$OUT"
