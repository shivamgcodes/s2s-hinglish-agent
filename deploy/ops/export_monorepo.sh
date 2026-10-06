#!/bin/bash
# Export the development folder (laptop hinglish/deploy_serverless) into a checkout of the monorepo
# github.com/shivamgcodes/s2s-hinglish-agent, then run the secret + size scan. Does NOT commit or push.
#
#   bash deploy/ops/export_monorepo.sh OUT_DIR       (OUT_DIR is replaced, its .git kept)
#
# Layout (D-MONOREPO-LAYOUT 2026-10-07; the development folder has the same top level):
#   packages/   shared code, exactly once (needle_router, personaplex_lora: pip-installable; hinglish_text)
#   research/   data generation, audio pipeline, PersonaPlex LoRA training, Needle N1/N2, eval harness, inference examples
#   deploy/     worker/, space/, client/, common/, ops/, tests/, run_local.sh, run_all_cpu_tests.sh
#   .github/    CI (builds the worker image from deploy/ + packages/ only)
#   docs/       <- DECISIONS.md, GO_LIVE.md, DESIGN.md, README.md (as docs/SERVERLESS_NOTES.md), deploy/ops/monorepo/docs_README.md
#   README.md   <- deploy/ops/monorepo/README.md;  .gitignore <- deploy/ops/monorepo/gitignore
# Left out: results/ (raw outputs), .local-run/, .test-out/, venvs, node_modules/, client/dist (built by
# ops/stage_space.sh --build / run_local.sh setup / CI), client/.env.local (client/.env.production has the same
# non-secret value), space/static + space/common (staging copies), space/tests/out_e2e, caches, deploy/ops/monorepo.
# In a clone of the monorepo this script is not needed (edit the clone directly).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${1:?usage: export_monorepo.sh OUT_DIR}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
[ "$OUT" != "$ROOT" ] || { echo "OUT must not be this folder" >&2; exit 2; }
for d in packages research deploy .github; do [ -d "$ROOT/$d" ] || { echo "missing $ROOT/$d" >&2; exit 2; }; done
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
RS=(rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude '.venv*' --exclude .venvs --exclude node_modules
    --exclude .test-out --exclude .local-run)
"${RS[@]}" "$ROOT/packages/" "$OUT/packages/"
"${RS[@]}" "$ROOT/research/" "$OUT/research/"
"${RS[@]}" "$ROOT/.github/" "$OUT/.github/"
"${RS[@]}" --exclude /worker/build --exclude /space/static --exclude /space/common --exclude /space/tests/out_e2e \
  --exclude /client/dist --exclude /client/.env.local --exclude '*.pem' --exclude /ops/monorepo "$ROOT/deploy/" "$OUT/deploy/"
mkdir -p "$OUT/docs"
cp "$ROOT/DECISIONS.md" "$ROOT/GO_LIVE.md" "$ROOT/DESIGN.md" "$OUT/docs/"
cp "$ROOT/README.md" "$OUT/docs/SERVERLESS_NOTES.md"
cp "$ROOT/deploy/ops/monorepo/docs_README.md" "$OUT/docs/README.md"
cp "$ROOT/deploy/ops/monorepo/README.md" "$OUT/README.md"
cp "$ROOT/deploy/ops/monorepo/gitignore" "$OUT/.gitignore"
chmod u+w "$OUT/docs/DESIGN.md"
bash "$ROOT/deploy/ops/scan_secrets.sh" "$OUT"
