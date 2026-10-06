#!/bin/bash
# S2S serverless: fill the docker build context with the baked assets (DESIGN 6.2; layout = worker/Dockerfile header).
# Runs on runpod2 (the assets live there). Read-only on the sources; never writes under /root/deploy.
#
#   bash ops/prepare_build_context.sh            copy assets into worker/build/ (+ common/data/records_v4.json if missing)
#   bash ops/prepare_build_context.sh --check    only verify that worker/build/ is complete (exit 1 if not)
#   bash ops/prepare_build_context.sh --bundle /root/s2s_build_context.tar
#                                                also write a tar of the whole build context (repo code + worker/build,
#                                                no venvs/node_modules/tests output) to copy to a docker host:
#                                                scp runpod2:/root/s2s_build_context.tar . && mkdir ctx && tar -xf ... -C ctx
# Result (about 470 MB):
#   worker/build/v3_adapter/{config.json,lora.safetensors}           V3 LoRA step 200 (md5 checked against assets/V3_CKPT)
#   worker/build/tuned_full.cact                                     Needle weights
#   worker/build/cactus-needle/v3/3.0.2/{libneedle.so,needle3.cact}  Needle native lib + base weights (else fetched at runtime)
# Sources (env overrides): SRC_ADAPTER (default: path= line of /root/deploy/assets/V3_CKPT), SRC_CACT, SRC_NEEDLE_LIB,
#   SRC_RECORDS. worker/build/ is NOT part of the repo; it is excluded from persistence by size only.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=copy; BUNDLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check; shift ;;
    --bundle) BUNDLE=$2; shift 2 ;;
    -h|--help) sed -n 2,19p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
CKPT_FILE=/root/deploy/assets/V3_CKPT
SRC_ADAPTER=${SRC_ADAPTER:-$( [ -f $CKPT_FILE ] && sed -n 's/^path=//p' $CKPT_FILE || echo "")}
MD5_LORA=$( [ -f $CKPT_FILE ] && sed -n 's/^md5_lora=//p' $CKPT_FILE || echo "")
SRC_CACT=${SRC_CACT:-/root/needle/finetune/sweep/tuned_full.cact}
SRC_NEEDLE_LIB=${SRC_NEEDLE_LIB:-/root/.cache/cactus-needle/v3/3.0.2}
SRC_RECORDS=${SRC_RECORDS:-/root/deploy/assets/ws/hinglish/data/V4/records.json}
B=$ROOT/worker/build
NEED=(v3_adapter/config.json v3_adapter/lora.safetensors tuned_full.cact
      cactus-needle/v3/3.0.2/libneedle.so cactus-needle/v3/3.0.2/needle3.cact)

check() {
  local bad=0
  for f in "${NEED[@]}"; do
    if [ -s "$B/$f" ]; then printf '  ok   %-45s %s\n' "$f" "$(du -h "$B/$f" | cut -f1)"; else echo "  MISSING $B/$f"; bad=1; fi
  done
  for f in common/s2s_token.py common/session.py common/data/records_v4.json worker/vendor/moshi; do
    [ -e "$ROOT/$f" ] && echo "  ok   $f" || { echo "  MISSING $ROOT/$f"; bad=1; }
  done
  if [ -n "$MD5_LORA" ] && [ -s "$B/v3_adapter/lora.safetensors" ]; then
    local m; m=$(md5sum "$B/v3_adapter/lora.safetensors" | cut -d' ' -f1)
    [ "$m" = "$MD5_LORA" ] && echo "  ok   lora md5 $m (= V3_CKPT md5_lora)" || { echo "  BAD  lora md5 $m != $MD5_LORA"; bad=1; }
  fi
  return $bad
}

if [ $MODE = copy ]; then
  [ -n "$SRC_ADAPTER" ] && [ -s "$SRC_ADAPTER/lora.safetensors" ] || { echo "adapter not found: '$SRC_ADAPTER'" >&2; exit 1; }
  [ -s "$SRC_CACT" ] || { echo "Needle weights not found: $SRC_CACT" >&2; exit 1; }
  [ -s "$SRC_NEEDLE_LIB/libneedle.so" ] || { echo "Needle lib cache not found: $SRC_NEEDLE_LIB" >&2; exit 1; }
  mkdir -p "$B/v3_adapter" "$B/cactus-needle/v3/3.0.2"
  cp -u "$SRC_ADAPTER/config.json" "$SRC_ADAPTER/lora.safetensors" "$B/v3_adapter/"
  cp -u "$SRC_CACT" "$B/tuned_full.cact"
  cp -u "$SRC_NEEDLE_LIB/libneedle.so" "$SRC_NEEDLE_LIB/needle3.cact" "$B/cactus-needle/v3/3.0.2/"
  if [ ! -s "$ROOT/common/data/records_v4.json" ]; then
    mkdir -p "$ROOT/common/data"; cp "$SRC_RECORDS" "$ROOT/common/data/records_v4.json"
  fi
  echo "copied into $B (sources: $SRC_ADAPTER, $SRC_CACT, $SRC_NEEDLE_LIB)"
fi
echo "build context check ($ROOT):"
check || { echo "build context INCOMPLETE" >&2; exit 1; }

if [ -n "$BUNDLE" ]; then
  # only what worker/Dockerfile + space/Dockerfile need, plus docs/ops/tests (small)
  tar -C "$ROOT" -cf "$BUNDLE" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='node_modules' --exclude='.venv*' --exclude='worker/local' \
    --exclude='space/tests/out_e2e' \
    DESIGN.md DECISIONS.md common worker space ops tests $( [ -f "$ROOT/README.md" ] && echo README.md )
  echo "bundle: $BUNDLE ($(du -h "$BUNDLE" | cut -f1)); md5 $(md5sum "$BUNDLE" | cut -d' ' -f1)"
fi
