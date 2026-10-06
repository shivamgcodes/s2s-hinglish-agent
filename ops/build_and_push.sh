#!/bin/bash
# S2S serverless: build and push the GPU worker image (and optionally the Space image for launcher (c)).
# DRY RUN BY DEFAULT: prints every command. --run builds; --push also pushes. Needs a docker host with BuildKit,
# about 60 GB free disk and the repo + worker/build/ (ops/prepare_build_context.sh). There is no docker on the pods.
#
#   IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh                 # print only
#   IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh --run           # docker build
#   IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh --run --push    # build + push
#   ... --space-image docker.io/<you>/s2s-space    also build/push space/ as a plain container (launcher c)
# Build args (env): TORCH_INDEX=cu130 (default; cu128 untested, R5)  TRELIS_BF16=0|1 (1: -2.9 GB, see worker/Dockerfile)
#
# Registry: log in first (docker login). A PRIVATE image needs RunPod console -> Settings -> Container registry auth,
# and its id in RUNPOD_REGISTRY_AUTH_ID for ops/create_endpoint.sh. PersonaPlex weights are never in the image
# (gated; RunPod model cache), so a public image only redistributes the V3 LoRA, Needle weights and public Trelis:
# make it PRIVATE unless you are sure that is acceptable.
set -euo pipefail
RUN=0; PUSH=0; SPACE_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --push) PUSH=1; RUN=1; shift ;;
    --space-image) SPACE_IMAGE=$2; shift 2 ;;
    -h|--help) sed -n 2,18p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG="${IMAGE:-<registry>/s2s-worker}:${TAG:-v1}"
TI=${TORCH_INDEX:-cu130}; TB=${TRELIS_BF16:-0}
# Amendment W2.3: cu128 wheels also need the 12.8 CUDA base (the 13.0.3 base sets NVIDIA_REQUIRE_CUDA=cuda>=13.0)
EXTRA=""; [ "$TI" = cu128 ] && EXTRA="--build-arg CUDA_BASE=${CUDA_BASE:-nvidia/cuda:12.8.2-base-ubuntu24.04}"
BUILD_ID=$(date -u +%Y%m%d-%H%M)

CMDS=(
  "bash $ROOT/ops/prepare_build_context.sh --check"
  "docker buildx build --platform linux/amd64 -f $ROOT/worker/Dockerfile --build-arg TORCH_INDEX=$TI --build-arg TRELIS_BF16=$TB $EXTRA --label s2s.build=$BUILD_ID -t $IMG --load $ROOT"
  "docker image inspect $IMG --format '{{.Size}}'"
  "docker run --rm --entrypoint bash $IMG -c '/opt/venv-pp/bin/python /opt/s2s/worker/paths.py && grep -rIl -e /root/deploy -e /root/needle -e /workspace -e /root/hf /opt/s2s/worker --include=*.py --include=*.sh | grep -v /local/ || echo path-grep-clean'   # hits are OK if they are only comments citing DEP1 (DESIGN 1.2); tests/test_mock_e2e.py has the exact check"
  "docker run --rm --entrypoint bash $IMG -c 'env | grep -iE \"token|secret|api_key\" || echo no-secret-env'"
)
[ $PUSH = 1 ] && CMDS+=("docker push $IMG")
if [ -n "$SPACE_IMAGE" ]; then
  CMDS+=("bash $ROOT/space/stage_common.sh"
         "docker buildx build --platform linux/amd64 -t $SPACE_IMAGE:${TAG:-v1} --load $ROOT/space")
  [ $PUSH = 1 ] && CMDS+=("docker push $SPACE_IMAGE:${TAG:-v1}")
fi

echo "# worker image: $IMG   (TORCH_INDEX=$TI TRELIS_BF16=$TB)"
for c in "${CMDS[@]}"; do echo "  $c"; done
cat <<'EOF'
# expected: image about 13-14 GB (about 11 GB with TRELIS_BF16=1); "no-secret-env"; the path grep may list files whose
#   only hits are comments/docstrings citing the DEP1 origin (allowed by DESIGN 1.2) -- review them, not a failure.
# There is no GPU-free container smoke test: the real check is the first RunPod cold start (TEST_PLAN.md step 3).
EOF
if [ $RUN != 1 ]; then
  echo "# DRY RUN: nothing executed. Add --run (build) or --push (build + push)."
  exit 0
fi
[ -n "${IMAGE:-}" ] || { echo "set IMAGE (e.g. docker.io/<you>/s2s-worker)" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker not found on this host" >&2; exit 2; }
for c in "${CMDS[@]}"; do
  echo "+ $c"
  bash -c "$c"
done
echo "built: $IMG  (build $BUILD_ID)"
