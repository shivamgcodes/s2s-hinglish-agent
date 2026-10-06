#!/bin/bash
# setup_container.sh - prepare a FRESH RunPod container on this volume for the S2S stack. Safe to re-run.
#
#   bash /workspace/setup_container.sh            # apt packages + import checks (no pip installs, no downloads)
#   bash /workspace/setup_container.sh --rebuild  # also rebuild any missing venv / model via the existing setup scripts
#
# What lives where:
#   /workspace (persistent): venv-pp (PersonaPlex), venv-tts (Kokoro + Gemma via Transformers),
#                            venv-vllm (Gemma via vLLM), venv-asr (faster-whisper), hf/ (PersonaPlex + Kokoro weights),
#                            gemma/models/ (Gemma 4 12B and 31B).
#   Lost on every container restart: apt packages (libopus-dev, espeak-ng), /root, /dev/shm.
set -uo pipefail
REBUILD=0; [ "${1:-}" = "--rebuild" ] && REBUILD=1
ok=0; fail=0
pass() { echo "  OK    $*"; ok=$((ok + 1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail + 1)); }

echo "== 1. system packages"
need=()
for p in libopus-dev espeak-ng; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if [ ${#need[@]} -gt 0 ]; then
  echo "  installing: ${need[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq "${need[@]}" >/dev/null 2>&1
fi
for p in libopus-dev espeak-ng; do dpkg -s "$p" >/dev/null 2>&1 && pass "$p" || bad "$p not installed"; done
command -v tmux >/dev/null && pass "tmux" || bad "tmux missing"

echo "== 2. GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  /'

if [ $REBUILD = 1 ]; then
  echo "== rebuild (only what is missing)"
  [ -x /workspace/venv-vllm/bin/python ] && [ -d /workspace/gemma/models/gemma-4-31B-it ] && [ -d /workspace/gemma/models/gemma-4-12B-it ] \
    || bash /workspace/gemma/vllm/setup_vllm.sh
  [ -x /workspace/venv-tts/bin/python ] || bash /workspace/gemma/setup_gemma.sh venv
  [ -x /workspace/venv-pp/bin/python ] || echo "  venv-pp missing: rebuild manually (see /workspace/personaplex-gate0/BLACKWELL_UPGRADE.txt)"
fi

echo "== 3. venv import checks"
check() {  # name, python code
  local v=/workspace/$1/bin/python
  if [ ! -x "$v" ]; then bad "$1 does not exist"; return; fi
  out=$(cd /tmp && HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf "$v" -c "$2" 2>&1 | grep -v -i -E "warn|deprecat" | tail -1)
  [ "${PIPESTATUS[0]:-0}" = 0 ] && [[ "$out" == OK* ]] && pass "$1: ${out#OK }" || bad "$1: $out"
}
check venv-pp   'import torch, moshi; assert torch.cuda.is_available(); print("OK torch", torch.__version__, "| moshi", moshi.__version__)'
check venv-tts  'import torch, kokoro, transformers; assert torch.cuda.is_available(); print("OK torch", torch.__version__, "| kokoro", kokoro.__version__, "| transformers", transformers.__version__)'
check venv-vllm 'import torch, vllm; assert torch.cuda.is_available(); print("OK torch", torch.__version__, "| vllm", vllm.__version__)'
if [ -x /workspace/venv-asr/bin/python ]; then
  check venv-asr 'import faster_whisper; print("OK faster_whisper", faster_whisper.__version__)'
fi

echo "== 4. models on the volume"
for d in /workspace/hf/hub/models--nvidia--personaplex-7b-v1 /workspace/hf/hub/models--hexgrad--Kokoro-82M \
         /workspace/gemma/models/gemma-4-12B-it /workspace/gemma/models/gemma-4-31B-it; do
  [ -d "$d" ] && pass "$d" || bad "$d missing (run with --rebuild, or the matching download script)"
done

cat <<'ENV'
== 5. environment each stack needs (not exported permanently; set per command)
  PersonaPlex (venv-pp):   HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
  Kokoro (venv-tts):       HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
  Gemma, Transformers:     /workspace/venv-tts/bin/python /workspace/gemma/run_gemma.py ...
  Gemma, vLLM (venv-vllm): VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1  (both already set inside run_gemma_vllm.py)
  pip/uv: PIP_NO_CACHE_DIR=1 or UV_CACHE_DIR=/dev/shm/... keeps caches off the volume
  Never run venv-pp's `pip install /workspace/personaplex/moshi/.` again (it downgrades torch below Blackwell support).
ENV
echo "== summary: $ok OK, $fail FAIL"
[ $fail = 0 ]
