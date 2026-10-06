#!/usr/bin/env bash
# Run the whole Hinglish full-duplex demo on ONE local NVIDIA GPU: no RunPod, no Hugging Face Space (DECISIONS D-LOCAL).
# The worker stack (PersonaPlex 7B + V4 LoRA, Trelis ASR, Needle v2 router) runs as plain processes in uv venvs, and the
# page backend (space/) runs next to it in S2S_MODE=local and serves the web page (client/, built by setup).
#
#   ./run_local.sh setup     venvs + model downloads + web client build (once; ~30 GB disk; needs HF_TOKEN, see below)
#   ./run_local.sh run       start the worker + the page; open http://localhost:7860 ; Ctrl-C stops everything
#   ./run_local.sh check     print what setup found / is missing (no downloads)
#
# Requirements: Linux x86_64, one NVIDIA GPU with >= 24 GB (32 GB comfortable; the stack uses ~23.4 GB), a driver for
# CUDA 12.8+ (cu128 wheels) or 13.0+ (cu130), apt packages libopus0 ffmpeg gcc curl, and Node.js >= 18 with npm (setup
# builds the web page from client/ once; or set S2S_STATIC to an already built client). One checkout of
# github.com/shivamgcodes/s2s-hinglish-agent holds everything (worker/, space/, client/, common/).
# HF_TOKEN: your own Hugging Face read token. Accept the NVIDIA Open Model License on
# https://huggingface.co/nvidia/personaplex-7b-v1 first (the repo is gated). The token is only passed to the `hf`
# downloader during setup; it is not stored and not needed by `run`.
#
# Env (all optional):
#   S2S_LOCAL_HOME   where venvs/models/logs go (default <repo>/.local-run)
#   TORCH_INDEX      cu128 | cu130 (default: cu130 if nvidia-smi reports CUDA >= 13.0, else cu128)
#   S2S_PP_DIR       an existing PersonaPlex snapshot dir (model.safetensors, tokenizer files, voices/ or voices.tgz)
#   S2S_ASR_DIR      an existing Trelis/whisper-hinglish-preview snapshot dir
#   S2S_LORA_REPO    HF repo with the V4 LoRA (default shivamgupta/personaplex-hinglish-v4-lora; files config.json +
#                    lora.safetensors under S2S_LORA_SUBDIR, default the repo root)
#   S2S_ROUTER_REPO  HF repo with the Needle v2 weights (default shivamgupta/needle-hinglish-router-v2; file
#                    S2S_ROUTER_FILE, default tuned_full.cact)
#   S2S_ADAPTER / S2S_NEEDLE_V2_WEIGHTS   use local files instead of the two repos above
#   WORKER_PORT (8000), SPACE_PORT (7860), HOST (127.0.0.1; 0.0.0.0 to expose the page on the LAN)
#   S2S_SPACE_DIR    the page backend dir (default <repo>/space)
#   S2S_STATIC       a built web client dir (default <repo>/client/dist, built by setup)
#   S2S_TURN_FILL_DEFAULT  ticker | off (the page's top-right toggle changes it at runtime)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$HERE/worker/stack.sh" ]; then ROOT=$HERE; W=$HERE/worker          # repo root (monorepo, D-MONOREPO)
else echo "cannot find worker/stack.sh next to $0" >&2; exit 2; fi
CMD=${1:-help}
H=${S2S_LOCAL_HOME:-$ROOT/.local-run}
PP_REPO=nvidia/personaplex-7b-v1
PP_REVISION=fdaf4090a61cb315c138a1faee287ffd6c716309          # = Dockerfile ARG PP_REVISION (the tested snapshot)
TRELIS_REPO=Trelis/whisper-hinglish-preview
TRELIS_REVISION=eab1188fd2d0e91f2584229b32b3bfe1901c896c
LORA_REPO=${S2S_LORA_REPO:-shivamgupta/personaplex-hinglish-v4-lora}
LORA_SUBDIR=${S2S_LORA_SUBDIR:-}
ROUTER_REPO=${S2S_ROUTER_REPO:-shivamgupta/needle-hinglish-router-v2}
ROUTER_FILE=${S2S_ROUTER_FILE:-tuned_full.cact}
PP_DIR=${S2S_PP_DIR:-$H/models/personaplex}
ASR_DIR=${S2S_ASR_DIR:-$H/models/trelis}
ADAPTER=${S2S_ADAPTER:-$H/assets/v4_adapter}
NEEDLE_V2=${S2S_NEEDLE_V2_WEIGHTS:-$H/assets/needle_v2/tuned_full.cact}
WORKER_PORT=${WORKER_PORT:-8000}; SPACE_PORT=${SPACE_PORT:-7860}; HOST=${HOST:-127.0.0.1}
if [ -z "${S2S_SPACE_DIR:-}" ] && [ -f "$ROOT/space/app/main.py" ]; then S2S_SPACE_DIR=$ROOT/space; fi
CLIENT=$ROOT/client
static_dir() {   # the built web client: S2S_STATIC, else client/dist (built by setup)
  if [ -n "${S2S_STATIC:-}" ]; then echo "$S2S_STATIC"; else echo "$CLIENT/dist"; fi
}
V=$H/venvs
say() { echo "[run_local] $*"; }
die() { echo "[run_local] ERROR: $*" >&2; exit 1; }

torch_index() {
  if [ -n "${TORCH_INDEX:-}" ]; then echo "$TORCH_INDEX"; return; fi
  local cv; cv=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1/p' | head -1)
  if [ -n "$cv" ] && [ "$cv" -ge 13 ]; then echo cu130; else echo cu128; fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then UV=$(command -v uv); return; fi
  UV=$H/bin/uv
  [ -x "$UV" ] && return
  say "installing uv 0.9.0 into $H/bin"
  mkdir -p "$H/bin"
  curl -LsSf https://astral.sh/uv/0.9.0/install.sh | env UV_INSTALL_DIR="$H/bin" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
}

hf_get() {   # hf_get <python> <repo> <revision|""> <local dir> <files...>   (token from HF_TOKEN, never printed)
  local py=$1 repo=$2 rev=$3 dir=$4; shift 4
  local revarg=(); [ -n "$rev" ] && revarg=(--revision "$rev")
  HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 "$(dirname "$py")/hf" download "$repo" "${revarg[@]}" "$@" --local-dir "$dir" >/dev/null
}

check() {
  local bad=0 f
  for f in "$V/pp/bin/python" "$V/asr/bin/python" "$V/needle/bin/python" "$V/space/bin/python"; do
    [ -x "$f" ] && echo "  ok      $f" || { echo "  MISSING $f"; bad=1; }
  done
  for f in model.safetensors tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model voices/NATF2.pt; do
    [ -e "$PP_DIR/$f" ] && echo "  ok      $PP_DIR/$f" || { echo "  MISSING $PP_DIR/$f"; bad=1; }
  done
  for f in "$ASR_DIR/config.json" "$ASR_DIR/model.safetensors.index.json" "$ADAPTER/config.json" "$ADAPTER/lora.safetensors" "$NEEDLE_V2"; do
    [ -e "$f" ] && echo "  ok      $f" || { echo "  MISSING $f"; bad=1; }
  done
  [ -n "${S2S_SPACE_DIR:-}" ] && [ -f "$S2S_SPACE_DIR/app/main.py" ] && echo "  ok      page backend: $S2S_SPACE_DIR" \
    || { echo "  MISSING the page backend (space/app/main.py; or set S2S_SPACE_DIR)"; bad=1; }
  [ -f "$(static_dir)/index.html" ] && echo "  ok      web client: $(static_dir)" \
    || { echo "  MISSING the built web client $(static_dir)/index.html (setup builds it with npm)"; bad=1; }
  return $bad
}

setup() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: an NVIDIA GPU + driver is required"
  command -v gcc >/dev/null || die "gcc not found (torch.compile needs a C compiler): sudo apt install gcc"
  ldconfig -p 2>/dev/null | grep -q libopus || say "WARNING: libopus not found (sudo apt install libopus0)"
  [ -n "${S2S_SPACE_DIR:-}" ] && [ -f "$S2S_SPACE_DIR/app/main.py" ] || die "no page backend at $ROOT/space (run this from a full checkout, or set S2S_SPACE_DIR)"
  if [ ! -f "$(static_dir)/index.html" ]; then   # web client (React + Vite) -> client/dist, once
    command -v npm >/dev/null || die "the web client is not built and npm is not installed: install Node.js >= 18 (e.g. https://nodejs.org), or set S2S_STATIC to a built client"
    say "web client: npm ci && npm run build in $CLIENT"
    ( cd "$CLIENT" && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null ) || die "client build failed (cd client && npm ci && npm run build)"
  fi
  ensure_uv
  local idx; idx=$(torch_index)
  say "torch index $idx; installing into $H"
  mkdir -p "$V" "$H/tmp"
  export UV_CACHE_DIR=$H/uv-cache UV_LINK_MODE=hardlink UV_PYTHON_INSTALL_DIR=$H/python
  local IDX=(--index-url "https://download.pytorch.org/whl/$idx" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match)
  local f
  for f in pp asr; do
    if [ "$idx" = cu130 ]; then cp "$W/requirements-$f.txt" "$H/tmp/req-$f.txt"
    else   # same idea as the Dockerfile's non-cu130 branch; the newest cu128 torch is 2.11.0 (tested on an RTX 5090, driver 570)
      grep -vE '^(nvidia-|cuda-|torch==|torchaudio==|triton==|setuptools==)' "$W/requirements-$f.txt" > "$H/tmp/req-$f.txt"
      printf 'torch==2.11.0\ntorchaudio==2.11.0\n' >> "$H/tmp/req-$f.txt"
    fi
  done
  if [ ! -x "$V/pp/bin/python" ] || ! "$V/pp/bin/python" -c "import torch, moshi, sphn" 2>/dev/null; then
    say "venv pp (PersonaPlex / moshi)"
    "$UV" venv -q --allow-existing -p 3.11 "$V/pp"
    "$UV" pip install -q -p "$V/pp/bin/python" "${IDX[@]}" -r "$H/tmp/req-pp.txt"
    "$UV" pip install -q -p "$V/pp/bin/python" --no-deps "$W/vendor/moshi"
  fi
  if [ ! -x "$V/asr/bin/python" ] || ! "$V/asr/bin/python" -c "import torch, transformers" 2>/dev/null; then
    say "venv asr (Trelis Whisper)"
    "$UV" venv -q --allow-existing -p 3.11 "$V/asr"
    "$UV" pip install -q -p "$V/asr/bin/python" "${IDX[@]}" -r "$H/tmp/req-asr.txt"
  fi
  if [ ! -x "$V/needle/bin/python" ] || ! "$V/needle/bin/python" -c "import needle, websockets" 2>/dev/null; then
    say "venv needle (router, CPU)"
    "$UV" venv -q --allow-existing -p 3.12 "$V/needle"
    "$UV" pip install -q -p "$V/needle/bin/python" -r "$W/requirements-needle.txt"
  fi
  if [ ! -x "$V/space/bin/python" ] || ! "$V/space/bin/python" -c "import aiohttp" 2>/dev/null; then
    say "venv space (web page backend)"
    "$UV" venv -q --allow-existing -p 3.11 "$V/space"
    "$UV" pip install -q -p "$V/space/bin/python" -r "$S2S_SPACE_DIR/requirements.txt"
  fi
  rm -rf "$H/uv-cache" "$H/tmp"
  "$V/pp/bin/python" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU (driver too old for this torch index? set TORCH_INDEX=cu128)'; print('[run_local] venv pp: torch', torch.__version__, torch.cuda.get_device_name(0))"
  # the Needle native lib: cactus-needle downloads it on first use; fetch it now (the stack runs offline)
  NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 "$V/needle/bin/python" -c "import needle; print('[run_local] needle lib', needle._library_path(3))" \
    || say "WARNING: Needle native lib not fetched now; the router fetches it on its first call"

  if [ ! -e "$PP_DIR/model.safetensors" ]; then
    [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set: PersonaPlex is gated (accept the license at https://huggingface.co/$PP_REPO, then export HF_TOKEN=<your read token>)"
    say "PersonaPlex 7B ($PP_REPO @ ${PP_REVISION:0:8}, ~16.4 GB) -> $PP_DIR"
    hf_get "$V/asr/bin/python" "$PP_REPO" "$PP_REVISION" "$PP_DIR" model.safetensors \
      tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model voices.tgz \
      || die "PersonaPlex download failed: is the license accepted for this token's account?"
  fi
  if [ ! -e "$PP_DIR/voices/NATF2.pt" ]; then
    [ -w "$PP_DIR" ] || die "$PP_DIR/voices/ is missing and $PP_DIR is not writable: extract voices.tgz there yourself"
    tar -xzf "$PP_DIR/voices.tgz" -C "$PP_DIR"
  fi
  if [ ! -e "$ASR_DIR/model.safetensors.index.json" ]; then
    say "Trelis whisper-hinglish-preview (public, ~5.8 GB) -> $ASR_DIR"
    hf_get "$V/asr/bin/python" "$TRELIS_REPO" "$TRELIS_REVISION" "$ASR_DIR" added_tokens.json config.json \
      generation_config.json merges.txt model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
      model.safetensors.index.json normalizer.json preprocessor_config.json special_tokens_map.json \
      tokenizer_config.json vocab.json
  fi
  if [ ! -e "$ADAPTER/lora.safetensors" ]; then
    say "V4 LoRA ($LORA_REPO${LORA_SUBDIR:+/$LORA_SUBDIR}, ~370 MB) -> $ADAPTER"
    local p=${LORA_SUBDIR:+$LORA_SUBDIR/}
    hf_get "$V/asr/bin/python" "$LORA_REPO" "" "$H/dl/lora" "${p}config.json" "${p}lora.safetensors"
    mkdir -p "$ADAPTER"; mv "$H/dl/lora/${p}config.json" "$H/dl/lora/${p}lora.safetensors" "$ADAPTER/"
  fi
  if [ ! -e "$NEEDLE_V2" ]; then
    say "Needle v2 router weights ($ROUTER_REPO/$ROUTER_FILE, ~63 MB) -> $NEEDLE_V2"
    hf_get "$V/asr/bin/python" "$ROUTER_REPO" "" "$H/dl/router" "$ROUTER_FILE"
    mkdir -p "$(dirname "$NEEDLE_V2")"; mv "$H/dl/router/$ROUTER_FILE" "$NEEDLE_V2"
  fi
  rm -rf "$H/dl"
  check >/dev/null && say "setup complete. Start with: $0 run" || { check; die "setup incomplete (see MISSING above)"; }
}

run() {
  check >/dev/null || { check; die "run '$0 setup' first"; }
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge 22000 ] || say "WARNING: only ${free} MiB free on GPU 0; the stack needs ~23.4 GB"
  mkdir -p "$H/logs" "$H/s2s-tmp"
  [ -s "$H/session_secret" ] || (umask 077; "$V/space/bin/python" -c "import secrets; print(secrets.token_urlsafe(48))" > "$H/session_secret")
  local SECRET; SECRET=$(cat "$H/session_secret")
  local STATIC; STATIC=$(static_dir)
  [ -f "$STATIC/index.html" ] || die "no built web client in $STATIC (run '$0 setup', or cd client && npm ci && npm run build)"

  say "worker: http://127.0.0.1:$WORKER_PORT (logs $H/logs); first start loads + compiles for ~2-5 min"
  env S2S_MODE=lb PORT="$WORKER_PORT" S2S_INTERNAL_PORT="${S2S_INTERNAL_PORT:-8999}" S2S_ASR_PORT="${S2S_ASR_PORT:-8996}" \
    S2S_ROUTER_PORT="${S2S_ROUTER_PORT:-8995}" S2S_SESSION_SECRET="$SECRET" S2S_AUDIENCE=local RUNPOD_POD_ID=local-gpu \
    S2S_SKIP_FETCH_ASSETS=1 S2S_ADAPTER="$ADAPTER" S2S_NEEDLE_V2_WEIGHTS="$NEEDLE_V2" S2S_ROUTER=v2 \
    S2S_PP_DIR="$PP_DIR" S2S_VOICES="$PP_DIR/voices" S2S_ASR_DIR="$ASR_DIR" S2S_ASR_HF_HOME="$H/hf-asr" \
    S2S_PY_PP="$V/pp/bin/python" S2S_PY_ASR="$V/asr/bin/python" S2S_PY_NEEDLE="$V/needle/bin/python" \
    S2S_TMP="$H/s2s-tmp" S2S_LOGS="$H/logs" S2S_ASSETS="$H/assets" PYTHONDONTWRITEBYTECODE=1 \
    setsid bash "$W/stack.sh" > "$H/logs/stack.log" 2>&1 &
  WPID=$!
  say "page:   http://$HOST:$SPACE_PORT  (backend log $H/logs/space.log)"
  ( cd "$S2S_SPACE_DIR" && exec env S2S_MODE=local S2S_WORKER_URL="http://127.0.0.1:$WORKER_PORT" \
      S2S_SESSION_SECRET="$SECRET" S2S_AUDIENCE=local HOST="$HOST" PORT="$SPACE_PORT" S2S_STATIC="$STATIC" \
      S2S_PASSCODE="${S2S_PASSCODE:-}" WAKE_TIMEOUT_S="${WAKE_TIMEOUT_S:-1200}" CALL_MAX_S="${CALL_MAX_S:-300}" \
      S2S_TURN_FILL_DEFAULT="${S2S_TURN_FILL_DEFAULT:-ticker}" PYTHONDONTWRITEBYTECODE=1 \
      "$V/space/bin/python" -u -m app.main ) > "$H/logs/space.log" 2>&1 &
  SPID=$!
  stop() {
    trap - INT TERM EXIT
    say "stopping"
    kill -TERM -- "-$WPID" 2>/dev/null || kill -TERM "$WPID" 2>/dev/null || true
    kill -TERM "$SPID" 2>/dev/null || true
    wait "$WPID" "$SPID" 2>/dev/null || true
    say "stopped"
    exit 0
  }
  trap stop INT TERM EXIT
  local i code
  for i in $(seq 1 600); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "127.0.0.1:$WORKER_PORT/ping" || true)
    [ "$code" = 200 ] && break
    kill -0 "$WPID" 2>/dev/null || { tail -30 "$H/logs/stack.log"; die "the worker stack exited (log above)"; }
    kill -0 "$SPID" 2>/dev/null || { tail -30 "$H/logs/space.log"; die "the page backend exited (log above)"; }
    [ $((i % 15)) = 0 ] && say "loading... $(curl -s --max-time 3 "127.0.0.1:$WORKER_PORT/status" | head -c 160)"
    sleep 2
  done
  [ "$code" = 200 ] || die "the worker did not become ready in 20 min (see $H/logs/stack.log)"
  say "READY. Open http://localhost:$SPACE_PORT , pick a record, press Connect. Use headphones. Ctrl-C stops."
  wait -n "$WPID" "$SPID" 2>/dev/null || true
  say "a process exited; see $H/logs/ (stopping the rest)"
}

case "$CMD" in
  setup) setup ;;
  run) run ;;
  check) check && say "everything in place" ;;
  *) sed -n 2,30p "$0"; exit 2 ;;
esac
