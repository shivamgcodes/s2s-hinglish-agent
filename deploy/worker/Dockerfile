# syntax=docker/dockerfile:1.7
# S2S serverless GPU worker image (DESIGN.md section 6.2). One image, both modes (S2S_MODE=lb | queue).
#
# 2026-10-06 "BAKE" (DECISIONS D-BAKE; supersedes "WEIGHTS"): ALL weights are baked into the image, which is built by
# GitHub Actions (.github/workflows/build-worker.yml in the s2s-hinglish-agent monorepo, D-MONOREPO) and pushed to a
# PRIVATE Docker Hub repo; RunPod uses "Deploy from a Docker image" with a registry credential. No model cache, no
# network volume, no HF token on the endpoint. Build context = the repo root, -f deploy/worker/Dockerfile (D-MONOREPO-LAYOUT
# 2026-10-07: deploy/{worker,common} + packages/{needle_router,personaplex_lora}; research/ never enters
# the context, see deploy/worker/Dockerfile.dockerignore). In-image layout unchanged: /opt/s2s/{worker,common}, plus
# /opt/s2s/packages/ (the shared code, D-SINGLE-SOURCE).
#   - PersonaPlex 7B (gated, nvidia/personaplex-7b-v1, pinned revision): only the 4 files the engine loads
#     (model.safetensors, the Mimi tokenizer, the SPM model, voices.tgz -> pre-extracted voices/) -> /opt/pp (S2S_PP_DIR)
#   - V4_A2 step-600 LoRA (shivamgupta/personaplex-hinglish-v4-lora, root files), Needle v2 + N1 .cact
#     (shivamgupta/needle-hinglish-router-v2: root tuned_full.cact + n1/tuned_full.cact) and libneedle.so (from the
#     public Cactus-Compute/needle3 wheel): fetched by worker/fetch_assets.py with S2S_ASSETS_FROM_HF=1, ONLY the files
#     listed in worker/assets_manifest.json (each pinned to a revision + md5; checkpoint folders are forbidden), into
#     /opt/s2s/assets + the cactus-needle cache dir; the fetched list + bytes are written to /opt/s2s/assets/fetched.json.
#     At runtime fetch_assets.py finds them "in place" and does nothing.
#   - Trelis/whisper-hinglish-preview (public, pinned revision) -> /opt/hf-asr/trelis
# GitHub workflow: BuildKit builds + pushes :base-<sha12> with BAKE_PP=0, then crane appends /opt/pp as one layer
# (see .github/workflows/build-worker.yml). A plain local build (BAKE_PP=1) bakes everything in one go.
# The HF token is a BuildKit SECRET (docker build --secret id=hf_token,env=HF_TOKEN or ,src=FILE): it is mounted at
# /run/secrets/hf_token only for the two RUNs that download, never an ARG/ENV, never in a layer, and those RUNs do not
# use `set -x`. Runtime is fully offline (HF_HUB_OFFLINE=1).
# Local build elsewhere (repo root): DOCKER_BUILDKIT=1 docker build -f deploy/worker/Dockerfile --secret id=hf_token,env=HF_TOKEN -t s2s-worker .
#
# Not in any layer: the HF token, the RunPod key, S2S_SESSION_SECRET (all runtime env / build secret only).
# The image is PRIVATE because it contains the gated PersonaPlex weights and the private adapter: never push it to a
# public repo.
#
# ARGs (build args; the workflow passes none, so the defaults below are what gets built):
#   TORCH_INDEX=cu130 (default, as tested on DEP1; endpoint needs the CUDA 13.0+ filter) | cu128 (UNTESTED: wider host
#               pool; verify on a pod first, DESIGN R5). cu128 ALSO needs CUDA_BASE=nvidia/cuda:12.8.2-base-ubuntu24.04:
#               the 13.0.3 base sets NVIDIA_REQUIRE_CUDA=cuda>=13.0, so older drivers would be refused.
#   TRELIS_BF16=0 (default: fp32 snapshot, 5.8 GB, as on DEP1) | 1 (convert to bf16 at build time, about 2.9 GB;
#               asr_service loads bf16 on cuda anyway; UNTESTED). Saves ~3 GB of image/pull.
#   PP_REVISION = the PersonaPlex snapshot that DEP1 and the GPU smoke ran (fdaf4090...).
#   BAKE_PP=1 (default: PersonaPlex downloaded in this build) | 0 (skipped: the GitHub workflow builds the base with 0
#               and then APPENDS /opt/pp as one streamed layer with crane, because on the 72 GB runner disk a BuildKit build
#               cannot hold the 16 GB layer and its compressed blob at the same time; DECISIONS D-BAKE attempt 1).

ARG CUDA_BASE=nvidia/cuda:13.0.3-base-ubuntu24.04
FROM ${CUDA_BASE}

ARG TORCH_INDEX=cu130
ARG TRELIS_REPO=Trelis/whisper-hinglish-preview
ARG TRELIS_REVISION=eab1188fd2d0e91f2584229b32b3bfe1901c896c
ARG TRELIS_BF16=0
ARG UV_VERSION=0.9.0
ARG PP_REPO=nvidia/personaplex-7b-v1
ARG PP_REVISION=fdaf4090a61cb315c138a1faee287ffd6c716309
ARG BAKE_PP=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=hardlink \
    UV_COMPILE_BYTECODE=1 \
    HOME=/root

# apt: libopus (sphn Opus), ffmpeg (DEP1 build_envs.sh), curl (stack.sh health polls), ca-certificates,
# gcc + libc6-dev: moshi wraps gating/rope/seanet in torch_compile_lazy -> inductor/Triton compile a C launcher at the
# first warmup and fail with "Failed to find C compiler" without one (runpod2 has gcc; the -base- image does not).
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends libopus0 libopus-dev ffmpeg curl ca-certificates gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# uv (static binary) + python 3.11 / 3.12 managed by uv under /opt/uv-python (the venvs point there)
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
 && uv python install 3.11 3.12

WORKDIR /opt/s2s
COPY deploy/worker/requirements-pp.txt deploy/worker/requirements-asr.txt deploy/worker/requirements-needle.txt deploy/worker/requirements-rp.txt /opt/s2s/worker/
COPY deploy/worker/vendor/moshi /opt/s2s/worker/vendor/moshi

# (TORCH_INDEX != cu130, untested: torch pulls its own CUDA wheels; the cu13 nvidia/cuda pins and +cu130 tags are dropped)
# venv-pp + venv-asr in ONE layer with ONE uv cache: UV_LINK_MODE=hardlink stores torch (+ nvidia libs) once
# (DESIGN 6.2 "One torch copy"). The cache is a plain dir inside this RUN (a --mount cache would break the hardlinks)
# and is deleted at the end of the same RUN, so the hardlinked files are the only copy in the layer.
RUN set -eux; \
    export UV_CACHE_DIR=/opt/uv-cache; \
    IDX="--index-url https://download.pytorch.org/whl/${TORCH_INDEX} --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"; \
    if [ "${TORCH_INDEX}" = cu130 ]; then \
      cp worker/requirements-pp.txt /tmp/req-pp.txt; cp worker/requirements-asr.txt /tmp/req-asr.txt; \
    else \
      for f in pp asr; do grep -vE '^(nvidia-|cuda-|torch==|torchaudio==|triton==)' worker/requirements-$f.txt > /tmp/req-$f.txt; \
        echo "torch==2.14.1" >> /tmp/req-$f.txt; echo "torchaudio==2.11.0" >> /tmp/req-$f.txt; done; \
    fi; \
    uv venv -p 3.11 /opt/venv-pp; \
    uv pip install -p /opt/venv-pp/bin/python $IDX -r /tmp/req-pp.txt; \
    uv pip install -p /opt/venv-pp/bin/python --no-deps /opt/s2s/worker/vendor/moshi; \
    uv venv -p 3.11 /opt/venv-asr; \
    uv pip install -p /opt/venv-asr/bin/python $IDX -r /tmp/req-asr.txt; \
    /opt/venv-pp/bin/python -c "import torch, moshi, sphn, aiohttp; print('venv-pp torch', torch.__version__, torch.version.cuda)"; \
    /opt/venv-asr/bin/python -c "import torch, transformers; print('venv-asr torch', torch.__version__, 'transformers', transformers.__version__)"; \
    rm -rf /opt/uv-cache /tmp/req-*.txt /root/.cache/uv

# venv-needle (py3.12, CPU only, no jax: the Needle runtime is a ctypes lib) + venv-rp (runpod SDK, queue handler)
RUN set -eux; \
    export UV_CACHE_DIR=/opt/uv-cache; \
    uv venv -p 3.12 /opt/venv-needle; \
    uv pip install -p /opt/venv-needle/bin/python -r worker/requirements-needle.txt; \
    uv venv -p 3.11 /opt/venv-rp; \
    uv pip install -p /opt/venv-rp/bin/python -r worker/requirements-rp.txt; \
    /opt/venv-needle/bin/python -c "import needle, websockets, numpy; print('venv-needle ok')"; \
    /opt/venv-rp/bin/python -c "import runpod; print('venv-rp runpod', runpod.__version__)"; \
    rm -rf /opt/uv-cache /root/.cache/uv

# Trelis ASR, public, no token, pinned revision -> /opt/hf-asr/trelis (S2S_ASR_DIR); optional bf16 conversion
RUN set -eux; \
    HF_HUB_DISABLE_TELEMETRY=1 /opt/venv-asr/bin/hf download "${TRELIS_REPO}" --revision "${TRELIS_REVISION}" \
        added_tokens.json config.json generation_config.json merges.txt model-00001-of-00002.safetensors \
        model-00002-of-00002.safetensors model.safetensors.index.json normalizer.json preprocessor_config.json \
        special_tokens_map.json tokenizer_config.json vocab.json --local-dir /opt/hf-asr/trelis; \
    rm -rf /opt/hf-asr/trelis/.cache /root/.cache/huggingface; \
    if [ "${TRELIS_BF16}" = 1 ]; then \
      /opt/venv-asr/bin/python -c "import torch; from transformers import WhisperForConditionalGeneration as W, WhisperProcessor as P; m=W.from_pretrained('/opt/hf-asr/trelis', torch_dtype=torch.bfloat16); p=P.from_pretrained('/opt/hf-asr/trelis'); m.save_pretrained('/opt/hf-asr/trelis-bf16'); p.save_pretrained('/opt/hf-asr/trelis-bf16')"; \
      cp /opt/hf-asr/trelis/generation_config.json /opt/hf-asr/trelis-bf16/ 2>/dev/null || true; \
      rm -rf /opt/hf-asr/trelis; mv /opt/hf-asr/trelis-bf16 /opt/hf-asr/trelis; \
    fi; \
    du -sh /opt/hf-asr/trelis

# PersonaPlex 7B (gated): the token comes from the BuildKit secret only (no set -x in this RUN). Own layer (~16.4 GB).
# BAKE_PP=0: skipped here; the workflow appends the same /opt/pp tree (same 4 files, sha256-checked) as a layer.
RUN --mount=type=secret,id=hf_token,required=true \
    set -eu; \
    if [ "${BAKE_PP}" != 1 ]; then echo "BAKE_PP=${BAKE_PP}: PersonaPlex not baked by this build (appended later)"; exit 0; fi; \
    HF_TOKEN="$(tr -d ' \r\n' < /run/secrets/hf_token)"; export HF_TOKEN; \
    export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/tmp/hf-home HF_XET_CACHE=/tmp/hf-xet \
           HF_XET_CHUNK_CACHE_SIZE_BYTES=0; \
    /opt/venv-asr/bin/hf download "${PP_REPO}" --revision "${PP_REVISION}" \
        model.safetensors tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model voices.tgz \
        --local-dir /opt/pp > /dev/null; \
    unset HF_TOKEN; \
    tar -xzf /opt/pp/voices.tgz -C /opt/pp; test -f /opt/pp/voices/NATF2.pt; \
    rm -rf /opt/pp/.cache /tmp/hf-home /tmp/hf-xet /root/.cache/huggingface; \
    ls -la /opt/pp; du -sh /opt/pp

# Assets (V4 adapter, Needle v2 + N1 .cact, libneedle.so) via fetch_assets.py itself, from the model repos listed in
# assets_manifest.json (allowlist + pinned revisions + md5): same dest layout + checks as at runtime. The token (needed
# while the model repos are private) comes from the BuildKit secret only (no set -x).
COPY deploy/worker/paths.py deploy/worker/fetch_assets.py deploy/worker/assets_manifest.json /opt/s2s/worker/
RUN --mount=type=secret,id=hf_token,required=true \
    set -eu; \
    HF_TOKEN="$(tr -d ' \r\n' < /run/secrets/hf_token)"; export HF_TOKEN; \
    export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/tmp/hf-home HF_XET_CACHE=/tmp/hf-xet \
           HF_XET_CHUNK_CACHE_SIZE_BYTES=0 S2S_ROOT=/opt/s2s S2S_ASSETS=/opt/s2s/assets S2S_TMP=/tmp/s2s S2S_ASSETS_FROM_HF=1; \
    /opt/venv-asr/bin/python /opt/s2s/worker/fetch_assets.py; \
    unset HF_TOKEN S2S_ASSETS_FROM_HF; \
    test -s /opt/s2s/assets/fetched.json; cat /opt/s2s/assets/fetched.json; \
    rm -rf /tmp/s2s /tmp/hf-home /tmp/hf-xet /root/.cache/huggingface; \
    S2S_ROOT=/opt/s2s /opt/venv-asr/bin/python /opt/s2s/worker/fetch_assets.py --check; \
    du -sh /opt/s2s/assets /root/.cache/cactus-needle

# The Needle engine must resolve to the baked, md5-checked lib WITHOUT network (the package's own download is no longer
# run at build time; D-ASSET-SWITCH 2026-10-07).
RUN --network=none NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 /opt/venv-needle/bin/python -c "import needle; p=needle._library_path(3); print('needle lib', p); assert p == '/root/.cache/cactus-needle/v3/3.0.2/libneedle.so', p"

# code last (changes most often; the weight layers above stay cached when only code changes)
# (explicit dirs: a plain `COPY worker` would put worker/build into a second layer)
COPY deploy/common /opt/s2s/common
COPY deploy/worker/paths.py deploy/worker/resolve_models.py deploy/worker/fetch_assets.py deploy/worker/assets_manifest.json deploy/worker/rp_handler.py deploy/worker/entrypoint.sh deploy/worker/stack.sh /opt/s2s/worker/
COPY deploy/worker/server /opt/s2s/worker/server
COPY deploy/worker/router /opt/s2s/worker/router
COPY packages/personaplex_lora /opt/s2s/packages/personaplex_lora
COPY packages/needle_router /opt/s2s/packages/needle_router
RUN set -eux; \
    chmod +x /opt/s2s/worker/entrypoint.sh /opt/s2s/worker/stack.sh; \
    mkdir -p /opt/s2s/assets; test -f /opt/s2s/common/data/records_v4.json; command -v gcc; \
    S2S_ROOT=/opt/s2s /opt/venv-asr/bin/python /opt/s2s/worker/fetch_assets.py --check; \
    if [ "${BAKE_PP}" = 1 ]; then S2S_PP_DIR=/opt/pp HF_HUB_OFFLINE=1 /opt/venv-pp/bin/python /opt/s2s/worker/resolve_models.py --no-write; fi; \
    test -f "$(/opt/venv-pp/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')/Python.h"; \
    /opt/venv-pp/bin/python /opt/s2s/worker/paths.py; \
    /opt/venv-needle/bin/python -c "import sys; sys.path.insert(0,'/opt/s2s/worker/router'); import router_core, trigger, stub_tools; print('router imports ok')"; \
    S2S_ROOT=/opt/s2s /opt/venv-needle/bin/python -c "import sys; sys.path.insert(0,'/opt/s2s/worker/router'); import router_core as rc; m,w=rc.load_router('v2'); import os, n1path, numconv; assert n1path.N1_DIR=='/opt/s2s/packages/needle_router/n1', n1path.N1_DIR; assert m.__file__=='/opt/s2s/packages/needle_router/v2/router_v2.py', m.__file__; assert os.path.isfile(w), w; assert 'hindi_share' not in sys.modules; m1,w1=rc.load_router('n1'); assert m1.__file__=='/opt/s2s/packages/needle_router/n1/router.py', m1.__file__; assert os.path.isfile(w1), w1; print('router v2 + n1 imports ok', m.__file__, w, m1.__file__, w1, sorted(m.tools_v2.AGENT_TYPES)[:2])"; \
    /opt/venv-pp/bin/python -c "import sys; sys.path.insert(0,'/opt/s2s/common'); import session; assert session.ascii_prompt.__module__=='role_prompt'; import role_prompt; assert role_prompt.__file__=='/opt/s2s/packages/personaplex_lora/role_prompt.py', role_prompt.__file__; c=session.session_config('food_01','g2'); assert c['voice']=='NATM1.pt'; print('session + role_prompt ok')"; \
    test -f /opt/s2s/packages/personaplex_lora/trainer/merge_lora.py && test -f /opt/s2s/packages/personaplex_lora/trainer/voice_codes/NATF2.columns.pt

ENV S2S_ROOT=/opt/s2s \
    S2S_MODE=lb \
    S2S_ASR_HF_HOME=/opt/hf-asr \
    S2S_ASR_DIR=/opt/hf-asr/trelis \
    S2S_PY_PP=/opt/venv-pp/bin/python \
    S2S_PY_ASR=/opt/venv-asr/bin/python \
    S2S_PY_NEEDLE=/opt/venv-needle/bin/python \
    S2S_PY_RP=/opt/venv-rp/bin/python \
    S2S_PP_DIR=/opt/pp \
    S2S_VOICES=/opt/pp/voices \
    S2S_ASSETS=/opt/s2s/assets \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NEEDLE_TELEMETRY=0 \
    DO_NOT_TRACK=1 \
    JAX_PLATFORMS=cpu

# LB: PORT defaults to 80 ([LB-OV]); queue: 8765 (expose it as a TCP port on the endpoint). Set by entrypoint.sh.
EXPOSE 80 8765
ENTRYPOINT ["bash", "/opt/s2s/worker/entrypoint.sh"]
