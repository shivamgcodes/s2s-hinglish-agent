# Source on runpod2 before any needle finetune/build/val step (GPU: RTX 4090 24 GB sm_89 since 2026-10-05 pod switch; was L4).
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN JAX_PLATFORMS CUDA_VISIBLE_DEVICES
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
N2_ROOT=${N2_ROOT:-/root/n2}; NEEDLE_VENV=${NEEDLE_VENV:-/root/venv-needle}
export HF_HOME=$N2_ROOT/needle_n1/finetune/hf_home
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH=$NEEDLE_VENV/bin:$PATH
CKPT=$N2_ROOT/needle_n1/finetune/needle3.safetensors
