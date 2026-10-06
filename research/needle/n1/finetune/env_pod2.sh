# Source on pod2 before any needle finetune/build/val step (GPU, RTX 3090).
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN JAX_PLATFORMS CUDA_VISIBLE_DEVICES
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
N1_ROOT=${N1_ROOT:-/root/needle}; NEEDLE_VENV=${NEEDLE_VENV:-/root/venv-needle}
export HF_HOME=$N1_ROOT/finetune/hf_home
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH=$NEEDLE_VENV/bin:$PATH
CKPT=$N1_ROOT/finetune/needle3.safetensors
