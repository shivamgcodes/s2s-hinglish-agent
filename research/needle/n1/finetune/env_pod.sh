# Source on the pod before any needle finetune/build/val step (F-section).
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
N1_ROOT=${N1_ROOT:-/workspace/hinglish/needle}
export HF_HOME=$N1_ROOT/finetune/hf_home
export PATH=/workspace/venv-tts/bin:$PATH
CKPT=$N1_ROOT/finetune/needle3.safetensors
