#!/bin/bash
# Builds /workspace/venv-ft. Never touches venv-pp or /workspace/personaplex.
set -ex
export UV_LINK_MODE=copy UV_CACHE_DIR=/workspace/.uvcache-ft
V=/workspace/venv-ft
[ -x $V/bin/python ] || uv venv -p /usr/bin/python3.11 $V
PY=$V/bin/python
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.14.1+cu130" "torchaudio==2.11.0+cu130" "triton==3.8.0"
uv pip install --python $PY \
  "numpy==2.1.3" "safetensors==0.4.5" "huggingface-hub==0.24.7" "einops==0.7.0" "sentencepiece==0.2.0" \
  "sphn==0.1.12" fire simple-parsing pyyaml tqdm tensorboard matplotlib pyloudnorm pandas "accelerate==1.15.0" "aiohttp>=3.10.5,<3.11"
# PersonaPlex moshi + moshi-finetune via .pth (no writes into either tree)
SP=$($PY -c "import site;print(site.getsitepackages()[0])")
echo /workspace/personaplex/moshi > $SP/personaplex_moshi.pth
echo /workspace/moshi-finetune > $SP/moshi_finetune.pth
$PY -c "import torch,moshi,finetune;print('torch',torch.__version__,torch.version.cuda);print('moshi',moshi.__file__);print('finetune',finetune.__file__)"
$V/bin/pip --version || uv pip install --python $PY pip
echo BUILD_OK
