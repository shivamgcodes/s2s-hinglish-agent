import importlib, traceback, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("dev", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul ok", float((a @ a).float().abs().mean()) > 0)
for m in ["moshi", "moshi.models.loaders", "moshi.models.lm", "moshi.offline", "sentencepiece", "sphn", "safetensors", "fire", "simple_parsing", "yaml", "tensorboard", "matplotlib", "pyloudnorm",
          "finetune.args", "finetune.loss", "finetune.data.dataset", "finetune.data.interleaver", "finetune.wrapped_model", "finetune.checkpointing", "train"]:
    try:
        mod = importlib.import_module(m); print("OK  ", m, getattr(mod, "__file__", ""))
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e).splitlines()[0])
from moshi.models import loaders
print("has CheckpointInfo:", hasattr(loaders, "CheckpointInfo"))
