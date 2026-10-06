"""S2S serverless worker: the ONE env-driven path + port layout (DESIGN.md section 1.2).

Stdlib only; imported by every worker process (venv-pp py3.11, venv-asr py3.11, venv-needle py3.12, rp_handler):
    sys.path.insert(0, <worker dir>); import paths

Every DEP1 hard-coded location (the deploy tree, the needle tree, the pod1 workspace tree, the HF cache dir) is
replaced by a name below. Defaults are the image layout (S2S_ROOT=/opt/s2s, this file at /opt/s2s/worker/paths.py);
worker/local/runpod2.env sets them for a run outside docker on runpod2. Nothing here reads a secret.
"""
import os
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent


def _env(name, default):
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


ROOT = Path(_env("S2S_ROOT", str(WORKER_DIR.parent)))                  # /opt/s2s in the image
COMMON = Path(_env("S2S_COMMON", str(ROOT / "common")))                 # s2s_token.py, session.py, data/
ASSETS = Path(_env("S2S_ASSETS", str(ROOT / "assets")))                 # fetched at start (fetch_assets.py): v4_adapter/, tuned_full.cact

# adapter: a dir with config.json + lora.safetensors | a .safetensors | a DEP1 key=value file | '' = base | 'premerged'
ADAPTER = os.environ.get("S2S_ADAPTER", str(ASSETS / "v4_adapter"))    # 2026-10-06: V4_A2 step 600 (was the baked V3 LoRA)
HINGLISH = Path(_env("S2S_HINGLISH", str(WORKER_DIR / "hinglish")))     # trainer/merge_lora.py, infer/lora_merge.py
MERGE_LORA = HINGLISH / "trainer" / "merge_lora.py"
VOICE_CODES = HINGLISH / "trainer" / "voice_codes"
NEEDLE = Path(_env("S2S_NEEDLE", str(WORKER_DIR / "needle")))          # runtime subset of the needle tree
NEEDLE_WEIGHTS = _env("S2S_NEEDLE_WEIGHTS", str(ASSETS / "tuned_full.cact"))   # Needle N1 (S2S_ROUTER=n1)
# D-ROUTER-V2 (2026-10-06): Needle v2 router (default). Code = worker/needle_v2/{schema,numconv} (sibling dirs, path-edit
# copies of hinglish/needle_v2); it uses the N1 runtime subset in NEEDLE (resolver.py, build_data.py, romanise.py).
ROUTER = _env("S2S_ROUTER", "v2").strip().lower()                       # v2 | n1 (rollback)
NEEDLE_V2 = Path(_env("S2S_NEEDLE_V2", str(WORKER_DIR / "needle_v2")))
NEEDLE_V2_WEIGHTS = _env("S2S_NEEDLE_V2_WEIGHTS", str(ASSETS / "needle_v2" / "tuned_full.cact"))
RECORDS = _env("S2S_RECORDS", str(COMMON / "data" / "records_v4.json"))
NEEDLE_RECORDS = _env("S2S_NEEDLE_RECORDS", str(NEEDLE / "src" / "records.json"))   # V3 records (offline eval)

# PersonaPlex (resolve_models.py writes S2S_PP_DIR / S2S_VOICES into MODELS_ENV; stack.sh sources it)
PP_REPO = _env("S2S_PP_REPO", "nvidia/personaplex-7b-v1")
PP_DIR = os.environ.get("S2S_PP_DIR", "")
VOICES = os.environ.get("S2S_VOICES", "")
RUNPOD_VOLUME = Path(_env("S2S_RUNPOD_VOLUME", "/runpod-volume"))       # overridable for tests
MODEL_CACHE = Path(_env("S2S_MODEL_CACHE", str(RUNPOD_VOLUME / "huggingface-cache" / "hub")))   # [CACHE]
VOLUME_HF = Path(_env("S2S_VOLUME_HF", str(RUNPOD_VOLUME / "hf" / "hub")))                       # [VOLCACHE]
TMP = Path(_env("S2S_TMP", "/tmp/s2s"))
MODELS_ENV = Path(_env("S2S_MODELS_ENV", str(TMP / "models.env")))
DOWNLOAD_DIR = Path(_env("S2S_DOWNLOAD_DIR", str(TMP / "hf")))

# ASR (Trelis baked in the image under its own HF home; S2S_ASR_DIR = explicit snapshot dir, option B)
ASR_HF_HOME = _env("S2S_ASR_HF_HOME", "/opt/hf-asr")
ASR_MODEL = _env("S2S_ASR_DIR", "Trelis/whisper-hinglish-preview")

LOGS = Path(_env("S2S_LOGS", str(TMP / "logs")))

# mock engine (tests only; never in the image): DEP1 V3 replays + inputs + the PersonaPlex SPM tokenizer
MOCK_REPLAY = Path(_env("S2S_MOCK_REPLAY", str(ASSETS / "mock" / "replay")))
MOCK_INPUTS = Path(_env("S2S_MOCK_INPUTS", str(ASSETS / "mock" / "inputs")))
MOCK_SPM = _env("S2S_MOCK_SPM", str(Path(PP_DIR) / "tokenizer_spm_32k_3.model") if PP_DIR else "")

# ports (image defaults = DEP1 ports; worker/local/run_local.sh moves them off the live DEP1 demo)
PORT = int(_env("PORT", "80"))                        # public: /ping /status /session/* /api/chat /metrics
INTERNAL_PORT = int(_env("S2S_INTERNAL_PORT", "8999"))
ASR_PORT = int(_env("S2S_ASR_PORT", "8996"))
ROUTER_PORT = int(_env("S2S_ROUTER_PORT", "8995"))


def ensure_common_on_path():
    import sys
    c = str(COMMON)
    if c not in sys.path:
        sys.path.insert(0, c)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper():
            print(f"{k:16s} {v}")
