"""S2S serverless worker: fork of DEP1 server/engine.py (PersonaPlex Engine, DEP1 INTERFACE.md section 8).

    from engine import Engine, FrameOut, InjectPlan
    eng = Engine(adapter=paths.ADAPTER, device="cuda", pp_files={"moshi":..,"mimi":..,"tokenizer":..}, voice_prompt_dir=..)
    sid = eng.start_session(cfg); out = eng.step_frame(pcm1920); ...; eng.end_session()

Changes from DEP1 (DESIGN.md 3.1 "Engine"): explicit file paths instead of hf_hub_download (resolve_models.py finds
them: RunPod model cache / network volume / explicit dir), merge_lora path from paths.py, adapter "premerged" = the
moshi weights already contain the V3 LoRA (DESIGN 6.4 option A; no merge). Sampling, merge, warmup and the step
loop are unchanged.

Same model loading / merge / warmup / prompt phases / per-frame loop as tests/driver.py and moshi/server.py
(model code untouched; moshi.server is never imported because it runs main() at import).
All GPU calls of one Engine must come from ONE thread (the server constructs it on its GPU worker thread).
A serverless worker owns its GPU (no flock); on runpod2 run it via worker/local/run_local.sh (takes the GPU lock).
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

import numpy as np  # noqa: E402
import sentencepiece  # noqa: E402
import torch  # noqa: E402

from moshi.models import LMGen, loaders  # noqa: E402
from moshi.offline import seed_all, warmup, wrap_with_system_tags  # noqa: E402

from core import CONTEXT_MARGIN, FRAME_SIZE, EngineBase, FrameOut, InjectPlan  # noqa: E402,F401

DEFAULT_ADAPTER = paths.ADAPTER
MERGE_LORA = paths.MERGE_LORA
PREMERGED = "premerged"
SAMPLING = dict(use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)  # tests/driver.py values


class SessionAborted(Exception):
    pass


def resolve_adapter(adapter):
    """V3_CKPT is a key=value file (path=...); a dir or .safetensors is used as is; None/'' = base model;
    'premerged' = the moshi weights already carry the adapter (no merge)."""
    if not adapter or adapter == PREMERGED:
        return None
    p = Path(adapter)
    if p.is_file() and p.suffix != ".safetensors":
        kv = dict(line.split("=", 1) for line in p.read_text().splitlines() if "=" in line)
        return kv["path"].strip()
    return str(p)


def pp_files_from_dir(d):
    """{moshi, mimi, tokenizer} file paths inside one PersonaPlex snapshot dir (loaders.* names)."""
    d = Path(d)
    files = {"moshi": d / loaders.MOSHI_NAME, "mimi": d / loaders.MIMI_NAME, "tokenizer": d / loaders.TEXT_TOKENIZER_NAME}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"PersonaPlex files missing: {missing}")
    return {k: str(v) for k, v in files.items()}


def _merge(lm, path):
    spec = importlib.util.spec_from_file_location("merge_lora", MERGE_LORA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.merge_into(lm, path)


class Engine(EngineBase):
    def __init__(self, adapter=DEFAULT_ADAPTER, device="cuda", pp_files=None,
                 voice_prompt_dir=None, sampling=None, smi=True, torch_threads=None):
        t0 = time.time()
        # the container has a ~31-CPU CFS quota (not 256): keep torch's CPU pool small
        torch.set_num_threads(int(torch_threads or os.environ.get("DEP1_TORCH_THREADS", "4")))
        self.device = torch.device(device)
        pp_files = pp_files or pp_files_from_dir(paths.PP_DIR)
        self.pp_files = dict(pp_files)
        seed_all(42424242)  # stock server.py startup seed
        self.voice_prompt_dir = voice_prompt_dir or paths.VOICES
        if not self.voice_prompt_dir or not os.path.isdir(self.voice_prompt_dir):
            raise FileNotFoundError(f"voice prompt dir {self.voice_prompt_dir!r} (run resolve_models.py first)")
        mimi_w = pp_files["mimi"]
        self.mimi = loaders.get_mimi(mimi_w, self.device)
        self.other_mimi = loaders.get_mimi(mimi_w, self.device)
        spm = sentencepiece.SentencePieceProcessor(pp_files["tokenizer"])
        self.lm = loaders.get_moshi_lm(pp_files["moshi"], device=self.device, cpu_offload=False)
        self.lm.eval()
        self.adapter = resolve_adapter(adapter)
        self.merge_info = _merge(self.lm, self.adapter) if self.adapter else None
        self.sampling = dict(SAMPLING, **(sampling or {}))
        self.lm_gen = LMGen(self.lm, audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate, device=self.device, frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=False, **self.sampling)
        assert int(self.mimi.sample_rate / self.mimi.frame_rate) == FRAME_SIZE
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        with torch.no_grad():
            warmup(self.mimi, self.other_mimi, self.lm_gen, self.device, FRAME_SIZE)
        self._attn = self._find_attn()
        super().__init__(spm, context=self._attn.kv_cache.capacity if self._attn else 3000, smi=smi)
        self.load_s = round(time.time() - t0, 1)

    # ---- helpers ------------------------------------------------------------------------------------
    def _find_attn(self):
        """Streaming state of the first main-transformer attention layer (holds the RingKVCache)."""
        for name, m in self.lm.transformer.named_modules():
            st = getattr(m, "_streaming_state", None)
            if st is not None and hasattr(st, "kv_cache"):
                return st
        return None

    def kv_used(self) -> int:
        if self._attn is None:
            return int(self.lm_gen._streaming_state.offset) - 1
        # GPU tensor (offset_cpu is a python int that CUDA-graph replays do not advance)
        return int(self._attn.kv_cache.end_offset.item())

    def vram(self) -> dict:
        return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "nvidia_smi_used_mib": self._smi.value}

    # ---- prompt phase -------------------------------------------------------------------------------
    @torch.no_grad()
    def _prompt_phase(self, cfg, should_abort):
        seed = cfg.get("seed")
        if seed is not None and int(seed) != -1:
            seed_all(int(seed))
        text = cfg.get("role_prompt") or ""
        ids = self.spm.encode(wrap_with_system_tags(text)) if text.strip() else []
        self.prompt_tokens, self.prompt_unk = len(ids), ids.count(0)
        voice = cfg.get("voice") or "NATF2.pt"
        vp = os.path.join(self.voice_prompt_dir, voice)
        if not os.path.exists(vp):
            raise FileNotFoundError(f"voice prompt {voice!r} not found in {self.voice_prompt_dir}")
        g = self.lm_gen
        if vp.endswith(".pt"):
            g.load_voice_prompt_embeddings(vp)
        else:
            g.load_voice_prompt(vp)
        g.text_prompt_tokens = ids
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        g.reset_streaming()
        torch.cuda.reset_peak_memory_stats()
        for core in (g._step_voice_prompt_core(self.mimi), g._step_audio_silence_core(),
                     g._step_text_prompt_core(), g._step_audio_silence_core()):
            for _ in core:
                if should_abort is not None and should_abort():
                    raise SessionAborted()
        self.mimi.reset_streaming()   # as stock: Mimi reused for the voice prompt, reset before conversation
        torch.cuda.synchronize()

    # ---- one frame ----------------------------------------------------------------------------------
    @torch.no_grad()
    def _model_step(self, pcm, forced_token):
        chunk = torch.from_numpy(pcm).to(self.device)[None, None]
        tt = None
        if forced_token is not None:
            tt = torch.tensor([int(forced_token)], dtype=torch.long, device=self.device)
        torch.cuda.synchronize()
        a = time.perf_counter()
        codes = self.mimi.encode(chunk)
        _ = self.other_mimi.encode(chunk)
        torch.cuda.synchronize()
        b = time.perf_counter()
        tokens = self.lm_gen.step(codes[:, :, 0:1], text_token=tt)
        torch.cuda.synchronize()
        c = time.perf_counter()
        if tokens is None:  # cannot happen after the prompt phase (offset > max_delay)
            raise RuntimeError("lm_gen.step returned None during conversation")
        pcm_t = self.mimi.decode(tokens[:, 1:9])
        _ = self.other_mimi.decode(tokens[:, 1:9])
        out = pcm_t[0, 0].float().cpu().numpy().astype(np.float32)
        tok = int(tokens[0, 0, 0].item())
        d = time.perf_counter()
        return tok, out, (c - b) * 1000.0, (d - a) * 1000.0
