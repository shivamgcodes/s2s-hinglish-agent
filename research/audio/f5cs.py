"""IndicF5 code-switch TTS backend (Tharshan/indicf5_hindi-english_code_switch), run in /workspace/venv-f5.

  python f5cs.py tts ROOT --try K      # synthesize every plan chunk needing try K (same files/contract as synth.py tts)
  python f5cs.py refstats              # CPU: per-voice reference rate (syllables/s)
Model loads once per process; chunks are grouped per reference voice; F5_BATCH (default 1) items per sampler call.

Duration fix: the repo's generate() sizes output from UTF-8 BYTE length (Devanagari = 3 bytes/char, Latin = 1),
which mis-sizes any input whose script mix differs from the reference transcript. Here the model's CFM sampler is
called directly with an explicit per-item duration:
    frames = ref_len + (syllables(text) * frames_per_syllable(voice) * F5_RATE_SCALE + EDGE_S) / speed
frames_per_syllable is measured on the voice's own reference clip (voiced frames / syllables(transcript)), clamped
to a natural conversational range [RATE_MIN, RATE_MAX] syllables/s. Seeds: torch.manual_seed per batch
(seed_of(first chunk stem, try)); the CFM sampler then re-seeds every item with that seed (repo behaviour), so
results are deterministic given the batch composition.
Post: per-chunk trim (synth.trim) + loudness normalisation to TARGET_DBFS RMS (peak-limited to 0.99), 24 kHz PCM16.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tts_backends  # noqa: E402

SR = 24000
HOP = 256
HUB = Path(os.environ.get("HF_HOME", "/workspace/hf")) / "hub"
REPO = "Tharshan/indicf5_hindi-english_code_switch"
VOICE_DIR = HERE / "voices_f5"
TARGET_DBFS = float(os.environ.get("F5_TARGET_DBFS", "-20"))
# engine params: defaults from tts_backends.f5_params() (env); cmd_tts overrides them from plan.json "f5" (fixed at plan time)
_P = tts_backends.f5_params()
RATE_SCALE, RATE_MIN, RATE_MAX = _P["rate_scale"], _P["rate_min"], _P["rate_max"]
EDGE_S = _P["edge_s"]  # extra frames for onset/offset room (review fix: 0.25 -> 0.8, clipped endings)
TAIL_DB, TAIL_EXTRA_S = _P["tail_db"], _P["tail_extra_s"]
BATCH = int(os.environ.get("F5_BATCH", "1"))  # batch>1 measured slower per chunk AND truncates endings (see F5_BACKEND.md)
NFE, CFG, SWAY = _P["nfe"], 2.0, -1.0


def apply_params(p):
    global RATE_SCALE, RATE_MIN, RATE_MAX, EDGE_S, TAIL_DB, TAIL_EXTRA_S, NFE
    RATE_SCALE, RATE_MIN, RATE_MAX = p["rate_scale"], p["rate_min"], p["rate_max"]
    EDGE_S, TAIL_DB, TAIL_EXTRA_S, NFE = p["edge_s"], p["tail_db"], p["tail_extra_s"], p["nfe"]


def end_level_db(a, sr=SR):
    """Level of the last 40 ms vs the chunk's p95 10-ms frame RMS (after trim). Kokoro V1: never > -15 dB;
    f5cs at edge 0.25: 15% > -15 dB = speech cut by the end of the generation buffer."""
    import numpy as np
    w = int(0.01 * sr)
    n = len(a) // w
    if n < 8:
        return -99.0
    r = np.sqrt((a[: n * w].reshape(n, w) ** 2).mean(1) + 1e-12)
    return float(20 * np.log10(r[-4:].max() / np.percentile(r, 95)))


def snapshot():
    d = HUB / f"models--{REPO.replace('/', '--')}" / "snapshots"
    return d / sorted(os.listdir(d))[0]


def load_voices():
    return json.loads((VOICE_DIR / "voices_f5.json").read_text(encoding="utf-8"))


def voiced_seconds(a, sr=SR):
    import numpy as np
    w = int(0.02 * sr)
    n = len(a) // w
    rms = np.sqrt((a[: n * w].reshape(n, w) ** 2).mean(1) + 1e-12)
    idx = np.where(rms > rms.max() * 10 ** (-35 / 20))[0]
    return (idx[-1] - idx[0] + 1) * 0.02 if len(idx) else len(a) / sr


def ref_rate(key, v):
    import soundfile as sf
    a, sr = sf.read(str(VOICE_DIR / v["wav"]), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    vs = voiced_seconds(a, sr)
    syl = tts_backends.syllables(v["transcript"])
    return {"voice": key, "dur": round(len(a) / sr, 2), "voiced": round(vs, 2), "syl": syl,
            "rate_syl_s": round(syl / vs, 2)}


class F5CS:
    def __init__(self, device="cuda"):
        import torch
        S = snapshot()
        pk = HERE / "pkgs"
        pk.mkdir(exist_ok=True)
        link = pk / "codeswitch"
        if not link.exists():
            link.symlink_to(S)
        sys.path.insert(0, str(pk))
        from codeswitch.model import IndicF5Hinglish  # AutoModel(trust_remote_code) fails on transformers 4.49
        self.torch = torch
        self.device = device
        self.m = IndicF5Hinglish.from_pretrained(str(S), device=device)
        self.vocab = self.m.model.vocab_char_map
        self.revision = S.name
        self.voices_meta = load_voices()
        self._v = {}

    def voice(self, key):
        if key in self._v:
            return self._v[key]
        meta = self.voices_meta[key]
        cond, rms = self.m._load_ref(str(VOICE_DIR / meta["wav"]))  # (1, n) wave, RMS-raised to 0.1 if quieter
        mel = self.m.model.mel_spec(cond).permute(0, 2, 1)  # (1, frames, 100)
        ref_text = meta["transcript"].strip()
        if not ref_text.endswith((" ", ".", "।", "!", "?")):
            ref_text += "। "
        elif not ref_text.endswith(" "):
            ref_text += " "
        st = ref_rate(key, meta)
        rate = min(max(st["rate_syl_s"], RATE_MIN), RATE_MAX)
        self._v[key] = {"mel": mel, "rms": float(rms), "ref_len": mel.shape[1], "ref_text": ref_text,
                        "fps": SR / HOP / rate, "rate_used": rate, **st}
        bad = self.unknown_chars(ref_text)
        assert not bad, f"voice {key} transcript has chars outside vocab: {bad}"
        return self._v[key]

    def unknown_chars(self, text):
        return sorted({c for c in text if c not in self.vocab})

    def frames_for(self, v, text, speed, extra_s=0.0):
        syl = tts_backends.syllables(text)
        return v["ref_len"] + int(round((syl * v["fps"] * RATE_SCALE + (EDGE_S + extra_s) * SR / HOP) / speed))

    def render(self, key, texts, speeds, seed, extra_s=0.0):
        """Batch-render texts with one voice -> list of float32 numpy waves (24 kHz)."""
        torch = self.torch
        v = self.voice(key)
        for t in texts:
            bad = self.unknown_chars(t)
            if bad:
                raise ValueError(f"chars outside F5 vocab {bad} in {t!r}")
        B = len(texts)
        dur = torch.tensor([self.frames_for(v, t, s, extra_s) for t, s in zip(texts, speeds)], device=self.device)
        cond = v["mel"].expand(B, -1, -1).contiguous()
        lens = torch.full((B,), v["ref_len"], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            gen, _ = self.m.model.sample(cond=cond, text=[v["ref_text"] + t for t in texts], duration=dur, lens=lens,
                                         steps=NFE, cfg_strength=CFG, sway_sampling_coef=SWAY, seed=seed)
            outs = []
            for i in range(B):
                mel = gen[i:i + 1, v["ref_len"]:int(dur[i]), :].to(torch.float32).permute(0, 2, 1)
                w = self.m.vocoder.decode(mel).squeeze().float().cpu().numpy()
                if v["rms"] < 0.1:
                    w = w * v["rms"] / 0.1
                outs.append(w.astype("float32"))
        return outs


def loudnorm(a, target_dbfs=TARGET_DBFS):
    import numpy as np
    r = float(np.sqrt(np.mean(a ** 2) + 1e-12))
    a = a * (10 ** (target_dbfs / 20) / r)
    pk = float(np.abs(a).max()) if len(a) else 0.0
    if pk > 0.99:
        a = a * (0.99 / pk)
    return a


def cmd_tts(root, k, shard=0, nshards=1):
    """shard/nshards (D2): this process renders plan entries with index % nshards == shard (partition of the FULL
    plan, then needs_try), so N processes under one gpu.lock split the work without overlap. Batch 1 + per-stem
    seeds make the audio independent of the sharding."""
    import numpy as np
    import soundfile as sf
    import torch
    import synth  # stdlib-level import: plan helpers + trim (no kokoro import at module level)
    root = Path(root)
    plan = json.loads((root / "work" / "plan.json").read_text(encoding="utf-8"))
    wrong = {p.get("backend", "kokoro") for p in plan} - {"f5cs"}
    if wrong:
        raise SystemExit(f"plan.json has non-f5cs chunks {wrong}; re-run plan with TTS_BACKEND=f5cs")
    ps_ = {json.dumps(p["f5"], sort_keys=True) for p in plan if "f5" in p}
    if len(ps_) > 1:
        raise SystemExit("plan.json mixes f5 params; re-run plan")
    if ps_:
        apply_params(json.loads(ps_.pop()))
    if not 0 <= shard < nshards:
        raise SystemExit(f"bad shard {shard}/{nshards}")
    todo = [p for i, p in enumerate(plan) if i % nshards == shard and synth.needs_try(root, p, k)]
    print(f"f5cs tts try {k} shard {shard}/{nshards}: {len(todo)} chunks to synthesize", flush=True)
    if not todo:
        return
    t0 = time.time()
    eng = F5CS("cuda" if torch.cuda.is_available() else "cpu")
    for key in sorted({p["voice"] for p in todo}):
        v = eng.voice(key)
        print(f"  voice {key}: ref {v['dur']}s voiced {v['voiced']}s syl {v['syl']} rate {v['rate_syl_s']} "
              f"-> used {v['rate_used']} syl/s", flush=True)
    eng.render(todo[0]["voice"], ["नमस्ते।"], [1.0], 0)  # warm-up
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    by_v = {}
    for p in todo:
        by_v.setdefault(p["voice"], []).append(p)
    t1 = time.time()
    audio_s, n_done, n_tail, n_tail_fixed = 0.0, 0, 0, 0
    for key, ps in sorted(by_v.items()):
        ps.sort(key=lambda p: tts_backends.syllables(p["tts_input"]))
        for b in range(0, len(ps), BATCH):
            batch = ps[b:b + BATCH]
            stem0 = synth.chunk_stem(root, batch[0], k)
            seed = synth.seed_of(stem0.parent.name, stem0.name, k) % (2 ** 31)
            waves = eng.render(key, [p["tts_input"] for p in batch], [p["speeds"][k] for p in batch], seed)
            for p, a in zip(batch, waves):
                out = synth.chunk_stem(root, p, k)
                out.parent.mkdir(parents=True, exist_ok=True)
                a = synth.trim(a)
                e0 = end_level_db(a)
                if e0 > TAIL_DB:  # clipped ending: re-render once (same seed) with more frames, keep the cleaner end
                    n_tail += 1
                    a2 = synth.trim(eng.render(key, [p["tts_input"]], [p["speeds"][k]], seed, TAIL_EXTRA_S)[0])
                    if end_level_db(a2) < e0:
                        a = a2
                        n_tail_fixed += end_level_db(a2) <= TAIL_DB
                a = loudnorm(a) if len(a) else np.zeros(int(0.2 * SR), np.float32)
                a = np.clip(a, -1, 1).astype(np.float32)
                sf.write(str(out) + ".tmp.wav", a, SR, subtype="PCM_16")
                Path(str(out) + ".tmp.wav").rename(out.with_suffix(".wav"))
                audio_s += len(a) / SR
                n_done += 1
            if n_done % 200 < len(batch):
                print(f"  {n_done}/{len(todo)}  {time.time() - t1:.0f}s", flush=True)
    wall = time.time() - t1
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    rec = {"stage": f"tts_try{k}", "backend": "f5cs", **({"shard": shard, "nshards": nshards} if nshards > 1 else {}),
           "chunks": len(todo), "audio_s": round(audio_s, 1),
           "wall_s": round(wall, 1), "load_s": round(t_load, 1), "x_realtime": round(audio_s / max(wall, 1e-6), 2),
           "s_per_chunk": round(wall / max(1, len(todo)), 3), "batch": BATCH, "nfe": NFE, "peak_vram_gib": peak,
           "rate_scale": RATE_SCALE, "edge_s": EDGE_S, "tail_rerender": n_tail, "tail_fixed": n_tail_fixed,
           "model_revision": eng.revision}
    print(rec, flush=True)
    with open(root / "work" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tts", "refstats"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--try", dest="k", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "tts":
        cmd_tts(a.args[0], a.k, a.shard, a.nshards)
    else:
        for k, v in load_voices().items():
            print(ref_rate(k, v))
