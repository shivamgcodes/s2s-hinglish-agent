"""Test driver for PersonaPlex (copy of /workspace/personaplex-gate0/exp2/driver.py, original untouched).

Same model loading, warmup, prompt phases and per-frame loop as moshi/offline.py; model code untouched.
Changes vs the Gate 0 driver:
  * every run is reseeded with its own seed just before its prompt phase (server.py style), including the
    first one, so results do not depend on run order or on how runs are sharded across processes;
  * optional adapter: --adapter PATH merges LoRA into the base weights in place after loading (see
    merge_adapter); --moshi-weight PATH loads a full merged checkpoint instead of the HF one;
  * runs whose outputs already exist are skipped (resumable); per-run outputs:
      <name>.wav                      model audio (mono 24 kHz)
      <name>_stereo.wav               L/ch0 = model (agent), R/ch1 = input (user)  [dataset convention]
      <name>.json                     text stream, one entry per 80 ms frame (piece, "PAD" or "EPAD")
      <name>.frames.txt               frame view: frame, time, token
      <name>.meta.json                prompt, seed, voice, timing, VRAM, turn events
  * input is fed as one continuous stream (FixedFeeder) when the run has "input_wav"; runs with "turns"
    use the Gate 0 TurnFeeder (used only for the Gate 0 substitute clips).

Usage:  python driver.py SPEC.json OUT_DIR [--adapter PATH] [--moshi-weight PATH] [--customer-mode fixed|vad]
                        [--vad-params JSON]
  --customer-mode vad (added 2026-10-05, tests/VAD_GATING.md): runs with "input_wav" play the scripted customer
  turns REACTIVELY (tests/vad_gate.py VadFeeder: next turn ~0.3 s after the model's output goes quiet, with
  timeouts / barge-in / caps) instead of the fixed-time stream; extra outputs <name>.vad.json (timeline) and
  <name>.tmeta.json (the inputs meta with the ACTUAL turn times, read by score.py / v4_eval.py / judge.py), meta keys
  "customer_mode" and "vad". Default fixed = the code path above, unchanged.
SPEC = {"runs": [{"name", "voice", "prompt", "seed", "input_wav" | "turns"}]}  (turn clip paths absolute)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

from moshi.models import LMGen, loaders
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.models.lm import load_audio as lm_load_audio
from moshi.offline import (_get_voice_prompt_dir, decode_tokens_to_pcm, seed_all, warmup,
                           wrap_with_system_tags)

FRAME_RATE = 12.5
TEXT_SPECIAL = {0: "EPAD", 3: "PAD"}

QUIET_FRAMES = 19
NOSPEECH_FRAMES = 75
MAX_WAIT_FRAMES = 375
TAIL_QUIET_FRAMES = 38
TAIL_NOSPEECH_FRAMES = 100
MAX_FRAMES = 1500


class TurnFeeder:
    """Gate 0 turn-taking feeder (unchanged logic)."""

    def __init__(self, clips, frame_size):
        self.clips = clips
        self.frame_size = frame_size
        self.fed, self.events = [], []
        self.idx, self.pos = 0, None
        self.spoke, self.quiet, self.waited = False, 0, 0

    def observe(self, is_piece):
        if is_piece:
            self.quiet = 0
            if self.pos is None:
                self.spoke = True
        else:
            self.quiet += 1

    def _release_reason(self):
        quiet_needed = self.clips[self.idx][2]
        if self.spoke and self.quiet >= quiet_needed:
            return f"model quiet for {quiet_needed / FRAME_RATE:.1f} s after speaking"
        if not self.spoke and self.waited >= NOSPEECH_FRAMES:
            return "model silent for 6 s"
        if self.waited >= MAX_WAIT_FRAMES:
            return "30 s wait cap"
        return None

    def _tail_done(self):
        if self.spoke and self.quiet >= TAIL_QUIET_FRAMES:
            return True
        if not self.spoke and self.waited >= TAIL_NOSPEECH_FRAMES:
            return True
        return self.waited >= MAX_WAIT_FRAMES

    def frames(self):
        zeros = np.zeros((1, self.frame_size), dtype=np.float32)
        while len(self.fed) < MAX_FRAMES:
            if self.pos is None:
                if self.idx < len(self.clips):
                    reason = self._release_reason()
                    if reason is not None:
                        self.pos = 0
                        self.events.append({"clip": self.clips[self.idx][0], "start_frame": len(self.fed),
                                            "released_by": reason})
                elif self._tail_done():
                    return
            if self.pos is None:
                frame = zeros
                self.waited += 1
            else:
                clip = self.clips[self.idx][1]
                chunk = clip[self.pos:self.pos + self.frame_size]
                frame = np.zeros((1, self.frame_size), dtype=np.float32)
                frame[0, :len(chunk)] = chunk
                self.pos += self.frame_size
                if self.pos >= len(clip):
                    self.events[-1]["end_frame"] = len(self.fed) + 1
                    self.pos, self.idx = None, self.idx + 1
                    self.spoke, self.waited = False, 0
            self.fed.append(frame[0].copy())
            yield frame


class FixedFeeder:
    def __init__(self, audio, frame_size):
        self.audio, self.frame_size = audio, frame_size
        self.fed, self.events = [], []

    def observe(self, is_piece):
        pass

    def frames(self):
        for frame in lm_iterate_audio(self.audio, sample_interval_size=self.frame_size, pad=True):
            self.fed.append(np.asarray(frame[0], dtype=np.float32))
            yield frame


def merge_adapter(lm, path):
    """Merge a LoRA adapter into lm in place. Contract (documented in NOTES.md, tests section):
    PATH = dir with lora.safetensors (+ optional config.json) or the .safetensors file itself.
    Keys:  <param>.lora_A [r, in] and <param>.lora_B [out, r]  (a trailing '.weight' after lora_A/lora_B is
           accepted), where <param> is an exact name from lm.named_parameters() (e.g.
           transformer.layers.0.self_attn.in_proj_weight); W += scaling * B @ A.
           Any key that exactly equals a parameter name replaces that parameter (config B embeddings).
    scaling: config.json lora.scaling, else lora.alpha / lora.rank, else 2.0 (rank 64 / alpha 128).
    If /workspace/hinglish/trainer/merge_lora.py defines merge_into(lm, path), that is used instead."""
    import importlib.util
    hook = Path(os.environ.get("PP_MERGE_LORA_HOOK") or Path(__file__).resolve().parents[2] / "packages" / "personaplex_lora" / "trainer" / "merge_lora.py")
    if hook.exists():
        spec = importlib.util.spec_from_file_location("merge_lora", hook)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "merge_into"):
            print(f"[driver] merging adapter via {hook}", flush=True)
            return mod.merge_into(lm, path)
    from safetensors.torch import load_file
    p = Path(path)
    st = p / "lora.safetensors" if p.is_dir() else p
    cfg_p = (p if p.is_dir() else p.parent) / "config.json"
    scaling = 2.0
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())
        lc = cfg.get("lora", cfg)
        if "scaling" in lc:
            scaling = float(lc["scaling"])
        elif "alpha" in lc and "rank" in lc:
            scaling = float(lc["alpha"]) / float(lc["rank"])
    sd = load_file(str(st))
    params = dict(lm.named_parameters())
    pairs, replaced, unused = {}, 0, []
    for k, v in sd.items():
        kk = k[:-len(".weight")] if k.endswith((".lora_A.weight", ".lora_B.weight")) else k
        if kk.endswith((".lora_A", ".lora_B")):
            pairs.setdefault(kk[:-7], {})[kk[-1]] = v
        elif k in params:
            with torch.no_grad():
                params[k].copy_(v.to(params[k].dtype))
            replaced += 1
        else:
            unused.append(k)
    merged = 0
    with torch.no_grad():
        for base, ab in pairs.items():
            name = base if base in params else base + ".weight"
            if name not in params or "A" not in ab or "B" not in ab:
                unused.append(base)
                continue
            W = params[name]
            delta = (ab["B"].to(W.device, torch.float32) @ ab["A"].to(W.device, torch.float32)) * scaling
            W.add_(delta.reshape(W.shape).to(W.dtype))
            merged += 1
    print(f"[driver] adapter {st}: merged {merged} LoRA pairs (scaling {scaling}), replaced {replaced} tensors, "
          f"unused keys {len(unused)} {unused[:5]}", flush=True)
    if merged + replaced == 0:
        raise RuntimeError("adapter merged nothing; key naming does not match the contract")
    return {"merged": merged, "replaced": replaced, "unused": len(unused), "scaling": scaling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out_dir")
    ap.add_argument("--adapter")
    ap.add_argument("--moshi-weight")
    ap.add_argument("--customer-mode", choices=["fixed", "vad"], default="fixed")
    ap.add_argument("--vad-params", default="{}", help="JSON overrides of vad_gate.DEFAULTS")
    a = ap.parse_args()
    vad = a.customer_mode == "vad"
    if vad:
        import vad_gate
    spec = json.loads(Path(a.spec).read_text())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [r for r in spec["runs"] if not (out_dir / f"{r['name']}.meta.json").exists()]
    print(f"[driver] {len(runs)} of {len(spec['runs'])} runs to do (pid {os.getpid()})", flush=True)
    if not runs:
        return
    device, hf_repo = "cuda", loaders.DEFAULT_REPO
    t_load = time.time()
    seed_all(runs[0]["seed"])
    voice_dir = Path(_get_voice_prompt_dir(None, hf_repo))
    hf_hub_download(hf_repo, "config.json")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME))
    moshi_weight = a.moshi_weight or hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=False)
    lm.eval()
    merge_info = merge_adapter(lm, a.adapter) if a.adapter else None
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), sample_rate=mimi.sample_rate,
                   device=device, frame_rate=mimi.frame_rate, save_voice_prompt_embeddings=False,
                   use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, other_mimi, lm_gen, device, frame_size)
    load_s = time.time() - t_load
    print(f"[driver] model ready in {load_s:.1f} s, frame_size={frame_size}, "
          f"alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    with torch.no_grad():
        for r in runs:
            name = r["name"]
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            seed_all(r["seed"])
            prompt_ids = text_tokenizer.encode(wrap_with_system_tags(r["prompt"]))
            if prompt_ids.count(0):
                print(f"[driver] WARNING {name}: prompt has {prompt_ids.count(0)} unk tokens", flush=True)
            lm_gen.load_voice_prompt_embeddings(str(voice_dir / r["voice"]))
            lm_gen.text_prompt_tokens = prompt_ids
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()
            t_prompt = time.time() - t0
            if vad and "input_wav" in r:
                tmeta = json.loads(Path(r["input_wav"]).with_suffix(".meta.json").read_text())
                cin = np.asarray(lm_load_audio(r["input_wav"], mimi.sample_rate)[0], dtype=np.float32)
                vp = json.loads(a.vad_params)
                if vp.pop("tail_match_fixed", False):  # v2: same post-call time as the fixed input
                    vp["tail_max"] = round(tmeta["duration"] - max(t["end"] for t in tmeta["turns"]
                                                                   if t["speaker"] == "customer"), 3)
                clips = vad_gate.cut_customer_turns(cin, mimi.sample_rate, tmeta["turns"],
                                                    pad_s=vp.get("pad_s", vad_gate.DEFAULTS["pad_s"]))
                feeder = vad_gate.VadFeeder(clips, tmeta["turns"], frame_size, **vp)
            elif "input_wav" in r:
                feeder = FixedFeeder(lm_load_audio(r["input_wav"], mimi.sample_rate), frame_size)
            else:
                clips = []
                for turn in r["turns"]:
                    path, quiet = (turn, QUIET_FRAMES) if isinstance(turn, str) else \
                        (turn["clip"], int(round(turn["quiet_s"] * FRAME_RATE)))
                    audio = lm_load_audio(str(path), mimi.sample_rate)
                    clips.append((Path(path).stem, np.asarray(audio[0], dtype=np.float32), quiet))
                feeder = TurnFeeder(clips, frame_size)

            tokens_out, pcm_out, skipped = [], [], 0
            t1 = time.time()
            for user_encoded in lm_encode_from_sphn(mimi, feeder.frames(), max_batch=1):
                for c in range(user_encoded.shape[-1]):
                    tokens = lm_gen.step(user_encoded[:, :, c:c + 1])
                    if tokens is None:
                        skipped += 1
                        if vad:
                            feeder.observe_vad(-200.0, None)
                        else:
                            feeder.observe(False)
                        continue
                    pcm_out.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))
                    tid = tokens[0, 0, 0].item()
                    is_piece = tid not in TEXT_SPECIAL
                    tokens_out.append(text_tokenizer.id_to_piece(tid).replace("▁", " ") if is_piece
                                      else TEXT_SPECIAL[tid])
                    if vad:
                        feeder.observe_vad(vad_gate.frame_db(pcm_out[-1]), tokens_out[-1])
                    else:
                        feeder.observe(is_piece)
            wall = time.time() - t1

            out = np.concatenate(pcm_out, axis=-1).astype(np.float32)
            inp = np.concatenate(feeder.fed, axis=-1).astype(np.float32)
            n = max(len(out), len(inp))
            st = np.zeros((2, n), np.float32)
            st[0, :len(out)] = out
            st[1, :len(inp)] = inp
            sphn.write_wav(str(out_dir / f"{name}.wav"), out, mimi.sample_rate)
            sphn.write_wav(str(out_dir / f"{name}_stereo.wav"), st, mimi.sample_rate)
            (out_dir / f"{name}.json").write_text(json.dumps(tokens_out, ensure_ascii=False))
            with open(out_dir / f"{name}.frames.txt", "w", encoding="utf-8") as f:
                f.write("# frame\ttime_s\ttoken   (12.5 Hz; time = model output time, input starts at 0)\n")
                for i, t in enumerate(tokens_out):
                    f.write(f"{i}\t{i / FRAME_RATE:.2f}\t{t!r}\n")
            text = "".join(t for t in tokens_out if t not in ("PAD", "EPAD")).strip()
            meta = {"name": name, "voice": r["voice"], "prompt": r["prompt"], "seed": r["seed"],
                    "input_wav": r.get("input_wav"), "adapter": a.adapter, "moshi_weight": a.moshi_weight,
                    "merge": merge_info, "prompt_tokens": len(prompt_ids), "prompt_unk_tokens": prompt_ids.count(0),
                    "frames": len(tokens_out), "frames_fed": len(feeder.fed), "skipped_steps": skipped,
                    "duration_s": len(tokens_out) / FRAME_RATE, "prompt_phase_s": round(t_prompt, 2),
                    "generation_wall_s": round(wall, 2), "rtf": round(wall / max(len(tokens_out) / FRAME_RATE, 1e-6), 3),
                    "load_s": round(load_s, 1), "pid": os.getpid(),
                    "max_mem_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "mem_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                    "events": feeder.events, "text": text}
            if vad and "input_wav" in r:
                tl = feeder.timeline
                meta["customer_mode"] = "vad"
                meta["vad"] = {k: tl[k] for k in ("stop_reason", "overlap_s", "model_active_s", "n_barge_in",
                                                  "n_timeout_silent", "unplayed_turns", "params")}
                (out_dir / f"{name}.vad.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False))
                tm2 = dict(tmeta)
                tm2["turns"] = feeder.effective_turns()
                tm2["duration"] = round(len(feeder.fed) / FRAME_RATE, 3)
                tm2["customer_mode"] = "vad"
                (out_dir / f"{name}.tmeta.json").write_text(json.dumps(tm2, indent=1, ensure_ascii=False))
            (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            print(f"[driver] {name}: {meta['frames']} frames ({meta['duration_s']:.1f} s) in {wall:.1f} s "
                  f"(rtf {meta['rtf']}), prompt phase {t_prompt:.1f} s, peak {meta['max_mem_alloc_gib']} GiB", flush=True)
            print(f"[driver]   TEXT: {text[:300]}", flush=True)
    print("[driver] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
