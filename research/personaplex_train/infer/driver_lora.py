"""Turn-taking offline driver for PersonaPlex (Gate 0 experiments).

Built from the repo's moshi/offline.py: same model loading, warmup, prompt phases and per-frame
loop, with the model code untouched. Two differences:

1. The model is loaded once and reused for several experiments, resetting streaming state between
   them the way moshi/server.py does between connections.
2. The user side can be a list of separate clips ("turns") instead of one wav. The next clip is
   released only when the model's text stream has gone quiet, so the dialogue order holds no
   matter how long the model talks. Silence (zeros) is fed while waiting.

Usage:
  python driver.py experiments.json out_dir

Adapter version (copy of /workspace/personaplex-gate0/exp2/driver.py; the original is unchanged):
  python driver_lora.py experiments.json out_dir [ADAPTER]
ADAPTER = a trainer run dir or checkpoint dir holding lora.safetensors + config.json, or a
lora.safetensors path. The base model is loaded with the PersonaPlex loaders exactly as before and
the adapter is merged in place (infer/lora_merge.py). Omitted / "none" / "base" = base model.
For ft_embed adapters the voice-prompt replay embeddings are recomputed with the merged tables.
"""
import json
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('PP_LORA_INFER_DIR') or str(_REPO / 'packages/personaplex_lora/infer'))
import lora_merge  # noqa: E402

FRAME_RATE = 12.5
TEXT_SPECIAL = {0: "EPAD", 3: "PAD"}

# Turn-taking rule, in frames (80 ms each).
QUIET_FRAMES = 19        # default: 1.5 s without a text piece, after the model has spoken, releases the next clip
NOSPEECH_FRAMES = 75     # 6 s with the model saying nothing at all also releases it
MAX_WAIT_FRAMES = 375    # 30 s cap on any single wait (the user then talks over the model)
TAIL_QUIET_FRAMES = 38   # after the last clip, stop once the model has been quiet for 3 s
TAIL_NOSPEECH_FRAMES = 100
MAX_FRAMES = 1500        # 120 s hard cap per experiment


class TurnFeeder:
    """Yields one 80 ms user-audio frame per model step, deciding live when each clip starts."""

    def __init__(self, clips, frame_size):
        self.clips = clips                # list of (name, mono float32 array, quiet frames before release)
        self.frame_size = frame_size
        self.fed = []                     # frames actually fed, for the left channel
        self.events = []                  # {"clip", "start_frame", "end_frame", "released_by"}
        self.idx = 0
        self.pos = None                   # sample offset in the current clip while playing
        self.spoke = False                # model emitted a text piece during the current wait
        self.quiet = 0                    # frames since the model's last text piece
        self.waited = 0

    def observe(self, is_piece):
        """Called after every model step with whether it emitted a real text piece."""
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
    """Feeds one wav exactly as offline.py does (used to check the driver against the baseline)."""

    def __init__(self, audio, frame_size):
        self.audio = audio
        self.frame_size = frame_size
        self.fed = []
        self.events = []

    def observe(self, is_piece):
        pass

    def frames(self):
        for frame in lm_iterate_audio(self.audio, sample_interval_size=self.frame_size, pad=True):
            self.fed.append(np.asarray(frame[0], dtype=np.float32))
            yield frame


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed, device, hf_repo = spec["seed"], "cuda", loaders.DEFAULT_REPO
    base = Path(sys.argv[1]).parent

    # Same order of operations as offline.py so the first experiment is comparable with it.
    seed_all(seed)
    voice_dir = Path(_get_voice_prompt_dir(None, hf_repo))
    hf_hub_download(hf_repo, "config.json")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(hf_hub_download(hf_repo, loaders.MOSHI_NAME), device=device, cpu_offload=False)
    lm.eval()
    adapter = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("", "none", "base") else None
    merge_info = None
    if adapter:
        sd, acfg = lora_merge.load_adapter(adapter)
        merge_info = lora_merge.merge_into(lm, sd, acfg)
        del sd
        print(f"[driver] merged adapter {adapter}: {merge_info}", flush=True)
    else:
        print("[driver] base model (no adapter)", flush=True)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, other_mimi, lm_gen, device, frame_size)
    print(f"[driver] model ready, frame_size={frame_size}", flush=True)

    with torch.no_grad():
        for n, exp in enumerate(spec["experiments"]):
            name = exp["name"]
            # offline.py seeds once per process; server.py reseeds per connection. A fixed-wav
            # baseline run first keeps the offline.py behaviour, everything else follows server.py.
            exp_seed = exp.get("seed", seed)
            if not (n == 0 and "input_wav" in exp):
                seed_all(exp_seed)
            prompt_ids = text_tokenizer.encode(wrap_with_system_tags(exp["prompt"]))
            lm_gen.load_voice_prompt_embeddings(str(voice_dir / exp["voice"]))
            if merge_info and merge_info["ft_embed"]:
                lm_gen.voice_prompt_embeddings = lora_merge.voice_embeddings_from_codes(lm, exp["voice"])
            lm_gen.text_prompt_tokens = prompt_ids
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            if "input_wav" in exp:
                feeder = FixedFeeder(lm_load_audio(exp["input_wav"], mimi.sample_rate), frame_size)
            else:
                clips = []
                for turn in exp["turns"]:
                    # A turn is a clip path, or {"clip": path, "quiet_s": seconds of model silence to wait for}.
                    path, quiet = (turn, QUIET_FRAMES) if isinstance(turn, str) else \
                        (turn["clip"], int(round(turn["quiet_s"] * FRAME_RATE)))
                    audio = lm_load_audio(str(base / path), mimi.sample_rate)
                    clips.append((Path(path).stem, np.asarray(audio[0], dtype=np.float32), quiet))
                feeder = TurnFeeder(clips, frame_size)

            tokens_out, pcm_out, skipped = [], [], 0
            t0 = time.time()
            for user_encoded in lm_encode_from_sphn(mimi, feeder.frames(), max_batch=1):
                for c in range(user_encoded.shape[-1]):
                    tokens = lm_gen.step(user_encoded[:, :, c:c + 1])
                    if tokens is None:
                        skipped += 1
                        feeder.observe(False)
                        continue
                    pcm_out.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))
                    tid = tokens[0, 0, 0].item()
                    is_piece = tid not in TEXT_SPECIAL
                    tokens_out.append(text_tokenizer.id_to_piece(tid).replace("▁", " ") if is_piece
                                      else TEXT_SPECIAL[tid])
                    feeder.observe(is_piece)
            wall = time.time() - t0

            sphn.write_wav(str(out_dir / f"{name}.wav"), np.concatenate(pcm_out, axis=-1), mimi.sample_rate)
            sphn.write_wav(str(out_dir / f"{name}_input.wav"), np.concatenate(feeder.fed, axis=-1), mimi.sample_rate)
            (out_dir / f"{name}.json").write_text(json.dumps(tokens_out, ensure_ascii=False))
            meta = {
                "name": name, "voice": exp["voice"], "prompt": exp["prompt"], "seed": exp_seed,
                "prompt_tokens": len(prompt_ids), "prompt_unk_tokens": prompt_ids.count(0),
                "frames": len(tokens_out), "frames_fed": len(feeder.fed), "skipped_steps": skipped,
                "duration_s": len(tokens_out) / FRAME_RATE, "generation_wall_s": round(wall, 2),
                "events": feeder.events, "adapter": adapter, "merge": merge_info,
            }
            (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            text = "".join(t for t in tokens_out if t not in ("PAD", "EPAD")).strip()
            print(f"[driver] {name}: {meta['frames']} frames ({meta['duration_s']:.1f} s) in {wall:.1f} s, "
                  f"prompt {len(prompt_ids)} tokens ({prompt_ids.count(0)} unk), skipped {skipped}", flush=True)
            for e in feeder.events:
                print(f"[driver]   user clip {e['clip']}: frames {e['start_frame']}-{e.get('end_frame')} "
                      f"({e['start_frame'] / FRAME_RATE:.1f} s), released by: {e['released_by']}", flush=True)
            print(f"[driver]   TEXT: {text}", flush=True)
    print("[driver] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
