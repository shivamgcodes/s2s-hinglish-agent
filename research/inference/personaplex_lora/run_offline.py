"""Offline run: base PersonaPlex 7B + this Hinglish LoRA (merged in memory) on one customer-side wav.

Same loading, warmup, prompt phases and per-frame loop as the stock PersonaPlex `moshi/offline.py`
(github.com/NVIDIA/personaplex) and as our test driver / live worker; the only addition is the in-memory LoRA merge.

    pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora" safetensors
    python run_offline.py --output out.wav          # defaults: the HF repo's examples/food_23_g1_customer_30s.wav + food_23.json, g1
    python run_offline.py --input my.wav --record my_record.json --pairing g2 --adapter ''   # '' = base model

D-LEAN-HF (2026-10-07): this is the one copy of the HF card's offline example; the merge + role-prompt code is the
`personaplex_lora` package (packages/personaplex_lora), the adapter + examples come from the HF repo via hf_hub_download.
Needs: the PersonaPlex `moshi` package (pip install <personaplex>/moshi), access to the gated
nvidia/personaplex-7b-v1 repo (accept the NVIDIA Open Model License, then `huggingface-cli login` or HF_TOKEN), one CUDA GPU
with about 20 GB free.
Writes <output> (agent audio, 24 kHz mono) and <output>.json (one text token per 80 ms frame: piece, PAD or EPAD).
"""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download
from moshi.models import LMGen, loaders
from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
from moshi.offline import decode_tokens_to_pcm, seed_all, warmup, wrap_with_system_tags

try:
    import personaplex_lora  # pip-installed (see above)
except ImportError:          # running from a monorepo checkout without installing: use packages/ in place
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages"))
    import personaplex_lora
from personaplex_lora import role_prompt  # noqa: E402
from personaplex_lora.infer import lora_merge  # noqa: E402

LORA_REPO = personaplex_lora.HF_REPO                 # shivamgupta/personaplex-hinglish-v4-lora
BASE_REPO = personaplex_lora.BASE_REPO
BASE_REVISION = personaplex_lora.BASE_REVISION       # the base snapshot the adapter was trained on
SAMPLING = dict(use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)   # as in training evals + demo
TEXT_SPECIAL = {0: "EPAD", 3: "PAD"}


def base_file(name):
    return hf_hub_download(BASE_REPO, name, revision=BASE_REVISION)


def lora_file(name):
    return hf_hub_download(LORA_REPO, name)


def voice_dir():
    tgz = Path(base_file("voices.tgz"))
    d = tgz.parent / "voices"
    if not d.exists():
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(path=tgz.parent)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="customer-side audio (any sample rate; resampled to 24 kHz); default: the HF repo's "
                    "examples/food_23_g1_customer_30s.wav")
    ap.add_argument("--output", default="out.wav")
    ap.add_argument("--record", help="record JSON with role_prompts; default: the HF repo's examples/food_23.json")
    ap.add_argument("--pairing", default="g1", choices=role_prompt.PAIRINGS)
    ap.add_argument("--role-prompt", help="role prompt text (overrides --record)")
    ap.add_argument("--voice", help="voice prompt file name in the base repo's voices/ (default from --pairing)")
    ap.add_argument("--adapter", default="hf", help="dir with lora.safetensors + config.json; 'hf' = download from "
                    f"{LORA_REPO} (default); '' = base model")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--max-seconds", type=float, default=0, help="cut the input to this many seconds (0 = all)")
    a = ap.parse_args()
    a.input = a.input or lora_file("examples/food_23_g1_customer_30s.wav")
    if not a.role_prompt and not a.record:
        a.record = lora_file("examples/food_23.json")
    if a.adapter == "hf":
        lora_file("lora.safetensors")
        a.adapter = str(Path(lora_file("config.json")).parent)

    if a.role_prompt:
        prompt = role_prompt.ascii_prompt(a.role_prompt)
    else:
        prompt = role_prompt.build_role_prompt(json.loads(Path(a.record).read_text()), a.pairing)
    voice = a.voice or role_prompt.voice_for(a.pairing)
    device = "cuda"
    seed_all(a.seed)

    mimi_w = base_file(loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_w, device)
    other_mimi = loaders.get_mimi(mimi_w, device)
    spm = sentencepiece.SentencePieceProcessor(base_file(loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(base_file(loaders.MOSHI_NAME), device=device, cpu_offload=False)
    lm.eval()
    if a.adapter:
        sd, cfg = lora_merge.load_adapter(a.adapter)
        assert not cfg.get("ft_embed"), "ft_embed adapters: use personaplex_lora.merge_adapter (voice-embedding fix)"
        print("merged:", lora_merge.merge_into(lm, sd, cfg), flush=True)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)   # 1920 samples = 80 ms at 24 kHz
    lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), sample_rate=mimi.sample_rate,
                   device=device, frame_rate=mimi.frame_rate, save_voice_prompt_embeddings=False, **SAMPLING)
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    with torch.no_grad():
        seed_all(a.seed)
        lm_gen.load_voice_prompt_embeddings(str(voice_dir() / voice))
        lm_gen.text_prompt_tokens = spm.encode(wrap_with_system_tags(prompt))
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)   # voice prompt, silence, role prompt, silence
        mimi.reset_streaming()

        audio = load_audio(a.input, mimi.sample_rate)
        if a.max_seconds:
            audio = audio[:, : int(a.max_seconds * mimi.sample_rate)]
        pcm, text = [], []
        for enc in encode_from_sphn(mimi, _iterate_audio(audio, sample_interval_size=frame_size, pad=True),
                                    max_batch=1):
            for c in range(enc.shape[-1]):
                tokens = lm_gen.step(enc[:, :, c:c + 1])
                if tokens is None:
                    continue
                pcm.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))
                tid = tokens[0, 0, 0].item()
                text.append(TEXT_SPECIAL.get(tid) or spm.id_to_piece(tid).replace("▁", " "))

    sphn.write_wav(a.output, np.concatenate(pcm, axis=-1).astype(np.float32), mimi.sample_rate)
    Path(a.output + ".json").write_text(json.dumps(text, ensure_ascii=False))
    print("agent text:", "".join(t for t in text if t not in ("PAD", "EPAD")).strip())


if __name__ == "__main__":
    main()
