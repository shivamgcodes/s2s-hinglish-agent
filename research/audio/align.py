"""MMS_FA word timestamps per utterance (venv-pp, read-only use; torchaudio 2.11 MMS_FA, GPU).

  TORCH_HOME=/workspace/torch /workspace/venv-pp/bin/python align.py ROOT
For each work/utt/<call>/tNN.wav: align the whole utterance (chunks + gaps; CTC absorbs gaps) against
the turn's text_roman expanded to spoken roman tokens (tts_norm.roman_words), then merge expanded tokens
back to the ORIGINAL text_roman words (so "A1234" / "Rs 540" stay one transcript word).
On failure: even spacing (by character length) over the voiced region, flag "align_fallback".
Writes work/align/<call>/tNN.json: {words:[{word,start,end,score}], method, mean_score, flags}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_norm  # noqa: E402

LOW_SCORE = 0.3


def voiced_region(wav, sr):
    win = int(0.01 * sr)
    n = len(wav) // win
    if n == 0:
        return 0.0, len(wav) / sr
    rms = np.sqrt((wav[: n * win].reshape(n, win) ** 2).mean(1) + 1e-12)
    idx = np.where(rms > rms.max() * 0.03)[0]
    if len(idx) == 0:
        return 0.0, len(wav) / sr
    return idx[0] * win / sr, (idx[-1] + 1) * win / sr


def even_spacing(words, a, b):
    lens = np.array([max(1, len(w)) for w in words], float)
    edges = a + np.concatenate([[0], np.cumsum(lens)]) / lens.sum() * (b - a)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(words))]


def main():
    root = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(dev).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    t_load = time.time() - t0
    utts = sorted((root / "work" / "utt").glob("*/t*.wav"))
    n_done = n_fb = n_low = 0
    audio_s = 0.0
    t1 = time.time()
    for wp in utts:
        out = root / "work" / "align" / wp.parent.name / (wp.stem + ".json")
        if out.exists():
            continue
        meta = json.loads(wp.with_suffix(".json").read_text(encoding="utf-8"))
        text_roman = " ".join(c["text_roman"] for c in meta["chunks"])
        groups = tts_norm.roman_words(text_roman)
        wav, sr = sphn.read(str(wp))
        wav = wav[0]
        audio_s += len(wav) / sr
        flags = []
        words = []
        method = "mms_fa"
        try:
            if not groups:
                raise ValueError("no alignable words")
            flat = [t for _, toks in groups for t in toks]
            x = torch.from_numpy(wav).float()[None]
            x16 = torchaudio.functional.resample(x, sr, bundle.sample_rate).to(dev)
            with torch.inference_mode():
                emission, _ = model(x16)
            spans = aligner(emission[0], tokenizer(flat))
            ratio = x16.shape[1] / emission.shape[1] / bundle.sample_rate
            k = 0
            for orig, toks in groups:
                ws = spans[k:k + len(toks)]
                k += len(toks)
                st = ws[0][0].start * ratio
                en = ws[-1][-1].end * ratio
                sc = float(np.mean([s.score for w in ws for s in w]))
                words.append({"word": orig, "start": round(st, 3), "end": round(max(en, st + 0.02), 3),
                              "score": round(sc, 3)})
            mean_score = float(np.mean([w["score"] for w in words]))
            if mean_score < LOW_SCORE:
                flags.append("align_low_score")
                n_low += 1
        except Exception as e:  # noqa: BLE001
            method = "even_spacing"
            flags.append(f"align_fallback:{type(e).__name__}:{str(e)[:80]}")
            n_fb += 1
            ow = text_roman.split()
            a, b = voiced_region(wav, sr)
            words = [{"word": w, "start": round(s, 3), "end": round(e_, 3), "score": None}
                     for w, (s, e_) in zip(ow, even_spacing(ow, a, b))]
            mean_score = None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"call_id": meta["call_id"], "turn": meta["turn"], "speaker": meta["speaker"],
                                   "duration": meta["duration"], "method": method, "mean_score": mean_score,
                                   "flags": flags, "words": words}, ensure_ascii=False, indent=1), encoding="utf-8")
        n_done += 1
    wall = time.time() - t1
    rec = {"stage": "align", "utts": n_done, "fallback": n_fb, "low_score": n_low, "audio_s": round(audio_s, 1),
           "wall_s": round(wall, 1), "load_s": round(t_load, 1), "x_realtime": round(audio_s / max(wall, 1e-6), 1)}
    print(rec, flush=True)
    with open(root / "work" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
