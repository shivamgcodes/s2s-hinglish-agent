"""Transcript-back check for the IndicF5 clips: faster-whisper large-v3, same settings as the Kokoro check
(beam 5, temperature 0, no conditioning on previous text, no VAD), on the GPU if it works, else CPU int8.

Run on the pod:
  HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 /workspace/venv-asr/bin/python scripts/asr.py

language="hi" for groups A, B, D and the long inputs; "en" for group C. Also transcribes the last 8 s of each
long-input clip separately (the Kokoro check did this to see whether the last sentence is present).
Writes transcripts.json.
"""
import json
import time
from importlib.metadata import version
from pathlib import Path

import wave

import numpy as np
from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent.parent
MODEL = "large-v3"


def load():
    for device, ctype in (("cuda", "float16"), ("cpu", "int8")):
        try:
            t0 = time.time()
            m = WhisperModel(MODEL, device=device, compute_type=ctype, cpu_threads=16)
            m.transcribe(np.zeros(16000, dtype=np.float32), language="hi")  # forces kernels to load
            print(f"{MODEL} loaded on {device} ({ctype}) in {time.time() - t0:.1f} s", flush=True)
            return m, device, ctype
        except Exception as e:  # e.g. missing cuDNN/cuBLAS libraries for CTranslate2
            print(f"{device} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    raise SystemExit("no device worked")


def run(model, audio, lang):
    segments, _ = model.transcribe(audio, language=lang, beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False, vad_filter=False)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    return " ".join(s["text"] for s in segs).strip(), segs


def main():
    model, device, ctype = load()
    out = {"_meta": {"model": MODEL, "device": device, "compute_type": ctype, "beam_size": 5,
                     "faster_whisper": version("faster-whisper"), "ctranslate2": version("ctranslate2")}}
    for wav in sorted((HERE / "out").glob("*.wav")):
        lang = "en" if wav.stem.startswith("C") else "hi"
        text, segs = run(model, str(wav), lang)
        out[wav.stem] = {"language": lang, "text": text, "segments": segs}
        if wav.stem.startswith("long_"):
            with wave.open(str(wav)) as w:  # 16-bit mono written by synth.py
                sr = w.getframerate()
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            tail = audio[-8 * sr:]
            # faster-whisper wants 16 kHz float input when given an array
            idx = np.linspace(0, len(tail) - 1, int(len(tail) * 16000 / sr)).astype(np.int64)
            out[wav.stem]["tail_8s_text"] = run(model, tail[idx], lang)[0]
        print(f"{wav.stem} [{lang}]: {text}", flush=True)
    (HERE / "transcripts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ASR DONE", flush=True)


if __name__ == "__main__":
    main()
