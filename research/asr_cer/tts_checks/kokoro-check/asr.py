"""Transcript-back check: faster-whisper large-v3 (multilingual) on CPU over every clip in out/.

Run on the pod:
  CUDA_VISIBLE_DEVICES="" HF_HOME=/root/hf-asr /workspace/venv-asr/bin/python asr.py

language="hi" for groups A, B, D and the long-input clips; language="en" for group C.
Writes transcripts.json: {clip_stem: {"language", "text", "segments": [{start, end, text}]}}.
"""
import json
import time
from importlib.metadata import version
from pathlib import Path

from faster_whisper import WhisperModel

HERE = Path(__file__).parent
MODEL = "large-v3"


def main():
    t0 = time.time()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=16)
    print(f"{MODEL} loaded on CPU (int8) in {time.time() - t0:.1f} s", flush=True)
    out = {"_meta": {"model": MODEL, "device": "cpu", "compute_type": "int8", "beam_size": 5,
                     "faster_whisper": version("faster-whisper"), "ctranslate2": version("ctranslate2")}}
    for wav in sorted((HERE / "out").glob("*.wav")):
        lang = "en" if wav.stem.startswith("C") else "hi"
        segments, _ = model.transcribe(str(wav), language=lang, beam_size=5, temperature=0.0,
                                       condition_on_previous_text=False, vad_filter=False)
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        out[wav.stem] = {"language": lang, "text": " ".join(s["text"] for s in segs).strip(), "segments": segs}
        print(f"{wav.stem} [{lang}]: {out[wav.stem]['text']}", flush=True)
    (HERE / "transcripts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ASR DONE", flush=True)


if __name__ == "__main__":
    main()
