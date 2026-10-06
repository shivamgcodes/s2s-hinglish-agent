"""Whisper large-v3 (faster-whisper, venv-asr) on every clip in out/: language hi, beam 5, temperature 0,
no conditioning on previous text, no VAD (same settings as indicf5-check/scripts/asr.py). GPU fp16.
Writes /workspace/hinglish/tts_ab_f5/asr.json."""
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")) / "tts_ab_f5"
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
out = {"_meta": {"model": "large-v3", "device": "cuda", "compute_type": "float16", "language": "hi", "beam_size": 5}}
for wav in sorted((ROOT / "out").glob("*.wav")):
    segs, _ = m.transcribe(str(wav), language="hi", beam_size=5, temperature=0.0,
                           condition_on_previous_text=False, vad_filter=False)
    out[wav.name] = " ".join(s.text.strip() for s in segs).strip()
    print(wav.name, "|", out[wav.name], flush=True)
(ROOT / "asr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("ASR DONE")
