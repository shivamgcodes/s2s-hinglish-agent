"""Metric 6 helper: faster-whisper large-v3 (language hi) on every model output wav of the given tags.

  flock /workspace/hinglish/gpu.lock /workspace/venv-asr/bin/python asr_pass.py TAG [TAG ...] [--nshards N --shard I]
Writes <run>.asr.json {"text", "language", "duration"} next to <run>.wav (skips existing). Same decoding
settings as audio/asr.py (beam 5, temp 0, no previous-text conditioning), no initial prompt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    todo = []
    for tag in a.tags:
        for m in sorted((OUT / tag).rglob("*.meta.json")):
            wav = m.with_name(m.name.replace(".meta.json", ".wav"))
            asr = m.with_name(m.name.replace(".meta.json", ".asr.json"))
            if wav.exists() and not asr.exists():
                todo.append((wav, asr))
    todo = todo[a.shard::a.nshards]
    print(f"[asr_pass] {len(todo)} wavs", flush=True)
    if not todo:
        return
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    t0, audio_s = time.time(), 0.0
    for wav, asr in todo:
        segs, info = model.transcribe(str(wav), language="hi", beam_size=5, temperature=0.0,
                                      condition_on_previous_text=False, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        audio_s += info.duration
        tmp = asr.with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": text, "language": "hi", "duration": round(info.duration, 2)},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.rename(asr)
    wall = time.time() - t0
    print(f"[asr_pass] {len(todo)} wavs, {audio_s:.0f} s audio in {wall:.0f} s ({audio_s / max(wall, 1e-6):.1f}x rt)",
          flush=True)


if __name__ == "__main__":
    main()
