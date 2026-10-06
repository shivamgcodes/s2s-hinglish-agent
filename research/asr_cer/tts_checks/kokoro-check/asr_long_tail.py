"""Extra check for the long-input clips: word timestamps for the whole clip, and a separate transcript of
the last 8 s, to tell audio truncation apart from Whisper stopping early."""
import json, wave
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
HERE = Path(__file__).parent
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=16)
out = {}
for v in ["hf_beta", "hm_omega"]:
    p = HERE / "out" / f"long_A_{v}.wav"
    w = wave.open(str(p)); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    segs, _ = m.transcribe(str(p), language="hi", beam_size=5, temperature=0.0, word_timestamps=True,
                           condition_on_previous_text=False)
    words = [(round(wd.start, 2), round(wd.end, 2), wd.word) for s in segs for wd in (s.words or [])]
    tail = x[-8 * sr:]
    segs2, _ = m.transcribe(tail, language="hi", beam_size=5, temperature=0.0, condition_on_previous_text=False)
    tail_text = " ".join(s.text.strip() for s in segs2)
    rms = [round(float(np.sqrt((x[i:i + sr] ** 2).mean())), 4) for i in range(0, len(x), sr)]
    out[v] = {"duration_s": round(len(x) / sr, 2), "last_word": words[-1] if words else None, "n_words": len(words),
              "last_8s_transcript": tail_text, "rms_per_second": rms}
    print(v, json.dumps(out[v], ensure_ascii=False))
(HERE / "long_tail_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
