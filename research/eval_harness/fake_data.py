"""Self-test only: a fake assemble.py-shaped data root from audio/testdata TTS chunks (try0), no alignment.

  /workspace/venv-tts/bin/python fake_data.py  -> /workspace/hinglish/tests/selftest/data/{calls.jsonl,stereo/*}
"""
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path("/workspace/hinglish/audio/testdata")
DST = Path("/workspace/hinglish/tests/selftest/data")
SR = 24000


def main():
    (DST / "stereo").mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "calls.jsonl", DST / "calls.jsonl")
    for line in open(SRC / "calls.jsonl", encoding="utf-8"):
        c = json.loads(line)
        cid = c["call_id"]
        t, pieces, tl = 1.0, [], []
        for ti, turn in enumerate(c["turns"]):
            chunks = sorted((SRC / "out/work/chunks" / cid).glob(f"t{ti:02d}_c*_try0.wav"))
            a = np.concatenate([np.concatenate([sf.read(str(p), dtype="float32")[0], np.zeros(int(0.15 * SR), np.float32)])
                                for p in chunks])
            ch = 0 if turn["speaker"] == "agent" else 1
            pieces.append((ch, t, a))
            tl.append({"turn": ti, "speaker": turn["speaker"], "start": round(t, 3), "end": round(t + len(a) / SR, 3)})
            t += len(a) / SR + (turn.get("pause_after_s") or 0.4)
        out = np.zeros((int((t + 0.5) * SR), 2), np.float32)
        for ch, st, a in pieces:
            i = int(st * SR)
            out[i:i + len(a), ch] += a
        sf.write(str(DST / "stereo" / f"{cid}.wav"), out, SR, subtype="PCM_16")
        (DST / "stereo" / f"{cid}.json").write_text(json.dumps({"alignments": [], "turns": tl}))
        print(cid, round(t, 1), "s")


if __name__ == "__main__":
    main()
