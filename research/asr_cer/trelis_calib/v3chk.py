import sys, json
import os  # monorepo shim: trelis_m1/synth/tts_norm/tts_backends are in research/audio ($AUDIO_DIR); data under $HINGLISH_ROOT
from pathlib import Path as _P
HINGLISH_ROOT = os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(_P(__file__).resolve().parents[3] / "research" / "audio"))  # was /workspace/hinglish/tmp_trelis/audio_new (same synth.py md5)
sys.path.insert(0, AUDIO_DIR)
import synth
R = HINGLISH_ROOT + "/data/V3"
P = json.load(open(R + "/work/plan.json"))
d = n = 0
for p in P:
    rs = [synth.asr_result(R, p, j) for j in range(3)]
    for k in (1, 2):
        prev = rs[:k]
        if all(r is not None for r in prev):
            n += 1
            a = min(r["cer"] for r in prev) > 0.35
            b = all(r["cer"] > synth.cer_max_of(r) for r in prev)
            d += a != b
print("V3 old-vs-new retry rule: compared", n, "disagreements", d)
