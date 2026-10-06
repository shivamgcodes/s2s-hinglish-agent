"""Script decision test for V3 F5 input (venv-f5 render; venv-asr score). Writes only under audio/f5_test/.
  python f5_scripttest.py render     # S1-S8 x 4 voices x {deva, mixed}, duration fix, batched per voice
  python f5_scripttest.py score      # (venv-asr) Whisper large-v3 hi + CER (ref = all-Devanagari spoken form)
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/workspace/hinglish/tts_ab_f5/scripts")
import tts_backends  # noqa: E402

import os
TAG = os.environ.get("F5_TEST_TAG", "b8")
OUT = HERE / ("f5_test" if TAG == "b8" else f"f5_test_{TAG}")
VOICES = ["ritu_hinglish", "orato_male", "orato_female", "fleurs_hi_m1559"]


def items():
    import common  # tts_ab_f5 comparison: SENTS, MIXED
    sents = json.loads(Path("/workspace/hinglish/tts_ab_f5/sents.json").read_text(encoding="utf-8"))
    res = []
    for s in sents:
        res.append({"key": s["key"], "deva": tts_backends.f5_text(s["text_tts"]), "mixed": common.MIXED[s["key"]],
                    "kokoro_dur": s.get("dataset_dur", 0.0)})
    return res


def render():
    import numpy as np
    import soundfile as sf
    import torch
    import f5cs
    import synth
    OUT.mkdir(parents=True, exist_ok=True)
    its = items()
    t0 = time.time()
    eng = f5cs.F5CS("cuda")
    load_s = time.time() - t0
    eng.render("ritu_hinglish", ["नमस्ते।"], [1.0], 0)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    timing = []
    for v in VOICES:
        vi = eng.voice(v)
        for kind in ("deva", "mixed"):
            texts = [it[kind] for it in its]
            torch.cuda.synchronize()
            t = time.time()
            if TAG == "b8":
                waves = eng.render(v, texts, [1.0] * len(texts), 1234)
            else:
                waves = [eng.render(v, [x], [1.0], 1234)[0] for x in texts]
            torch.cuda.synchronize()
            wall = time.time() - t
            timing.append({"voice": v, "kind": kind, "n": len(texts), "wall_s": round(wall, 2),
                           "s_per_chunk": round(wall / len(texts), 3)})
            for it, a in zip(its, waves):
                a = f5cs.loudnorm(synth.trim(a))
                name = f"{it['key']}_{v}_{kind}.wav"
                sf.write(str(OUT / name), np.clip(a, -1, 1), 24000, subtype="PCM_16")
                rows.append({"file": name, "key": it["key"], "voice": v, "kind": kind, "input": it[kind],
                             "ref_deva": it["deva"], "dur": round(len(a) / 24000, 3), "kokoro_dur": it["kokoro_dur"],
                             "syl": tts_backends.syllables(it[kind]), "rate_used": vi["rate_used"]})
    # batch vs single: S1, S3, S6 deva, ritu + fleurs, batch of 1 with the same seed
    for v in (("ritu_hinglish", "fleurs_hi_m1559") if TAG == "b8" else ()):
        for it in its:
            if it["key"] not in ("S1", "S3", "S6"):
                continue
            t = time.time()
            a = eng.render(v, [it["deva"]], [1.0], 1234)[0]
            torch.cuda.synchronize()
            timing.append({"voice": v, "kind": "deva_single", "n": 1, "wall_s": round(time.time() - t, 2)})
            a = f5cs.loudnorm(synth.trim(a))
            name = f"{it['key']}_{v}_devasingle.wav"
            sf.write(str(OUT / name), np.clip(a, -1, 1), 24000, subtype="PCM_16")
            rows.append({"file": name, "key": it["key"], "voice": v, "kind": "deva_single", "input": it["deva"],
                         "ref_deva": it["deva"], "dur": round(len(a) / 24000, 3), "kokoro_dur": it["kokoro_dur"],
                         "syl": tts_backends.syllables(it["deva"]), "rate_used": eng.voice(v)["rate_used"]})
    info = {"load_s": round(load_s, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "timing": timing, "voices": {v: {k: eng.voice(v)[k] for k in ("dur", "voiced", "syl", "rate_syl_s", "rate_used")}
                                         for v in VOICES}}
    (OUT / "render.json").write_text(json.dumps({"info": info, "rows": rows}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps(info, indent=1))


def score():
    import numpy as np
    from faster_whisper import WhisperModel, decode_audio
    import tts_norm
    sys.path.insert(0, "/workspace/hinglish/tts_ab_f5/scripts")
    import common
    d = json.loads((OUT / "render.json").read_text(encoding="utf-8"))
    m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for r in d["rows"]:
        segs, _ = m.transcribe(str(OUT / r["file"]), language="hi", beam_size=5, temperature=0.0,
                               condition_on_previous_text=False, vad_filter=False)
        r["asr"] = " ".join(s.text.strip() for s in segs).strip()
        r["cer"] = round(tts_norm.cer_best(r["ref_deva"], r["asr"]), 4)
        a = decode_audio(str(OUT / r["file"]), sampling_rate=24000)
        r["sil80"] = len(common.silences(a))
    (OUT / "scored.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print("rate", {v: d["info"]["voices"][v]["rate_used"] for v in VOICES})
    print(f"{'voice':18s} {'kind':12s} n  meanDur  kokDur  meanCER  maxCER  nCER>.35  sil80  syl/s")
    for v in VOICES:
        for kind in ("deva", "mixed", "deva_single"):
            rs = [r for r in d["rows"] if r["voice"] == v and r["kind"] == kind]
            if not rs:
                continue
            print(f"{v:18s} {kind:12s} {len(rs)}  {np.mean([r['dur'] for r in rs]):6.2f}  "
                  f"{np.mean([r['kokoro_dur'] for r in rs]):6.2f}  {np.mean([r['cer'] for r in rs]):6.3f}  "
                  f"{max(r['cer'] for r in rs):6.3f}  {sum(r['cer'] > 0.35 for r in rs):3d}  "
                  f"{np.mean([r['sil80'] for r in rs]):5.2f}  {np.mean([r['syl'] / r['dur'] for r in rs]):5.2f}")
    for r in d["rows"]:
        print(r["file"], r["dur"], r["cer"], "|", r["asr"])


if __name__ == "__main__":
    render() if sys.argv[1] == "render" else score()
