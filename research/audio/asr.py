"""Chunk ASR QC: transcription + CER of every synthesized chunk lacking a score.

  /workspace/venv-asr/bin/python asr.py ROOT [--shard I --nshards N]
Writes work/chunks/<call>/<stem>.asr.json (resumable: chunks with an .asr.json are skipped).

Engine is chosen per ROOT (user 2026-10-04, NOTES "D2 / V4 audio QC: Trelis ASR"):
* ROOT/work/asr_config.json with {"model": "trelis", ...} (V4): Trelis/whisper-hinglish-preview (<|hi|><|mixedcode|>,
  romanised/mixed Hinglish output) scored with CER-study method 1 against the chunk's text_roman (trelis_m1.py).
  Record: {text, cer, wer, scorer: "trelis_m1", cer_max, num_ok, ref, ref_m1, hyp_m1, ...}. Needs transformers>=5:
  if this interpreter (venv-asr) lacks it, the process re-execs itself under /workspace/venv-tts/bin/python (same pid,
  so run_variant.sh's `wait $p` still works). Config "procs" (default 1) = how many of the N shard processes do work;
  shards >= procs exit 0 at once (decided from the config file, never from live GPU memory, so all shards agree and
  no chunk is left unscored). "batch" = Trelis batch size (default 16; halves itself on CUDA OOM).
* otherwise (V1/V3/CONTROL, unchanged): faster-whisper large-v3 (language hi), CER = tts_norm.cer_best =
  min(Devanagari CER, folded-Latin CER) after clean_hyp; ref = tts_input (spoken form). Chunks with plan "lang": "en"
  (kokoro_en control) use language="en" and tts_backends.cer_en. Records carry no "scorer" key.
Downstream (synth.needs_try / build, assemble, make_stats_v4) read the threshold per record via synth.cer_max_of().
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_norm  # noqa: E402
import tts_backends  # noqa: E402
import trelis_m1  # noqa: E402

TRELIS_PY = "/workspace/venv-tts/bin/python"


def write_rec(stem, rec):
    tmp = stem.with_suffix(".asr.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.rename(stem.with_suffix(".asr.json"))


def run_trelis(a, root, plan, todo_all, cfg):
    procs = max(1, min(int(cfg.get("procs", 1)), a.nshards))
    if a.shard >= procs:
        print(f"asr shard {a.shard}/{a.nshards}: trelis procs={procs} -> this shard idle", flush=True)
        return
    todo = todo_all[a.shard::procs]
    print(f"asr shard {a.shard}/{a.nshards} (trelis, active procs {procs}): {len(todo)} chunks", flush=True)
    if not todo:
        return
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("soundfile") is None:
        if os.environ.get("ASR_TRELIS_REEXEC"):
            raise SystemExit("trelis: transformers/soundfile missing even after re-exec")
        os.environ["ASR_TRELIS_REEXEC"] = "1"
        sys.stdout.flush()
        os.execv(TRELIS_PY, [TRELIS_PY, str(Path(__file__).resolve()), *sys.argv[1:]])
    cer_max = float(cfg["cer_max"])
    lex = trelis_m1.build_lexicon(plan)
    t0 = time.time()
    T = trelis_m1.Transcriber(device=a.device, batch=int(cfg.get("batch", 16)))
    t_load = time.time() - t0
    t1 = time.time()
    audio_s, n = 0.0, 0
    for i in range(0, len(todo), 64):
        part = todo[i:i + 64]
        auds = [T.load16(stem.with_suffix(".wav")) for _, stem in part]
        texts = T.transcribe(auds)
        for (p, stem), au, text in zip(part, auds, texts):
            audio_s += len(au) / 16000
            try:
                s = trelis_m1.score(p["text_roman"], text, lex)
                rec = {"text": text, "cer": s["cer"], "wer": s["wer"], "scorer": trelis_m1.SCORER,
                       "asr_model": trelis_m1.REPO, "cer_max": cer_max, "num_ok": s["num_ok"], "ref": p["text_roman"],
                       "ref_m1": s["ref_m1"], "hyp_m1": s["hyp_m1"], "hyp_deva_tokens": s["hyp_deva_tokens"],
                       "hyp_oov_tokens": s["hyp_oov_tokens"]}
            except Exception as e:  # noqa: BLE001  an odd transcript must not kill the whole shard
                rec = {"text": text, "cer": 1.0, "scorer": trelis_m1.SCORER, "asr_model": trelis_m1.REPO,
                       "cer_max": cer_max, "cer_error": f"{type(e).__name__}:{e}"[:120], "ref": p["text_roman"]}
            write_rec(stem, rec)
            n += 1
        if (i // 64) % 4 == 3:
            print(f"  {n}/{len(todo)}", flush=True)
    wall = time.time() - t1
    rec = {"stage": f"asr_shard{a.shard}of{a.nshards}", "engine": "trelis_m1", "active_procs": procs,
           "chunks": len(todo), "audio_s": round(audio_s, 1), "wall_s": round(wall, 1), "load_s": round(t_load, 1),
           "x_realtime": round(audio_s / max(wall, 1e-6), 1)}
    print(rec, flush=True)
    with open(root / "work" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--cpu-threads", type=int, default=32)
    a = ap.parse_args()
    root = Path(a.root)
    plan = json.loads((root / "work" / "plan.json").read_text(encoding="utf-8"))
    todo = []
    for p in plan:
        for k in range(len(p["speeds"])):
            stem = root / "work" / "chunks" / p["call_id"] / f"t{p['turn']:02d}_c{p['chunk']}_try{k}"
            if stem.with_suffix(".wav").exists() and not stem.with_suffix(".asr.json").exists():
                todo.append((p, stem))
    cfg = trelis_m1.load_config(root)
    if cfg and cfg.get("model") == "trelis" and any(p.get("lang", "hi") != "en" for p in plan):
        return run_trelis(a, root, plan, todo, cfg)
    todo = todo[a.shard::a.nshards]
    print(f"asr shard {a.shard}/{a.nshards}: {len(todo)} chunks", flush=True)
    if not todo:
        return
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = (WhisperModel("large-v3", device="cuda", compute_type="float16") if a.device == "cuda" else
             WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=a.cpu_threads))
    t_load = time.time() - t0
    t1 = time.time()
    audio_s = 0.0
    for i, (p, stem) in enumerate(todo):
        lang = p.get("lang", "hi")
        segs, info = model.transcribe(str(stem.with_suffix(".wav")), language=lang, beam_size=a.beam,
                                      temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        audio_s += info.duration
        try:
            if lang == "en":
                rec = {"text": text, "cer": round(tts_backends.cer_en(p["tts_input"], text), 4), "lang": "en",
                       "ref": p["tts_input"]}
            else:
                rec = {"text": text, "cer": round(tts_norm.cer_best(p["tts_input"], text), 4),
                       "cer_deva": round(tts_norm.cer(p["tts_input"], tts_norm.clean_hyp(text)), 4),
                       "cer_latin": round(tts_norm.cer_latin(p["tts_input"], tts_norm.clean_hyp(text)), 4),
                       "latin_chars": len(tts_norm.LATIN_RE.findall(text)), "ref": p["tts_input"]}
        except Exception as e:  # noqa: BLE001  review fix: an odd transcript must not kill the whole shard
            rec = {"text": text, "cer": 1.0, "cer_error": f"{type(e).__name__}:{e}"[:120], "ref": p["tts_input"]}
        write_rec(stem, rec)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    wall = time.time() - t1
    rec = {"stage": f"asr_shard{a.shard}of{a.nshards}", "chunks": len(todo), "audio_s": round(audio_s, 1),
           "wall_s": round(wall, 1), "load_s": round(t_load, 1), "x_realtime": round(audio_s / max(wall, 1e-6), 1)}
    print(rec, flush=True)
    with open(root / "work" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
