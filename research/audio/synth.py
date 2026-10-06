"""Kokoro synthesis with chunking + Whisper-CER reject loop (stages; see run_variant.sh).

  python synth.py plan  CALLS.jsonl ROOT        # stdlib: chunk every line, write ROOT/work/plan.json
  python synth.py tts   ROOT --try K            # venv-tts, GPU: synthesize chunks needing try K
  (asr.py ROOT)                                 # venv-asr, GPU: whisper + CER for every chunk wav
  python synth.py build ROOT                    # stdlib+numpy+soundfile: best try per chunk -> utterance wav

Kokoro native rate is 24 kHz mono float32 (written as 24 kHz PCM16, no resampling needed).
Try 0 = speed 1.0; tries 1..MAX_TRIES-1 = speed uniform(0.95,1.05) seeded per chunk. A chunk is retried
while its best CER > CER_MAX; after MAX_TRIES (3 attempts total) the best is kept and flagged.
Threshold per .asr.json record via cer_max_of(): ROOTs with work/asr_config.json (V4, user 2026-10-04) are scored by
Trelis Whisper-Hinglish + CER-study method 1 vs text_roman (asr.py / trelis_m1.py) with their own cer_max.
Resumable: every stage skips outputs that exist.

TTS backend switch (env TTS_BACKEND at `plan` time, default kokoro = V1 behaviour, byte-identical plan.json):
  kokoro     Kokoro lang 'h' (V1).                      tts stage: venv-tts
  f5cs       IndicF5 code-switch, all-Devanagari input. tts stage: venv-f5 (delegates to f5cs.py)
  kokoro_en  Kokoro lang 'a', English CONTROL calls.    tts stage: venv-tts
Non-kokoro chunks carry "backend" and "lang" in plan.json (asr.py uses lang). See F5_BACKEND.md.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_norm  # noqa: E402
import tts_backends  # noqa: E402

SR = 24000
CER_MAX = 0.35  # stock-Whisper (faster-whisper large-v3) cer_best records (V1/V3/CONTROL)
MAX_TRIES = 3
MAX_WORDS, MAX_DEVA = 12, 70
GAP_RANGE = (0.12, 0.20)
CONJ = {"और", "लेकिन", "पर", "तो", "क्योंकि", "ताकि", "बट", "एंड", "सो", "ऑर", "फिर", "जिससे", "वरना"}
BOUND_PUNCT = tuple(",।?!;:.—")
CONJ_EN = {"and", "but", "so", "because", "or", "then"}


def set_backend(backend):
    """Chunk limits for the backend (kokoro: the original constants above, unchanged)."""
    global MAX_WORDS, MAX_DEVA, SPOKEN_MAX, CONJ
    if backend not in tts_backends.BACKENDS:
        raise SystemExit(f"TTS_BACKEND must be one of {tts_backends.BACKENDS}, got {backend!r}")
    set_limits(backend)
    if backend == "kokoro_en":
        CONJ = CONJ | CONJ_EN


def set_limits(key):
    """Set the chunk limits (MAX_WORDS/MAX_DEVA/SPOKEN_MAX) from tts_backends.LIMITS[key]. D2: cmd_plan calls this
    for EVERY call (backend key or 'f5cs_long'), so a long-band call never leaks its limits into the next call."""
    global MAX_WORDS, MAX_DEVA, SPOKEN_MAX
    lim = tts_backends.LIMITS[key]
    MAX_WORDS, MAX_DEVA, SPOKEN_MAX = lim["max_words"], lim["max_deva"], lim["spoken_max"]


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def deva_len(s):
    return len(tts_norm.DEVA_RE.findall(s))


POSTP = {"से", "में", "को", "का", "की", "के", "पे", "है", "हैं", "था", "थी", "थे", "वाला", "वाली", "वाले", "लिए"}
SPOKEN_MAX = 100  # also split when the spoken (number-expanded) form is long


def _spoken_len(ws):
    return deva_len(tts_norm.tts_text(" ".join(ws)))


def _numish(w):
    return bool(re.search(r"[0-9०-९A-Za-z₹]", w))


def choose_splits(words):
    """Return (split indices (split before index j), forced flag)."""
    n = len(words)
    text = " ".join(words)
    spoken = _spoken_len(words)
    if n <= MAX_WORDS and deva_len(text) <= MAX_DEVA and spoken <= SPOKEN_MAX:
        return [], False
    k = 2 if (n <= 2 * MAX_WORDS and deva_len(text) <= 2 * MAX_DEVA and spoken <= 2 * SPOKEN_MAX) else 3
    k = min(k, max(1, n // 2))
    if k < 2:
        return [], False
    cand = {}
    for j in range(2, n - 1):
        prev_core = tts_norm._split_punct(words[j - 1])[1].lower()
        cur_core = tts_norm._split_punct(words[j])[1].lower()
        if prev_core in tts_norm.CURRENCY or cur_core in tts_norm.RUPEE_WORDS or cur_core in tts_norm.AMPM:
            continue
        if _numish(words[j - 1]) and _numish(words[j]):
            continue  # never split inside "6E 2134", "98765 43210", "DL 01 AB 4821"
        if words[j - 1].endswith(BOUND_PUNCT):
            cand[j] = 0
        elif cur_core in CONJ:
            cand[j] = 8  # prefer punctuation
        elif prev_core in POSTP:
            cand[j] = 14  # Hindi phrase boundary (after postposition / copula)
    best = None
    cl = sorted(cand)
    for kk in (k, 2) if k == 3 else (k,):
        for combo in itertools.combinations(cl, kk - 1):
            bounds = [0, *combo, n]
            if any(bounds[i + 1] - bounds[i] < 2 for i in range(kk)):
                continue
            lens = [_spoken_len(words[bounds[i]:bounds[i + 1]]) for i in range(kk)]
            cost = max(lens) + sum(cand[c] for c in combo)
            if best is None or cost < best[0]:
                best = (cost, list(combo))
        if best is not None:
            break
    if best is None:
        safe = [j for j in range(2, n - 1) if not (_numish(words[j - 1]) and _numish(words[j]))] or [n // 2]
        return [min(safe, key=lambda j: abs(j - n / 2))], True
    return best[1], False


def map_roman(tw, rw, splits):
    """Split roman words at positions matching the tts splits."""
    if not splits:
        return [rw], False
    if len(tw) == len(rw):
        return _cut(rw, splits), False
    tp = [j for j in range(1, len(tw)) if tw[j - 1].endswith(BOUND_PUNCT)]
    rp = [j for j in range(1, len(rw)) if rw[j - 1].endswith(BOUND_PUNCT)]
    if len(tp) == len(rp) and all(s in tp for s in splits):
        return _cut(rw, [rp[tp.index(s)] for s in splits]), False
    return _cut(rw, [max(1, min(len(rw) - 1, round(s * len(rw) / len(tw)))) for s in splits]), True


def _cut(ws, splits):
    b = [0, *splits, len(ws)]
    return [ws[b[i]:b[i + 1]] for i in range(len(b) - 1)]


def spoken_form(backend, ttext):
    if backend == "f5cs":
        return tts_backends.f5_text(ttext)
    if backend == "kokoro_en":
        return tts_backends.en_text(ttext)
    return tts_norm.tts_text(ttext)


def cmd_plan(calls_path, root):
    backend = os.environ.get("TTS_BACKEND", "kokoro")
    set_backend(backend)
    f5p = tts_backends.f5_params() if backend == "f5cs" else None
    root = Path(root)
    (root / "work").mkdir(parents=True, exist_ok=True)
    plan = []
    for line in open(calls_path, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        lkey = tts_backends.limits_key(backend, c)  # D2: per-call limits (== backend except f5cs long band)
        set_limits(lkey)
        for ti, t in enumerate(c["turns"]):
            voice = c["voices"]["agent" if t["speaker"] == "agent" else "customer"]
            if backend != "kokoro":  # v3-data fix: map V1 Kokoro Hindi keys to the backend's voices (kokoro path unchanged)
                voice = tts_backends.backend_voice(backend, c, t["speaker"], voice)
            tw, rw = t["text_tts"].split(), t["text_roman"].split()
            splits, forced = choose_splits(tw)
            rchunks, approx = map_roman(tw, rw, splits)
            tchunks = _cut(tw, splits)
            flags = (["forced_split"] if forced else []) + (["roman_map_approx"] if approx else [])
            if len(tw) != len(rw):
                flags.append(f"wordcount_mismatch_{len(tw)}v{len(rw)}")
            for ci, (tc, rc) in enumerate(zip(tchunks, rchunks)):
                ttext = " ".join(tc)
                spoken = spoken_form(backend, ttext)
                cflags = list(flags) + (["long_chunk"] if len(spoken) > 200 else [])
                if backend == "f5cs" and re.search(r"[A-Za-z]", tts_norm.tts_text(ttext)):
                    cflags.append("f5_latin_spelled")
                rng = random.Random(seed_of(c["call_id"], ti, ci))
                speeds = [1.0] + [round(rng.uniform(0.95, 1.05), 3) for _ in range(MAX_TRIES - 1)]
                plan.append({"call_id": c["call_id"], "turn": ti, "chunk": ci, "n_chunks": len(tchunks),
                             "speaker": t["speaker"], "voice": voice, "text_tts": ttext, "tts_input": spoken,
                             "text_roman": " ".join(rc), "speeds": speeds, "flags": cflags,
                             "gap_after": round(rng.uniform(*GAP_RANGE), 3) if ci < len(tchunks) - 1 else 0.0})
                if backend != "kokoro":
                    plan[-1].update({"backend": backend, "lang": tts_backends.LANG[backend]})
                if backend == "f5cs":
                    plan[-1]["f5"] = f5p
                if lkey != backend:  # D2: only long-band chunks carry this (V1/V3 plan.json unchanged)
                    plan[-1]["limits"] = lkey
    # stale-output guard: if a call's chunk texts/voices/speeds changed (regenerated call, new chunking or
    # normalisation), wipe its chunks/utt/align so a resumed run cannot reuse old audio under new text.
    import shutil
    by_call = {}
    for p in plan:
        by_call.setdefault(p["call_id"], []).append((p["turn"], p["chunk"], p["voice"], p["tts_input"], p["speeds"],
                                                     p["text_roman"]) + ((backend,) if backend != "kokoro" else ())
                                                    + ((json.dumps(f5p, sort_keys=True),) if f5p else ()))
    wiped = 0
    for cid, items in by_call.items():
        sig = hashlib.md5(json.dumps(items, ensure_ascii=False).encode()).hexdigest()
        sf_ = root / "work" / "chunks" / cid / ".sig"
        if sf_.exists() and sf_.read_text() == sig:
            continue
        if sf_.exists() or (root / "work" / "chunks" / cid).exists():
            wiped += 1
        for sub in ("chunks", "utt", "align"):
            shutil.rmtree(root / "work" / sub / cid, ignore_errors=True)
        sf_.parent.mkdir(parents=True, exist_ok=True)
        sf_.write_text(sig)
    print(f"plan: {wiped} calls changed since last run -> their chunks/utt/align wiped")
    (root / "work" / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=0), encoding="utf-8")
    n_split = sum(1 for p in plan if p["chunk"] == 1)
    print(f"plan: backend={backend}, {len(plan)} chunks, {n_split} lines split, -> {root/'work'/'plan.json'}")


def chunk_stem(root, p, k):
    return Path(root) / "work" / "chunks" / p["call_id"] / f"t{p['turn']:02d}_c{p['chunk']}_try{k}"


def cer_max_of(r):
    """Reject threshold for one .asr.json record. Trelis/method-1 records (scorer 'trelis_m1', V4 from 2026-10-04)
    carry their own cer_max (from ROOT/work/asr_config.json); old stock-Whisper records have no scorer -> CER_MAX."""
    return float(r["cer_max"]) if r.get("scorer") else CER_MAX


def asr_result(root, p, k):
    f = chunk_stem(root, p, k).with_suffix(".asr.json")
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def needs_try(root, p, k):
    if chunk_stem(root, p, k).with_suffix(".wav").exists():
        return False
    if k == 0:
        return True
    prev = [asr_result(root, p, j) for j in range(k)]
    if any(r is None for r in prev):
        return False  # previous try not yet scored
    return all(r["cer"] > cer_max_of(r) for r in prev)  # == min(cer) > CER_MAX for old records


def trim(audio, sr=SR, margin=0.04):
    import numpy as np
    win = int(0.01 * sr)
    n = len(audio) // win
    if n == 0:
        return audio
    rms = np.sqrt((audio[: n * win].reshape(n, win) ** 2).mean(1) + 1e-12)
    thr = max(rms.max() * 0.03, 1e-4)
    idx = np.where(rms > thr)[0]
    if len(idx) == 0:
        return audio
    a = max(0, idx[0] * win - int(margin * sr))
    b = min(len(audio), (idx[-1] + 1) * win + int(margin * sr))
    return audio[a:b]


def cmd_tts(root, k, shard=0, nshards=1):
    plan = json.loads((Path(root) / "work" / "plan.json").read_text(encoding="utf-8"))
    backends = {p.get("backend", "kokoro") for p in plan}
    if len(backends) != 1:
        raise SystemExit(f"plan.json mixes backends {backends}")
    backend = backends.pop()
    if backend == "f5cs":
        import f5cs  # venv-f5
        return f5cs.cmd_tts(root, k, shard, nshards)
    if nshards != 1:
        raise SystemExit("--nshards > 1 is implemented for f5cs only")
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    lang, warm = ("a", "Hello.") if backend == "kokoro_en" else ("h", "नमस्ते।")
    todo = [p for p in plan if needs_try(root, p, k)]
    print(f"tts try {k}: {len(todo)} chunks to synthesize", flush=True)
    if not todo:
        return
    t0 = time.time()
    pipe = KPipeline(lang_code=lang, device="cuda" if torch.cuda.is_available() else "cpu",
                     repo_id="hexgrad/Kokoro-82M")
    for v in sorted({p["voice"] for p in todo}):
        list(pipe(warm, voice=v))
    t_load = time.time() - t0
    t1 = time.time()
    audio_s = 0.0
    for i, p in enumerate(todo):
        out = chunk_stem(root, p, k)
        out.parent.mkdir(parents=True, exist_ok=True)
        pieces = [np.asarray(r.audio, dtype=np.float32) for r in pipe(p["tts_input"], voice=p["voice"],
                                                                      speed=p["speeds"][k]) if r.audio is not None]
        a = trim(np.concatenate(pieces)) if pieces else np.zeros(int(0.2 * SR), np.float32)
        if backend == "kokoro_en" and pieces:  # control: same -20 dBFS RMS as f5cs (V1 kokoro path unchanged)
            a = a * (10 ** (-20 / 20) / float(np.sqrt(np.mean(a ** 2) + 1e-12)))
            a = a * min(1.0, 0.99 / max(float(np.abs(a).max()), 1e-9))
        a = np.clip(a, -1, 1)
        sf.write(str(out) + ".tmp.wav", a, SR, subtype="PCM_16")
        Path(str(out) + ".tmp.wav").rename(out.with_suffix(".wav"))
        audio_s += len(a) / SR
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    wall = time.time() - t1
    rec = {"stage": f"tts_try{k}", **({"backend": backend} if backend != "kokoro" else {}), "chunks": len(todo), "audio_s": round(audio_s, 1), "wall_s": round(wall, 1),
           "load_s": round(t_load, 1), "x_realtime": round(audio_s / max(wall, 1e-6), 1)}
    print(rec, flush=True)
    with open(Path(root) / "work" / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_build(root):
    """Pick best try per chunk, concatenate chunks with gaps -> work/utt/<call>/tNN.wav + .json."""
    import numpy as np
    import soundfile as sf
    root = Path(root)
    plan = json.loads((root / "work" / "plan.json").read_text(encoding="utf-8"))
    by_utt = {}
    for p in plan:
        by_utt.setdefault((p["call_id"], p["turn"]), []).append(p)
    missing = 0
    stats = {"utts": 0, "chunks": 0, "flag_cer": 0, "retried": 0}
    for (cid, ti), chunks in sorted(by_utt.items()):
        chunks.sort(key=lambda p: p["chunk"])
        audio, meta = [], []
        t = 0.0
        ok = True
        for p in chunks:
            tries = [(k, asr_result(root, p, k)) for k in range(MAX_TRIES)]
            tries = [(k, r) for k, r in tries if r is not None and chunk_stem(root, p, k).with_suffix(".wav").exists()]
            if not tries:
                ok = False
                break
            k, r = min(tries, key=lambda x: (x[1]["cer"], x[0]))
            a, sr = sf.read(str(chunk_stem(root, p, k).with_suffix(".wav")), dtype="float32")
            assert sr == SR
            flags = list(p["flags"])
            cmax = cer_max_of(r)
            if r["cer"] > cmax:
                flags.append("cer_reject_kept_best")
                stats["flag_cer"] += 1
            if len(tries) > 1:
                stats["retried"] += 1
            meta.append({"chunk": p["chunk"], "start": round(t, 3), "end": round(t + len(a) / SR, 3),
                         "try": k, "n_tries": len(tries), "speed": p["speeds"][k], "cer": r["cer"],
                         "cer_all": [round(x[1]["cer"], 3) for x in tries], "asr": r["text"],
                         "tts_input": p["tts_input"], "text_tts": p["text_tts"], "text_roman": p["text_roman"],
                         "flags": flags})
            if r.get("scorer"):  # Trelis/method-1 record: threshold + scorer travel with the chunk (assemble reads them)
                if r.get("num_ok") is False:
                    flags.append("num_mismatch")
                meta[-1].update({"scorer": r["scorer"], "cer_max": cmax, "wer": r.get("wer"), "num_ok": r.get("num_ok")})
            audio.append(a)
            t += len(a) / SR
            if p["gap_after"]:
                audio.append(np.zeros(int(p["gap_after"] * SR), np.float32))
                t += p["gap_after"]
            stats["chunks"] += 1
        if not ok:
            missing += 1
            continue
        d = root / "work" / "utt" / cid
        d.mkdir(parents=True, exist_ok=True)
        wav = np.concatenate(audio)
        sf.write(str(d / f"t{ti:02d}.wav"), wav, SR, subtype="PCM_16")
        (d / f"t{ti:02d}.json").write_text(json.dumps({"call_id": cid, "turn": ti, "speaker": chunks[0]["speaker"],
                                                        "duration": round(len(wav) / SR, 3), "chunks": meta},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
        stats["utts"] += 1
    print(f"build: {stats}, utterances missing chunks: {missing}")
    if missing:
        sys.exit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "tts", "build"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--try", dest="k", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)  # D2: f5cs only; shard I of N over plan.json indices (i % N == I)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.args[0], a.args[1])
    elif a.cmd == "tts":
        cmd_tts(a.args[0], a.k, a.shard, a.nshards)
    else:
        cmd_build(a.args[0])
