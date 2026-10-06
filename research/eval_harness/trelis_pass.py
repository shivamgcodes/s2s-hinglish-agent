"""D2 spec section 8: Trelis Whisper-Hinglish ASR on every model output wav of the given tags, silence-trimmed.

  flock /workspace/hinglish/gpu.lock /workspace/venv-tts/bin/python trelis_pass.py TAG [TAG ...] [--batch 8] [--limit N]
Writes <run>.trelis.json next to <run>.wav (skips existing; <run>.asr.json = score.py's large-v3 m6 file is NOT touched).

Model / decoding = cer_study/trelis/TRANSCRIBE.md (tests/trelis_tx.py): Trelis/whisper-hinglish-preview, bf16, decoder
prompt <|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>, greedy, max_new_tokens 440 (trelis_m1's
Transcriber batching/OOM halving, its _gen replaced by a 440-token copy);
24 kHz -> 16 kHz FFT resample; chunks <= 28 s cut at the lowest-energy 20 ms frame in the 18-28 s window.

Silence trimming (defined here, 2026-10-05; the CER study transcribed untrimmed wavs): 20 ms frames, frame level =
20*log10(RMS) dBFS on the 16 kHz signal. Active = level > -50 dBFS (PersonaPlex output silence floor is ~-74 dBFS;
speech frames are mostly -40..-10). Active frames are dilated by 0.2 s on each side; the merged regions are kept, every
gap between regions is replaced by 0.3 s of digital silence, leading/trailing silence dropped. Files with no active
frame get text "" (no ASR call).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[2]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_AUDIO_DIR') or str(_REPO / 'research/audio'))
from tcommon import OUT  # noqa: E402

THR_DB, PAD_S, GAP_S, HOP = -50.0, 0.2, 0.3, 320  # HOP = 20 ms at 16 kHz


def trim(a, np):
    n = len(a) // HOP
    if n == 0:
        return a[:0], 0
    db = 20 * np.log10(np.sqrt((a[:n * HOP].reshape(n, HOP) ** 2).mean(1)) + 1e-9)
    act = db > THR_DB
    pad = int(round(PAD_S * 16000 / HOP))
    if not act.any():
        return a[:0], 0
    idx = np.flatnonzero(act)
    dil = np.zeros(n, bool)
    for i in idx:
        dil[max(0, i - pad):i + pad + 1] = True
    regs, i = [], 0
    while i < n:
        if dil[i]:
            j = i
            while j < n and dil[j]:
                j += 1
            regs.append((i * HOP, j * HOP))
            i = j
        else:
            i += 1
    gap = np.zeros(int(GAP_S * 16000), np.float32)
    parts = []
    for k, (s, e) in enumerate(regs):
        if k:
            parts.append(gap)
        parts.append(a[s:e])
    return np.concatenate(parts).astype(np.float32), len(regs)


def chunks(a, np, sr=16000):
    out, i = [], 0
    while len(a) - i > 28 * sr:
        w = a[i + 18 * sr:i + 28 * sr]
        hop = sr // 50
        e = np.array([np.mean(w[k:k + hop] ** 2) for k in range(0, len(w) - hop, hop)])
        cut = i + 18 * sr + int(np.argmin(e)) * hop
        out.append(a[i:cut])
        i = cut
    out.append(a[i:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    todo = []
    for tag in a.tags:
        for m in sorted((OUT / tag).rglob("*.meta.json")):
            if "logs" in m.relative_to(OUT / tag).parts:
                continue
            wav = m.with_name(m.name.replace(".meta.json", ".wav"))
            out = m.with_name(m.name.replace(".meta.json", ".trelis.json"))
            if wav.exists() and not out.exists():
                todo.append((wav, out))
    if a.limit:
        todo = todo[:a.limit]
    print(f"[trelis_pass] {len(todo)} wavs", flush=True)
    if not todo:
        return
    import numpy as np
    import trelis_m1
    tx = trelis_m1.Transcriber(batch=a.batch)

    def gen440(audios):  # = Transcriber._gen with the study's max_new_tokens 440 (28 s chunks; 224 can truncate)
        torch = tx.torch
        feat = tx.proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        feat = feat.to(tx.device, tx.dtype)
        dec = torch.tensor([tx.prompt] * len(audios), device=tx.device)
        with torch.inference_mode():
            o = tx.model.generate(input_features=feat, decoder_input_ids=dec, max_new_tokens=440)
        return [tx.proc.tokenizer.decode(r, skip_special_tokens=True).strip() for r in o]
    tx._gen = gen440
    t0, audio_s, kept_s = time.time(), 0.0, 0.0
    for n, (wav, out) in enumerate(todo):
        import soundfile as sf  # trelis_m1.Transcriber.load16 truncates at 30 s: load + resample here
        y, sr = sf.read(str(wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        n_out = int(round(len(y) * 16000 / sr))
        X = np.fft.rfft(y)
        x = (np.fft.irfft(X[: n_out // 2 + 1], n_out) * (n_out / len(y))).astype(np.float32)
        tr, nreg = trim(x, np)
        cs = chunks(tr, np) if len(tr) else []
        cs = [c for c in cs if len(c) >= 1600]  # < 0.1 s tail chunk: nothing to transcribe
        texts = tx.transcribe(cs) if cs else []
        rec = {"text": " ".join(t for t in texts if t).strip(), "chunk_texts": texts, "n_chunks": len(cs),
               "dur_s": round(len(x) / 16000, 2), "trimmed_s": round(len(tr) / 16000, 2), "n_regions": nreg,
               "model": trelis_m1.REPO, "decode": "greedy, <|hi|><|mixedcode|>, max_new_tokens 440",
               "trim": {"thr_dbfs": THR_DB, "pad_s": PAD_S, "gap_s": GAP_S}}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out)
        audio_s += rec["dur_s"]
        kept_s += rec["trimmed_s"]
        if n % 20 == 0:
            print(f"[trelis_pass] {n + 1}/{len(todo)} {wav.name} kept {rec['trimmed_s']}/{rec['dur_s']} s "
                  f"{rec['n_chunks']} chunks", flush=True)
    wall = time.time() - t0
    print(f"[trelis_pass] {len(todo)} wavs, {audio_s:.0f} s audio ({kept_s:.0f} s after trim) in {wall:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
