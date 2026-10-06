"""QC artefacts.

  /workspace/venv-tts/bin/python qc.py plots ROOT [N=10]
      N alignment PNGs (word boundaries over the utterance waveform) -> ROOT/qc/align_<call>_tNN.png
      Picks a spread: truncated agent turns first, then split lines, then the rest, across calls/speakers.
  flock /workspace/hinglish/gpu.lock /workspace/venv-pp/bin/python qc.py mimi ROOT [N=4]
      Mimi (PersonaPlex loader, 8 codebooks) encode->decode of N agent-channel clips (ch0, 1 s .. 21 s)
      -> ROOT/qc/mimi_<call>_orig.wav / mimi_<call>.wav (24 kHz mono); prints a log-spectral distance.
"""
import json
import sys
from pathlib import Path


def plots(root, n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    root = Path(root)
    als = sorted((root / "work" / "align").glob("*/t*.json"))
    infos = []
    for p in als:
        a = json.loads(p.read_text(encoding="utf-8"))
        u = json.loads((root / "work" / "utt" / p.parent.name / p.name).read_text(encoding="utf-8"))
        infos.append((p, a, len(u["chunks"])))
    calls = {}
    for i in infos:
        calls.setdefault(i[1]["call_id"], []).append(i)
    pri = sorted(infos, key=lambda i: (-(len(i[1]["flags"]) > 0), -(i[2] > 1), i[1]["turn"] % 3, str(i[0])))
    chosen, seen = [], set()
    for i in pri:  # round-robin over calls, prefer flagged / multi-chunk
        key = (i[1]["call_id"], i[1]["speaker"], i[2] > 1)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) >= n:
            break
    for i in pri:
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    (root / "qc").mkdir(parents=True, exist_ok=True)
    outs = []
    for p, a, nch in chosen:
        wav, sr = sf.read(str(root / "work" / "utt" / p.parent.name / (p.stem + ".wav")), dtype="float32")
        t = np.arange(len(wav)) / sr
        fig, ax = plt.subplots(figsize=(max(10, len(wav) / sr * 1.6), 3.2))
        ax.plot(t, wav, lw=0.4, color="0.4")
        for k, w in enumerate(a["words"]):
            ax.axvline(w["start"], color="tab:green", lw=0.8)
            ax.axvline(w["end"], color="tab:red", lw=0.6, ls="--")
            ax.text((w["start"] + w["end"]) / 2, 0.85 if k % 2 else 0.7, w["word"], ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())
        ax.set_xlim(0, len(wav) / sr)
        ax.set_title(f"{a['call_id']} turn {a['turn']} ({a['speaker']}), {nch} chunk(s), method={a['method']}, "
                     f"mean score={a['mean_score']}, flags={a['flags']}", fontsize=8)
        ax.set_xlabel("s")
        fig.tight_layout()
        out = root / "qc" / f"align_{a['call_id']}_t{a['turn']:02d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        outs.append(str(out))
    print("\n".join(outs))


def mimi(root, n=4):
    import numpy as np
    import sphn
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    root = Path(root)
    wavs = sorted((root / "stereo").glob("*.wav"))[:n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), dev)
    m.set_num_codebooks(8)
    for w in wavs:
        x, sr = sphn.read(str(w), sample_rate=m.sample_rate)
        clip = x[0, int(1.0 * sr): int(21.0 * sr)]
        clip = clip[: len(clip) // 1920 * 1920]
        with torch.no_grad():
            codes = m.encode(torch.from_numpy(clip).float().to(dev)[None, None])
            y = m.decode(codes)[0, 0].float().cpu().numpy()
        y = y[: len(clip)]
        sphn.write_wav(str(root / "qc" / f"mimi_{w.stem}_orig.wav"), clip.astype(np.float32), sr)
        sphn.write_wav(str(root / "qc" / f"mimi_{w.stem}.wav"), y.astype(np.float32), sr)

        def logspec(s):
            fr = np.lib.stride_tricks.sliding_window_view(s, 1024)[::256] * np.hanning(1024)
            return np.log10(np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-8)
        L = min(len(clip), len(y))
        lsd = float(np.mean(np.sqrt(np.mean((logspec(clip[:L]) - logspec(y[:L])) ** 2, axis=1))))
        print(f"{w.stem}: codes {tuple(codes.shape)}, {len(clip)/sr:.1f} s, log-spectral distance {lsd:.3f}")


if __name__ == "__main__":
    cmd, root = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if cmd == "plots":
        plots(root, k or 10)
    else:
        mimi(root, k or 4)
