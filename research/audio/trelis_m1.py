"""Trelis Whisper-Hinglish chunk QC, scored with CER-study "method 1" (user 2026-10-04; see NOTES.md
"D2 / V4 audio QC: Trelis ASR").

Model: Trelis/whisper-hinglish-preview, bf16, decoder prompt <|startoftranscript|><|hi|><|mixedcode|><|transcribe|>
<|notimestamps|>, greedy (as tests/trelis_tx.py / cer_study/trelis/TRANSCRIBE.md). Audio 24 kHz -> 16 kHz FFT resample
(venv-tts has no torchaudio/scipy). Needs transformers>=5 + soundfile: runs in venv-tts (asr.py re-execs there).

Method 1 (cer_study/COMPARISON.md), automated:
  hyp: Trelis text; every Devanagari token romanised (lexicon Devanagari->roman built from this ROOT's plan.json
       text_tts/text_roman word pairs, most frequent spelling; OOV tokens -> tts_norm.deva_to_latin).
  ref: chunk text_roman.
  both: tts_backends.en_text (digits/IDs/amounts -> English spoken words, as the TTS spoke them; Trelis writes numbers
       as English words), lowercase, every char not [a-z0-9'] -> space, whitespace collapsed, runs of >=2 single-letter
       tokens joined ("f d four" == "fd four"). Spaces count. CER/WER = plain Levenshtein / len(ref).
Deviations from the hand-scored study: romanisation is automatic (lexicon) not by hand; numbers compared in spoken
form (the study charged digits vs number words; that was its one large artefact, ~0.19 CER on ecom_11).
Stdlib-only at import (numpy/torch/transformers imported inside Transcriber).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_norm  # noqa: E402
import tts_backends  # noqa: E402

REPO = "Trelis/whisper-hinglish-preview"
SCORER = "trelis_m1"
DEVA_TOK = re.compile(r"[ऀ-ॿ]")
_STRIP = "".join(sorted(set(tts_norm.PUNCT + "|")))


def _core(w):
    return w.strip(_STRIP).lower()


def build_lexicon(plan):
    """Devanagari word -> most frequent roman spelling, from chunks whose text_tts/text_roman word counts match."""
    cnt = defaultdict(Counter)
    for p in plan:
        tw, rw = p["text_tts"].split(), p["text_roman"].split()
        if len(tw) != len(rw):
            continue
        for a, b in zip(tw, rw):
            a, b = _core(a), _core(b)
            if a and b and DEVA_TOK.search(a) and not DEVA_TOK.search(b) and not re.search(r"[0-9]", a + b):
                cnt[a][b] += 1
    return {a: c.most_common(1)[0][0] for a, c in cnt.items()}


def romanise(text, lex):
    """Romanise Devanagari tokens of a Trelis transcript; Latin tokens unchanged. Returns (text, n_deva, n_oov)."""
    text = text.translate(tts_norm.DEVA_DIGITS)
    out, nd, oov = [], 0, 0
    for w in text.split():
        if not DEVA_TOK.search(w):
            out.append(w)
            continue
        nd += 1
        lead = w[: len(w) - len(w.lstrip(_STRIP))]
        trail = w[len(w.rstrip(_STRIP)):]
        c = _core(w)
        r = lex.get(c)
        if r is None:
            oov += 1
            r = tts_norm.deva_to_latin(c)
        out.append(lead + r + trail)
    return " ".join(out), nd, oov


def m1_norm(text):
    try:
        t = tts_backends.en_text(text)
    except Exception:  # noqa: BLE001  odd token in a transcript: score the raw text
        t = text
    t = t.lower()
    t = re.sub(r"[^a-z0-9']", " ", t)
    toks = t.split()
    out, run = [], []
    for w in toks + [None]:
        if w is not None and len(w) == 1 and w.isalpha():
            run.append(w)
            continue
        if run:
            out.extend(["".join(run)] if len(run) >= 2 else run)
            run = []
        if w is not None:
            out.append(w)
    return " ".join(out)


NUM_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
                "lakh crore double triple".split())


def num_seq(m1_text):
    return [w for w in m1_text.split() if w in NUM_WORDS]


def score(text_roman, hyp, lex):
    hyp_r, nd, oov = romanise(hyp, lex)
    r, h = m1_norm(text_roman), m1_norm(hyp_r)
    if not r:
        c = w = 0.0 if not h else 1.0
    else:
        c = tts_norm.levenshtein(r, h) / len(r)
        rw, hw = r.split(), h.split()
        w = tts_norm.levenshtein(rw, hw) / len(rw)
    # num_ok (informational only, not used for gating): the spoken number-word sequence matches exactly. Catches
    # single dropped/repeated digits that cost only ~0.05 CER; None when the reference has no number words.
    num_ok = (num_seq(r) == num_seq(h)) if num_seq(r) else None
    return {"cer": round(c, 4), "wer": round(w, 4), "num_ok": num_ok, "ref_m1": r, "hyp_m1": h, "hyp_deva_tokens": nd,
            "hyp_oov_tokens": oov}


class Transcriber:
    def __init__(self, device="cuda", batch=16):
        import numpy as np
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()  # keep audio.log readable
        hf_logging.set_verbosity_error()
        self.np, self.torch, self.device, self.batch = np, torch, device, batch
        self.proc = WhisperProcessor.from_pretrained(REPO)
        dt = torch.bfloat16 if device == "cuda" else torch.float32
        self.dtype = dt
        self.model = WhisperForConditionalGeneration.from_pretrained(REPO, dtype=dt).to(device).eval()
        ids = self.proc.tokenizer.convert_tokens_to_ids
        mc = self.proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
        self.prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"),
                       ids("<|notimestamps|>")]

    def load16(self, path):
        import soundfile as sf
        np = self.np
        a, sr = sf.read(str(path), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != 16000:
            n_out = int(round(len(a) * 16000 / sr))
            X = np.fft.rfft(a)
            a = (np.fft.irfft(X[: n_out // 2 + 1], n_out) * (n_out / len(a))).astype(np.float32)
        return a[: 30 * 16000]  # chunks are <= ~15 s; Whisper's window is 30 s

    def _gen(self, audios):
        torch = self.torch
        feat = self.proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        feat = feat.to(self.device, self.dtype)
        dec = torch.tensor([self.prompt] * len(audios), device=self.device)
        with torch.inference_mode():
            out = self.model.generate(input_features=feat, decoder_input_ids=dec, max_new_tokens=224)
        return [self.proc.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]

    def transcribe(self, audios):
        """List of 16 kHz arrays -> texts. Halves the batch on CUDA OOM."""
        res, i, b = [], 0, self.batch
        while i < len(audios):
            try:
                res.extend(self._gen(audios[i:i + b]))
                i += b
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if b == 1:
                    raise
                b = max(1, b // 2)
                print(f"  OOM -> batch {b}", flush=True)
        return res


def load_config(root):
    f = Path(root) / "work" / "asr_config.json"
    return json.loads(f.read_text()) if f.exists() else None
