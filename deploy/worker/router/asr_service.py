"""S2S serverless worker: copy of DEP1 router/asr_service.py (path edits only). ASR service (venv-asr),
127.0.0.1:$S2S_ASR_PORT (default 8996; DEP1 INTERFACE.md sections 1 and 9).

  POST /asr?lang=hi   body = float32 LE mono 24 kHz PCM
       -> {"text": romanised, "text_raw": ASR output, "backend": str, "audio_s": float, "latency_ms": float}
  GET  /health        -> {"ok": true, "backend": ..., "device": ...}

Backends (one per process, --backend):
  trelis           Trelis/whisper-hinglish-preview, bf16, <|hi|><|mixedcode|> prompt (tests/trelis_tx.py usage),
                   greedy, max_new_tokens 440, GPU (--device cuda) or CPU (--device cpu, fp32).
  fw-small / fw-medium   faster-whisper (Systran CT2), language hi, beam 5, int8 on CPU, vad_filter on,
                   condition_on_previous_text off.
Romanisation: <S2S_NEEDLE>/romanise.py (N1 rendering b: lexicon from train calls + rule fallback); Latin tokens
are kept. Audio longer than 30 s is cut to its LAST 30 s (Whisper's window; the feature extractor would
otherwise keep the first 30 s). Resampling 24 kHz -> 16 kHz is the FFT band-limited resample used by
tests/trelis_tx.py and needle/asr_turns.py.

S2S: the Trelis weights come from S2S_ASR_HF_HOME (baked into the image, HF_HUB_OFFLINE=1) or S2S_ASR_DIR (an
explicit snapshot dir, DESIGN 6.4 option B). fw-* backends are NOT baked: they need network or a pre-filled HF home.
Run (GPU):  <venv-asr>/bin/python asr_service.py --backend trelis --device cuda   (started by worker/stack.sh)
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

os.environ.setdefault("HF_HOME", paths.ASR_HF_HOME)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(paths.NEEDLE))
import romanise  # noqa: E402

SR_IN = 24000
SR_ASR = 16000
MAX_S = 30.0
_LEX = romanise.load_lex()


def to16k(a24):
    a = np.asarray(a24, dtype=np.float32)
    if len(a) == 0:
        return a
    n_out = int(round(len(a) * SR_ASR / SR_IN))
    X = np.fft.rfft(a)
    return (np.fft.irfft(X[: n_out // 2 + 1], n_out) * (n_out / len(a))).astype(np.float32)


def roman(text):
    return romanise.romanise(text, _LEX)


class TrelisBackend:
    repo = paths.ASR_MODEL   # repo id (resolved in HF_HOME) or a snapshot dir

    def __init__(self, device="cuda", max_new_tokens=440):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        torch.set_num_threads(4)   # container CFS quota is 31 CPUs (INTERFACE_ISSUES I1); the GPU does the work
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.proc = WhisperProcessor.from_pretrained(self.repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.repo, torch_dtype=self.dtype).to(device).eval()
        ids = self.proc.tokenizer.convert_tokens_to_ids
        mc = self.proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
        self.prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
        self.max_new_tokens = max_new_tokens
        self.name = "trelis"

    def transcribe16(self, a16):
        torch = self.torch
        feat = self.proc.feature_extractor(a16, sampling_rate=SR_ASR, return_tensors="pt").input_features
        feat = feat.to(self.device, self.dtype)
        with torch.no_grad():
            out = self.model.generate(input_features=feat, decoder_input_ids=torch.tensor([self.prompt], device=self.device),
                                      max_new_tokens=self.max_new_tokens)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return self.proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()


class FasterWhisperBackend:
    def __init__(self, size="small", device="cpu", threads=16, beam=5, vad=True):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16",
                                  cpu_threads=threads)
        self.beam, self.vad = beam, vad
        self.name = f"faster-whisper-{size}"

    def transcribe16(self, a16, lang="hi"):
        segs, _ = self.model.transcribe(a16, language=lang, beam_size=self.beam, temperature=0.0,
                                        condition_on_previous_text=False, vad_filter=self.vad)
        return " ".join(s.text.strip() for s in segs).strip()


def make_backend(name, device="cuda", threads=16):
    if name == "trelis":
        return TrelisBackend(device)
    if name.startswith("fw-"):
        return FasterWhisperBackend(name[3:], "cpu" if device == "cpu" else device, threads)
    raise ValueError(name)


class ASR:
    def __init__(self, backend):
        self.b = backend
        self.lock = threading.Lock()

    def __call__(self, pcm24, lang="hi"):
        pcm24 = np.asarray(pcm24, dtype=np.float32)
        if len(pcm24) > MAX_S * SR_IN:
            pcm24 = pcm24[-int(MAX_S * SR_IN):]
        with self.lock:
            t = time.perf_counter()
            a16 = to16k(pcm24)
            raw = self.b.transcribe16(a16) if len(a16) else ""
            lat = (time.perf_counter() - t) * 1000
        return {"text": roman(raw), "text_raw": raw, "backend": self.b.name, "audio_s": round(len(pcm24) / SR_IN, 2),
                "latency_ms": round(lat, 1)}


def serve(asr, host="127.0.0.1", port=8996, info=None):
    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                return self._json(200, {"ok": True, **(info or {})})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/asr":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            lang = parse_qs(u.query).get("lang", ["hi"])[0]
            try:
                pcm = np.frombuffer(body, dtype="<f4")
                self._json(200, asr(pcm, lang))
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("asr_service %s\n" % (fmt % args))

    httpd = ThreadingHTTPServer((host, port), H)
    print(f"asr_service listening on http://{host}:{port} {info}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fw-small", choices=["trelis", "fw-small", "fw-medium"])
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda for trelis, cpu for fw-*)")
    ap.add_argument("--threads", type=int, default=8, help="faster-whisper CPU threads (31-CPU container quota)")
    ap.add_argument("--port", type=int, default=paths.ASR_PORT)
    a = ap.parse_args()
    dev = a.device or ("cuda" if a.backend == "trelis" else "cpu")
    t = time.time()
    asr = ASR(make_backend(a.backend, dev, a.threads))
    asr(np.zeros(SR_IN, np.float32))   # warm-up
    serve(asr, port=a.port, info={"backend": asr.b.name, "device": dev, "load_s": round(time.time() - t, 1)})
