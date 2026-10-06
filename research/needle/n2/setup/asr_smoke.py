"""Smoke: Trelis whisper-hinglish-preview on a 30 s customer-channel window (same recipe as
deploy/router/asr_service.py TrelisBackend: bf16, <|hi|><|mixedcode|> prompt, greedy, max_new_tokens 440,
24k->16k FFT resample, last 30 s). Clean vs Opus round-trip."""
import json, os, sys, time
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # opus_rt.py (was /root/n2/tools; same file as setup/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir
from opus_rt import opus_roundtrip
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR_IN, SR_ASR = 24000, 16000
def to16k(a):
    n = int(round(len(a) * SR_ASR / SR_IN)); X = np.fft.rfft(a)
    return (np.fft.irfft(X[: n // 2 + 1], n) * (n / len(a))).astype(np.float32)

cid = sys.argv[1] if len(sys.argv) > 1 else "air_01_g1"
meta = json.load(open(f"{N2_ROOT}/V4/stereo/{cid}.json"))
wav, sr = sf.read(f"{N2_ROOT}/V4/stereo/{cid}.wav", dtype="float32"); assert sr == SR_IN
cl = [t for t in meta["turns"] if "check_line" in t.get("tags", [])]
end = cl[0]["start"] if cl else min(meta["duration"], 30.0)
seg = wav[int(max(0, end - 30) * sr): int(end * sr), 1]  # ch1 = customer
print("call", cid, "check_line_start", end, "window_s", len(seg) / sr)

repo = "Trelis/whisper-hinglish-preview"
t = time.time()
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
ids = proc.tokenizer.convert_tokens_to_ids
mc = proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
print("load_s", round(time.time() - t, 2), "prompt", prompt)

def tx(a24):
    feat = proc.feature_extractor(to16k(a24), sampling_rate=SR_ASR, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = model.generate(input_features=feat, decoder_input_ids=torch.tensor([prompt], device="cuda"), max_new_tokens=440)
    torch.cuda.synchronize()
    return proc.tokenizer.decode(out[0], skip_special_tokens=True).strip(), time.time() - t, out.shape[1]

torch.cuda.reset_peak_memory_stats()
for i in range(3):
    txt, dt, n = tx(seg); print(f"clean run{i}: {dt:.2f}s tokens={n}")
print("CLEAN:", txt)
t = time.time(); deg, nbytes = opus_roundtrip(seg); print(f"opus rt {time.time()-t:.2f}s bytes={nbytes} corr={np.corrcoef(seg, deg)[0,1]:.3f}")
txt2, dt, n = tx(deg); print(f"opus: {dt:.2f}s tokens={n}\nOPUS:", txt2)
print("peak_alloc_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), "peak_reserved_GB", round(torch.cuda.max_memory_reserved() / 1e9, 2))
print("customer turns in window:", [x["text_roman"] for x in meta["turns"] if x["speaker"] == "customer" and x["end"] > end - 30 and x["start"] < end])
