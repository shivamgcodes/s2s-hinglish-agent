# Transcribe 10 CER-study model wavs with Trelis/whisper-hinglish-preview (model-card usage, <|hi|><|mixedcode|> prompt),
# long audio split into <=28 s chunks at the lowest-energy point in the 18-28 s window.
import json, os, sys, time, numpy as np, soundfile as sf, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
repo = "Trelis/whisper-hinglish-preview"
pairs = json.loads(sys.argv[1])
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda").eval()
ids = proc.tokenizer.convert_tokens_to_ids
mc = proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
def load16(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:
        n_out = int(round(len(a) * 16000 / sr)); X = np.fft.rfft(a)  # FFT (band-limited) resample
        a = (np.fft.irfft(X[:n_out // 2 + 1], n_out) * (n_out / len(a))).astype(np.float32)
    return a
def chunks(a, sr=16000):
    out, i = [], 0
    while len(a) - i > 28 * sr:
        w = a[i + 18 * sr:i + 28 * sr]; hop = sr // 50
        e = np.array([np.mean(w[k:k + hop] ** 2) for k in range(0, len(w) - hop, hop)])
        cut = i + 18 * sr + int(np.argmin(e)) * hop
        out.append(a[i:cut]); i = cut
    out.append(a[i:]); return out
res = {}
for p in pairs:
    t0 = time.time()
    a = load16(f"{os.environ.get('HINGLISH_ROOT', '/workspace/hinglish')}/tests/out/{p['model']}/V1/{p['call_id']}_s1001.wav")
    texts = []
    for c in chunks(a):
        feat = proc.feature_extractor(c, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
        out = model.generate(input_features=feat, decoder_input_ids=torch.tensor([prompt]).to("cuda"), max_new_tokens=440)
        texts.append(proc.tokenizer.decode(out[0], skip_special_tokens=True).strip())
    res[p["pair_id"]] = {"text": " ".join(texts), "n_chunks": len(texts), "dur_s": round(len(a) / 16000, 2), "sec": round(time.time() - t0, 1)}
    print(p["pair_id"], res[p["pair_id"]]["n_chunks"], res[p["pair_id"]]["sec"], flush=True)
print("JSON" + json.dumps(res, ensure_ascii=False))
