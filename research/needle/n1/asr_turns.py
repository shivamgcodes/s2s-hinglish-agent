# N1 rendering (b): Trelis/whisper-hinglish-preview ASR of every CUSTOMER turn of V1 and V3 calls.
# Logic copied from /workspace/hinglish/tests/trelis_tx.py (model-card usage, <|hi|><|mixedcode|> prompt,
# bf16, FFT band-limited resample to 16 kHz). Differences: per-turn wavs are <= ~11 s so no chunking;
# clips are batched (BATCH per generate call); greedy decoding, max_new_tokens=200.
# Input wavs: /workspace/hinglish/data/<V>/work/utt/<call_id>/t<turn:02d>.wav (24 kHz mono, one per turn,
# turn index = index into calls.jsonl turns[]). Read-only on data/.
# Output: /workspace/hinglish/needle/asr_turns_raw.jsonl (asr_text only); romanise.py adds asr_roman.
# Run: flock /workspace/hinglish/gpu.lock /workspace/venv-tts/bin/python -u asr_turns.py
import json, sys, time, os, numpy as np, soundfile as sf, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

HINGLISH_ROOT = os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")
D = f"{HINGLISH_ROOT}/data"
OUT = f"{HINGLISH_ROOT}/needle/asr_turns_raw.jsonl"
BATCH = 32
repo = "Trelis/whisper-hinglish-preview"

held = set(json.load(open(f"{D}/holdout.json"))["test_scenarios"])
items = []
for V in ["V1", "V3"]:
    for l in open(f"{D}/{V}/calls.jsonl"):
        c = json.loads(l)
        for i, t in enumerate(c["turns"]):
            if t["speaker"] != "customer":
                continue
            items.append({"variant": V, "call_id": c["call_id"], "scenario_id": c["scenario_id"],
                          "split": "heldout" if c["scenario_id"] in held else "train", "turn": i,
                          "wav": f"{D}/{V}/work/utt/{c['call_id']}/t{i:02d}.wav", "text_roman": t["text_roman"]})
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r["variant"], r["call_id"], r["turn"]))
todo = [it for it in items if (it["variant"], it["call_id"], it["turn"]) not in done]
print(f"items {len(items)} done {len(done)} todo {len(todo)}", flush=True)

t_load = time.time()
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda").eval()
ids = proc.tokenizer.convert_tokens_to_ids
mc = proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
print(f"model loaded {time.time() - t_load:.1f}s", flush=True)

def load16(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:
        n_out = int(round(len(a) * 16000 / sr)); X = np.fft.rfft(a)  # FFT (band-limited) resample
        a = (np.fft.irfft(X[:n_out // 2 + 1], n_out) * (n_out / len(a))).astype(np.float32)
    return a

t0 = time.time(); audio_s = 0.0
with open(OUT, "a") as f:
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        auds = [load16(it["wav"]) for it in chunk]
        audio_s += sum(len(a) for a in auds) / 16000
        feat = proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
        dec = torch.tensor([prompt] * len(chunk)).to("cuda")
        with torch.no_grad():
            out = model.generate(input_features=feat, decoder_input_ids=dec, max_new_tokens=200)
        for it, a, o in zip(chunk, auds, out):
            r = dict(it, dur_s=round(len(a) / 16000, 3), asr_text=proc.tokenizer.decode(o, skip_special_tokens=True).strip())
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        print(f"{b + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
wall = time.time() - t0
print("TIMING " + json.dumps({"clips": len(todo), "audio_s": round(audio_s, 1), "wall_s": round(wall, 1),
                              "load_s": round(time.time() - t_load - wall, 1), "x_realtime": round(audio_s / max(wall, 1e-9), 1)}), flush=True)
