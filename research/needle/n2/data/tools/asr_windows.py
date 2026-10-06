"""Trelis whisper-hinglish-preview on every N2 window (clean_16k and opus_16k), router recipe
(deploy/router/asr_service.py / tools/asr_smoke.py): bf16, <|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>,
greedy, max_new_tokens 440. Inputs are the 16 kHz files (clean_16k = to16k(ch1 24k cut); opus_16k = to16k(opus_24k), exactly).
Usage: asr_windows.py <variant clean|opus> <batch> [--check N] [--limit N]
Writes /root/n2/data/asr/<variant>.jsonl {id, variant, asr_raw, n_tokens, hit_cap, batch_s, batch}."""
import json, os, sys, time
import numpy as np, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir

variant, B = sys.argv[1], int(sys.argv[2])
check = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 0
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
W = f"{N2_ROOT}/windows/"
meta = [json.loads(l) for l in open(W + "meta.jsonl")]
if limit: meta = meta[:limit]
key = {"clean": "clean_16k", "opus": "opus_16k"}[variant]
repo = "Trelis/whisper-hinglish-preview"
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
ids = proc.tokenizer.convert_tokens_to_ids
mc = proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
MAXNEW = 440

def load(m):
    p = m["files"][key]; p = p if p.startswith("/") else W + p
    a, sr = sf.read(p, dtype="float32"); assert sr == 16000 and a.ndim == 1, (p, sr, a.shape)
    return a[-30 * 16000:]

def tx(arrs):
    feat = proc.feature_extractor(arrs, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
    dec = torch.tensor([prompt] * len(arrs), device="cuda")
    with torch.no_grad():
        out = model.generate(input_features=feat, decoder_input_ids=dec, max_new_tokens=MAXNEW)
    res = []
    eos = proc.tokenizer.eos_token_id
    for row in out:
        gen = row[len(prompt):].tolist() if row.shape[0] > len(prompt) and row[:len(prompt)].tolist() == prompt else row.tolist()
        n = sum(1 for t in gen if t != eos and t != proc.tokenizer.pad_token_id)
        res.append((proc.tokenizer.decode(row, skip_special_tokens=True).strip(), n))
    return res

print("prompt", prompt, "n", len(meta), "variant", variant, "batch", B, flush=True)
tx([load(meta[0])])  # warm
if check:
    sub = meta[:check]; arrs = [load(m) for m in sub]
    single = [tx([a])[0][0] for a in arrs]
    batched = []
    for i in range(0, len(arrs), B): batched += [r[0] for r in tx(arrs[i:i + B])]
    same = sum(a == b for a, b in zip(single, batched))
    print(f"CHECK batched==single {same}/{len(sub)}", flush=True)
    for m, a, b in zip(sub, single, batched):
        if a != b: print("DIFF", m["id"], "\n  1:", a, "\n  B:", b, flush=True)
    json.dump({"variant": variant, "batch": B, "n": len(sub), "same": same,
               "diffs": [{"id": m["id"], "single": a, "batched": b} for m, a, b in zip(sub, single, batched) if a != b]},
              open(f"{N2_ROOT}/data/asr/check_{variant}_b{B}.json", "w"), indent=1, ensure_ascii=False)
    sys.exit(0)
out = open(f"{N2_ROOT}/data/asr/{variant}.jsonl", "w")
t0 = time.time(); audio_s = 0.0
for i in range(0, len(meta), B):
    sub = meta[i:i + B]; arrs = [load(m) for m in sub]; audio_s += sum(len(a) for a in arrs) / 16000
    torch.cuda.synchronize(); t = time.time(); r = tx(arrs); torch.cuda.synchronize(); dt = time.time() - t
    for m, (txt, n) in zip(sub, r):
        out.write(json.dumps({"id": m["id"], "variant": variant, "asr_raw": txt, "n_tokens": n, "hit_cap": n >= MAXNEW - 1,
                              "batch_s": round(dt, 3), "batch": len(sub)}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"{i + len(sub)}/{len(meta)} batch {dt:.2f}s", flush=True)
el = time.time() - t0
print(f"DONE {variant}: {len(meta)} clips {el:.1f}s = {len(meta)/el:.2f} clips/s, audio {audio_s/60:.1f} min, RTF {el/audio_s:.4f}, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
