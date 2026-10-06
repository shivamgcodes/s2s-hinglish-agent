# Calibration: Trelis+m1 vs the stored stock-Whisper (large-v3) CER on V3 chunks, plus a V4 try0 sample.
# READ-ONLY on data/: writes only to /workspace/hinglish/tmp_trelis/calib.jsonl
import json, random, re, sys, time
from pathlib import Path
import os  # monorepo shim: trelis_m1/synth/tts_norm/tts_backends are in research/audio ($AUDIO_DIR); data under $HINGLISH_ROOT
from pathlib import Path as _P
HINGLISH_ROOT = os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(_P(__file__).resolve().parents[3] / "research" / "audio"))  # was /workspace/hinglish/audio
sys.path.insert(0, AUDIO_DIR)
import trelis_m1 as tm

H = Path(HINGLISH_ROOT) / "data"
OUT = Path(HINGLISH_ROOT) / "tmp_trelis" / "calib.jsonl"
N_V3_RAND, N_V4 = int(sys.argv[1]), int(sys.argv[2])
random.seed(7)
items = []
for var, n_rand in (("V3", N_V3_RAND), ("V4", N_V4)):
    plan = json.loads((H / var / "work/plan.json").read_text(encoding="utf-8"))
    lex = tm.build_lexicon(plan)
    print(var, "lexicon", len(lex), flush=True)
    cand, bad = [], []
    for p in plan:
        for k in range(3):
            st = H / var / "work/chunks" / p["call_id"] / f"t{p['turn']:02d}_c{p['chunk']}_try{k}"
            if not st.with_suffix(".wav").exists():
                continue
            old = json.loads(st.with_suffix(".asr.json").read_text()) if st.with_suffix(".asr.json").exists() else None
            it = (var, p, k, st, old, lex)
            if var == "V3" and old is None:
                continue
            if var == "V3" and k > 0:
                bad.append(it)  # every retry (and below, its try0)
            elif k == 0:
                cand.append(it)
    if var == "V3":
        retried = {(i[1]["call_id"], i[1]["turn"], i[1]["chunk"]) for i in bad}
        sel = bad + [i for i in cand if (i[1]["call_id"], i[1]["turn"], i[1]["chunk"]) in retried]
        rest = [i for i in cand if (i[1]["call_id"], i[1]["turn"], i[1]["chunk"]) not in retried]
        sel += random.sample(rest, n_rand)
    else:
        sel = random.sample(cand, min(n_rand, len(cand)))
    items += sel
print("items", len(items), flush=True)
t0 = time.time()
T = tm.Transcriber(batch=16)
print("load", round(time.time() - t0, 1), flush=True)
import torch
t1 = time.time()
done = 0
with open(OUT, "w") as f:
    for i in range(0, len(items), 64):
        part = items[i:i + 64]
        auds = [T.load16(it[3].with_suffix(".wav")) for it in part]
        texts = T.transcribe(auds)
        for it, a, tx in zip(part, auds, texts):
            var, p, k, st, old, lex = it
            s = tm.score(p["text_roman"], tx, lex)
            f.write(json.dumps({"var": var, "id": f"{p['call_id']}|{p['turn']}|{p['chunk']}", "k": k, "dur": len(a) / 16000,
                                "old_cer": old["cer"] if old else None, "old_text": old["text"] if old else None,
                                "trelis": tx, **s, "flags": p["flags"], "digits": bool(re.search(r"[0-9₹]", p["text_roman"])),
                                "text_roman": p["text_roman"]}, ensure_ascii=False) + "\n")
        done += len(part)
        print(done, round(time.time() - t1, 1), "maxmem_GB", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
print("audio_s", sum(1 for _ in items), "wall", round(time.time() - t1, 1))
