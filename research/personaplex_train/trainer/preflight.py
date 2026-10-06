"""CPU preflight for a training manifest (run by train_run.sh before taking the GPU lock).
Fails on anything that would crash the data loader mid-run; prints stats."""
import json
import os
import sys

import sentencepiece
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.offline import wrap_with_system_tags

man = sys.argv[1]
dur_chunk = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
sp = sentencepiece.SentencePieceProcessor(hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME))
errs, n, long_calls, unk_words, maxP, words = [], 0, 0, 0, 0, 0
for line in open(man):
    if not line.strip():
        continue
    r = json.loads(line)
    n += 1
    p = r["path"]
    if not os.path.isabs(p) or not os.path.exists(p):
        errs.append(f"bad wav path {p}")
        continue
    js = os.path.splitext(p)[0] + ".json"
    try:
        d = json.load(open(js))
    except Exception as e:
        errs.append(f"{js}: {e}")
        continue
    if "text_conditions" in d:
        errs.append(f"{js}: text_conditions key")
    rp = d.get("role_prompt")
    if not rp:
        errs.append(f"{js}: no role_prompt")
    else:
        ids = sp.encode(wrap_with_system_tags(rp))
        if 0 in ids:
            errs.append(f"{js}: role_prompt has unk tokens")
        maxP = max(maxP, 63 + len(ids))
    v = d.get("voice_prompt") or {"f": "NATF2", "m": "NATM1"}.get(d.get("agent_gender"))
    if v is None or os.path.splitext(os.path.basename(v))[0] not in ("NATF2", "NATM1"):
        errs.append(f"{js}: voice unresolvable ({v})")
    al = d.get("alignments", [])
    if any(al[i][1][0] > al[i + 1][1][0] for i in range(len(al) - 1)):
        errs.append(f"{js}: alignments not sorted")
    for w in al:
        if w[2] == "SPEAKER_MAIN":
            words += 1
            if 0 in sp.encode(w[0].strip()):
                unk_words += 1
    if r["duration"] > dur_chunk:
        long_calls += 1
print(f"[preflight] {man}: {n} calls, {words} agent words, {unk_words} agent words with unk tokens (-> EPAD), "
      f"{long_calls} calls > {dur_chunk:.0f} s (2nd chunk, prefix again), max prefix P={maxP}")
for e in errs[:20]:
    print("[preflight] ERROR", e)
if errs:
    print(f"[preflight] FAILED: {len(errs)} errors")
    sys.exit(1)
print("[preflight] OK")
