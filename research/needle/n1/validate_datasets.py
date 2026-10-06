"""Validate the N1 Needle files with the INSTALLED cactus-needle loader (run with system python3; see sys.path note).
- every line yields an example through needle.model.finetune.read_examples (no skipped lines)
- render_example + _encode work; token length measured with Needle's SentencePiece tokenizer
- schema: query/tools/answers/system only; answer names are in the row's tools; arguments are properties of
  that tool; required args present; system pinned with exactly one date fact
- sidecar meta aligned line by line; scenario disjointness (train/val/test) re-asserted
Writes validate_datasets.json."""
import collections, json, os, sys
# The needle venv has no numpy/sentencepiece (finetune extras not installed); run with system python3 and
# append the INSTALLED package so its own finetune.py / tokenizer.py code is what validates the files.
sys.path.append(os.environ.get("NEEDLE_SITE_PACKAGES", "/mnt/drive2/venv/needle-exp0/lib/python3.11/site-packages"))
from needle.model import finetune as ft
from needle.model.tokenizer import SANTokenizer

HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was this script's own dir
TOK = os.path.join(HERE, "vendor", "needle_tokenizer.model")
FILES = ["train", "val", "test_heldout_a", "test_heldout_b", "test_n0"]
PIN = "date: 2026-10-04 Sun 10:00; user: "
tok = SANTokenizer(TOK)
report = {}
scen = {}
for name in FILES:
    path = os.path.join(HERE, name + ".jsonl")
    lines = [l for l in open(path) if l.strip()]
    meta = [json.loads(l) for l in open(path.replace(".jsonl", "_meta.jsonl"))]
    exs = list(ft.read_examples(path, report=True))
    assert len(exs) == len(lines) == len(meta), (name, len(exs), len(lines), len(meta))
    lens = []
    per_agent_max = collections.Counter()
    for ex, m in zip(exs, meta):
        assert set(ex) == {"query", "tools", "answers", "system"}, ex.keys()
        assert ex["query"].strip() and ex["system"].startswith(PIN) and ex["system"].count("date:") == 1
        tools = {t["name"]: t for t in ex["tools"]}
        assert len(tools) == len(ex["tools"])
        for c in ex["answers"]:
            assert set(c) == {"name", "arguments"} and c["name"] in tools, c
            props = tools[c["name"]]["parameters"]["properties"]
            assert set(c["arguments"]) <= set(props), (c, list(props))
            assert set(tools[c["name"]]["parameters"].get("required", [])) <= set(c["arguments"]), c
            assert all(isinstance(v, str) and v.strip() for v in c["arguments"].values()), c
        prompt, target = ft.render_example(ex)
        n = len(tok.encode(prompt)) + len(tok.encode(target)) + 2  # + BOS/EOS as in _encode
        lens.append(n)
        at = m["agent_type"]
        per_agent_max[at] = max(per_agent_max[at], n)
        ids, mask = ft._encode(tok, ex, 2048)
        assert sum(mask) > 0
    if "scenario_id" in meta[0]:
        scen[name] = {m["scenario_id"] for m in meta}
    lens.sort()
    report[name] = {"lines": len(lines), "loaded": len(exs), "tok_max": lens[-1],
                    "tok_median": lens[len(lens) // 2], "tok_p95": lens[int(0.95 * (len(lens) - 1))],
                    "over_1024": sum(x > 1024 for x in lens), "per_agent_max": dict(per_agent_max)}
    print(name, report[name], flush=True)
assert not scen["train"] & scen["val"]
assert not (scen["train"] | scen["val"]) & scen["test_heldout_a"]
assert scen["test_heldout_a"] == scen["test_heldout_b"] and len(scen["test_heldout_a"]) == 25
report["scenario_counts"] = {k: len(v) for k, v in scen.items()}
report["scenario_overlap"] = 0
json.dump(report, open(os.path.join(HERE, "validate_datasets.json"), "w"), indent=1)
print("OK", report["scenario_counts"])
