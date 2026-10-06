"""Validate every row file with the installed Needle loader (N1 validate_datasets.py method) and compute the
DATA_STATS numbers. Run with /root/venv-needle/bin/python. Writes /root/n2/data/stats.json."""
import collections, glob, json, os, re, sys
from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[5]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
from needle.model import finetune as ft
from needle.model.tokenizer import SANTokenizer
import router_v2, targets, tools_v2  # noqa: E402

D = f"{N2_ROOT}/data/rows"
tok = SANTokenizer(f"{N2_ROOT}/needle_n1/vendor/needle_tokenizer.model")
PIN = "date: 2026-10-04 Sun 10:00; user: "
S = {"files": {}}
for path in sorted(glob.glob(D + "/*.jsonl")):
    if path.endswith("_meta.jsonl"):
        continue
    name = os.path.basename(path)[:-6]
    lines = [l for l in open(path) if l.strip()]
    meta = [json.loads(l) for l in open(path[:-6] + "_meta.jsonl")]
    exs = list(ft.read_examples(path, report=True))
    assert len(exs) == len(lines) == len(meta), (name, len(exs), len(lines), len(meta))
    lens, per_agent = [], collections.Counter()
    for ex, m in zip(exs, meta):
        assert set(ex) == {"query", "tools", "answers", "system"} and ex["query"].strip()
        assert ex["system"].startswith(PIN) and ex["system"].count("date:") == 1
        tools = {t["name"]: t for t in ex["tools"]}
        assert len(ex["answers"]) == 1
        for c in ex["answers"]:
            assert set(c) == {"name", "arguments"} and c["name"] in tools, c
            props = tools[c["name"]]["parameters"]["properties"]
            assert set(c["arguments"]) <= set(props), (c, list(props))
            assert set(tools[c["name"]]["parameters"].get("required", [])) <= set(c["arguments"]), (name, c)
            assert all(isinstance(v, str) and v.strip() for v in c["arguments"].values()), c
        prompt, target = ft.render_example(ex)
        n = len(tok.encode(prompt)) + len(tok.encode(target)) + 2  # + BOS/EOS as in _encode
        lens.append(n)
        per_agent[m["agent_type"]] = max(per_agent[m["agent_type"]], n)
        ids, mask = ft._encode(tok, ex, 2048)
        assert sum(mask) > 0
    s = sorted(lens)
    S["files"][name] = {"rows": len(lines), "windows": len({m["id"] for m in meta}),
                        "scenarios": len({m["scenario_id"] for m in meta}),
                        "tok_max": s[-1], "tok_p95": s[int(0.95 * (len(s) - 1))], "tok_median": s[len(s) // 2],
                        "over_1024": sum(x > 1024 for x in s), "per_agent_max": dict(per_agent)}
    print(name, S["files"][name], flush=True)

M = lambda k: [json.loads(l) for l in open(f"{D}/{k}_meta.jsonl")]  # noqa: E731
sc = {k: {m["scenario_id"] for m in M(k)} for k in ("train", "val", "test_opus")}
assert not sc["train"] & sc["val"] and not sc["train"] & sc["test_opus"] and not sc["val"] & sc["test_opus"]
S["scenarios"] = {k: len(v) for k, v in sc.items()}

# windows per split / agent / tool (one variant)
allm = M("train") + M("val") + M("test_opus")
win = {m["id"]: m for m in allm}
S["windows_by_split_agent"] = collections.defaultdict(collections.Counter)
S["windows_by_split_tool"] = collections.defaultdict(collections.Counter)
for m in win.values():
    S["windows_by_split_agent"][m["agent_type"]][m["split"]] += 1
    S["windows_by_split_tool"][m["tool"]][m["split"]] += 1

# grounding of NEW args, per kind group, per variant, per split
GROUP = {"phone": "phone", "email": "email", "address": "address/location", "location": "address/location",
         "address_or_registered": "address/location", "instruction": "instruction/reason", "reason": "instruction/reason"}
gr = collections.defaultdict(lambda: [0, 0])
refm = collections.defaultdict(lambda: [0, 0])
for k in ("train", "val_clean", "val_opus", "val_exact", "test_clean", "test_opus", "test_exact"):
    pass
for k in ("train", "val_exact", "test_clean", "test_opus", "test_exact", "val_clean", "val_opus"):
    for m in M(k):
        if k == "train" and m["variant"] == "exact":
            continue
        kinds = {a: kd for a, kd, g in tools_v2.arg_specs(m["agent_type"], m["tool"])}
        sp = "test" if m["split"] == "test" else "train+val"
        for a, ok in m["grounded"].items():
            for key in ((sp, m["variant"], GROUP.get(kinds[a], kinds[a])), ("all", m["variant"], GROUP.get(kinds[a], kinds[a]))):
                gr[key][0] += ok; gr[key][1] += 1
        for a, ok in m["ref_mentioned"].items():
            key = ("all", m["variant"], kinds[a])
            refm[key][0] += ok; refm[key][1] += 1
S["grounded"] = {"|".join(k): v for k, v in sorted(gr.items())}
S["ref_mentioned"] = {"|".join(k): v for k, v in sorted(refm.items())}

# phone/email-grounded rows (train_grounded filter)
S["phone_email_grounded"] = {}
for k in ("train", "val_clean", "val_opus", "test_clean", "test_opus", "test_exact"):
    mm = M(k)
    pe = [m for m in mm if any(kd in ("phone", "email") for a, kd, g in tools_v2.arg_specs(m["agent_type"], m["tool"]) if g in m["gold_args_raw"])]
    S["phone_email_grounded"][k] = {"rows": len(mm), "rows_with_phone_or_email": len(pe),
                                    "grounded": sum(m["phone_email_grounded"] for m in pe)}


# WER-ish: prepared ASR vs prepared gold customer text, token level
def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


exact = {m["id"]: m for m in M("test_exact") + M("val_exact")}
ex_train = {}
for m in M("train"):
    pass
wer = collections.defaultdict(lambda: [0, 0])
# exact text for train windows: rebuild from windows meta (train has no exact file)
wm = {json.loads(l)["id"]: json.loads(l) for l in open(f"{N2_ROOT}/windows/meta.jsonl")}
for m in allm + M("train") + M("val_clean") + M("test_clean"):
    pass
seen = set()
for k in ("train", "val_clean", "val_opus", "test_clean", "test_opus"):
    for m in M(k):
        key = (m["id"], m["variant"])
        if key in seen:
            continue
        seen.add(key)
        w = wm[m["id"]]
        ref = router_v2.prepare_transcript(" ".join(t["text_roman"] for t in w["customer_turns_in_window"]))
        hyp = router_v2.prepare_transcript(m["raw_text"])
        r, h = toks(ref), toks(hyp)
        e = ed(r, h)
        full = not m["exact_has_partial_turn"]
        sp = "test" if m["split"] == "test" else "train+val"
        for kk in ((m["variant"], "all", "all"), (m["variant"], sp, "all")) + (
                ((m["variant"], "all", "no_partial_turn"), (m["variant"], sp, "no_partial_turn")) if full else ()):
            wer["|".join(kk)][0] += e; wer["|".join(kk)][1] += len(r)
S["wer"] = {k: {"edits": v[0], "ref_words": v[1], "wer": round(v[0] / max(1, v[1]), 4)} for k, v in sorted(wer.items())}

# ASR run facts
asr = {}
for v in ("clean", "opus"):
    rows = [json.loads(l) for l in open(f"{N2_ROOT}/data/asr/{v}.jsonl")]
    asr[v] = {"clips": len(rows), "hit_cap": sum(r["hit_cap"] for r in rows), "empty": sum(not r["asr_raw"].strip() for r in rows),
              "tokens_max": max(r["n_tokens"] for r in rows), "tokens_median": sorted(r["n_tokens"] for r in rows)[len(rows) // 2]}
S["asr"] = asr
S["prior_writes_windows"] = sum(bool(m["prior_writes"]) for m in win.values())
json.dump(S, open(f"{N2_ROOT}/data/stats.json", "w"), indent=1)
print(json.dumps({k: S[k] for k in ("scenarios", "grounded", "phone_email_grounded", "wer", "asr", "prior_writes_windows")}, indent=1))
print("VALID OK")
