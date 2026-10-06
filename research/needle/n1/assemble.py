"""N1 final dataset assembly: examples_raw.jsonl + asr_turns.jsonl + tests_n0_relabelled.jsonl
-> train.jsonl, val.jsonl, test_heldout_a.jsonl, test_heldout_b.jsonl, test_n0.jsonl
   (+ line-aligned *_meta.jsonl sidecars with traceability / gold fields for eval).

Needle line format (installed cactus-needle 3.0.6, finetune.render_example / read_examples):
  {"query": str, "tools": [tool dict, ...], "answers": [{"name", "arguments"}], "system": str}
No other keys in the Needle files; everything else goes to the sidecar (same line number).

Decisions (logged in DECISIONS.md, section "§2-3 Final assembly"):
- A1 target = answers_omit_default: order_ref is left out when the customer did not name the entity
  (tools.py description "omit if the customer did not name it"; guide rule "every argument is a span").
- A2 rendering (b) value args (phone/address/instruction/email/reason) keep the canonical gold value
  (digits-only phone etc.) although the ASR says numbers as words; probe in experiments/asr_span_probe.*.
- A3 rendering (b) order_ref is re-extracted from asr_roman: spoken-ID span first (letters + exactly 4
  single-digit words, copied verbatim), then build_data.extract_order_ref (name/ordinal/possessive);
  default -> omitted. The verbatim span is kept even when resolver.resolve misses (router and
  resolver are scored separately in §5).
- A4 rendering (c) uses the (a) target (build_data protected the order_ref and arg spans in (c)).
- A5 train/val = renderings a, b, c of every positive and every neg_selected negative.
- Pure python, deterministic; sorted by (example_id, rendering).
"""
import collections
import hashlib
import json
import os
import re
import sys

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))  # build_data, resolver, router, tools, src/records.json
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (data, results/); was this script's own dir
sys.path.insert(0, N1_PKG)
import build_data  # noqa: E402  (extract_order_ref, PINNED_DATE)
import resolver  # noqa: E402
import tools as toolmod  # noqa: E402

SRC = os.path.join(HERE, "src")
OUT = {k: os.path.join(HERE, f"{k}.jsonl") for k in
       ("train", "val", "test_heldout_a", "test_heldout_b", "test_n0")}

DIGIT_W = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
# letters of a spoken ID: up to 2 short tokens before the 4 digit words ("F D", "efdi", "are di", "i c", "rd")
ID_PREFIX_STOP = {"is", "hai", "order", "id", "my", "the", "ki", "ka", "ke", "no", "number", "ride", "cancel",
                  "please", "plan", "to", "se", "aur", "and", "of", "for", "me", "mera", "meri", "mere",
                  "booking", "ticket", "flight", "pnr", "this", "it", "ji", "haan", "toh", "ye", "yeh", "wo",
                  "woh", "a", "an", "in", "on", "at", "be", "do", "kar", "ko", "check", "with", "par", "pe"}
SPOKEN_RUN = re.compile(rf"(?<![A-Za-z]){DIGIT_W}(?:\s+{DIGIT_W})*(?![A-Za-z])", re.I)


def spoken_id_span(text):
    """Last maximal run of EXACTLY 4 single-digit words (IDs are XXdddd; phones are 10-digit runs), extended
    left over up to 2 short (<=4 letters) non-stopword alphabetic tokens. Returns the verbatim span or None."""
    best = None
    for m in SPOKEN_RUN.finditer(text):
        if len(m.group(0).split()) != 4:
            continue
        s = m.start()
        toks = list(re.finditer(r"[A-Za-z]+", text[:s]))
        # part of a longer number the ASR split ("nine eight zero zero zero to one seven six zero"):
        # a digit word or digit within the 3 tokens before the run -> not an ID
        prev = " ".join(t.group(0) for t in toks[-3:])
        if re.search(rf"\b{DIGIT_W}\b", prev, re.I) or re.search(r"\d", text[max(0, s - 12):s]):
            continue
        take = 0
        for t in reversed(toks[-2:]):
            between = text[t.end():s if take == 0 else toks[len(toks) - take].start()]
            if not re.fullmatch(r"\s+", between):
                break
            w = t.group(0)
            if len(w) > 4 or w.lower() in ID_PREFIX_STOP:
                break
            take += 1
        start = toks[len(toks) - take].start() if take else s
        best = text[start:m.end()]
    return best


def sha(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def needle_row(query, agent_type, answers, system):
    return {"query": query, "tools": toolmod.TOOLS[agent_type], "answers": answers, "system": system}


def omit_ref(answers):
    return [{"name": c["name"], "arguments": {k: v for k, v in c["arguments"].items() if k != "order_ref"}}
            for c in answers]


def b_target(row, b_text, rec):
    """(b) target: tool + gold value args from answers_omit_default, order_ref re-extracted from b_text."""
    base = row["answers_omit_default"]
    assert len(base) == 1
    call = base[0]
    value_args = [str(v) for k, v in row["gold"]["args_raw"].items()
                  if k not in build_data.ID_ARGS and k not in build_data.ENTITY_ARGS]
    sp = spoken_id_span(b_text)
    if sp:
        ref, rule = sp, "spoken_id"
    else:
        ref, rule, _ = build_data.extract_order_ref(b_text, rec, row["agent_type"], protect=value_args)
    args = {k: v for k, v in call["arguments"].items() if k != "order_ref"}  # (a) ref must not leak into (b)
    if rule != "default":
        assert ref in b_text, (row["example_id"], ref, b_text)
        args = {"order_ref": ref, **args}
    res = resolver.resolve(ref if rule != "default" else None, rec, row["agent_type"])
    return [{"name": call["name"], "arguments": args}], rule, res["id"]


def main():
    records = json.load(open(os.path.join(N1_PKG, "src", "records.json")))
    holdout = json.load(open(os.path.join(SRC, "holdout.json")))
    raw = [json.loads(l) for l in open(os.path.join(HERE, "examples_raw.jsonl"))]
    asr = {}
    for l in open(os.path.join(HERE, "asr_turns.jsonl")):
        x = json.loads(l)
        asr[(x["variant"], x["call_id"], x["turn"])] = x

    rows = [r for r in raw if r["type"] == "positive" or r.get("neg_selected")]
    rows.sort(key=lambda r: r["example_id"])
    out = {k: [] for k in OUT}
    meta = {k: [] for k in OUT}
    b_stats = collections.Counter()
    b_rule = collections.Counter()

    for r in rows:
        rec = records[r["scenario_id"]]
        assert r["system"].startswith(build_data.PINNED_DATE + "; user: ")
        assert r["system"] == build_data.system_text(rec)
        texts = {"a": r["renderings"]["a"], "c": r["renderings"]["c"]}
        bt = []
        for t in r["turn_idx"]:
            a = asr[(r["variant"], r["call_id"], t)]
            assert a["scenario_id"] == r["scenario_id"]
            bt.append(a["asr_roman"].strip())
        texts["b"] = " ".join(bt)
        assert all(texts[k].strip() for k in "abc"), r["example_id"]
        pos = r["type"] == "positive"
        ans = {}
        bmeta = {}
        if pos:
            ans["a"] = ans["c"] = r["answers_omit_default"]
            ans["b"], rule_b, rid_b = b_target(r, texts["b"], rec)
            bmeta = {"order_ref_rule_b": rule_b, "resolver_id_b": rid_b,
                     "resolver_ok_b": rid_b == r["gold"]["entity_id"]}
            b_rule[(r["split"], rule_b)] += 1
            b_stats[("resolver_ok_b", r["split"], rid_b == r["gold"]["entity_id"])] += 1
            # order_ref and value args of (a)/(c) are spans of their query (phone: digits of the query)
            for k in "ac":
                for c in ans[k]:
                    for an, v in c["arguments"].items():
                        if an == "phone":
                            assert v in re.sub(r"\D", "", texts[k]), (r["example_id"], k, v)
                        elif an == "order_ref":
                            assert v.lower() in texts[k].lower(), (r["example_id"], k, v)
        else:
            ans = {"a": [], "b": [], "c": []}
        m = {"example_id": r["example_id"], "type": r["type"], "split": r["split"], "variant": r["variant"],
             "scenario_id": r["scenario_id"], "call_id": r["call_id"], "agent_type": r["agent_type"],
             "is_test_call": r["is_test_call"], "language": r["language"], "hindi_share": r["hindi_share"],
             "turn_idx": r["turn_idx"], "wav": r["wav"], "shape": r.get("shape"),
             "args_grounded": r.get("args_grounded"), "in_write_window": r.get("in_write_window"),
             "order_ref_rule": r.get("order_ref_rule"),
             "gold_entity_id": (r.get("gold") or {}).get("entity_id"),
             "gold_args_raw": (r.get("gold") or {}).get("args_raw"),
             "answers_with_default": r.get("answers")}
        if r["split"] in ("train", "val"):
            for k in "abc":
                out[r["split"]].append(needle_row(texts[k], r["agent_type"], ans[k], r["system"]))
                meta[r["split"]].append({**m, "rendering": k, **(bmeta if k == "b" else {})})
        else:
            for k in "ab":
                f = f"test_heldout_{k}"
                out[f].append(needle_row(texts[k], r["agent_type"], ans[k], r["system"]))
                meta[f].append({**m, "rendering": k, **(bmeta if k == "b" else {})})

    # N0 relabelled 60 (food tool set; resolver.N0_RECORD is the eval record)
    for x in map(json.loads, open(os.path.join(HERE, "tests_n0_relabelled.jsonl"))):
        assert "default" not in x["order_ref_rules"]
        assert x["system"].startswith(build_data.PINNED_DATE + "; user: ")
        out["test_n0"].append(needle_row(x["text"], "food_delivery_support", x["expected_calls"], x["system"]))
        meta["test_n0"].append({"id": x["id"], "lang": x["lang"], "kind": x["kind"],
                                "expected_entity_ids": x["expected_entity_ids"],
                                "order_ref_rules": x["order_ref_rules"],
                                "n0_expected_calls": x["n0_expected_calls"],
                                "agent_type": "food_delivery_support", "eval_record": "resolver.N0_RECORD"})

    # scenario disjointness
    scen = {k: {m["scenario_id"] for m in meta[k]} for k in ("train", "val", "test_heldout_a", "test_heldout_b")}
    assert not (scen["train"] & scen["val"]) and not (scen["train"] & scen["test_heldout_a"]) \
        and not (scen["val"] & scen["test_heldout_a"]), "scenario overlap"
    assert scen["test_heldout_a"] == scen["test_heldout_b"] == set(holdout["test_scenarios"])
    assert len(scen["test_heldout_a"]) == 25
    assert not ((scen["train"] | scen["val"]) & set(holdout["test_scenarios"]))
    assert (scen["train"] | scen["val"]) <= set(holdout["train_scenarios"])

    for k, p in OUT.items():
        with open(p, "w") as f:
            for row in out[k]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(p.replace(".jsonl", "_meta.jsonl"), "w") as f:
            for row in meta[k]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        assert len(out[k]) == len(meta[k])
    summary = {"counts": {k: len(v) for k, v in out.items()},
               "scenarios": {k: sorted(v) for k, v in scen.items()},
               "b_order_ref_rule": {f"{a}|{b}": n for (a, b), n in sorted(b_rule.items())},
               "b_resolver": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(b_stats.items())},
               "md5": {k: sha(p) for k, p in OUT.items()}}
    json.dump(summary, open(os.path.join(HERE, "assemble_summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("counts", "b_order_ref_rule", "b_resolver", "md5")}, indent=1))


if __name__ == "__main__":
    main()
