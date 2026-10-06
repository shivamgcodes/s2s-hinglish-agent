"""Append/replace the '## Final Needle datasets' section of DATA_STATS.md from the assembled files.
Run after assemble.py (and after build_data.py, which rewrites DATA_STATS.md from scratch)."""
import collections, json, os, re

HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir; was this script's own dir
FILES = ["train", "val", "test_heldout_a", "test_heldout_b", "test_n0"]
MARK = "## Final Needle datasets"


def load(name):
    return list(zip(map(json.loads, open(os.path.join(HERE, name + ".jsonl"))),
                    map(json.loads, open(os.path.join(HERE, name + "_meta.jsonl")))))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


D = {f: load(f) for f in FILES}
summ = json.load(open(os.path.join(HERE, "assemble_summary.json")))
val = json.load(open(os.path.join(HERE, "validate_datasets.json")))
L = [MARK, "",
     "Built by assemble.py from examples_raw.jsonl (+ asr_turns.jsonl for rendering b, tests_n0_relabelled.jsonl), "
     "validated by validate_datasets.py with the installed cactus-needle 3.0.6 `finetune.read_examples` / "
     "`render_example` / `_encode` and Needle's SentencePiece tokenizer (vendor/needle_tokenizer.model). "
     "Counts below are Needle lines (one line = one example in one rendering). Each file has a line-aligned "
     "`<name>_meta.jsonl` sidecar (example_id, split, variant, scenario/call, rendering, language, gold entity id, "
     "wav paths, ...). Line format: `{query, tools, answers, system}` only.", ""]

# overview
rows = []
for f in FILES:
    d = D[f]
    pos = sum(1 for r, m in d if r["answers"])
    neg = sum(1 for r, m in d if not r["answers"])
    multi = sum(1 for r, m in d if len(r["answers"]) > 1)
    rend = collections.Counter(m.get("rendering", "a") for r, m in d)
    lang = collections.Counter(m.get("language") or {"en": "english", "hinglish": "hinglish"}[m["lang"]] for r, m in d)
    scen = len({m["scenario_id"] for r, m in d}) if "scenario_id" in d[0][1] else "-"
    rows.append([f, len(d), pos, neg, multi, scen, " ".join(f"{k}:{v}" for k, v in sorted(rend.items())),
                 lang.get("english", 0), lang.get("hinglish", 0), val[f]["tok_max"], summ["md5"][f][:8]])
L += [table(["file", "lines", "positives", "negatives (answers [])", "multi-call", "scenarios", "renderings",
             "english", "hinglish", "max tokens", "md5"], rows), ""]
L += ["- train/val: renderings a, b, c of every positive and every neg_selected negative (V1 + V3). "
      "test_heldout_a / _b: renderings a / b of the same 216 positives + 324 negatives (all 4 calls of each of the "
      "25 held-out scenarios, V1 + V3). test_n0: the relabelled 60 (food tool set, eval record `resolver.N0_RECORD`).",
      f"- Scenarios: train {len(summ['scenarios']['train'])}, val {len(summ['scenarios']['val'])} "
      f"({', '.join(summ['scenarios']['val'])}), test 25 (= holdout.json test_scenarios). Overlap train/val/test = 0 "
      "(asserted in assemble.py and validate_datasets.py). val = 5/47 training scenarios = 10.6% (90/10 by scenario).",
      "- Multi-call: 0 in train/val/test_heldout (no customer turn carries two writes); test_n0 has "
      f"{sum(1 for r, m in D['test_n0'] if len(r['answers']) > 1)} multi-call items.",
      "- Language = hindi_share of the (a) text < 0.10 -> english (build_data); the same label is kept for the b/c "
      "line of an example. test_n0 uses its own lang field.",
      f"- Token length (prompt + target + BOS/EOS): max {max(val[f]['tok_max'] for f in FILES)} (cab), 0 lines over "
      "the default `--max-len 1024`.", ""]

# per tool
tools = sorted({c["name"] for f in FILES for r, m in D[f] for c in r["answers"]})
rows = []
for t in tools:
    rows.append([t] + [sum(1 for r, m in D[f] for c in r["answers"] if c["name"] == t) for f in FILES])
rows.append(["(empty list)"] + [sum(1 for r, m in D[f] if not r["answers"]) for f in FILES])
L += ["### Calls per tool (lines; train/val count each rendering)", "",
      table(["tool"] + FILES, rows), "",
      "val has 0 positives for add_delivery_instruction, add_driver_instruction, cancel_ride, cancel_subscription, "
      "change_drop_location, pause_subscription, request_refund, request_ticket_cancellation (builder's 1-scenario-"
      "per-agent pick, kept; see DECISIONS.md). Val loss therefore does not cover those tools.", ""]

# per agent type x type
rows = []
for at in sorted({m["agent_type"] for f in FILES for r, m in D[f]}):
    rows.append([at] + [f"{sum(1 for r, m in D[f] if m['agent_type'] == at and r['answers'])}/"
                        f"{sum(1 for r, m in D[f] if m['agent_type'] == at and not r['answers'])}" for f in FILES])
L += ["### Positives/negatives per agent type (lines)", "", table(["agent type"] + FILES, rows), ""]

# order_ref presence per rendering
rows = []
for f in FILES:
    for k in sorted({m.get("rendering", "a") for r, m in D[f]}):
        sel = [(r, m) for r, m in D[f] if r["answers"] and m.get("rendering", "a") == k]
        has = sum(1 for r, m in sel for c in r["answers"] if "order_ref" in c["arguments"])
        calls = sum(len(r["answers"]) for r, m in sel)
        rows.append([f, k, calls, has, calls - has])
L += ["### order_ref in the target (A1: omitted when the customer did not name the entity)", "",
      table(["file", "rendering", "calls", "with order_ref", "order_ref omitted"], rows), ""]

# rendering b details
rb = collections.Counter()
for k, v in summ["b_order_ref_rule"].items():
    rb[tuple(k.split("|"))] = v
rows = [[s, r, rb[(s, r)]] for s, r in sorted(rb)]
L += ["### Rendering (b) order_ref (re-extracted from asr_roman, A3)", "",
      table(["split", "rule", "positives"], rows), "",
      "resolver.resolve(order_ref_b, record) == gold entity: " +
      ", ".join(f"{k.split('|')[1]} {v}" for k, v in summ["b_resolver"].items() if k.endswith("True")) +
      " (all 480; misses: " + str(sum(v for k, v in summ["b_resolver"].items() if k.endswith("False"))) + "). "
      "56 spoken-ID spans resolve through the resolver's id rule (e.g. 'efdi one nine zero four' -> FD1904, "
      "'are di four two seven seven' -> RD4277); 4 (a)-id rows fall back to a possessive ref ('meri ride') because "
      "the ASR dropped/garbled the ID.", ""]

# b value-arg literalness (A2)
cnt = collections.Counter()
for f in ("train", "val", "test_heldout_b"):
    for r, m in D[f]:
        if m.get("rendering") != "b" or not r["answers"]:
            continue
        q = r["query"]
        for c in r["answers"]:
            for an, v in c["arguments"].items():
                if an == "order_ref":
                    continue
                if an == "phone":
                    lit = v in re.sub(r"\D", "", q)
                else:
                    tv = toks(v)
                    lit = len(tv & toks(q)) / max(1, len(tv)) >= 0.6
                cnt[(an, lit)] += 1
rows = [[an, cnt[(an, True)], cnt[(an, False)]] for an in sorted({a for a, _ in cnt})]
L += ["### Rendering (b) value args vs the ASR text (A2: gold canonical values kept)", "",
      table(["arg", ">= 0.6 token overlap / digits present", "not literal in the (b) query"], rows), "",
      "Phones are never literal in (b): the ASR writes them as number words. The tuned model has to normalise "
      "number words to the digits the tool description asks for; eval must report shipped vs suppressed calls "
      "on (b) separately (experiments/asr_span_probe.* was inconclusive on whether the engine suppresses such calls).",
      ""]
text = open(os.path.join(HERE, "DATA_STATS.md")).read()
if MARK in text:
    text = text[:text.index(MARK)].rstrip() + "\n"
open(os.path.join(HERE, "DATA_STATS.md"), "w").write(text.rstrip() + "\n\n" + "\n".join(L))
print("\n".join(L))
