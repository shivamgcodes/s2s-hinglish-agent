"""Metric 2 Gemma-judge pass (separate batched GPU step; run after all PersonaPlex runs).

  cd /workspace/hinglish/tests && flock /workspace/hinglish/gpu.lock /workspace/venv-vllm/bin/python judge.py TAG [TAG ...]
Loads Gemma 4 31B once (gen/common.LLM), judges every run text stream lacking <run>.judge.json, temperature 0,
JSON-schema output. Writes <run>.judge.json {"hindi_verb_final_clauses", "english_clauses", "total_clauses",
"r2", "example"}.

Naturalness mode (D2 spec section 7 re-rank; added 2026-10-05; the default mode above is unchanged):
  VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf flock /workspace/hinglish/gpu.lock \
    /workspace/venv-vllm/bin/python judge.py --naturalness [--variant V4] [--seed 1001] [--dest-root DIR] TAG [TAG ...]
Segments each agent text stream <tag>/<variant>/*_s<seed>.json deterministically with tcommon.segments() (new
utterance after >= 15 PAD/EPAD frames or after a piece ending in . ? !), then asks Gemma (temperature 0, one prompt
per call) for one label per utterance with the v3_hinglish_review naturalness rubric (NATURAL / STIFF / PEPPERED /
NONSENSE, + trunc / EN / hindi_content flags). Writes <run>.nat.json next to the run, or under
<dest-root>/<tag>/<variant>/ when --dest-root is given (used to judge existing runs without touching their dirs).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[2]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_GEN_DIR') or str(_REPO / 'research/data_gen/gen'))
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
from tcommon import OUT, jload, text_from_tokens  # noqa: E402

PROMPT = """You are a strict linguist. Below is the text a call-centre agent spoke in one phone call, written in
romanised Hinglish (Hindi words in Latin letters) or English. Split it into clauses and classify each clause:
- HINDI_VERB_FINAL: the clause follows Hindi grammar: Hindi word order with the verb (or auxiliary hai/hoon/tha/
  raha/kar diya/sakti hoon...) at the end, Hindi postpositions (ko, se, mein, ke liye, ka/ki/ke, par, wala) and Hindi
  verb conjugation. English nouns inside such a clause are fine ("aapka order cancel ho gaya hai").
- ENGLISH: English grammar and word order, even if Hindi nouns or fillers are dropped in ("your order is ready, ji").
- Ignore fragments of fewer than 3 words and greetings like "hello" or "thank you".
r2 = true if at least one clause is HINDI_VERB_FINAL.
example = the clearest HINDI_VERB_FINAL clause copied verbatim, or "" if none.

TEXT:
{text}
"""

SCHEMA = {"type": "object", "properties": {
    "hindi_verb_final_clauses": {"type": "integer"}, "english_clauses": {"type": "integer"},
    "r2": {"type": "boolean"}, "example": {"type": "string"}},
    "required": ["hindi_verb_final_clauses", "english_clauses", "r2", "example"]}


NAT_PROMPT = """You review the speech of a phone customer-support agent in India. The agent is supposed to talk like a
real Delhi/NCR call-centre agent: natural Hinglish (romanised Hindi + English) with Hindi grammar, or the short
English formulas such agents really use. Below is what the agent said in ONE call, already split into numbered
utterances (automatic split; a line may be a fragment). Brand: {brand}. Agent: {agent_g}. Customer: {cust_g}.

Label EVERY utterance with exactly one label, judging the text as written:
- NATURAL: a real Hinglish support agent would say this line as it is. Hindi-framed Hinglish with correct grammar
  ("Aapki booking cancel ho gayi hai", "Ek minute, main check karke batata hoon"), or a pure-English formula agents
  really say ("Thank you for calling X, take care") -> NATURAL with en=true.
- STIFF: the meaning is clear but the line is awkward, bookish or unidiomatic, OR it has a meaning-preserving
  grammar slip: wrong verb gender for the agent ({agent_g} agent must say {verb_ok}), wrong address for the customer
  (should be {addr}), noun-gender/number agreement ("order ki status", "request submit ho gaya").
- PEPPERED: English sentence grammar and word order with Hindi words grafted on for flavour ("Sure, let me check
  that for you, bas ek second", "Your order is confirmed ji").
- NONSENSE: no coherent meaning: word salad, a verb or connective that destroys the meaning ("aapko Terminal 3 se
  taaki kar sakti hoon"), a role mix-up (the agent says what only the customer would say), or a cut-off fragment
  with no predicate ("Flight number", "Please aapka", "Shree") -> NONSENSE with trunc=true.
trunc=true only for cut-off fragments; en=true only for pure-English NATURAL lines.
hindi_content=true if a Hindi verb, copula or question frame carries the line's information ("cancel ho gayi hai",
"aapka ETA 12 minutes hai"); false if Hindi is only greeting/filler/politeness (ji, haan, ek minute, koi baat nahi)
or the line has no Hindi. Bare "please aapka X bataiye" requests count as false.
Return {n} labels, i = 1..{n}, in order.

UTTERANCES:
{lines}
"""

NAT_LABELS = ["NATURAL", "STIFF", "PEPPERED", "NONSENSE"]


def nat_schema(n):
    item = {"type": "object", "properties": {"i": {"type": "integer"}, "label": {"type": "string", "enum": NAT_LABELS},
                                             "trunc": {"type": "boolean"}, "en": {"type": "boolean"},
                                             "hindi_content": {"type": "boolean"}},
            "required": ["i", "label", "trunc", "en", "hindi_content"]}
    return {"type": "object", "properties": {"labels": {"type": "array", "items": item, "minItems": n, "maxItems": n}},
            "required": ["labels"]}


_LLM = []


def get_llm():
    """One Gemma load per process (shared by the r2, naturalness and call-judge passes of --suite)."""
    if not _LLM:
        import common
        _LLM.append(common.LLM(gpu_util=0.9, max_model_len=8192))
    return _LLM[0]


def naturalness(argv):
    import argparse
    from tcommon import DATA, load_calls, segments
    ap = argparse.ArgumentParser()
    ap.add_argument("--naturalness", action="store_true")
    ap.add_argument("--variant", default="V4")
    ap.add_argument("--seed", default="1001")
    ap.add_argument("--dest-root")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args(argv)
    calls = load_calls(DATA / a.variant / "calls.jsonl")
    todo = []
    seeds = [x for x in str(a.seed).split(",") if x]  # comma list accepted (section 8); one seed as before
    metas = [(tag, sd, m) for tag in a.tags for sd in seeds
             for m in sorted((OUT / tag / a.variant).glob(f"*_s{sd}.meta.json"))]
    for tag, sd, m in metas:
        if True:
            js = m.with_name(m.name.replace(".meta.json", ".json"))
            odir = Path(a.dest_root) / tag / a.variant if a.dest_root else m.parent
            out = odir / m.name.replace(".meta.json", ".nat.json")
            if not js.exists() or out.exists():
                continue
            cid = m.name[:-len(f"_s{sd}.meta.json")]
            c = calls[cid]
            segs = segments(jload(js))
            ag, cg = c.get("agent_gender", "f"), c.get("customer_gender", "m")
            p = NAT_PROMPT.format(
                brand=c.get("brand"), agent_g="female" if ag == "f" else "male",
                cust_g="female" if cg == "f" else "male",
                verb_ok="sakti/deti/karti/rahi" if ag == "f" else "sakta/deta/karta/raha",
                addr="ma'am" if cg == "f" else "sir", n=len(segs),
                lines="\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(segs)))
            todo.append({"out": out, "segs": segs, "prompt": p, "call_id": cid, "tag": tag})
    print(f"[judge-nat] {len(todo)} runs", flush=True)
    if not todo:
        return
    for t in [t for t in todo if not t["segs"]]:
        t["out"].parent.mkdir(parents=True, exist_ok=True)
        t["out"].write_text(json.dumps({"call_id": t["call_id"], "n": 0, "utterances": [], "note": "no text"}))
    todo = [t for t in todo if t["segs"]]
    if not todo:
        return
    llm = get_llm()
    bad = 0
    for attempt in range(a.retries + 1):
        if not todo:
            break
        # one vLLM batch, per-prompt schema (common.LLM.chat accepts a schema list; section 8: the earlier
        # one-batch-per-segment-count loop ran ~30 small serial batches per tag)
        retry, ts = [], todo
        rs = llm.chat([t["prompt"] for t in ts], schema=[nat_schema(len(t["segs"])) for t in ts],
                      temperature=0.0 if attempt == 0 else 0.3, max_tokens=4000,
                      seeds=[attempt] * len(ts) if attempt else None)
        if True:
            for t, (txt, fin, _) in zip(ts, rs):
                n = len(t["segs"])
                try:
                    labs = json.loads(txt)["labels"]
                    assert len(labs) == n and all(l["label"] in NAT_LABELS for l in labs)
                except Exception:  # noqa: BLE001
                    t["err"] = {"raw": txt[:300], "finish": fin}
                    retry.append(t)
                    continue
                utts = [{"i": k + 1, "start_f": s["start_f"], "end_f": s["end_f"], "text": s["text"],
                         "label": l["label"], "trunc": bool(l["trunc"]) and l["label"] == "NONSENSE",
                         "en": bool(l["en"]), "hindi_content": bool(l["hindi_content"])}
                        for k, (s, l) in enumerate(zip(t["segs"], labs))]
                cnt = {x: sum(u["label"] == x for u in utts) for x in NAT_LABELS}
                t["out"].parent.mkdir(parents=True, exist_ok=True)
                t["out"].write_text(json.dumps({"call_id": t["call_id"], "tag": t["tag"], "n": n, "counts": cnt,
                                                "trunc": sum(u["trunc"] for u in utts),
                                                "hindi_content": sum(u["hindi_content"] for u in utts),
                                                "segmenter": "tcommon.segments(gap_frames=15)",
                                                "attempt": attempt, "utterances": utts}, ensure_ascii=False, indent=1))
        todo = retry
    for t in todo:
        bad += 1
        t["out"].parent.mkdir(parents=True, exist_ok=True)
        t["out"].write_text(json.dumps({"call_id": t["call_id"], "tag": t["tag"], "error": t.get("err")}))
    print(f"[judge-nat] done, {bad} unparsable after retries", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--suite" in argv:
        return suite(argv)
    if "--calljudge" in argv:
        return calljudge(argv)
    if "--naturalness" in argv:
        return naturalness(argv)
    tags = argv
    todo = []
    for tag in tags:
        for m in sorted((OUT / tag).rglob("*.meta.json")):
            js = m.with_name(m.name.replace(".meta.json", ".json"))
            out = m.with_name(m.name.replace(".meta.json", ".judge.json"))
            if js.exists() and not out.exists():
                todo.append((text_from_tokens(jload(js)), out))
    print(f"[judge] {len(todo)} runs", flush=True)
    if not todo:
        return
    empty = [(t, o) for t, o in todo if len(t.split()) < 3]
    for _, o in empty:
        o.write_text(json.dumps({"hindi_verb_final_clauses": 0, "english_clauses": 0, "total_clauses": 0,
                                 "r2": False, "example": "", "note": "fewer than 3 words"}))
    todo = [(t, o) for t, o in todo if len(t.split()) >= 3]
    llm = get_llm()
    res = llm.chat([PROMPT.format(text=t[:6000]) for t, _ in todo], schema=SCHEMA, temperature=0.0, max_tokens=400)
    bad = 0
    for (t, o), (txt, fin, _) in zip(todo, res):
        try:
            d = json.loads(txt)
            d["total_clauses"] = d["hindi_verb_final_clauses"] + d["english_clauses"]
        except Exception:  # noqa: BLE001
            d, bad = {"error": txt[:300], "finish": fin}, bad + 1
        o.write_text(json.dumps(d, ensure_ascii=False))
    print(f"[judge] done, {bad} unparsable", flush=True)


CALL_PROMPT = """You evaluate ONE phone call handled by a customer-support AGENT for {brand} ({agent_g} agent, {cust_g}
customer). The agent is a speech model; its lines below are its own text stream, split automatically at pauses, so a
line may be a fragment. The agent was given this Information (the only true facts it has):
{info}

Write actions the agent was expected to perform in this call (tool and the values involved):
{writes}

THE CALL, in time order (CUSTOMER = the real caller, AGENT = the model):
{timeline}

For reference only, an IDEAL agent for the same customer said:
{ideal}

Judge the AGENT lines only (not the ideal script; the agent may word things differently):
- task_completion: COMPLETE if the agent handled every customer request: answered the questions with the right
  information and, for every expected write, told the customer it was done (or being done) for the right value.
  PARTIAL if it handled some but not all. FAILED if it handled none, or its replies are mostly irrelevant or
  incoherent.
- writes: one entry per expected write, in the order listed: confirmed = the agent told the customer this action was
  done / updated / cancelled / raised; value_correct = the agent's confirmation or acknowledgement does not state a
  value different from the customer's (false if it states a wrong value; true if it states the right one or none).
- fact_consistency: CONSISTENT if every concrete fact the agent states (numbers, IDs, names, dates, times, places,
  prices, statuses, policies) matches the Information or what the customer said; MINOR if there are small slips
  (one digit off, a slightly wrong detail) that do not mislead the customer on the main point; CONTRADICTS if the
  agent states clearly wrong or invented facts.
- wrong_facts: each wrong or invented fact the agent stated, as a short verbatim quote (empty list if none).
- n_facts_stated: number of concrete facts the agent stated.
- reason: one short sentence.
"""


def call_schema(nw):
    w = {"type": "object", "properties": {"i": {"type": "integer"}, "confirmed": {"type": "boolean"},
                                          "value_correct": {"type": "boolean"}},
         "required": ["i", "confirmed", "value_correct"]}
    return {"type": "object", "properties": {
        "task_completion": {"type": "string", "enum": ["COMPLETE", "PARTIAL", "FAILED"]},
        "writes": {"type": "array", "items": w, "minItems": nw, "maxItems": nw},
        "fact_consistency": {"type": "string", "enum": ["CONSISTENT", "MINOR", "CONTRADICTS"]},
        "wrong_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "n_facts_stated": {"type": "integer"}, "reason": {"type": "string"}},
        "required": ["task_completion", "writes", "fact_consistency", "wrong_facts", "n_facts_stated", "reason"]}


def build_call_prompt(tm, c, rm, tokens):
    from tcommon import FRAME_RATE, segments
    skipped = rm.get("skipped_steps", 0)
    items = [(t["start"], "CUSTOMER: " + t["text_roman"]) for t in tm["turns"]
             if t["speaker"] == "customer" and not t.get("unplayed")]  # unplayed: vad mode, hard cap only
    items += [((s["start_f"] + skipped) / FRAME_RATE, "AGENT: " + s["text"]) for s in segments(tokens)]
    timeline = "\n".join(f"[{t:5.1f} s] {x}" for t, x in sorted(items, key=lambda z: z[0]))
    writes = tm.get("writes") or []
    wtxt = "\n".join(f"{i + 1}. {w['tool']}(" + ", ".join(f"{k}={v}" for k, v in w["args"].items()) + ")"
                     for i, w in enumerate(writes)) or "(none: this call has no write action)"
    ag, cg = c.get("agent_gender", "f"), c.get("customer_gender", "m")
    p = CALL_PROMPT.format(brand=c.get("brand"), agent_g="female" if ag == "f" else "male",
                           cust_g="female" if cg == "f" else "male",
                           info=tm["role_prompt"].split("Information:", 1)[-1].strip(), writes=wtxt, timeline=timeline,
                           ideal="\n".join(t["text_roman"] for t in tm["turns"] if t["speaker"] == "agent"))
    return p, len(writes)


def calljudge(argv):
    """D2 spec section 8: Gemma task completion + fact consistency per call (added 2026-10-05).
      judge.py --calljudge [--variant V4] [--seed 1001,1002,1003] [--dry] TAG [TAG ...]  -> <run>.call.json
    Timeline: customer turns from tests/inputs/<V>/<call>.meta.json (start s, 2 s-lead input time) merged with the
    model's segments (tcommon.segments, start = (start_f + skipped_steps) / 12.5 s). Temperature 0, JSON schema."""
    import argparse
    from tcommon import DATA, load_calls, test_meta_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--calljudge", action="store_true")
    ap.add_argument("--variant", default="V4")
    ap.add_argument("--seed", default="1001,1002,1003")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args(argv)
    calls = load_calls(DATA / a.variant / "calls.jsonl")
    todo = []
    for tag in a.tags:
        for sd in [x for x in str(a.seed).split(",") if x]:
            for m in sorted((OUT / tag / a.variant).glob(f"*_s{sd}.meta.json")):
                js = m.with_name(m.name.replace(".meta.json", ".json"))
                out = m.with_name(m.name.replace(".meta.json", ".call.json"))
                if not js.exists() or out.exists():
                    continue
                cid = m.name[:-len(f"_s{sd}.meta.json")]
                p, nw = build_call_prompt(jload(test_meta_path(a.variant, cid, m)), calls[cid], jload(m),
                                          jload(js))
                todo.append({"out": out, "prompt": p, "nw": nw, "call_id": cid, "tag": tag, "seed": int(sd)})
    print(f"[judge-call] {len(todo)} runs", flush=True)
    if a.dry:
        if todo:
            print(todo[0]["prompt"])
        return
    if not todo:
        return
    llm = get_llm()
    for attempt in range(a.retries + 1):
        if not todo:
            break
        groups, retry = {}, []
        for t in todo:
            groups.setdefault(t["nw"], []).append(t)
        for nw, ts in groups.items():
            rs = llm.chat([t["prompt"] for t in ts], schema=call_schema(nw), temperature=0.0 if attempt == 0 else 0.3,
                          max_tokens=1500, seeds=[attempt] * len(ts) if attempt else None)
            for t, (txt, fin, _) in zip(ts, rs):
                try:
                    d = json.loads(txt)
                    assert len(d["writes"]) == nw
                except Exception:  # noqa: BLE001
                    t["err"] = {"raw": txt[:300], "finish": fin}
                    retry.append(t)
                    continue
                d.update({"call_id": t["call_id"], "tag": t["tag"], "seed": t["seed"], "attempt": attempt})
                t["out"].write_text(json.dumps(d, ensure_ascii=False, indent=1))
        todo = retry
    for t in todo:
        t["out"].write_text(json.dumps({"call_id": t["call_id"], "tag": t["tag"], "error": t.get("err")}))
    print(f"[judge-call] done, {len(todo)} unparsable after retries", flush=True)


def suite(argv):
    """D2 spec section 8: one Gemma load for all three passes.
      judge.py --suite [--variant V4] [--seed 1001,1002,1003] TAG [TAG ...]
    = default r2 pass (all runs of the tags, incl. gate0; feeds score.py m2) + --naturalness + --calljudge."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--variant", default="V4")
    ap.add_argument("--seed", default="1001,1002,1003")
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args(argv)
    opts = ["--variant", a.variant, "--seed", a.seed]
    main(list(a.tags))
    naturalness(["--naturalness"] + opts + list(a.tags))
    calljudge(["--calljudge"] + opts + list(a.tags))


if __name__ == "__main__":
    main()
