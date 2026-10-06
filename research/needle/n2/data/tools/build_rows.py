"""N2 data stage: Needle v2 rows (+ N1 comparison rows) from the router windows and their Trelis transcripts.
Run with /root/venv-needle/bin/python (needs needle finetune loader + tokenizer). Inputs:
  /root/n2/windows/meta.jsonl, /root/n2/V4/calls.jsonl, /root/n2/data/asr/{clean,opus}.jsonl
Every v2 row: query = router_v2.prepare_transcript(raw text) (numconv, then N1 romanise), system =
targets.system_text(call record), tools = tools_v2.tools_for(agent_type), answers = [targets.target_call(w)] canonical.
Outputs in /root/n2/data/rows (+ line-aligned *_meta.jsonl) and /root/n2/data/build_stats.json."""
import collections, json, os, re, sys
from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[5]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
import router_v2, targets, tools_v2, resolver_v2 as rv  # noqa: E402
import n1path  # noqa: E402,F401
import build_data as n1bd  # noqa: E402  (N1)
import tools as n1tools  # noqa: E402  (N1)

OUT = f"{N2_ROOT}/data/rows"
os.makedirs(OUT, exist_ok=True)
meta = [json.loads(l) for l in open(f"{N2_ROOT}/windows/meta.jsonl")]
calls = {c["call_id"]: c for c in map(json.loads, open(f"{N2_ROOT}/V4/calls.jsonl"))}
asr = {v: {r["id"]: r for r in map(json.loads, open(f"{N2_ROOT}/data/asr/{v}.jsonl"))} for v in ("clean", "opus")}
assert all(len(asr[v]) == len(meta) for v in asr), {v: len(asr[v]) for v in asr}

files = collections.defaultdict(list)  # name -> [(row, meta)]
st = collections.Counter()
nonidem = []
for m in meta:
    call = calls[m["call_id"]]
    rec, at = call["record"], m["agent_type"]
    ent = [e for e in targets.checkline_writes(call) if e["cl_idx"] == m["check_line_turn_idx"]]
    assert len(ent) == 1 and len(ent[0]["writes"]) == 1, m["id"]
    w = ent[0]["writes"][0]
    assert w["tool"] == m["writes"][0]["tool"] and w["args"] == m["writes"][0]["args"], m["id"]
    system = targets.system_text(rec)
    assert system == n1bd.system_text(rec)
    tools = tools_v2.tools_for(at)
    ans = targets.target_call(w, rec, at)
    gold = targets.gold_canonical(w, rec, at)
    # leak check: the gold NEW phone / email must not be in the model input system string
    for marg, kind, garg in tools_v2.arg_specs(at, w["tool"]):
        if garg in w["args"] and kind == "phone":
            assert targets.phone10(w["args"][garg]) not in re.sub(r"\D", "", system), ("phone leak", m["id"])
        if garg in w["args"] and kind == "email":
            assert targets.norm_email(w["args"][garg]) not in system.lower(), ("email leak", m["id"])
    prior = [{"tool": x["tool"], "args": x["args"]} for x in call.get("writes") or []
             if x["confirm_turn_idx"] < m["check_line_turn_idx"]]
    exact_raw = " ".join(t["text_roman"] for t in m["customer_turns_in_window"])
    exact_partial = any(t["coverage"] != "full" for t in m["customer_turns_in_window"])
    raws = {"clean": asr["clean"][m["id"]]["asr_raw"], "opus": asr["opus"][m["id"]]["asr_raw"], "exact": exact_raw}
    for var, raw in raws.items():
        q = router_v2.prepare_transcript(raw)
        if router_v2.prepare_transcript(q) != q:
            nonidem.append({"id": m["id"], "variant": var, "q": q, "qq": router_v2.prepare_transcript(q)})
        if not q.strip():
            st[f"empty_query_{var}"] += 1
            q = "..."
        gr = targets.grounded(w, rec, at, q)
        # targets.grounded email check keeps a spoken "dot" inside the local part ("pandey dot work 60" vs
        # pandey.work60), so it under-counts; re-check with the word "dot" removed (data-stage fix, schema untouched)
        for marg, kind, garg in tools_v2.arg_specs(at, w["tool"]):
            if kind == "email" and marg in gr and not gr[marg]:
                local = re.sub(r"[^a-z0-9]", "", targets.norm_email(w["args"][garg]).split("@")[0])
                qq = re.sub(r"[^a-z0-9]", "", re.sub(r"\bdot\b", "", q.lower()))
                gr[marg] = bool(local) and local in qq
        kinds = {marg: kind for marg, kind, garg in tools_v2.arg_specs(at, w["tool"])}
        ref_mentioned = {marg: targets._mentioned(kinds[marg], v, q) for marg, v in ans["arguments"].items()
                         if kinds[marg] in tools_v2.REF_KINDS and kinds[marg] != "phone_on_file"}
        pe_ok = all(v for k, v in gr.items() if kinds[k] in ("phone", "email"))
        row = {"query": q, "tools": tools, "answers": [ans], "system": system}
        mm = {"example_id": m["id"] + ":" + var, "id": m["id"], "call_id": m["call_id"], "scenario_id": m["scenario_id"],
              "split": m["split"], "variant": var, "agent_type": at, "record_id": m["record_id"], "tool": w["tool"],
              "check_line_ordinal": m["check_line_ordinal"], "check_line_turn_idx": m["check_line_turn_idx"],
              "window_len_s": m["window_len_s"], "in_holdout_test_calls": m["in_holdout_test_calls"],
              "gold_args_raw": w["args"], "gold_canonical": gold, "raw_text": raw,
              "grounded": gr, "phone_email_grounded": pe_ok, "ref_mentioned": ref_mentioned,
              "arg_coverage": m["writes"][0].get("arg_coverage"), "prior_writes": prior,
              "exact_has_partial_turn": exact_partial}
        sp = m["split"]
        if var != "exact" and sp == "train":
            files["train"].append((row, mm))
            if pe_ok:
                files["train_grounded"].append((row, mm))
        if sp in ("val", "test"):
            files[f"{sp}_{var}"].append((row, mm))
            if sp == "val" and var != "exact":
                files["val"].append((row, mm))
        # N1 comparison rows: 5 old agent types, test split, same windows
        if sp == "test" and at in n1tools.AGENT_TYPES:
            for suf, q1 in (("", rv.romanise_text(raw)), ("_nc", q)):
                q1 = q1 if q1.strip() else "..."
                call1, g1, rule1, flags1 = n1bd.build_target(w, dict(rec, brand=call["brand"]), at, q1)  # brand: N1 record_vocab needs it (target only, not model input)
                call1 = {"name": call1["name"], "arguments": {k: v for k, v in call1["arguments"].items() if v}}
                row1 = {"query": q1, "tools": n1tools.tools_for(at), "answers": [call1], "system": n1bd.system_text(rec)}
                mm1 = dict(mm, n1_query_style="romanise_only (N1 native)" if not suf else "numconv+romanise (v2 prepare)",
                           n1_order_ref_rule=rule1, n1_gold_entity_id=g1["entity_id"])
                files[f"n1_test_{var}{suf}"].append((row1, mm1))

for name, rows in files.items():
    with open(f"{OUT}/{name}.jsonl", "w") as f, open(f"{OUT}/{name}_meta.jsonl", "w") as g:
        for r, mm in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps(mm, ensure_ascii=False) + "\n")
    print(name, len(rows))
json.dump({"counts": {k: len(v) for k, v in files.items()}, "misc": dict(st), "nonidempotent": nonidem},
          open(f"{N2_ROOT}/data/build_stats.json", "w"), indent=1, ensure_ascii=False)
print("nonidempotent prepare:", len(nonidem), "misc", dict(st))
