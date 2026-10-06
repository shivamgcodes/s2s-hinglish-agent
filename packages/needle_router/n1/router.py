"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle/router.py (verbatim).

N1 action router: transcript -> resolved tool calls (spec §6).

    route(agent_type, record, transcript) -> [{name, arguments, order_ref, resolved_id, resolver_rule}]

Pipeline (the same one evaluate.py scores):
  1. tools   = tools.tools_for(agent_type)            (no tool takes an ID; every tool takes order_ref)
  2. system  = build_data.system_text(record)         ("date: 2026-10-04 Sun 10:00; user: customer phone ...")
     The date fact is pinned exactly as in training and eval (auto_date=False). Pass `date_fact=` to use a
     different date line; that input was not evaluated.
  3. Needle(tools, system, weights=DEFAULT_WEIGHTS).complete(transcript) -> function_calls (turn 1 only;
     nothing is executed). A local fine-tune has no confidence head (confidence None), so the gate is
     "a call shipped": every entry of function_calls is returned, suppressed_calls are not.
  4. resolver.resolve(order_ref, record, agent_type) -> resolved_id + rule, per call. order_ref may be
     omitted by the model ("my order" with nothing named); the resolver then defaults to the single active
     entity. resolved_id None (rule "ambiguous") means the server must ask which one.

`record` is one entry of src/records.json (keys used: information, phone, facts, distractors, primary_id,
...; see resolver.entities_from_record). CPU only. One Needle instance is cached per (agent_type, system,
weights); building one takes about 1 s, a call about 0.2-0.4 s on the laptop (REPORT.md, latency).

Known limit (REPORT.md): the tuned model fires a tool on many chit-chat/read turns, so call route() only
on the turns the agent flags with a check-line, as the client already does.
"""
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_data  # noqa: E402
import resolver  # noqa: E402
import tools  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(HERE, "tuned_full.cact")  # e10 LoRA, 20 layers (REPORT.md, chosen model)
MAX_NEW_TOKENS = 512
_CACHE = OrderedDict()
_CACHE_MAX = 8


def system_for(record, date_fact=None):
    s = build_data.system_text(record)
    if date_fact is not None:
        s = s.replace(build_data.PINNED_DATE, date_fact, 1)
    return s


def _agent(agent_type, system, weights):
    import needle
    key = (agent_type, system, weights)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    agent = needle.Needle(tools=tools.tools_for(agent_type), system=system, weights=weights, auto_date=False)
    assert agent._system_text == system, "system text not pinned"
    _CACHE[key] = agent
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)[1].close()
    return agent


def route_raw(agent_type, record, transcript, weights=DEFAULT_WEIGHTS, date_fact=None):
    """Needle's full response dict (function_calls, suppressed_calls, reasoning, ...) for one transcript."""
    if agent_type not in tools.AGENT_TYPES:
        raise ValueError(f"unknown agent_type {agent_type!r}; one of {tools.AGENT_TYPES}")
    agent = _agent(agent_type, system_for(record, date_fact), weights)
    agent.reset()
    return agent.complete(transcript, MAX_NEW_TOKENS) or {}


def resolve_calls(function_calls, record, agent_type):
    out = []
    for c in function_calls or []:
        args = dict(c.get("arguments") or {})
        ref = args.pop("order_ref", None) or None
        r = resolver.resolve(ref or "", record, agent_type)
        out.append({"name": c["name"], "arguments": args, "order_ref": ref,
                    "resolved_id": r["id"], "resolver_rule": r["rule"]})
    return out


def route(agent_type, record, transcript, weights=DEFAULT_WEIGHTS, date_fact=None):
    """Resolved calls for one customer transcript; [] when the model ships no call."""
    resp = route_raw(agent_type, record, transcript, weights, date_fact)
    return resolve_calls(resp.get("function_calls"), record, agent_type)


def close():
    while _CACHE:
        _CACHE.popitem()[1].close()


def _selftest(n_per_file=1):
    """Route the first positive row of each held-out file (and one more from a) and compare the model's
    function_calls with what evaluate.py recorded for the same row (results/tuned_full_<test>_raw.jsonl)."""
    recs = json.load(open(os.path.join(HERE, "src", "records.json"), encoding="utf-8"))
    picks = [("test_heldout_a", 0), ("test_heldout_a", 1), ("test_heldout_b", 0)]
    seen = {}
    ok = 0
    for test, k in picks:
        rows = [json.loads(l) for l in open(os.path.join(HERE, test + ".jsonl"), encoding="utf-8")]
        meta = [json.loads(l) for l in open(os.path.join(HERE, test + "_meta.jsonl"), encoding="utf-8")]
        pos = [i for i, m in enumerate(meta) if m["type"] != "negative"]
        i = pos[k * 37 % len(pos)]
        raw_path = os.path.join(HERE, "results", f"tuned_full_{test}_raw.jsonl")
        raw = [json.loads(l) for l in open(raw_path, encoding="utf-8")][1:] if os.path.exists(raw_path) else None
        m, row = meta[i], rows[i]
        rec = recs[m["scenario_id"]]
        resp = route_raw(m["agent_type"], rec, row["query"])
        calls = resolve_calls(resp.get("function_calls"), rec, m["agent_type"])
        same = raw is not None and raw[i]["response"]["function_calls"] == resp.get("function_calls")
        ok += same
        seen[(test, i)] = same
        print(json.dumps({"test": test, "row": i, "example_id": m["example_id"], "query": row["query"],
                          "gold": row["answers"], "gold_entity_id": m.get("gold_entity_id"),
                          "routed": calls, "reasoning": resp.get("reasoning"),
                          "identical_to_eval_run": same}, ensure_ascii=False, indent=1))
    print(f"identical to evaluate.py output: {ok}/{len(picks)}")
    close()


if __name__ == "__main__":
    _selftest()
