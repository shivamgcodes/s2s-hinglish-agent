"""S2S serverless worker: copy of needle_v2/schema/router_v2.py (path edits only: numconv = the sibling ../numconv,
no default weights path; the worker passes weights=).
N2 action router: record + customer transcript -> resolved write calls. Same interface as N1 router.py.

    route(agent_type, record, transcript, weights=DEFAULT_WEIGHTS, date_fact=None) -> [call, ...]
    route_raw(...)                -> Needle's full response dict
    resolve_calls(function_calls, record, agent_type) -> [call, ...]   (no model; used by eval/tests)
    prepare_transcript(text)      -> the text Needle sees (number converter applied)

Pipeline (what the data stage must reproduce for training rows):
  0. transcript = Trelis Whisper-Hinglish text of the last 30 s of the customer's audio before the check-line,
     as ONE clip; prepare_transcript() applies numconv (spoken Hindi/English numbers -> digits), then N1
     romanise (Devanagari -> Roman Hinglish).
  1. tools  = tools_v2.tools_for(agent_type)          (7 agent types; REF args + NEW args, see SCHEMA_V2.md)
  2. system = build_data.system_text(record)          (N1 system string: pinned date + the record's information
                                                       + customer phone; the record is in the input)
  3. Needle(tools, system, weights).complete(transcript) -> function_calls
  4. per call, per arg:
       REF -> resolver_v2.resolve(kind, value, record)    -> exact record value, or ASK
       NEW -> phone: 10 digits (ASK otherwise); email: lower-case, spoken "at the rate"/"dot" fixed, must
              contain one @ and a dot after it (ASK otherwise); address/location/instruction/reason: stripped
              text, ASK if empty; request_card_replacement.address omitted -> registered address on file
     a required NEW arg the model left out -> ASK.

Each returned call:
  {"name", "arguments": <model args as emitted>, "server_args": {<writes[] arg name>: value},
   "refs": {model_arg: {"kind", "ref", "value", "rule"}}, "ask": bool, "ask_reasons": [...],
   # N1 router.py compatibility:
   "order_ref": <first REF arg as emitted>, "resolved_id": <its resolved value>, "resolver_rule": <its rule>}
ask=True means the server must ask the caller (do not execute the write).
"""
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n1path  # noqa: E402,F401
import build_data  # noqa: E402  (N1)
import resolver_v2 as rv  # noqa: E402
import targets  # noqa: E402
import tools_v2  # noqa: E402

for _p in (os.path.join(os.path.dirname(HERE), "numconv"),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import numconv  # noqa: E402  (sibling worker's module; not reimplemented here)
except ImportError:  # pragma: no cover
    numconv = None

DEFAULT_WEIGHTS = os.environ.get("NEEDLE_V2_WEIGHTS", "")   # S2S: the worker passes weights= (paths.NEEDLE_V2_WEIGHTS)
MAX_NEW_TOKENS = 512
_CACHE = OrderedDict()
_CACHE_MAX = 8


def prepare_transcript(text):
    """The exact text Needle sees: numconv.convert (spoken numbers -> digits; it reads Devanagari number words),
    then N1 romanise.py on the remaining Devanagari (N1 trained on romanised Trelis text). The data stage must
    build training inputs with THIS function so training matches inference."""
    if numconv is None:
        raise ImportError("numconv not importable (expected ../numconv/numconv.py)")
    return rv.prepare(text)


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
    agent = needle.Needle(tools=tools_v2.tools_for(agent_type), system=system, weights=weights, auto_date=False)
    assert agent._system_text == system, "system text not pinned"
    _CACHE[key] = agent
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)[1].close()
    return agent


def route_raw(agent_type, record, transcript, weights=DEFAULT_WEIGHTS, date_fact=None, convert=True):
    if agent_type not in tools_v2.AGENT_TYPES:
        raise ValueError(f"unknown agent_type {agent_type!r}; one of {tools_v2.AGENT_TYPES}")
    text = prepare_transcript(transcript) if convert else transcript
    agent = _agent(agent_type, system_for(record, date_fact), weights)
    agent.reset()
    return agent.complete(text, MAX_NEW_TOKENS) or {}


_EMAIL_OK = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)+$")


def _new_value(kind, value):
    """(server value, ask_reason or None) for a NEW arg."""
    v = "" if value is None else str(value).strip()
    if kind == "phone":
        d = targets.phone10(v)
        return (d, None) if len(d) == 10 else (d or None, f"phone_not_10_digits:{len(d)}")
    if kind == "email":
        e = targets.norm_email(v)
        return (e, None) if _EMAIL_OK.match(e) else (e or None, "email_malformed")
    return (v, None) if v else (None, f"{kind}_missing")


def resolve_calls(function_calls, record, agent_type):
    out = []
    names = set(tools_v2.STATIC_WRITE[agent_type])
    for c in function_calls or []:
        name = c.get("name")
        args = c.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        call = {"name": name, "arguments": dict(args), "server_args": {}, "refs": {}, "ask": False,
                "ask_reasons": [], "order_ref": None, "resolved_id": None, "resolver_rule": None}
        if name not in names:
            call["ask"] = True
            call["ask_reasons"].append("unknown_tool")
            out.append(call)
            continue
        for marg, kind, garg in tools_v2.arg_specs(agent_type, name):
            v = args.get(marg)
            if kind in tools_v2.REF_KINDS:
                r = rv.resolve(kind, v, record, agent_type)
                call["refs"][marg] = {"kind": kind, "ref": v, "value": r["value"], "rule": r["rule"]}
                call["server_args"][garg] = r["value"]
                if r["ask"]:
                    call["ask_reasons"].append(f"{marg}:{r['rule']}")
                if call["resolver_rule"] is None:
                    call["order_ref"], call["resolved_id"], call["resolver_rule"] = v, r["value"], r["rule"]
            elif kind == "address_or_registered":
                r = rv.resolve_address(v, record)
                call["server_args"][garg] = r["value"]
                call["refs"][marg] = {"kind": kind, "ref": v, "value": r["value"], "rule": r["rule"]}
                if r["ask"]:
                    call["ask_reasons"].append(f"{marg}:{r['rule']}")
            else:
                val, why = _new_value(kind, v)
                call["server_args"][garg] = val
                if why:
                    call["ask_reasons"].append(f"{marg}:{why}")
        unknown = set(args) - {m for m, _, _ in tools_v2.arg_specs(agent_type, name)}
        if unknown:
            call["dropped_args"] = sorted(unknown)
        call["ask"] = bool(call["ask_reasons"])
        out.append(call)
    return out


def route(agent_type, record, transcript, weights=DEFAULT_WEIGHTS, date_fact=None):
    """Resolved calls for one customer transcript (raw Trelis text; numbers are converted here)."""
    resp = route_raw(agent_type, record, transcript, weights, date_fact)
    return resolve_calls(resp.get("function_calls"), record, agent_type)


def close():
    while _CACHE:
        _CACHE.popitem()[1].close()
