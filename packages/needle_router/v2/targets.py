"""Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle_v2/schema/targets.py (verbatim).

Target building for the N2 data stage: V4 gold writes[] -> Needle v2 target calls, per check-line.

    checkline_writes(call)            -> [{"cl_idx", "writes": [w, ...]}]   one entry per check-line turn
    target_call(w, record, agent_type, ref_style="canonical", transcript=None)
                                      -> {"name", "arguments"}   the Needle v2 target for one write
    gold_canonical(w, record, agent_type) -> {"tool", "args"}   gold in server shape, refs canonicalised
    grounded(w, record, agent_type, transcript) -> {model_arg: bool}  is each NEW value recoverable from the
                                         (number-converted) transcript the router sees
    system_text(record)               -> the Needle system string (record included; same as N1)

Check-line -> write mapping: a write belongs to the LAST check-line strictly between its caller_turn_idx and its
confirm_turn_idx. On V4 (645 calls, 647 writes, 647 check-lines) every check-line gets exactly one write; the
"any check-line in the span" rule would give 2 writes to 9 check-lines (air_15, where one caller turn asks for
cancel + refund and the refund is confirmed after the second check-line).

ref_style:
  "canonical" (default, DECISIONS 2026-10-05): every REF arg = the exact record value (ID, last 4, plan /
      service / pack name as written in the record), always present; phone_on_file -> "registered"/"alternate"
      (the model never writes an on-file number). A gold value that is a shortened form of the record value
      (sub_04: "CineMax Ultra" vs record "StreamBox CineMax Ultra") is replaced by the record value.
  "mentioned": a REF arg is present only if the canonical value (or an ID's digit block) occurs in the
      transcript, else omitted (the resolver then defaults). Needs transcript=.
NEW args: phone -> 10 digits, no spaces; email -> lower-case, as in gold; other text args -> the gold text
(the data stage may swap in the transcript span; grounded() tells it which values are recoverable).
"""
import re

import n1path  # noqa: F401
import build_data  # N1 (system_text)
import resolver_v2 as rv
import tools_v2

system_text = build_data.system_text


def checkline_writes(call):
    T = call["turns"]
    cls = [i for i, t in enumerate(T) if "check_line" in (t.get("tags") or [])]
    m = {cl: [] for cl in cls}
    for w in call.get("writes") or []:
        span = [cl for cl in cls if w["caller_turn_idx"] < cl < w["confirm_turn_idx"]]
        if span:
            m[max(span)].append(w)
    return [{"cl_idx": cl, "writes": m[cl]} for cl in cls]


def phone10(v):
    d = re.sub(r"\D", "", rv.compact(str(v or "")))
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d


def norm_email(v):
    s = " " + str(v or "").lower().strip() + " "
    s = re.sub(r"\s+at\s+the\s+rate\s+(of\s+)?|\s+at\s+", "@", s)
    s = re.sub(r"\s+dot\s+", ".", s)
    return re.sub(r"\s+", "", s)


def canonical_ref(kind, gold_value, record, agent_type):
    """Record value the gold points at (resolver on the gold value; falls back to the gold value)."""
    r = rv.resolve(kind, gold_value, record, agent_type)
    return r["value"] if r["value"] is not None else gold_value


def _role_word(record, value):
    for c in rv.candidates("phone_on_file", record):
        if c["value"] == value:
            return c["role"]
    return "registered"


def gold_canonical(w, record, agent_type):
    """Gold write in server shape: {"tool", "args"} with REF values canonicalised to the record value and phone
    digits-only. This is what both N1 and v2 predictions are scored against (score_map.py)."""
    args = {}
    for marg, kind, garg in tools_v2.arg_specs(agent_type, w["tool"]):
        if garg not in w["args"]:
            continue
        v = w["args"][garg]
        if kind in tools_v2.REF_KINDS:
            v = canonical_ref(kind, v, record, agent_type)
        elif kind == "phone":
            v = phone10(v)
        args[garg] = v
    extra = set(w["args"]) - set(args)
    assert not extra, (w, extra)
    return {"tool": w["tool"], "args": args}


def _mentioned(kind, value, transcript):
    t = rv.compact(transcript)
    v = rv.compact(value)
    if v and v in t:
        return True
    if kind in rv.ID_KINDS or kind == "card":
        d = re.sub(r"\D", "", value)
        return len(d) >= 3 and d in set(re.findall(r"\d+", t))
    return False


def target_call(w, record, agent_type, ref_style="canonical", transcript=None):
    g = gold_canonical(w, record, agent_type)
    out = {}
    for marg, kind, garg in tools_v2.arg_specs(agent_type, w["tool"]):
        if garg not in g["args"]:
            continue
        v = g["args"][garg]
        if kind in tools_v2.REF_KINDS:
            if kind == "phone_on_file":
                v = _role_word(record, v)
                if ref_style == "mentioned" and not re.search(r"\b(alternate|alt)\b", transcript or "", re.I) \
                        and v == "registered":
                    continue
            elif ref_style == "mentioned":
                if transcript is None:
                    raise ValueError("ref_style='mentioned' needs transcript")
                if not _mentioned(kind, v, transcript):
                    continue
            elif ref_style != "canonical":
                raise ValueError(ref_style)
        elif kind == "email":
            v = str(v).strip().lower()
        out[marg] = v
    return {"name": w["tool"], "arguments": out}


def _tok(s):
    return [t for t in re.findall(r"[a-z0-9]+", rv.numconv_text(str(s or "")).lower())]


def grounded(w, record, agent_type, transcript, min_recall=0.6):
    """{model_arg: bool} for NEW args: phone -> its 10 digits appear (spaces ignored) in the converted
    transcript; email -> local part appears; text args -> >= min_recall of the value's tokens appear."""
    t = rv.numconv_text(transcript or "")
    tdig = re.sub(r"\D", "", t)
    ttok = set(_tok(t))
    out = {}
    for marg, kind, garg in tools_v2.arg_specs(agent_type, w["tool"]):
        if kind in tools_v2.REF_KINDS or garg not in w["args"]:
            continue
        v = w["args"][garg]
        if kind == "phone":
            out[marg] = phone10(v) in tdig
        elif kind == "email":
            local = norm_email(v).split("@")[0]
            out[marg] = bool(local) and re.sub(r"[^a-z0-9]", "", local) in re.sub(r"[^a-z0-9]", "", t.lower())
        else:
            vt = _tok(v)
            out[marg] = bool(vt) and sum(x in ttok for x in vt) / len(vt) >= min_recall
    return out
