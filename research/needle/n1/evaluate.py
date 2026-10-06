"""N1 evaluator (spec §4 baseline + §5 metrics): run a Needle model over a test file and score it.

Usage (source ../../needle-exp0/env.sh first; run with $NEEDLE_PY, CPU only):
    $NEEDLE_PY evaluate.py run   --test test_n0 --tag base                 # base Needle 3
    $NEEDLE_PY evaluate.py run   --test test_heldout_a --tag tuned --weights tuned.cact
    $NEEDLE_PY evaluate.py score --test test_n0 --tag base                 # rescore an existing raw file
    $NEEDLE_PY evaluate.py summary --tag base                              # results/<tag>_summary.md

Files (results/ next to this script):
    <tag>_<test>_raw.jsonl   line 1 = run metadata, then one line per test row in file order, with the
                             engine's raw response to the query, flushed row by row
    <tag>_<test>.json        metrics, per-argument table, confusion matrix, every failure verbatim, per-row scores
    <tag>_summary.md         tables over all tests of one tag

Engine call. Per row: agent.reset(); agent.complete(query). Turn 1 only (the tools are JSON dicts with
no callables, so nothing is executed). Rows are grouped by (tools JSON, system text) in order of first
appearance; one Needle(tools=row.tools, system=row.system, auto_date=False) per group, checked so that
agent._system_text == row.system byte for byte (the pinned `date: 2026-10-04 Sun 10:00` fact, A8).
Each group gets one untimed warm-up query ("hi there", not in any test set) before its rows. Latency =
time.perf_counter() around complete(). Threads are not capped, the same as N0 (N0 only recorded the
thread count); threads and load average are recorded in the metadata.

Scoring rules (spec §5; the choices the spec leaves open are in DECISIONS.md section E):
  tool match       predicted function_calls names == gold names, same order and count; on a
                   negative row, correct iff function_calls == [] (= empty-list accuracy)
  real-arg match   (only when tool match) every non-order_ref argument over the union of gold and
                   predicted keys; a missing or extra argument fails
                     phone                digits-only equality after spoken digits are turned into digits
                     address, instruction token F1 >= 0.6 on normalised tokens
                     every other argument normalised equality (location, reason, email)
  order_ref        gold names it: normalised equality (spoken_to_compact of both sides);
                   gold omits it (A1): match iff the prediction omits it too (or sends "")
  arg match        real-arg match and order_ref match
  resolver         resolver.resolve(predicted order_ref, record, agent_type) == gold entity id, for each
                   aligned call; reported over tool-matched rows and over all positive rows
  correct          positive: tool match and real-arg match and resolver correct (what the server would do)
                   negative: function_calls == []
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
N1_PKG = os.environ.get("N1_DIR", str(REPO / "packages" / "needle_router" / "n1"))  # build_data, resolver, router, tools, src/records.json
HERE = os.environ.get("N1_WORKDIR", "/workspace/hinglish/needle")  # N1 working dir (data, results/); was this script's own dir
sys.path.insert(0, N1_PKG)
import resolver  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TESTS = ["test_n0", "test_heldout_a", "test_heldout_b"]
WARMUP_QUERY = "hi there"
OVERLAP_ARGS = {"address", "instruction"}
OVERLAP_MIN = 0.6
LANG = {"en": "english", "english": "english", "hinglish": "hinglish"}


# ------------------------------------------------------------------------------------- loading
def load_test(name):
    rows = [json.loads(l) for l in open(os.path.join(HERE, name + ".jsonl"), encoding="utf-8") if l.strip()]
    meta = [json.loads(l) for l in open(os.path.join(HERE, name + "_meta.jsonl"), encoding="utf-8") if l.strip()]
    assert len(rows) == len(meta), (name, len(rows), len(meta))
    return rows, meta


_RECORDS = None


def record_for(m):
    """The record the server resolves against: N0_RECORD for test_n0, records.json[scenario_id] else."""
    global _RECORDS
    if m.get("eval_record") == "resolver.N0_RECORD":
        return resolver.N0_RECORD
    if _RECORDS is None:
        _RECORDS = json.load(open(os.path.join(N1_PKG, "src", "records.json"), encoding="utf-8"))
    return _RECORDS[m["scenario_id"]]


def gold_entities(m, n_calls):
    if "expected_entity_ids" in m:
        return list(m["expected_entity_ids"])
    return [m.get("gold_entity_id")] * n_calls


def row_info(m, rendering_default):
    """Uniform per-row descriptors for both meta shapes (test_n0 vs held-out)."""
    if "kind" in m:  # test_n0
        positive = m["kind"] != "read"
        return {"row_id": m["id"], "language": LANG[m["lang"]], "kind": m["kind"], "positive": positive,
                "rendering": "a", "agent_type": m["agent_type"]}
    return {"row_id": m["example_id"], "language": LANG[m["language"]], "kind": m["type"],
            "positive": m["type"] != "negative", "rendering": m.get("rendering", rendering_default),
            "agent_type": m["agent_type"], "variant": m.get("variant"), "is_test_call": m.get("is_test_call"),
            "shape": m.get("shape"), "args_grounded": m.get("args_grounded")}


# ------------------------------------------------------------------------------------- running
def n_threads():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except OSError:
        return None


def cmd_run(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "source needle-exp0/env.sh (CPU only)"
    if args.weights and not args.weights.endswith(".cact"):
        sys.exit("--weights takes a built .cact; build an adapter first: "
                 "needle build --lora adapter.safetensors [--layers N] --out tuned.cact")
    import platform
    import needle

    rows, meta = load_test(args.test)
    groups = OrderedDict()
    for i, r in enumerate(rows):
        key = (json.dumps(r["tools"], ensure_ascii=False, separators=(",", ":")), r.get("system") or "")
        groups.setdefault(key, []).append(i)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{args.tag}_{args.test}_raw.jsonl")
    head = {
        "_meta": True, "tag": args.tag, "test": args.test, "n_rows": len(rows), "n_groups": len(groups),
        "weights_arg": args.weights, "needle_version": needle.__version__,
        "engine_lib": needle._library_path(3), "python": sys.version.split()[0],
        "machine": platform.platform(), "cpu_count": os.cpu_count(),
        "loadavg_start": os.getloadavg(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "auto_date": False, "warmup_query": WARMUP_QUERY, "max_new_tokens": args.max_new_tokens,
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HOME", "NEEDLE_TELEMETRY",
                                               "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    results = [None] * len(rows)
    group_meta = []
    t_all = time.perf_counter()
    for gi, ((tools_json, system), idxs) in enumerate(groups.items()):
        t0 = time.perf_counter()
        agent = needle.Needle(tools=rows[idxs[0]]["tools"], system=system, weights=args.weights,
                              auto_date=False)
        assert agent._system_text == system, "system text not pinned"
        init_s = time.perf_counter() - t0
        agent.reset()
        t0 = time.perf_counter()
        agent.complete(WARMUP_QUERY, args.max_new_tokens)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        if gi == 0:
            head["weights"] = args.weights or needle._loaded_base.get(3)
            head["threads_in_process_after_warmup"] = n_threads()
        group_meta.append({"group": gi, "n_rows": len(idxs), "init_s": init_s, "warmup_ms": warm_ms,
                           "agent_type": meta[idxs[0]].get("agent_type"),
                           "scenario_id": meta[idxs[0]].get("scenario_id")})
        for pos, i in enumerate(idxs):
            agent.reset()
            err, resp = None, None
            t0 = time.perf_counter()
            try:
                resp = agent.complete(rows[i]["query"], args.max_new_tokens)
            except Exception as exc:  # recorded, not hidden
                err = repr(exc)
            ms = (time.perf_counter() - t0) * 1000.0
            results[i] = {"i": i, "group": gi, "pos_in_group": pos, "latency_ms": ms, "error": err,
                          "response": json.loads(json.dumps(resp, ensure_ascii=False)) if resp else None}
            fc = (resp or {}).get("function_calls")
            print(f"[{i+1}/{len(rows)}] g{gi} {ms:6.0f}ms calls={json.dumps(fc, ensure_ascii=False)[:160]}",
                  flush=True)
        agent.close()
    head["loadavg_end"] = os.getloadavg()
    head["wall_s"] = time.perf_counter() - t_all
    head["groups"] = group_meta
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(head, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out_path)
    cmd_score(args)


# ------------------------------------------------------------------------------------- normalisation
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
# romanised Hindi number words; "do" (2) left out on purpose ("kar do"), as in N0's score.py
_HINDI = {"ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6,
          "saat": 7, "aath": 8, "nau": 9, "das": 10}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm_tokens(value):
    """Lower-case alnum tokens; English/Hindi number words -> digits, incl. 'X hundred [Y]' and
    'twenty five'. 'B-42' -> ['b', '42']; 'four hundred five' -> ['405']."""
    s = str(value or "").translate(_DEV_DIGITS).lower().replace("’", "'").replace("'", "")
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    toks, out, i = s.split(), [], 0
    while i < len(toks):
        t = toks[i]
        if t in _UNITS or t in _TENS or t in _HINDI:
            n = _UNITS.get(t, _TENS.get(t, _HINDI.get(t)))
            j = i + 1
            if t in _TENS and j < len(toks) and toks[j] in _UNITS and 1 <= _UNITS[toks[j]] <= 9:
                n += _UNITS[toks[j]]; j += 1
            if j < len(toks) and toks[j] == "hundred":
                n *= 100; j += 1
                if j < len(toks) and toks[j] in ("and",) and j + 1 < len(toks) and toks[j + 1] in _UNITS:
                    j += 1
                if j < len(toks) and (toks[j] in _UNITS or toks[j] in _TENS):
                    m = _UNITS.get(toks[j], _TENS.get(toks[j])); j += 1
                    if toks[j - 1] in _TENS and j < len(toks) and toks[j] in _UNITS and 1 <= _UNITS[toks[j]] <= 9:
                        m += _UNITS[toks[j]]; j += 1
                    n += m
            out.append(str(n)); i = j
            continue
        out.append(t); i += 1
    return out


def norm_eq_text(value):
    return " ".join(norm_tokens(value))


def token_f1(gold, pred):
    g, p = Counter(norm_tokens(gold)), Counter(norm_tokens(pred))
    if not g or not p:
        return 0.0
    common = sum((g & p).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _strip_phone(d):
    if len(d) == 12 and d.startswith("91"):
        return d[2:]
    if len(d) == 11 and d.startswith("0"):
        return d[1:]
    return d


def phone_digits(value):
    """Digits-only form. Spoken digits ('nine eight zero', 'double five', Hindi digit words) become
    digits via resolver.spoken_to_compact; returns that and the N0 norm (tens words) as candidates."""
    a = _strip_phone(re.sub(r"\D", "", resolver.spoken_to_compact(str(value or ""))))
    b = _strip_phone(re.sub(r"\D", "", norm_eq_text(value)))
    return {a, b} - {""}


def norm_email(value):
    s = str(value or "").lower().strip()
    s = re.sub(r"\s+at\s+the\s+rate\s+|\s+at\s+", "@", " " + s + " ").strip()
    s = re.sub(r"\s+dot\s+", ".", " " + s + " ").strip()
    return re.sub(r"\s+", "", s)


def norm_ref(value):
    return resolver.spoken_to_compact(str(value or ""))


def compare_arg(key, gold, pred):
    """-> (ok, detail dict)."""
    if pred is None or str(pred).strip() == "":
        return False, {"missing": True}
    if gold is None:
        return False, {"extra": True}
    if key == "phone":
        g, p = phone_digits(gold), phone_digits(pred)
        literal = re.sub(r"\D", "", str(pred)) == re.sub(r"\D", "", str(gold))
        return bool(g & p), {"gold_digits": sorted(g), "pred_digits": sorted(p), "literal_digits_equal": literal}
    f1 = token_f1(gold, pred)
    if key in OVERLAP_ARGS:
        return f1 >= OVERLAP_MIN, {"token_f1": round(f1, 4)}
    if key == "email":
        return norm_email(gold) == norm_email(pred), {"token_f1": round(f1, 4)}
    return norm_eq_text(gold) == norm_eq_text(pred), {"token_f1": round(f1, 4),
                                                      "f1_ge_0.6": f1 >= OVERLAP_MIN}


# ------------------------------------------------------------------------------------- scoring
def score_calls(gold_calls, got_calls, m, info, record):
    """Score one list of predicted calls against gold for one row."""
    gnames = [c["name"] for c in gold_calls]
    pnames = [c.get("name") for c in got_calls]
    s = {"pred_names": pnames, "gold_names": gnames, "tool_match": gnames == pnames}
    if not info["positive"]:
        s["tool_match"] = s["correct"] = len(got_calls) == 0
        return s
    ents = gold_entities(m, len(gold_calls))
    s.update(real_args_match=None, order_ref_match=None, args_match=None, resolver_ok=None,
             per_call=[], correct=False)
    if s["tool_match"]:
        real_ok_all, ref_ok_all, res_ok_all = True, True, True
        for gc, pc, ent in zip(gold_calls, got_calls, ents):
            ga, pa = gc.get("arguments") or {}, pc.get("arguments") or {}
            if not isinstance(pa, dict):
                pa = {}
            arg_res = {}
            for k in sorted((set(ga) | set(pa)) - {"order_ref"}):
                ok, det = compare_arg(k, ga.get(k), pa.get(k))
                arg_res[k] = {"ok": ok, "gold": ga.get(k), "pred": pa.get(k), **det}
                real_ok_all &= ok
            gref, pref = ga.get("order_ref"), pa.get("order_ref")
            pref_s = "" if pref is None else str(pref).strip()
            if gref:
                ref_ok = norm_ref(gref) == norm_ref(pref_s)
                ref_kind = "named"
            else:
                ref_ok = pref_s == ""
                ref_kind = "omitted"
            ref_ok_all &= ref_ok
            res = resolver.resolve(pref_s, record, info["agent_type"])
            res_ok = res["id"] == ent
            res_ok_all &= res_ok
            s["per_call"].append({"name": gc["name"], "args": arg_res,
                                  "order_ref": {"gold": gref, "pred": pref, "gold_kind": ref_kind, "ok": ref_ok},
                                  "resolver": {"pred_id": res["id"], "rule": res["rule"], "gold_id": ent,
                                               "ok": res_ok}})
        s["real_args_match"] = real_ok_all
        s["order_ref_match"] = ref_ok_all
        s["args_match"] = real_ok_all and ref_ok_all
        s["resolver_ok"] = res_ok_all
        s["correct"] = real_ok_all and res_ok_all
    # failure group
    if not s["correct"]:
        if not got_calls:
            s["group"] = "missed call"
        elif not s["tool_match"]:
            if len(pnames) < len(gnames) and all(n in gnames for n in pnames):
                s["group"] = "partial (fewer calls)"
            elif sorted(pnames) == sorted(gnames):
                s["group"] = "right tools, wrong order"
            else:
                s["group"] = "wrong tool"
        elif not s["real_args_match"]:
            s["group"] = "wrong argument"
        else:
            s["group"] = "resolver wrong (order_ref)"
    return s


def multi_class(gnames, pnames):
    if pnames == gnames:
        return "exact order"
    if sorted(pnames) == sorted(gnames):
        return "right set, wrong order"
    if pnames and len(pnames) < len(gnames) and all(n in gnames for n in pnames):
        return "partial (subset)"
    if not pnames:
        return "no call"
    return "other"


def pctile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def frac(k, n):
    return {"k": k, "n": n, "pct": round(100.0 * k / n, 1) if n else None}


def aggregate(rows):
    pos = [r for r in rows if r["info"]["positive"]]
    neg = [r for r in rows if not r["info"]["positive"]]
    tm = [r for r in pos if r["score"]["tool_match"]]
    named = [pc for r in tm for pc in r["score"]["per_call"] if pc["order_ref"]["gold_kind"] == "named"]
    omitted = [pc for r in tm for pc in r["score"]["per_call"] if pc["order_ref"]["gold_kind"] == "omitted"]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    shipped = [r for r in pos if r["pred_calls"]]
    withheld = [r for r in pos if not r["pred_calls"] and r["suppressed_calls"]]
    return {
        "rows": len(rows), "positives": len(pos), "negatives": len(neg),
        "correct_all_rows": frac(sum(r["score"]["correct"] for r in rows), len(rows)),
        "correct_positives": frac(sum(r["score"]["correct"] for r in pos), len(pos)),
        "tool_exact_match_positives": frac(len(tm), len(pos)),
        "real_args_match_of_tool_matched": frac(sum(r["score"]["real_args_match"] for r in tm), len(tm)),
        "args_match_strict_of_tool_matched": frac(sum(r["score"]["args_match"] for r in tm), len(tm)),
        "args_match_strict_of_positives": frac(sum(bool(r["score"]["args_match"]) for r in pos), len(pos)),
        "order_ref_match_gold_named_calls": frac(sum(pc["order_ref"]["ok"] for pc in named), len(named)),
        "order_ref_omitted_agree_calls": frac(sum(pc["order_ref"]["ok"] for pc in omitted), len(omitted)),
        "resolver_ok_of_tool_matched": frac(sum(r["score"]["resolver_ok"] for r in tm), len(tm)),
        "resolver_ok_of_positives": frac(sum(bool(r["score"]["resolver_ok"]) for r in pos), len(pos)),
        "empty_list_accuracy_negatives": frac(sum(r["score"]["correct"] for r in neg), len(neg)),
        "negatives_with_suppressed_calls_only": sum(1 for r in neg if not r["pred_calls"] and r["suppressed_calls"]),
        "positives_shipped": frac(len(shipped), len(pos)),
        "positives_withheld_only": frac(len(withheld), len(pos)),
        "positives_no_call_at_all": frac(sum(1 for r in pos if not r["pred_calls"] and not r["suppressed_calls"]), len(pos)),
        "correct_positives_if_suppressed_shipped": frac(sum(r["score_if_shipped"]["correct"] for r in pos), len(pos)),
        "tool_match_positives_if_suppressed_shipped": frac(sum(r["score_if_shipped"]["tool_match"] for r in pos), len(pos)),
        "errors": sum(1 for r in rows if r.get("error")),
        "latency_ms_median": round(pctile(lat, 0.5), 1) if lat else None,
        "latency_ms_p95": round(pctile(lat, 0.95), 1) if lat else None,
        "latency_ms_max": round(max(lat), 1) if lat else None,
        "latency_n": len(lat),
    }


def cmd_score(args):
    raw_path = os.path.join(RESULTS, f"{args.tag}_{args.test}_raw.jsonl")
    lines = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    head, raws = lines[0], lines[1:]
    rows, meta = load_test(args.test)
    assert len(raws) == len(rows)
    render_default = {"test_heldout_a": "a", "test_heldout_b": "b"}.get(args.test, "a")

    # gold sanity: resolving each gold order_ref gives the gold entity (480/480 held-out, 48/48 n0)
    # Not asserted: the 4 N0 noisy spellings (biriyani/dominoz) are a known resolver gap (DECISIONS.md
    # §2-3 "Resolver on the N0 relabel"); listed so router and resolver errors stay separate.
    gold_res_ok = gold_res_n = 0
    gold_res_fail = []
    scored = []
    for row, m, raw in zip(rows, meta, raws):
        info = row_info(m, render_default)
        record = record_for(m) if info["positive"] else None
        gold = row["answers"]
        if info["positive"]:
            for gc, ent in zip(gold, gold_entities(m, len(gold))):
                gold_res_n += 1
                gref = (gc["arguments"] or {}).get("order_ref", "")
                gr = resolver.resolve(gref, record, info["agent_type"])
                if gr["id"] == ent:
                    gold_res_ok += 1
                else:
                    gold_res_fail.append({"row_id": info["row_id"], "gold_order_ref": gref, "gold_id": ent,
                                          "resolved": gr["id"], "rule": gr["rule"]})
        resp = raw.get("response") or {}
        fcalls = resp.get("function_calls") or []
        supp = resp.get("suppressed_calls") or []
        s = score_calls(gold, fcalls, m, info, record)
        s_if = score_calls(gold, fcalls or supp, m, info, record)
        scored.append({"info": info, "query": row["query"], "gold": gold, "pred_calls": fcalls,
                       "suppressed_calls": supp, "reasoning": resp.get("reasoning"),
                       "confidence": resp.get("confidence"), "response": resp, "error": raw.get("error"),
                       "latency_ms": raw.get("latency_ms"), "pos_in_group": raw.get("pos_in_group"),
                       "score": s, "score_if_shipped": {k: s_if.get(k) for k in ("correct", "tool_match", "group")}})

    # breakdowns
    by = {"overall": aggregate(scored)}
    for key in ("language", "rendering", "kind", "variant"):
        vals = sorted({r["info"].get(key) for r in scored if r["info"].get(key) is not None})
        if len(vals) >= 1:
            by[f"by_{key}"] = {v: aggregate([r for r in scored if r["info"].get(key) == v]) for v in vals}
    by["by_rendering_language"] = {f"{rd}/{lg}": aggregate([r for r in scored if r["info"]["rendering"] == rd
                                                            and r["info"]["language"] == lg])
                                   for rd in sorted({r["info"]["rendering"] for r in scored})
                                   for lg in sorted({r["info"]["language"] for r in scored})
                                   if any(r["info"]["rendering"] == rd and r["info"]["language"] == lg
                                          for r in scored)}
    by["by_gold_tool"] = {}
    for name in sorted({"+".join(r["score"]["gold_names"]) for r in scored if r["info"]["positive"]}):
        sub = [r for r in scored if r["info"]["positive"] and "+".join(r["score"]["gold_names"]) == name]
        by["by_gold_tool"][name] = aggregate(sub)

    # latency excluding the first row after each group's warm-up (sensitivity check)
    lat_rest = [r["latency_ms"] for r in scored if r["pos_in_group"] and r["latency_ms"] is not None]
    latency_extra = {"excluding_first_row_of_each_group": {
        "n": len(lat_rest), "median": round(pctile(lat_rest, .5), 1) if lat_rest else None,
        "p95": round(pctile(lat_rest, .95), 1) if lat_rest else None}}

    # confusion matrix (row level: gold label vs predicted label; label = calls joined by '+', 'none' = [])
    conf = defaultdict(Counter)
    for r in scored:
        g = "+".join(r["score"]["gold_names"]) or "none"
        p = "+".join(r["score"]["pred_names"]) or "none"
        conf[g][p] += 1
    confusion = {g: dict(c) for g, c in sorted(conf.items())}

    # per-argument table over aligned calls of tool-matched rows
    per_arg = defaultdict(lambda: Counter())
    for r in scored:
        for pc in r["score"].get("per_call") or []:
            for k, a in pc["args"].items():
                t = per_arg[k]
                t["n"] += 1
                t["ok_rule"] += a["ok"]
                if "token_f1" in a:
                    t["f1_ge_0.6"] += a["token_f1"] >= OVERLAP_MIN
                    t["norm_equal"] += norm_eq_text(a["gold"]) == norm_eq_text(a["pred"]) if a["gold"] and a["pred"] else 0
                if "literal_digits_equal" in a:
                    t["literal_digits_equal"] += a["literal_digits_equal"]
                t["missing"] += bool(a.get("missing"))
                t["extra"] += bool(a.get("extra"))
            o = per_arg["order_ref (" + pc["order_ref"]["gold_kind"] + ")"]
            o["n"] += 1
            o["ok_rule"] += pc["order_ref"]["ok"]
            o["resolver_ok"] += pc["resolver"]["ok"]
    per_arg = {k: dict(v) for k, v in sorted(per_arg.items())}

    # multi-call order (gold rows with >1 call)
    multi = Counter(multi_class(r["score"]["gold_names"], r["score"]["pred_names"])
                    for r in scored if len(r["score"]["gold_names"]) > 1)

    # resolver rules used on tool-matched calls
    rules = Counter(pc["resolver"]["rule"] for r in scored for pc in r["score"].get("per_call") or [])

    failures = []
    for r in scored:
        if r["score"]["correct"]:
            continue
        failures.append({"row_id": r["info"]["row_id"], "language": r["info"]["language"],
                         "rendering": r["info"]["rendering"], "kind": r["info"]["kind"],
                         "group": r["score"].get("group", "spurious call on a negative"),
                         "query": r["query"], "gold": r["gold"], "function_calls": r["pred_calls"],
                         "suppressed_calls": r["suppressed_calls"], "reasoning": r["reasoning"],
                         "per_call": r["score"].get("per_call"), "error": r["error"],
                         "raw_response": r["response"]})
    fail_groups = Counter(f["group"] for f in failures)

    out = {"test": args.test, "tag": args.tag, "run_meta": head,
           "gold_resolver_sanity": frac(gold_res_ok, gold_res_n), "gold_resolver_failures": gold_res_fail,
           "metrics": by, "latency_extra": latency_extra, "multi_call_order": dict(multi),
           "confusion_matrix": confusion, "per_argument": per_arg, "resolver_rules_on_predictions": dict(rules),
           "failure_groups": dict(fail_groups), "failures": failures,
           "rows": [{"row_id": r["info"]["row_id"], **r["info"], "query": r["query"], "gold": r["gold"],
                     "function_calls": r["pred_calls"], "suppressed_calls": r["suppressed_calls"],
                     "reasoning": r["reasoning"], "confidence": r["confidence"], "latency_ms": r["latency_ms"],
                     "score": r["score"], "score_if_suppressed_shipped": r["score_if_shipped"]}
                    for r in scored]}
    path = os.path.join(RESULTS, f"{args.tag}_{args.test}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    o = by["overall"]
    print(f"{path}: correct {o['correct_all_rows']} | pos tool {o['tool_exact_match_positives']} | "
          f"resolver {o['resolver_ok_of_positives']} | neg empty {o['empty_list_accuracy_negatives']} | "
          f"lat med {o['latency_ms_median']} p95 {o['latency_ms_p95']}")


# ------------------------------------------------------------------------------------- summary
def _c(f):
    return "-" if not f or not f["n"] else f"{f['k']}/{f['n']} ({f['pct']:.1f}%)"


def cmd_summary(args):
    data = {}
    for t in TESTS:
        p = os.path.join(RESULTS, f"{args.tag}_{t}.json")
        if os.path.exists(p):
            data[t] = json.load(open(p, encoding="utf-8"))
    L = [f"# Needle evaluation summary: `{args.tag}`", ""]
    any_meta = next(iter(data.values()))["run_meta"] if data else {}
    L += [f"Weights: `{any_meta.get('weights')}`. Engine: `{any_meta.get('engine_lib')}`, cactus-needle "
          f"{any_meta.get('needle_version')}. CPU only, threads not capped (as in N0); "
          f"process threads after warm-up: {any_meta.get('threads_in_process_after_warmup')}. "
          "System text pinned per row (`auto_date=False`, `date: 2026-10-04 Sun 10:00`). "
          "One untimed warm-up query per (tools, system) group. Generated by `evaluate.py summary`; "
          "scoring rules are in the evaluate.py docstring and DECISIONS.md section E.", ""]
    L += ["Run metadata:", ""]
    for t, d in data.items():
        m = d["run_meta"]
        L.append(f"- `{t}`: {m['n_rows']} rows, {m['n_groups']} groups, started {m['started']}, wall "
                 f"{m.get('wall_s', 0):.0f} s, loadavg start {[round(x, 2) for x in m['loadavg_start']]}, "
                 f"end {[round(x, 2) for x in m.get('loadavg_end', [])]}; gold order_ref resolver sanity "
                 f"{_c(d['gold_resolver_sanity'])}")
        for g in d.get("gold_resolver_failures", []):
            L.append(f"  - gold order_ref not resolvable (resolver gap, counts against resolver on that row): "
                     f"{g['row_id']} `{g['gold_order_ref']}` -> {g['resolved']} ({g['rule']}), gold {g['gold_id']}")
    L.append("")

    cols = [("correct_positives", "correct (pos)"), ("tool_exact_match_positives", "tool match (pos)"),
            ("real_args_match_of_tool_matched", "real args (of tool ok)"),
            ("args_match_strict_of_tool_matched", "args incl. order_ref (of tool ok)"),
            ("resolver_ok_of_tool_matched", "resolver (of tool ok)"),
            ("resolver_ok_of_positives", "resolver (all pos)"),
            ("empty_list_accuracy_negatives", "empty list (neg)")]
    L += ["## Main table (test x rendering x language)", "",
          "correct (pos) = tool match and real-argument match and resolver gives the gold ID. "
          "Every cell is k/n.", "",
          "| test | slice | rows | " + " | ".join(c[1] for c in cols) + " | latency med / p95 ms |",
          "|---|---|---|" + "---|" * len(cols) + "---|"]
    for t, d in data.items():
        mets = d["metrics"]
        slices = [("all", mets["overall"])] + list(mets["by_rendering_language"].items())
        for name, a in slices:
            L.append(f"| {t} | {name} | {a['rows']} | " + " | ".join(_c(a[c[0]]) for c in cols)
                     + f" | {a['latency_ms_median']} / {a['latency_ms_p95']} |")
    L.append("")

    L += ["## Order_ref and shipped vs withheld", "",
          "| test | slice | order_ref = gold (gold named) | order_ref omitted when gold omits | shipped (pos) | "
          "withheld only (pos) | no call (pos) | correct if withheld had shipped (pos) | negatives with suppressed only |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, d in data.items():
        mets = d["metrics"]
        for name, a in [("all", mets["overall"])] + list(mets["by_rendering_language"].items()):
            L.append(f"| {t} | {name} | {_c(a['order_ref_match_gold_named_calls'])} | "
                     f"{_c(a['order_ref_omitted_agree_calls'])} | {_c(a['positives_shipped'])} | "
                     f"{_c(a['positives_withheld_only'])} | {_c(a['positives_no_call_at_all'])} | "
                     f"{_c(a['correct_positives_if_suppressed_shipped'])} | "
                     f"{a['negatives_with_suppressed_calls_only']} |")
    L.append("")

    if "test_n0" in data:
        d = data["test_n0"]
        L += ["## test_n0 by kind", "", "| kind | rows | correct (pos) / empty list (neg) | tool match | resolver (all pos) |",
              "|---|---|---|---|---|"]
        for k, a in d["metrics"]["by_kind"].items():
            main = a["correct_positives"] if a["positives"] else a["empty_list_accuracy_negatives"]
            L.append(f"| {k} | {a['rows']} | {_c(main)} | {_c(a['tool_exact_match_positives'])} | "
                     f"{_c(a['resolver_ok_of_positives'])} |")
        L += ["", "Multi-call order (8 gold multi-call rows): " +
              ", ".join(f"{k}: {v}" for k, v in sorted(d["multi_call_order"].items())), ""]

    L += ["## Latency (ms, complete() only, warm-up excluded)", "",
          "| test | n | median | p95 | max | median / p95 excluding first row of each group |", "|---|---|---|---|---|---|"]
    for t, d in data.items():
        a, e = d["metrics"]["overall"], d["latency_extra"]["excluding_first_row_of_each_group"]
        L.append(f"| {t} | {a['latency_n']} | {a['latency_ms_median']} | {a['latency_ms_p95']} | "
                 f"{a['latency_ms_max']} | {e['median']} / {e['p95']} (n={e['n']}) |")
    L.append("")

    L += ["## Per-gold-tool (positives)", ""]
    for t, d in data.items():
        L += [f"### {t}", "", "| gold tool | n | tool match | correct | real args (of tool ok) | resolver (of tool ok) | withheld only |",
              "|---|---|---|---|---|---|---|"]
        for name, a in d["metrics"]["by_gold_tool"].items():
            L.append(f"| {name} | {a['positives']} | {_c(a['tool_exact_match_positives'])} | "
                     f"{_c(a['correct_positives'])} | {_c(a['real_args_match_of_tool_matched'])} | "
                     f"{_c(a['resolver_ok_of_tool_matched'])} | {_c(a['positives_withheld_only'])} |")
        L.append("")

    L += ["## Per-argument (aligned calls of tool-matched rows)", "",
          "ok_rule = the headline rule (phone digits; address/instruction token F1 >= 0.6; other args normalised "
          "equality; order_ref normalised equality / omission agreement). f1_ge_0.6 is shown for the "
          "equality-scored args so the effect of the rule choice is visible.", ""]
    for t, d in data.items():
        L += [f"### {t}", "", "| argument | n | ok_rule | f1_ge_0.6 | literal digits equal | missing | extra | resolver ok |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in d["per_argument"].items():
            L.append(f"| {k} | {v.get('n', 0)} | {v.get('ok_rule', 0)} | {v.get('f1_ge_0.6', '-')} | "
                     f"{v.get('literal_digits_equal', '-')} | {v.get('missing', 0)} | {v.get('extra', 0)} | "
                     f"{v.get('resolver_ok', '-')} |")
        L.append("")

    L += ["## Confusion matrices (row level; rows = gold, columns = predicted function_calls; 'none' = [])", ""]
    for t, d in data.items():
        cm = d["confusion_matrix"]
        preds = sorted({p for c in cm.values() for p in c})
        L += [f"### {t}", "", "| gold \\ pred | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
        for g, c in cm.items():
            L.append(f"| {g} | " + " | ".join(str(c.get(p, "")) for p in preds) + " |")
        L.append("")

    L += ["## Failure groups", ""]
    for t, d in data.items():
        L.append(f"- `{t}`: {len(d['failures'])} failures: " +
                 ", ".join(f"{k} {v}" for k, v in sorted(d["failure_groups"].items(), key=lambda x: -x[1])))
    L += ["", "Every failure is in `results/" + args.tag + "_<test>.json` under `failures`, verbatim: query, gold, "
          "function_calls, suppressed_calls, the model's `reasoning` string and the raw response.", ""]
    if "test_n0" in data:
        L += ["### test_n0 failures (verbatim)", ""]
        for f in data["test_n0"]["failures"]:
            L.append(f"- **{f['row_id']}** ({f['group']}): query `{f['query']}`; gold "
                     f"`{json.dumps(f['gold'], ensure_ascii=False)}`; function_calls "
                     f"`{json.dumps(f['function_calls'], ensure_ascii=False)}`; suppressed "
                     f"`{json.dumps(f['suppressed_calls'], ensure_ascii=False)}`; reasoning `{f['reasoning']}`")
        L.append("")
    path = os.path.join(RESULTS, f"{args.tag}_summary.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "score"):
        p = sub.add_parser(c)
        p.add_argument("--test", required=True, choices=TESTS)
        p.add_argument("--tag", required=True)
        p.add_argument("--weights", default=None, help="built .cact (default: base Needle 3)")
        p.add_argument("--max-new-tokens", type=int, default=512)
    p = sub.add_parser("summary")
    p.add_argument("--tag", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
