"""D2 spec section 8: V4 test-set scoring for one or more tags, everything split by length_band. CPU only.

  python3 v4_eval.py TAG [TAG ...] [--md /workspace/hinglish/tests/V4_EVAL.md]
Per tag: runs score.py's own main() first (runs.csv / scores.csv / streams.txt / gate0_monologues.txt, unchanged
metrics m1-m5, m2 from <run>.judge.json; m6 = large-v3 is not run for V4, so it stays empty), then for every
<tag>/V4/<call>_s<seed> run adds:
 cer_*          Trelis Whisper-Hinglish CER/WER, CER-study method 1 (roman vs roman): hyp = <run>.trelis.json text
                (silence-trimmed audio, trelis_pass.py) with every Devanagari token romanised by trelis_m1.romanise
                (lexicon Devanagari->roman = most frequent text_roman spelling of that word in data/V4/work/plan.json
                text_tts/text_roman word pairs; OOV -> tts_norm.deva_to_latin, a fixed character table); ref = the
                model's own text stream (tcommon.text_from_tokens, as in cer_study). Both sides -> trelis_m1.m1_norm
                (tts_backends.en_text: digits/IDs -> English number words; lowercase; [^a-z0-9'] -> space; collapse;
                runs of >= 2 single letters joined). CER = Levenshtein chars (spaces count) / ref chars; WER on words.
 echo_*         argument echo per scripted write: the write's echo argument (common.ECHO_ARG[tool], the same
                argument the data's confirm lines had to echo) is checked with gen/validate.echo_ok (the data's
                write_echo rule: phone/long numbers = digit string contained; order/ride/reference IDs and PNR =
                contained ignoring spaces; email = local-part tokens + domain name; other values = >= 60% content-
                token overlap and every digit token present) against the model text in the confirmation slot = score.py's m3 window
                [end of the last customer turn before the write's check-line - 0.3 s, start of the first customer turn
                after the confirm turn] (segments by start time); passes if it holds on the raw text or on the
                text with spoken numbers turned into digits (score.normalise_numbers), or, for one-token values with
                digits, if the value is a contiguous run of whole model tokens ('four six five one'). echo_script_rate
                = the same rule on the scripted confirm turn (should be 1.0 by construction).
 tc_* / fc_*    Gemma call judge (<run>.call.json, judge.py --calljudge): task completion COMPLETE/PARTIAL/FAILED
                (tc_score 1/0.5/0), fact consistency CONSISTENT/MINOR/CONTRADICTS, per-write confirmed/value_correct.
 nat_*          Gemma four-way naturalness per line (<run>.nat.json, judge.py --naturalness, rerank rubric).
Aggregation, per band in (all, standard, long, mixed): per-run numbers -> per seed mean over calls -> mean and
population std over seeds (score.py convention); plus POOLED rates over all seeds (lines, writes, CER chars).
Outputs: out/<tag>/v4_runs.jsonl, v4_runs.csv, v4_scores.json, v4_scores.csv; --md writes the comparison.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[2]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_GEN_DIR') or str(_REPO / 'research/data_gen/gen'))
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_AUDIO_DIR') or str(_REPO / 'research/audio'))
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
import score as S  # noqa: E402
import validate as VAL  # noqa: E402  (gen/validate.py: echo_ok = the data's write_echo rule)
from common import ECHO_ARG  # noqa: E402
import trelis_m1  # noqa: E402
import tts_norm  # noqa: E402
from tcommon import (DATA, FRAME_RATE, INPUTS, OUT, jdump, jload, load_calls, segments, test_meta_path,  # noqa: E402
                     text_from_tokens)

V = "V4"
BANDS = ["all", "standard", "long", "mixed"]
DROP = {"at", "dot", "underscore", "dash", "hyphen"}
NAT = ["NATURAL", "STIFF", "PEPPERED", "NONSENSE"]


def ntoks(text, drop=True):
    toks, _ = S.normalise_numbers(text)
    return [t for t in toks if not (drop and t in DROP)]


def echo_write(tool, val, text):
    """validate.py's write_echo rule (gen/validate.echo_ok, used to build the data) on the raw model text, or on the
    text with spoken numbers turned into digits; for values with digits also the score.has_fact rule (a value read
    in pieces, e.g. 'four six five one')."""
    arg = ECHO_ARG[tool]
    norm = " ".join(S.normalise_numbers(text)[0])
    if VAL.echo_ok(tool, arg, val, text) or VAL.echo_ok(tool, arg, val, norm):
        return True
    key = S.compress(ntoks(val))
    return bool(any(ch.isdigit() for ch in val) and " " not in val.strip() and key
                and S.has_fact(key, ntoks(text), max_span=16))


def echo(tm, tokens, skipped):
    turns = tm["turns"]
    segs = segments(tokens)
    out = []
    for w in tm.get("writes") or []:
        ci = w.get("confirm_turn_idx")
        arg = ECHO_ARG.get(w["tool"])
        val = (w.get("args") or {}).get(arg)
        if ci is None or ci >= len(turns) or not val:
            continue
        chk = [t for t in turns[:ci] if "check_line" in t["tags"]]
        anchor = chk[-1]["idx"] if chk else ci
        prev_c = [t for t in turns[:anchor] if t["speaker"] == "customer"]
        t0 = (prev_c[-1]["end"] if prev_c else 0.0) - 0.3  # = score.py m3 window start (the confirmation slot)
        nxt = [t for t in turns[ci + 1:] if t["speaker"] == "customer"]
        t1 = nxt[0]["start"] if nxt else 1e9
        win = " ".join(s["text"] for s in segs if t0 <= (s["start_f"] + skipped) / FRAME_RATE <= t1)
        out.append({"tool": w["tool"], "arg": arg, "value": val, "echo": echo_write(w["tool"], val, win),
                    "script_echo": echo_write(w["tool"], val, turns[ci]["text_roman"]),
                    "window_s": [round(t0, 2), round(t1, 2) if t1 < 1e8 else None], "window_text": win[:300]})
    return out


def lexicon():
    plan = jload(DATA / V / "work" / "plan.json")
    plan = plan["chunks"] if isinstance(plan, dict) and "chunks" in plan else plan
    return trelis_m1.build_lexicon(plan)


def run_extra(tm, call, rmeta, tokens, trel, nat, cj, lex):
    r = {"band": call["length_band"]}
    text = text_from_tokens(tokens)
    if trel is not None:
        hyp_r, nd, oov = trelis_m1.romanise(trel.get("text", ""), lex)
        ref, hyp = trelis_m1.m1_norm(text), trelis_m1.m1_norm(hyp_r)
        ed = tts_norm.levenshtein(ref, hyp)
        wed = tts_norm.levenshtein(ref.split(), hyp.split())
        r.update({"cer_m1": round(ed / len(ref), 4) if ref else None,
                  "wer_m1": round(wed / len(ref.split()), 4) if ref else None,
                  "cer_edits": ed, "cer_ref_chars": len(ref), "wer_edits": wed, "wer_ref_words": len(ref.split()),
                  "cer_hyp_deva_tokens": nd, "cer_hyp_oov_tokens": oov, "cer_trimmed_s": trel.get("trimmed_s"),
                  "cer_ref_m1": ref, "cer_hyp_m1": hyp})
    e = echo(tm, tokens, rmeta.get("skipped_steps", 0))
    ch = [x for x in e if x["echo"] is not None]
    r.update({"echo_n": len(ch), "echo_ok": sum(x["echo"] for x in ch), "echo_script_ok": sum(x["script_echo"] for x in ch),
              "echo_rate": round(sum(x["echo"] for x in ch) / len(ch), 4) if ch else None,
              "echo_script_rate": round(sum(x["script_echo"] for x in ch) / len(ch), 4) if ch else None,
              "echo_detail": e})
    if nat is not None and "utterances" in nat:
        n = len(nat["utterances"])
        cnt = {k: sum(u["label"] == k for u in nat["utterances"]) for k in NAT}
        r.update({"nat_n": n, **{f"nat_{k.lower()}_n": cnt[k] for k in NAT},
                  "nat_trunc_n": sum(u.get("trunc", False) for u in nat["utterances"]),
                  "nat_natural_en_n": sum(u["label"] == "NATURAL" and u.get("en", False) for u in nat["utterances"]),
                  "nat_natural_hi_n": sum(u["label"] == "NATURAL" and not u.get("en", False) for u in nat["utterances"]),
                  "nat_hindi_content_n": sum(bool(u.get("hindi_content")) for u in nat["utterances"]),
                  **{f"nat_{k.lower()}": (round(cnt[k] / n, 4) if n else None) for k in NAT}})
    elif nat is not None:
        r["nat_error"] = True
    if cj is not None and "task_completion" in cj:
        tc, fc = cj["task_completion"], cj["fact_consistency"]
        ws = cj.get("writes") or []
        r.update({"tc_label": tc, "tc_complete": int(tc == "COMPLETE"), "tc_failed": int(tc == "FAILED"),
                  "tc_score": {"COMPLETE": 1.0, "PARTIAL": 0.5, "FAILED": 0.0}[tc],
                  "fc_label": fc, "fc_consistent": int(fc == "CONSISTENT"), "fc_contradicts": int(fc == "CONTRADICTS"),
                  "fc_n_wrong": len(cj.get("wrong_facts") or []), "fc_wrong_facts": cj.get("wrong_facts") or [],
                  "cj_writes_n": len(ws), "cj_writes_confirmed": sum(bool(w["confirmed"]) for w in ws),
                  "cj_writes_value_ok": sum(bool(w["confirmed"]) and bool(w["value_correct"]) for w in ws),
                  "cj_reason": cj.get("reason", "")})
        if ws:
            r["cj_confirm_rate"] = round(r["cj_writes_confirmed"] / len(ws), 4)
            r["cj_confirm_value_rate"] = round(r["cj_writes_value_ok"] / len(ws), 4)
    elif cj is not None:
        r["cj_error"] = True
    return r


SEED_KEYS = ["m1_hindi_recall", "m1_model_hindi_share", "m2_vf_frac", "m2_judge_r2", "m3_check_rate",
             "m3_check_sil_rate", "m4_read_rate", "m4_value_rate", "m4_flag", "m4_invented", "m5_greeting",
             "m5_degenerate", "cer_m1", "wer_m1", "echo_rate", "echo_script_rate", "nat_natural", "nat_stiff",
             "nat_peppered", "nat_nonsense", "tc_score", "tc_complete", "tc_failed", "fc_consistent",
             "fc_contradicts", "fc_n_wrong", "cj_confirm_rate", "cj_confirm_value_rate", "n_words"]
POOLED = {  # name: (numerator key, denominator key)
    "cer_m1_pooled": ("cer_edits", "cer_ref_chars"), "wer_m1_pooled": ("wer_edits", "wer_ref_words"),
    "echo_pooled": ("echo_ok", "echo_n"), "echo_script_pooled": ("echo_script_ok", "echo_n"),
    "nat_natural_pooled": ("nat_natural_n", "nat_n"), "nat_stiff_pooled": ("nat_stiff_n", "nat_n"),
    "nat_peppered_pooled": ("nat_peppered_n", "nat_n"), "nat_nonsense_pooled": ("nat_nonsense_n", "nat_n"),
    "nat_trunc_pooled": ("nat_trunc_n", "nat_n"),
    "nat_natural_en_pooled": ("nat_natural_en_n", "nat_n"), "nat_natural_hi_pooled": ("nat_natural_hi_n", "nat_n"),
    "nat_hindi_content_pooled": ("nat_hindi_content_n", "nat_n"), "m3_check_pooled": ("m3_checks", "m3_n_writes"),
    "cj_confirm_pooled": ("cj_writes_confirmed", "cj_writes_n"),
    "cj_confirm_value_pooled": ("cj_writes_value_ok", "cj_writes_n")}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(rows):
    out = {"n_runs": len(rows), "n_calls": len({r["call_id"] for r in rows}), "seeds": sorted({r["seed"] for r in rows})}
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    for k in SEED_KEYS:
        ps = [x for x in (mean([r.get(k) for r in rs]) for rs in by_seed.values()) if x is not None]
        if ps:
            m = sum(ps) / len(ps)
            out[k] = {"mean": round(m, 4), "std": round(math.sqrt(sum((x - m) ** 2 for x in ps) / len(ps)), 4),
                      "n_seeds": len(ps)}
    for name, (a, b) in POOLED.items():
        den = sum(r.get(b) or 0 for r in rows)
        if den:
            out[name] = {"rate": round(sum(r.get(a) or 0 for r in rows) / den, 4), "den": den}
    for lab, key in (("tc", "tc_label"), ("fc", "fc_label")):
        c = defaultdict(int)
        for r in rows:
            if r.get(key):
                c[r[key]] += 1
        if c:
            out[f"{lab}_counts"] = dict(c)
    return out


def score_tag(tag, calls, lex):
    old_argv = sys.argv
    sys.argv = ["score.py", tag]
    try:
        S.main()  # standard score.py outputs for the tag (new tag dirs only)
    finally:
        sys.argv = old_argv
    base = {(r["call_id"], r["seed"]): r for r in map(json.loads, open(OUT / tag / "runs.jsonl", encoding="utf-8"))
            if r["variant"] == V}
    rows, missing = [], defaultdict(int)
    for (cid, seed), b in sorted(base.items()):
        stem = OUT / tag / V / f"{cid}_s{seed}"
        tokens, rmeta = jload(f"{stem}.json"), jload(f"{stem}.meta.json")
        tm = jload(test_meta_path(V, cid, f"{stem}.meta.json"))  # vad-mode runs: actual turn times
        side = {}
        for suf in ("trelis", "nat", "call"):
            p = Path(f"{stem}.{suf}.json")
            side[suf] = jload(p) if p.exists() else None
            missing[suf] += side[suf] is None
        r = {**{k: v for k, v in b.items() if k not in ("m3_detail", "m4_detail", "text")},
             "m3_checks": round((b.get("m3_check_rate") or 0) * (b.get("m3_n_writes") or 0)),
             **run_extra(tm, calls[cid], rmeta, tokens, side["trelis"], side["nat"], side["call"], lex)}
        rows.append(r)
    tdir = OUT / tag
    with open(tdir / "v4_runs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["tag", "call_id", "seed", "band"] + SEED_KEYS + ["tc_label", "fc_label", "echo_n", "echo_ok", "nat_n",
                                                             "cer_ref_chars", "cer_edits", "cer_trimmed_s"]
    with open(tdir / "v4_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # only COMPLETE seeds (all test calls present) are aggregated: V4_A2 has seed 1001 only (user, 2026-10-05) plus
    # 7 leftover s1002/s1003 runs from the stopped 3-seed run, which are kept on disk but excluded here.
    n_calls = len({c for c, _ in base})
    per_seed = defaultdict(set)
    for r in rows:
        per_seed[r["seed"]].add(r["call_id"])
    complete = sorted(sd for sd, cs in per_seed.items() if len(cs) == n_calls)
    res = {"tag": tag, "variant": V, "missing_side_files": dict(missing), "n_test_calls": n_calls,
           "complete_seeds": complete,
           "incomplete_seed_runs_excluded": {str(sd): len(cs) for sd, cs in per_seed.items() if sd not in complete},
           "aggregation": "SEED_KEYS: per seed mean over calls, then mean and population std over seeds; *_pooled: "
                          "sum over all runs of the band. bands = all complete seeds; bands_s1001 = seed 1001 only",
           "bands": {}, "bands_s1001": {}}
    for key, seeds in (("bands", complete), ("bands_s1001", [1001] if 1001 in complete else [])):
        for band in BANDS:
            rs = [r for r in rows if r["seed"] in seeds and (band == "all" or r["band"] == band)]
            if rs:
                res[key][band] = agg(rs)
    jdump(tdir / "v4_scores.json", res)
    with open(tdir / "v4_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "band", "metric", "mean_or_rate", "std", "n_seeds_or_den", "seeds"])
        for key, lab in (("bands", "complete_seeds_" + "+".join(map(str, complete))), ("bands_s1001", "s1001")):
            for band, d in res[key].items():
                for k, v in d.items():
                    if isinstance(v, dict) and "mean" in v:
                        w.writerow([tag, band, k, v["mean"], v["std"], v["n_seeds"], lab])
                    elif isinstance(v, dict) and "rate" in v:
                        w.writerow([tag, band, k, v["rate"], "", v["den"], lab])
    print(f"[v4_eval] {tag}: {len(rows)} runs, complete seeds {complete}, excluded "
          f"{res['incomplete_seed_runs_excluded']}, missing side files {dict(missing)}", flush=True)
    return res


MD_COLS = [("NATURAL", "nat_natural_pooled"), ("- of which Hinglish (en=false)", "nat_natural_hi_pooled"),
           ("- of which pure English (en=true)", "nat_natural_en_pooled"), ("hindi_content lines", "nat_hindi_content_pooled"),
           ("NONSENSE", "nat_nonsense_pooled"), ("STIFF", "nat_stiff_pooled"),
           ("PEPPERED", "nat_peppered_pooled"), ("task score", "tc_score"), ("COMPLETE", "tc_complete"),
           ("FAILED", "tc_failed"), ("facts CONSISTENT", "fc_consistent"), ("facts CONTRADICTS", "fc_contradicts"),
           ("echo", "echo_pooled"), ("judge confirm+value", "cj_confirm_value_pooled"),
           ("check-line", "m3_check_rate"), ("read rate", "m4_read_rate"), ("value rate", "m4_value_rate"),
           ("invented flag", "m4_flag"), ("greeting", "m5_greeting"), ("degenerate", "m5_degenerate"),
           ("Hindi share", "m1_model_hindi_share"), ("r2", "m2_judge_r2"), ("Trelis CER m1", "cer_m1_pooled"),
           ("Trelis WER m1", "wer_m1_pooled")]


def fmt(v):
    if v is None:
        return "-"
    if "rate" in v:
        return f"{v['rate']:.3f}"
    return f"{v['mean']:.3f} ± {v['std']:.3f}"


def write_md(results, path):
    L = ["# V4 test set (D2 spec section 8)", "",
         "Generated by tests/v4_eval.py. 30 V4 holdout test calls. Section A compares all four tags on seed 1001 "
         "(user 2026-10-05: single seed; V4_A2 was run with seed 1001 only). Section B adds the 3-seed numbers "
         "(1001-1003) for the tags that have them. Cells `mean ± std`: per seed mean over calls, then mean and "
         "population std over seeds (std is 0 with one seed); plain numbers: pooled over all runs of the band "
         "(lines for naturalness, writes for echo / judge confirmations, characters/words for CER). Definitions: "
         "tests/v4_eval.py and tests/score.py docstrings; NOTES.md 'D2 / V4 section 8'.", ""]
    for sec, key, title in (("A", "bands_s1001", "seed 1001, all tags"),
                            ("B", "bands", "all complete seeds (3 seeds where available)")):
        L += [f"# {sec}. {title}", ""]
        for band in BANDS:
            rs = [(t, r[key].get(band), r["complete_seeds"] if key == "bands" else [1001])
                  for t, r in results.items() if r[key].get(band)]
            if sec == "B":
                rs = [x for x in rs if len(x[2]) > 1]
            if not rs:
                continue
            L += [f"## {sec}: band {band} ({rs[0][1]['n_calls']} calls)", "",
                  "| metric | " + " | ".join(f"{t} (n={d['n_runs']})" for t, d, _ in rs) + " |",
                  "|---|" + "---|" * len(rs)]
            for lab, k in MD_COLS:
                L.append(f"| {lab} | " + " | ".join(fmt(d.get(k)) for _, d, _ in rs) + " |")
            L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[v4_eval] wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args()
    calls = load_calls(DATA / V / "calls.jsonl")
    lex = lexicon()
    print(f"[v4_eval] lexicon {len(lex)} Devanagari words", flush=True)
    results = {t: score_tag(t, calls, lex) for t in a.tags}
    if a.md:
        write_md(results, a.md)


if __name__ == "__main__":
    main()
