"""D2 / V4 post-generation report (CPU, stdlib). Reads data/V4/{calls.jsonl, gen_log.jsonl, gen_rounds.json,
gen_missing.json, failed_calls.jsonl, holdout.json}; writes data/V4/{gen_report.json, SAMPLES.md (5 per length band,
picked by sha256 of call_id), dropped_call_ids.json}. Does not touch data/dropped_call_ids.json.
  python3 gen/v4_post.py [--no-write]"""
import collections, hashlib, json, re, sys, datetime
H = __import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish"); D = f"{H}/data/V4"
W = "--no-write" not in sys.argv
rd = lambda p: [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
calls = rd(f"{D}/calls.jsonl"); log = rd(f"{D}/gen_log.jsonl")
rounds = json.load(open(f"{D}/gen_rounds.json")); missing = json.load(open(f"{D}/gen_missing.json"))
hold = json.load(open(f"{D}/holdout.json"))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
_REPO = __import__('pathlib').Path(__file__).resolve().parents[3]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('HINGLISH_TEXT_PKG') or str(_REPO / 'packages/hinglish_text'))
import v4, generate
ids_all = list(generate.default_ids("V4", generate.inputs_for("V4", records=False)[0]))
band = {c: v4.length_band(c) for c in ids_all}
typ = lambda cid: cid.rsplit("_", 2)[0]
# round of each log line
lr = []
for r in rounds:
    lr += [(r["round"], r["seed_base"])] * (r["log_lines"][1] - r["log_lines"][0])
assert len(lr) == len(log), (len(lr), len(log))
rep = {"n_target": len(ids_all), "n_accepted": len(calls), "n_dropped": len(missing),
       "target_by_band": collections.Counter(band.values())}
cb = collections.Counter(c["length_band"] for c in calls)
rep["accepted_by_band"] = dict(cb)
rep["accepted_by_type"] = dict(collections.Counter(typ(c["call_id"]) for c in calls))
rep["accepted_by_type_band"] = {t: dict(collections.Counter(c["length_band"] for c in calls if typ(c["call_id"]) == t))
                                for t in sorted(rep["accepted_by_type"])}
assert all(c["length_band"] == band[c["call_id"]] for c in calls)
# acceptance round/attempt per accepted call
acc = {}
for (r, sb), e in zip(lr, log):
    if e["pass"] and e["call_id"] not in acc:
        acc[e["call_id"]] = (r, e["attempt"])
rep["accepted_round"] = dict(collections.Counter(f"round{acc[c['call_id']][0]}" for c in calls))
rep["accepted_round_attempt"] = dict(sorted(collections.Counter(f"r{acc[c['call_id']][0]}a{acc[c['call_id']][1]}" for c in calls).items()))
rep["first_try_by_band"] = {b: sum(1 for c in calls if c["length_band"] == b and acc[c["call_id"]] == (0, 1)) for b in cb}
rep["candidates_total"] = len(log); rep["candidates_by_round"] = dict(collections.Counter(f"round{r}" for r, _ in lr))
chk = lambda f: f.split(":", 1)[0].strip()
fail = collections.Counter(); fail_band = collections.defaultdict(collections.Counter); fail_first = collections.Counter()
for (r, sb), e in zip(lr, log):
    if e["pass"]:
        continue
    for k in {chk(f) for f in e["failures"]}:
        fail[k] += 1; fail_band[band[e["call_id"]]][k] += 1
        if r == 0 and e["attempt"] == 1:
            fail_first[k] += 1
rep["failed_candidates"] = sum(1 for e in log if not e["pass"])
rep["fail_by_check_all_candidates"] = dict(fail.most_common())
rep["fail_by_check_round0_attempt1"] = dict(fail_first.most_common())
rep["fail_by_check_per_band"] = {b: dict(v.most_common()) for b, v in fail_band.items()}
nr = [(e["call_id"], f) for e in log if not e["pass"] for f in e["failures"] if chk(f) == "numbers_regex"]
rep["numbers_regex_bank"] = sum(1 for c, _ in nr if c.startswith("bank_"))
rep["numbers_regex_other"] = sum(1 for c, _ in nr if not c.startswith("bank_"))
rep["numbers_regex_bank_mentions_4"] = sum(1 for c, f in nr if c.startswith("bank_") and re.search(r"\b4\b", f))
rep["numbers_regex_examples"] = [f"{c}: {f[:160]}" for c, f in nr[:8]]
# shape stats per band
def agw(c, spk):
    return [len(t["text_roman"].split()) for t in c["turns"] if t["speaker"] == spk]
st = {}
for b in cb:
    cs = [c for c in calls if c["length_band"] == b]
    est = [c["est_duration_s"] for c in cs]; wds = [sum(len(t["text_roman"].split()) for t in c["turns"]) for c in cs]
    aw = [w for c in cs for w in agw(c, "agent")]; cw = [w for c in cs for w in agw(c, "customer")]
    nt = [len(c["turns"]) for c in cs]
    st[b] = {"n": len(cs), "est_s_mean": round(sum(est) / len(est), 1), "est_s_min": min(est), "est_s_max": max(est),
             "words_mean": round(sum(wds) / len(wds), 1), "words_per_s_est": round(sum(wds) / sum(est), 2),
             "turns_mean": round(sum(nt) / len(nt), 1), "turns_hist": dict(sorted(collections.Counter(nt).items())),
             "agent_words_mean": round(sum(aw) / len(aw), 1), "cust_words_mean": round(sum(cw) / len(cw), 1),
             "est_over_100s": sum(1 for x in est if x > 100)}
rep["shape_per_band"] = st
rep["interrupt_calls"] = sum(1 for c in calls if any(t["truncated"] for t in c["turns"]))
rep["greeting_templates"] = dict(collections.Counter(c["greeting_template"] for c in calls))
rep["signoff_templates"] = dict(collections.Counter(c["signoff_template"] for c in calls))
# dropped
tc, vc = set(hold["test_calls"]), set(hold["val_calls"])
dr = []
for cid in missing:
    fs = collections.Counter(chk(f) for e in log if e["call_id"] == cid and not e["pass"] for f in set(e["failures"]))
    n = sum(1 for e in log if e["call_id"] == cid)
    split = "TEST" if cid in tc else "VAL" if cid in vc else "train" if cid.rsplit("_", 1)[0] in hold.get("train_scenarios", []) else "val/test-scenario"
    dr.append({"call_id": cid, "length_band": band[cid], "split": split, "candidates": n, "fail_by_check": dict(fs.most_common())})
rep["dropped"] = dr
rep["dropped_test_calls"] = [d["call_id"] for d in dr if d["split"] == "TEST"]
rep["dropped_val_calls"] = [d["call_id"] for d in dr if d["split"] == "VAL"]
rep["test_calls_present"] = sum(1 for c in calls if c["call_id"] in tc); rep["val_calls_present"] = sum(1 for c in calls if c["call_id"] in vc)
print(json.dumps(rep, indent=1, default=str))
if not W:
    sys.exit()
json.dump(rep, open(f"{D}/gen_report.json", "w"), indent=1, default=str)
now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
json.dump({"note": "D2 / V4 only: call_ids that failed every generation round (gen_loop V4, seed_base 0,101..606, top-up cap "
                   "6 rounds / 120 min). Separate from data/dropped_call_ids.json (V1/V3).",
           "dropped": [dict(d, dropped_by="V4", utc=now,
                            reason=f"failed all rounds ({d['candidates']} candidates); dominant failures " +
                                   ", ".join(f"{k} {v}" for k, v in list(d["fail_by_check"].items())[:3])) for d in dr],
           "ids": [d["call_id"] for d in dr]}, open(f"{D}/dropped_call_ids.json", "w"), indent=1)
hk = lambda s: hashlib.sha256(("v4_samples:" + s).encode()).hexdigest()
out = [f"# SAMPLES — V4 (5 calls per length band, picked by sha256 of call_id, of {len(calls)} accepted calls)\n"]
for b in ("standard", "long", "mixed"):
    cs = sorted((c for c in calls if c["length_band"] == b), key=lambda c: hk(c["call_id"]))[:5]
    out.append(f"\n# Band: {b} ({cb[b]} calls)\n")
    for c in sorted(cs, key=lambda c: c["call_id"]):
        out.append(f"## {c['call_id']}  [{b}]  (agent {c['agent_name']} {c['agent_gender']} / customer {c['customer_gender']}; "
                   f"Hindi share {c['hindi_token_share']}; est {c.get('est_duration_s')} s; {len(c['turns'])} turns; "
                   f"greeting t{c['greeting_template']} / sign-off t{c['signoff_template']})\n")
        out.append(f"**role_prompt:** {c['role_prompt']}\n")
        out.append("| # | spk | words | text_roman | text_tts | pause | trunc | overlap | tags |\n|---|---|---|---|---|---|---|---|---|")
        for i, t in enumerate(c["turns"]):
            out.append(f"| {i} | {t['speaker']} | {len(t['text_roman'].split())} | {t['text_roman']} | {t['text_tts']} | {t['pause_after_s']} | "
                       f"{'Y' if t['truncated'] else ''} | {t['overlap_offset_s'] or ''} | {','.join(t['tags'])} |")
        out.append(f"\n**writes:** `{json.dumps(c['writes'], ensure_ascii=False)}`\n")
open(f"{D}/SAMPLES.md", "w", encoding="utf-8").write("\n".join(out))
print("wrote gen_report.json, dropped_call_ids.json, SAMPLES.md")
