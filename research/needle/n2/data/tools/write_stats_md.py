"""Render /root/n2/data/DATA_STATS.md from stats.json, build_stats.json and the ASR logs."""
import json, os, re
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir

S = json.load(open(f"{N2_ROOT}/data/stats.json"))
B = json.load(open(f"{N2_ROOT}/data/build_stats.json"))
log = open(f"{N2_ROOT}/data/asr/asr_run.log").read()
done = re.findall(r"DONE .*", log)
checks = {}
for v, b in (("opus", 16), ("clean", 16)):
    try:
        c = json.load(open(f"{N2_ROOT}/data/asr/check_{v}_b{b}.json"))
        checks[v] = f"{c['same']}/{c['n']}"
    except FileNotFoundError:
        pass
M = lambda k: [json.loads(l) for l in open(f"{N2_ROOT}/data/rows/{k}_meta.jsonl")]  # noqa: E731
partial = {k: sum(m["exact_has_partial_turn"] for m in M(k)) for k in ("test_exact", "val_exact")}

L = []
p = L.append
p("# N2 data stage: DATA_STATS (2026-10-05)\n")
p("Built on runpod2 (RTX 4090, d7e199d0669a). Scripts: `tools/asr_windows.py` (Trelis), `tools/build_rows.py` "
  "(rows), `tools/stats_rows.py` (validation + numbers), `tools/write_stats_md.py` (this file).\n")
p("## Inputs and recipe\n")
p("- Windows: `/root/n2/windows/meta.jsonl` (647 check-line windows; customer channel, 30 s ending at the check-line start).")
p("- ASR: Trelis/whisper-hinglish-preview, router recipe (bf16, `<|hi|><|mixedcode|>`, greedy, max_new_tokens 440), "
  "one clip per window. clean = `windows/clean_16k`, opus = `windows/opus_16k` (= `to16k(opus_24k)` exactly, noise + Opus round trip).")
p(f"- Batch 16. Batched output was checked against batch size 1 (how the router runs): opus {checks.get('opus')} identical, "
  f"clean {checks.get('clean')} identical.")
for d in done:
    p(f"- {d}")
p(f"- Clips that hit the 440-token cap: clean {S['asr']['clean']['hit_cap']}, opus {S['asr']['opus']['hit_cap']}. "
  f"Empty transcripts: clean {S['asr']['clean']['empty']}, opus {S['asr']['opus']['empty']}. Generated tokens "
  f"median/max: clean {S['asr']['clean']['tokens_median']}/{S['asr']['clean']['tokens_max']}, opus "
  f"{S['asr']['opus']['tokens_median']}/{S['asr']['opus']['tokens_max']}.")
p("- Raw transcripts: `asr/clean.jsonl`, `asr/opus.jsonl` (`asr_raw`). Every row meta keeps `raw_text` next to the converted `query`.\n")
p("## Row format (N1 format: query, tools, answers, system)\n")
p("- v2 rows: `query = router_v2.prepare_transcript(raw)` (numconv, then N1 romanise); `system = targets.system_text(call record)` "
  "(the record is in the input, built from calls.jsonl, never from other records.json fields); `tools = tools_v2.tools_for(agent_type)`; "
  "`answers = [targets.target_call(w, record, agent_type)]` (canonical style). Check-line to write via `targets.checkline_writes`, "
  "asserted equal to windows meta on all 647. Leak check passed: no gold new phone or email appears in any `system`.")
p("- exact rows: `query = prepare_transcript(join(customer_turns_in_window.text_roman))` (gold script text, number-converted). "
  f"The first turn is often only partly inside the window and its full text is used, so exact text can hold words the audio lacks: "
  f"{partial['test_exact']}/136 test and {partial['val_exact']}/60 val windows have such a partial turn (`exact_has_partial_turn`).")
p("- N1 rows (`n1_test_*`): 5 old agent types only (bank_card_support and telecom_prepaid_support have no N1 schema), test split, same window ids. "
  "`tools = N1 tools.tools_for`, `system = N1 system_text`, answers from N1 `build_data.build_target` (order_ref extracted the N1 way). "
  "Query style: `n1_test_<v>` = romanise only (N1's native ASR input, `asr_roman`; no numconv) and `n1_test_<v>_nc` = the v2 `prepare_transcript` text, so the converter's effect on N1 can be measured alone. "
  "Score N1 with `schema/score_map.py` against `gold_canonical` in the meta, not against the N1 `answers`.")
p("- Meta sidecars (line-aligned `*_meta.jsonl`): example_id, id, call_id, scenario_id, split, variant, agent_type, record_id, tool, "
  "check_line_ordinal, gold_args_raw, gold_canonical, raw_text, grounded (per NEW arg), phone_email_grounded, ref_mentioned, "
  "arg_coverage, prior_writes (earlier confirmed writes in the call), exact_has_partial_turn.\n")
p("## Files (`rows/`) and token lengths\n")
p("Token length = Needle SentencePiece tokens of render_example(prompt + target) + 2, as N1 validate_datasets.py. "
  "Every file loads through the installed `needle.model.finetune.read_examples` with no skipped lines, and passes the N1 asserts "
  "(answer tool in tools, args are properties, required args present, no empty values).\n")
p("| file | rows | windows | scenarios | tok median | tok p95 | tok max | > 1024 |")
p("|---|---|---|---|---|---|---|---|")
for k, v in S["files"].items():
    p(f"| {k} | {v['rows']} | {v['windows']} | {v['scenarios']} | {v['tok_median']} | {v['tok_p95']} | {v['tok_max']} | {v['over_1024']} |")
p("\n- train = clean + opus of every train window (451 x 2). train_grounded = train minus rows whose gold phone or email is not literally recoverable from that row's query (see below).")
p("- val = val_clean + val_opus. Test is per variant: test_clean, test_opus, test_exact.")
p(f"- Scenarios: train {S['scenarios']['train']}, val {S['scenarios']['val']}, test {S['scenarios']['test_opus']}; no overlap (asserted).")
p("- All v2 rows fit in 1024 tokens (max 806), so seq_len 1024 like N1. One N1 exact row is 1028 tokens (N1 eval only, not training).\n")
p("## Windows per agent type and split\n")
p("| agent type | train | val | test |")
p("|---|---|---|---|")
for a, c in sorted(S["windows_by_split_agent"].items()):
    p(f"| {a} | {c.get('train', 0)} | {c.get('val', 0)} | {c.get('test', 0)} |")
p("\n## Windows per tool and split\n")
p("| tool | train | val | test |")
p("|---|---|---|---|")
for a, c in sorted(S["windows_by_split_tool"].items(), key=lambda x: -sum(x[1].values())):
    p(f"| {a} | {c.get('train', 0)} | {c.get('val', 0)} | {c.get('test', 0)} |")
p("\n## Gold NEW-value args present in the converted query\n")
p("`targets.grounded` on the query: phone = its 10 digits appear (spaces ignored); email = the local part appears (alnum only, with the spoken word `dot` removed; the data stage added that because `targets.grounded` alone counted `pandey dot work 60` as a miss); "
  "address/location/instruction/reason = at least 60% of the value's tokens appear. Phone and email are literal checks, text args are fuzzy, "
  "so the groups are not directly comparable. Exact covers val + test only.\n")
p("| scope | variant | phone | email | address/location | instruction/reason |")
p("|---|---|---|---|---|---|")
G = S["grounded"]
for sc in ("all", "train+val", "test"):
    for v in ("clean", "opus", "exact"):
        cells = []
        for g in ("phone", "email", "address/location", "instruction/reason"):
            x = G.get(f"{sc}|{v}|{g}")
            cells.append(f"{x[0]}/{x[1]} ({100 * x[0] / x[1]:.0f}%)" if x and x[1] else "-")
        if any(c != "-" for c in cells):
            p(f"| {sc} | {v} | " + " | ".join(cells) + " |")
pe = S["phone_email_grounded"]
p("\nRows whose phone/email is grounded: " + "; ".join(
    f"{k} {v['grounded']}/{v['rows_with_phone_or_email']}" for k, v in pe.items()) + ".\n")
p("What the misses are (read by hand on train rows):")
p("- Phone: Trelis drops or merges digits (9-digit or 8-digit runs such as `730010662` for 73030 10662, `99509461` for 99580 94461), "
  "or mishears one group (`98730 safe 893`). Clean and opus give the same phone count (104/126), so the misses come from Trelis on the TTS audio, not from the codec.")
p("- Email: the name part of the local part is spoken as Devanagari and romanised back with a different spelling (`rawark 972 at hotmail` for raowork972@hotmail.com, "
  "Devanagari `बनर्जी forty two eight yahoo` for banerjee42@yahoo.co.in). numconv recovers the digits and the `at`, but the spelling of the name part cannot be recovered from the text. "
  "All 4 test email rows are ungrounded in both clean and opus (one customer, banerjee42).")
p("- REF args (IDs, card last 4, plans, packs, services) are canonical record values in every target, whether or not the customer said them in the window. "
  "How often the canonical value (or an ID's digit block) is literally in the query, clean: " + ", ".join(
      f"{k.split('|')[2]} {v[0]}/{v[1]}" for k, v in S["ref_mentioned"].items() if k.startswith("all|clean|")) + ".\n")
p("## Trelis WER-ish vs gold customer text\n")
p("Token edit distance between prepare_transcript(ASR) and prepare_transcript(gold script text of the window), lower-case alnum tokens. "
  "It overstates real errors: romanisation spelling variants (bahut/bohot) count as errors, and in windows with a partial first turn the "
  "reference holds words that are outside the audio. The no_partial_turn rows are the fairer number.\n")
p("| variant | scope | windows | WER |")
p("|---|---|---|---|")
for k, v in S["wer"].items():
    var, sc, sub = k.split("|")
    p(f"| {var} | {sc} | {sub} | {v['wer']:.3f} ({v['edits']}/{v['ref_words']}) |")
p("\n## Notes and open items\n")
p(f"- `prepare_transcript` is not idempotent on {len(B['nonidempotent'])} queries (air_20_g3__cl2 clean+opus: `172 8` becomes `1728` on a second pass; "
  "ecom_23_g1__cl0 clean: `three` becomes `3`). Training and the router both apply it once, so they match; `grounded()` runs numconv again, which can differ on these 3 only.")
p(f"- {S['prior_writes_windows']} windows (check_line_ordinal >= 1) follow an earlier confirmed write in the same call (REVIEW-windows-1). "
  "The input does not list done writes, matching router_v2; `prior_writes` is in the meta for eval and for that decision.")
p("- Open decision for the training stage: train.jsonl (all rows, gold targets even when ASR lost a digit) vs train_grounded.jsonl "
  "(drops " + str(S['files']['train']['rows'] - S['files']['train_grounded']['rows']) + " rows with an ungrounded phone/email). Training a 10-digit gold phone on a 9-digit transcript teaches the model to invent a digit, "
  "and the router only asks when a phone is not 10 digits, so an invented digit would become a silent wrong write.")
p("- Phone grounding check: `targets.grounded` joins all digits of the query, so a dropped digit could in principle be filled by a neighbouring number. "
  "Checked: all 240 grounded phone args (all v2 files) also sit inside one contiguous digit run of the query, so there is no over-count.")
p("- N1 baseline contamination: 0/104 `n1_test_*` system strings occur in N1 train.jsonl/val.jsonl, and 0/16 test scenario ids are N1 train/val scenarios, so the N1 numbers on these rows are not inflated by record overlap.")
open(f"{N2_ROOT}/data/DATA_STATS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
