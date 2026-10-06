"""One caller record per scenario (72), shared across g1-g4 and V1/V2/V3 -> /workspace/hinglish/data/records.json

Deterministic: brand, agent names (f/m), customer surname + first names (m/f), phones, IDs, emails.
Gemma: the scenario-specific static-read facts, distractors, Information prose, and the values the caller will dictate.
Run (GPU, flock'd):  python records.py            (or imported by generate.py --with-records)
"""
import argparse
import glob
import json
import re

from common import (AGENT_PLAIN, BRANDS, CUSTOMER_FIRST_F, CUSTOMER_FIRST_M, CUSTOMER_LAST, DATA, ID_PREFIX,
                    PAIRINGS, TOOL_ARGS, extract_json, jdump, load_scenarios)

SPM = glob.glob("/workspace/hf/hub/models--nvidia--personaplex-7b-v1/snapshots/*/tokenizer_spm_32k_3.model")
_sp = None


def n_tokens(text):
    """Token count with the PersonaPlex text tokenizer (SentencePiece 32k)."""
    global _sp
    if _sp is None:
        import sentencepiece as spm
        _sp = spm.SentencePieceProcessor(model_file=SPM[0])
    return len(_sp.encode(text))


LETTERS = "BCDFHJKLMNPQRSTVWXZ"


def det_fields(scen):
    """Deterministic identity fields per scenario."""
    out = {}
    per_type = {}
    for i, (sid, s) in enumerate(scen.items()):
        k = per_type.get(s["agent_type"], 0)
        per_type[s["agent_type"]] = k + 1
        brand, an_f, an_m = BRANDS[s["agent_type"]][k % 3]
        last = CUSTOMER_LAST[i % len(CUSTOMER_LAST)]
        d4 = 1000 + (i * 113) % 9000
        d4b = 1000 + ((i + 500) * 113) % 9000
        if s["agent_type"] == "airport_ticket_counter":
            pre = LETTERS[(i * 7) % 19] + LETTERS[(i * 11 + 3) % 19]
            pre2 = LETTERS[(i * 5 + 1) % 19] + LETTERS[(i * 13 + 7) % 19]
        else:
            pre = pre2 = ID_PREFIX[s["agent_type"]]
        f = {
            "brand": brand, "agent_name_f": an_f, "agent_name_m": an_m,
            "agent_type_plain": AGENT_PLAIN[s["agent_type"]],
            "customer_last": last, "customer_first_m": CUSTOMER_FIRST_M[i % 24],
            "customer_first_f": CUSTOMER_FIRST_F[(i * 7) % 24],
            "phone": f"98000 0{1000 + i * 13:04d}",
            "primary_id": f"{pre}{d4}", "secondary_id": f"{pre2}{d4b}",
            "new_phone": f"98000 1{1000 + i * 17:04d}",
            "other_phone": f"98000 2{1000 + i * 19:04d}",
            "old_email": f"{last.lower()}{10 + i}@yahoo.com",
            "new_email": f"{last.lower()}.home{10 + i}@gmail.com",
        }
        if sid == "air_09":
            p = f["primary_id"]
            f["misread_id"] = p[:2] + p[3] + p[2] + p[4:]  # two digits swapped
        out[sid] = f
    return out


RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                                              "required": ["key", "value"]}},
        "distractors": {"type": "array", "items": {"type": "string"}},
        "information": {"type": "string"},
        "caller_values": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                                                      "required": ["key", "value"]}},
        "caller_situation": {"type": "string"},
    },
    "required": ["facts", "distractors", "information", "caller_values", "caller_situation"],
}

TYPE_HINTS = {
    "food_delivery_support": "Restaurant on the order should be a plausible Indian chain or local restaurant (e.g. Biryani Blues, Haldiram's, Domino's, Wow! Momo, Behrouz Biryani). Items with quantities, total in Rs, status (e.g. preparing / out for delivery / delivered), ETA in minutes, delivery address, payment method, existing delivery instruction if any. Distractor: a second, older order with secondary_id from another restaurant.",
    "ecommerce_support": "Product with brand, model, colour/size/variant (e.g. boAt Rockerz 450 headphones, Samsung Galaxy M35 128 GB), price in Rs, status (e.g. shipped / out for delivery / return picked up), delivery date as a weekday + date like 'Thursday 8 October', delivery address, payment method, phone on delivery. Distractor: a second order with secondary_id.",
    "cab_ride_support": "Ride with status (e.g. driver on the way / ride in progress / completed), ETA in minutes, driver first name, car model and colour, vehicle number in Delhi plate format like 'DL 01 AB 4821', pickup and drop (Delhi/NCR places, terminals/gates/hotels as needed), fare in Rs, payment method. For safety scenarios add: 'SOS button available in the app' and 'emergency helpline 112'. Distractor: a past ride with secondary_id.",
    "subscription_account_support": "Account ID (primary_id), plan name (e.g. Premium Monthly / Family Annual), price in Rs per cycle, billing cycle, status (active/paused), last renewal date and next renewal date (dates like '18 October'), payment method (e.g. UPI / HDFC credit card ending 4412), registered email (old_email) and phone. Distractor: e.g. the previous plan or a past invoice.",
    "airport_ticket_counter": "PNR (primary_id), flight number like '6E 2134' or 'AI 302' style but for this airline use a 2-letter code + number, route Delhi to <city>, date, departure time and boarding time (e.g. 6:40 PM / 6:00 PM), terminal and gate (e.g. Terminal 3, Gate 42B), seat (e.g. 18A window), checked and cabin baggage allowance (e.g. 15 kg checked, 7 kg cabin), bags checked count if needed, meal preference, onward connection flight/time/terminal if the scenario needs it, booking status, contact phone and email (old_email). Distractor: e.g. return flight or a co-passenger.",
}


def record_prompt(sid, s, f):
    writes = s["writes"]
    wtxt = "; ".join(f"{w}({', '.join(TOOL_ARGS[w])})" for w in writes) or "none"
    extra = ""
    if sid == "air_09":
        extra = f"\n- The passenger first misreads the PNR as {f['misread_id']} then corrects it to {f['primary_id']}: put misread_pnr in caller_values."
    if sid in ("food_08", "ecom_13"):
        extra = "\n- The caller first says a wrong flat/apartment number and corrects it: caller_values must have new_address_wrong (with the wrong number) and new_address (final corrected)."
    return f"""You are creating ONE realistic customer record for a synthetic customer-support call dataset (India, Delhi/NCR).

Company: {f['brand']} ({s['brand_context']}; agent type {s['agent_type']}).
Scenario {sid}: {s['title']}
Flow:
""" + "\n".join(f"- {x}" for x in s["flow"]) + f"""
Static-read tools this company has (facts the agent may read out): {', '.join(s['static_read'])}
Write actions in this scenario: {wtxt}

FIXED VALUES (use exactly, do not invent other IDs/phones/emails):
- main reference ID: {f['primary_id']}   (second/distractor reference: {f['secondary_id']})
- customer phone on record: {f['phone']}
- old email on record: {f['old_email']}
- if the caller gives a new own phone number: {f['new_phone']}; if the caller gives someone else's number (family member): {f['other_phone']}
- if the caller gives a new email: {f['new_email']}
Customer name is added separately; do NOT mention the customer's name anywhere.

Domain guidance: {TYPE_HINTS[s['agent_type']]}

Produce JSON:
- facts: every static-read fact this scenario's flow could need the agent to say (key/value; values short, concrete; times like 6:40 PM, money like Rs 540, dates like 18 October). Include the main reference ID and only facts that make sense for this scenario. If the flow says the agent tells an existing instruction, existing address, registered email etc., include it.
- distractors: 1-2 extra facts about this customer that the call does NOT need (e.g. an older order, a different saved address).
- information: ONE prose paragraph containing ALL facts and the distractors, compact, comma-separated style like "Order {f['primary_id']} from Biryani Blues: chicken biryani x2, raita x1, Rs 540, status preparing, ETA 15 minutes, delivery to 42 Green Park, New Delhi, paid by UPI. Older order {f['secondary_id']} ...". Max 90 words. No customer name, no phone (added separately). Do not mention tools, checking or instructions to the agent. Write the money as 'Rs 540' (never the rupee symbol).
- caller_values: the exact values the CALLER will dictate in this call for the write actions (e.g. new_address with flat/tower/sector/locality/city in Delhi/NCR, new_phone, family_phone, new_email, instruction as a SHORT rider/driver note of at most 8 words like "Call on arrival, do not ring bell", new pickup/drop location, refund reason in at most 8 words, affected items). Addresses/locations at most 8 words (e.g. "Flat 402, Tower B, Sector 62, Noida"). Empty list if the scenario has no write and the caller dictates nothing. Never put these in information unless the flow says they are already on record.{extra}
- caller_situation: one short sentence of the caller's situation (why they call, their mood), consistent with the facts.
Return only the JSON."""


def build_role_prompt(f, info, agent_gender, customer_gender):
    agent = f["agent_name_f"] if agent_gender == "f" else f["agent_name_m"]
    first = f["customer_first_f"] if customer_gender == "f" else f["customer_first_m"]
    return (f"You work for {f['brand']} which is a {f['agent_type_plain']} and your name is {agent}. "
            f"Information: Customer: {first} {f['customer_last']}, phone {f['phone']}. {info}")


def check_record(sid, s, f, r):
    errs = []
    info = r["information"]
    if f["primary_id"] not in info:
        errs.append(f"information must contain the main reference ID {f['primary_id']}")
    if "₹" in info:
        errs.append("use 'Rs', not the rupee symbol")
    if f["customer_first_m"] in info or f["customer_first_f"] in info or f["customer_last"] in info:
        errs.append("do not mention the customer's name in information")
    for n in (f["agent_name_f"], f["agent_name_m"]):
        if re.search(r"\b" + n + r"\b", info):
            errs.append(f"the name {n} is reserved for the support agent; use a different name (e.g. for the driver)")
    full = build_role_prompt(f, info, "f", "m").split("Information:", 1)[1]
    nt = n_tokens(full)
    if nt > 150:
        errs.append(f"information too long ({nt} tokens incl. customer line; max 150): shorten")
    ids = re.findall(r"\b[A-Z]{2}\d{4}\b", info)
    for i in ids:
        if i not in (f["primary_id"], f["secondary_id"], f.get("misread_id")):
            errs.append(f"unexpected ID {i}; only use {f['primary_id']} / {f['secondary_id']}")
    cv = {c["key"]: c["value"] for c in r["caller_values"]}
    for k, v in cv.items():
        lim = 8 if any(x in k for x in ("instruction", "reason", "address", "location", "pickup", "drop")) else 99
        if len(v.split()) > lim + 1:
            errs.append(f"caller value {k} has {len(v.split())} words; max {lim}")
    if s["writes"] and not cv and s["writes"] not in (["cancel_order"], ["cancel_ride"], ["cancel_subscription"],
                                                      ["pause_subscription"], ["request_ticket_cancellation"]):
        errs.append("caller_values empty but scenario has writes with values")
    return errs, nt


def make_records(llm, out_path=f"{DATA}/records.json", only=None):
    scen = load_scenarios()
    det = det_fields(scen)
    todo = [sid for sid in scen if (only is None or sid in only)]
    results, feedback = {}, {sid: "" for sid in todo}
    for attempt in range(3):
        if not todo:
            break
        prompts = [record_prompt(sid, scen[sid], det[sid]) + (f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {feedback[sid]}" if feedback[sid] else "")
                   for sid in todo]
        outs = llm.chat(prompts, schema=RECORD_SCHEMA, temperature=0.7, max_tokens=1500,
                        seeds=[1000 + attempt * 100 + i for i in range(len(todo))])
        nxt = []
        for sid, (txt, fin, _) in zip(todo, outs):
            try:
                r = extract_json(txt)
                errs, nt = check_record(sid, scen[sid], det[sid], r)
            except Exception as e:  # noqa
                errs, nt, r = [f"bad JSON ({fin}): {e}"], 0, None
            if errs and attempt < 2:
                feedback[sid] = "; ".join(errs)
                nxt.append(sid)
                print(f"[records] {sid} attempt {attempt + 1} rejected: {feedback[sid]}", flush=True)
                continue
            f = det[sid]
            if r and sid == "air_09":  # deterministic misread PNR
                r["caller_values"] = [c for c in r["caller_values"] if "misread" not in c["key"]] + \
                    [{"key": "misread_pnr", "value": f["misread_id"]}]
            rec = {"scenario_id": sid, "agent_type": scen[sid]["agent_type"], **f,
                   "facts": {c["key"]: c["value"] for c in r["facts"]} if r else {},
                   "distractors": r["distractors"] if r else [],
                   "caller_values": {c["key"]: c["value"] for c in r["caller_values"]} if r else {},
                   "caller_situation": r["caller_situation"] if r else "",
                   "information": r["information"] if r else "",
                   "information_tokens_spm": nt, "record_errors": errs,
                   "role_prompts": {g: build_role_prompt(f, r["information"] if r else "", p["agent_gender"], p["customer_gender"])
                                    for g, p in PAIRINGS.items()}}
            results[sid] = rec
        todo = nxt
    allrec = {}
    try:
        allrec = json.load(open(out_path))
    except Exception:  # noqa
        pass
    allrec.update(results)
    allrec = {sid: allrec[sid] for sid in scen if sid in allrec}
    # uniqueness check across records
    ids = [r["primary_id"] for r in allrec.values()] + [r["secondary_id"] for r in allrec.values()]
    assert len(ids) == len(set(ids)), "duplicate IDs across records"
    jdump(out_path, allrec)
    bad = [sid for sid, r in allrec.items() if r["record_errors"]]
    print(f"[records] wrote {len(allrec)} records to {out_path}; with residual errors: {bad}", flush=True)
    return allrec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    # D2 / V4 (2026-10-04): diversity-enforced records for scenario_creation_v2.json -> data/V4/. Default path unchanged.
    ap.add_argument("--v2", action="store_true", help="V4 records (records_v2.py) -> data/V4/records.json + record_map.json")
    ap.add_argument("--scen", default=None, help="--v2: scenario file (default guidance/scenario_creation_v2.json)")
    ap.add_argument("--out-dir", default=None, help="--v2: output dir (default data/V4)")
    ap.add_argument("--dry", action="store_true", help="--v2: CPU only; print det fields + one prompt, no Gemma")
    ap.add_argument("--repair", action="store_true", help="--v2: D2 repair run on the existing data/V4/records.json "
                    "(redo residual / over-cap / scenario-conflict records only; backups run1_*); with --dry: plan only")
    ap.add_argument("--family", action="store_true", help="--v2 --repair: D2 review fixes (family reuse cap, pooled ecom "
                    "products, duplicate-ID check, Information clean-up + payment-bank spread)")
    a = ap.parse_args()
    if a.v2:
        import records_v2 as RV
        scen_path, out_dir = a.scen or RV.SCEN_V2, a.out_dir or RV.V4_DIR
        if a.repair:
            if a.dry:
                RV.repair_records_v2(None, scen_path=scen_path, out_dir=out_dir, dry=True, family=a.family)
            else:
                from common import LLM
                recs = RV.repair_records_v2(LLM(), scen_path=scen_path, out_dir=out_dir, family=a.family)
                RV.write_record_map(recs, out_dir=out_dir)
        elif a.dry:
            sc = RV.load_scenarios_v2(scen_path)
            det = RV.det_fields_v2(sc)
            sid = (a.only or list(sc))[0]
            print(json.dumps(det[sid], indent=1, ensure_ascii=False))
            print(RV.record_prompt_v2(sid, sc[sid], det[sid]))
        else:
            from common import LLM
            recs = RV.make_records_v2(LLM(), scen_path=scen_path, out_dir=out_dir, only=a.only)
            RV.write_record_map(recs, out_dir=out_dir)
    else:
        from common import LLM
        make_records(LLM(), only=a.only)
