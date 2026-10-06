"""Timeline + stereo wav + alignment JSON + manifests + per-call QC stats (venv-tts, CPU).

  /workspace/venv-tts/bin/python assemble.py CALLS.jsonl ROOT [--holdout /workspace/hinglish/data/holdout.json]
                                             [--interrupt-mode before_cut|literal]

Channel convention (moshi-finetune README + annotate.py channel=0): LEFT/ch0 = model = AGENT,
RIGHT/ch1 = user = CUSTOMER. 24 kHz stereo PCM16.
Alignment JSON (moshi-finetune format): {"alignments": [[word, [start, end], speaker], ...]} sorted by start
across both speakers; agent = "SPEAKER_MAIN" (kept by the trainer's keep_main_only), customer = "SPEAKER_USER".
Extra top-level keys (ignored by the trainer): call_id, variant, role_prompt, agent_gender, voice_prompt,
voices, turns (timeline), duration, flags.
Timeline: 1.0 s lead silence; after each turn gap = pause_after_s if set (check_line: forced into 0.8-1.5;
others: into 0.2-2.0), else default U(0.3,0.6) / U(0.8,1.5) after check_line; 0.5 s tail.
Interruption (agent turn truncated, next customer turn overlap_offset_s > 0): agent audio hard-cut at the end
of its last aligned word (+30 ms, 10 ms fade). Customer onset:
  before_cut (default): onset = cut - overlap_offset_s  (exactly overlap_offset_s of overlap; text already cut)
  literal:              onset = agent_start + overlap_offset_s
Outputs: ROOT/stereo/<call>.wav/.json, ROOT/manifest/all.jsonl, ROOT/train/train.jsonl (non-held-out),
ROOT/test/test.jsonl (the test-call list), ROOT/heldout/heldout_all.jsonl, ROOT/qc/qc_stats.jsonl,
ROOT/qc/qc_summary.json. Manifest lines: {"path": <absolute wav path>, "duration": seconds}.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
LEAD, TAIL = 1.0, 0.5
CER_MAX = 0.35
MAX_WINDOW = 100.0  # moshi-finetune duration_sec default


def rng_for(*p):
    return random.Random(int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16))


def gap_after(call_id, ti, turn):
    is_check = "check_line" in (turn.get("tags") or [])
    p = turn.get("pause_after_s")
    r = rng_for(call_id, ti, "gap")
    if is_check:
        return float(p) if p is not None and 0.8 <= p <= 1.5 else r.uniform(0.8, 1.5)
    if p is not None and 0.2 <= p <= 2.0:
        return float(p)
    return r.uniform(0.3, 0.6)


def build_call(call, root, mode):
    cid = call["call_id"]
    utt_dir, al_dir = root / "work" / "utt" / cid, root / "work" / "align" / cid
    turns = call["turns"]
    flags = []
    pieces = []  # (channel, start_s, audio)
    words = []
    timeline = []
    t_end = LEAD  # end of all audio placed so far (+gap)
    cut_info = {}
    stats = {"chunks": 0, "chunks_split_lines": 0, "cer": [], "cer_flagged": 0, "align_fallback": 0,
             "align_low": 0, "interruptions": 0}
    for ti, turn in enumerate(turns):
        a, sr = sf.read(str(utt_dir / f"t{ti:02d}.wav"), dtype="float32")
        assert sr == SR
        um = json.loads((utt_dir / f"t{ti:02d}.json").read_text(encoding="utf-8"))
        al = json.loads((al_dir / f"t{ti:02d}.json").read_text(encoding="utf-8"))
        stats["chunks"] += len(um["chunks"])
        stats["chunks_split_lines"] += int(len(um["chunks"]) > 1)
        for c in um["chunks"]:
            stats["cer"].append(c["cer"])
            if c["cer"] > c.get("cer_max", CER_MAX):  # per-chunk threshold (Trelis/method-1 chunks carry cer_max)
                stats["cer_flagged"] += 1
                flags.append(f"t{ti}c{c['chunk']}:cer={c['cer']:.2f}")
            for f in c["flags"]:
                if f.startswith(("forced_split", "roman_map_approx", "wordcount", "long_chunk", "num_mismatch")):
                    flags.append(f"t{ti}c{c['chunk']}:{f}")
        if al["method"] != "mms_fa":
            stats["align_fallback"] += 1
            flags.append(f"t{ti}:align_fallback")
        elif "align_low_score" in al["flags"]:
            stats["align_low"] += 1
            flags.append(f"t{ti}:align_low_score={al['mean_score']:.2f}")
        uwords = al["words"]
        start = t_end
        prev = turns[ti - 1] if ti > 0 else None
        if (prev is not None and prev["speaker"] == "agent" and prev.get("truncated")
                and turn["speaker"] == "customer" and (turn.get("overlap_offset_s") or 0) > 0 and ti - 1 in cut_info):
            ag_start, cut = cut_info[ti - 1]
            off = float(turn["overlap_offset_s"])
            start = ag_start + cut - off if mode == "before_cut" else ag_start + off
            start = max(start, ag_start + 0.1)
            stats["interruptions"] += 1
        if turn["speaker"] == "agent" and turn.get("truncated"):
            if uwords:
                cut = min(len(a) / SR, uwords[-1]["end"] + 0.03)
                n = int(cut * SR)
                a = a[:n].copy()
                fade = min(int(0.01 * SR), len(a))
                a[len(a) - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            cut_info[ti] = (start, len(a) / SR)
        ch = 0 if turn["speaker"] == "agent" else 1
        pieces.append((ch, start, a))
        dur = len(a) / SR
        spk = "SPEAKER_MAIN" if ch == 0 else "SPEAKER_USER"
        for w in uwords:
            if w["start"] >= dur:
                continue
            words.append([w["word"], [round(start + w["start"], 3), round(start + min(w["end"], dur), 3)], spk])
        timeline.append({"turn": ti, "speaker": turn["speaker"], "start": round(start, 3),
                         "end": round(start + dur, 3), "truncated": bool(turn.get("truncated")),
                         "tags": turn.get("tags") or [], "text_roman": turn["text_roman"]})
        t_end = max(t_end, start + dur) + gap_after(cid, ti, turn)
    total = max(e["end"] for e in timeline) + TAIL
    n = int(np.ceil(total * SR))
    out = np.zeros((n, 2), np.float32)
    for ch, st, a in pieces:
        i = int(round(st * SR))
        out[i:i + len(a), ch] += a
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
        flags.append("rescaled_peak")
    words.sort(key=lambda w: (w[1][0], w[1][1]))
    sdir = root / "stereo"
    sdir.mkdir(parents=True, exist_ok=True)
    wav_path = (sdir / f"{cid}.wav").resolve()
    sf.write(str(wav_path), out, SR, subtype="PCM_16")
    duration = n / SR
    agent_s = sum(e["end"] - e["start"] for e in timeline if e["speaker"] == "agent")
    cust_s = sum(e["end"] - e["start"] for e in timeline if e["speaker"] == "customer")
    if duration > MAX_WINDOW:
        flags.append(f"over_{int(MAX_WINDOW)}s")
    ag = call.get("agent_gender", "f")
    doc = {"alignments": words, "call_id": cid, "variant": call.get("variant"), "scenario_id": call.get("scenario_id"),
           "role_prompt": call.get("role_prompt"), "agent_gender": ag,
           "voice_prompt": "NATF2" if ag == "f" else "NATM1", "voices": plan_voices(root, cid) or call.get("voices"),
           "duration": round(duration, 3), "turns": timeline, "flags": flags, "interrupt_mode": mode}
    (sdir / f"{cid}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    cers = stats.pop("cer")
    qc = {"call_id": cid, "duration": round(duration, 2), "agent_speech_s": round(agent_s, 2),
          "customer_speech_s": round(cust_s, 2), "n_turns": len(turns), "n_agent_words": sum(w[2] == "SPEAKER_MAIN" for w in words),
          "cer_mean": round(float(np.mean(cers)), 4), "cer_max": round(float(np.max(cers)), 4),
          "over_100s": duration > MAX_WINDOW, **stats, "flags": flags}
    return str(wav_path), duration, qc


_PV = {}


def plan_voices(root, cid):
    """v3-data fix: TTS voices actually used (work/plan.json) for non-kokoro backends; None for kokoro (V1 unchanged)."""
    key = str(root)
    if key not in _PV:
        pp = Path(root) / "work" / "plan.json"
        m = {}
        if pp.exists():
            for ch in json.loads(pp.read_text(encoding="utf-8")):
                if ch.get("backend", "kokoro") != "kokoro":
                    m.setdefault(ch["call_id"], {})[ch["speaker"]] = ch["voice"]
        _PV[key] = m
    return _PV[key].get(cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("root")
    ap.add_argument("--holdout", default=__import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/data/holdout.json")
    ap.add_argument("--interrupt-mode", default="before_cut", choices=["before_cut", "literal"])
    a = ap.parse_args()
    root = Path(a.root)
    calls = [json.loads(l) for l in open(a.calls, encoding="utf-8") if l.strip()]
    ho = json.loads(Path(a.holdout).read_text()) if Path(a.holdout).exists() else None
    test_sc = set(ho["test_scenarios"]) if ho else set()
    test_calls = set(ho["test_calls"]) if ho else set()
    # D2 (holdout_v2.py, data/V4/holdout.json): a separate VAL split. Train excludes test AND val scenarios;
    # val/val.jsonl = all calls of the val scenarios (disjoint from test, so val never contains test calls).
    # Holdout files without "val_scenarios" (data/holdout.json) give exactly the old manifests.
    val_sc = set(ho.get("val_scenarios") or []) if ho else set()
    assert not val_sc & test_sc, "val and test scenarios overlap"
    # D2 review: V4 calls need the V4 holdout (val split). Without it the old data/holdout.json would put V4 val/test
    # scenarios into train and write no val/val.jsonl. Fail before writing anything. V1/V3/CONTROL never carry V4.
    if any(c.get("variant") == "V4" for c in calls) and not val_sc:
        raise SystemExit(f"V4 calls but holdout {a.holdout} has no val_scenarios: pass --holdout /workspace/hinglish/data/V4/holdout.json "
                         "(run_variant.sh: HOLDOUT=...)")
    rows, qcs, failed = [], [], []
    for c in calls:
        try:
            p, d, qc = build_call(c, root, a.interrupt_mode)
        except FileNotFoundError as e:
            failed.append((c["call_id"], str(e)[:120]))
            continue
        rows.append((c, {"path": p, "duration": round(d, 3)}))
        qcs.append(qc)

    def dump(rel, sel):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            for c, r in rows:
                if sel(c):
                    fh.write(json.dumps(r) + "\n")

    dump("manifest/all.jsonl", lambda c: True)
    dump("train/train.jsonl", lambda c: c["scenario_id"] not in test_sc and c["scenario_id"] not in val_sc)
    if val_sc:
        dump("val/val.jsonl", lambda c: c["scenario_id"] in val_sc)
    dump("test/test.jsonl", lambda c: c["call_id"] in test_calls)
    dump("heldout/heldout_all.jsonl", lambda c: c["scenario_id"] in test_sc)
    (root / "qc").mkdir(parents=True, exist_ok=True)
    with open(root / "qc" / "qc_stats.jsonl", "w") as fh:
        for q in qcs:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    summ = {"calls": len(qcs), "failed_missing_inputs": failed, "hours": round(sum(q["duration"] for q in qcs) / 3600, 3),
            "agent_hours": round(sum(q["agent_speech_s"] for q in qcs) / 3600, 3),
            "chunks": sum(q["chunks"] for q in qcs), "cer_flagged_chunks": sum(q["cer_flagged"] for q in qcs),
            "align_fallback_utts": sum(q["align_fallback"] for q in qcs), "align_low_utts": sum(q["align_low"] for q in qcs),
            "over_100s_calls": sum(q["over_100s"] for q in qcs), "interruptions": sum(q["interruptions"] for q in qcs),
            "mean_cer": round(float(np.mean([q["cer_mean"] for q in qcs])), 4) if qcs else None,
            "train_calls": sum(1 for c, _ in rows if c["scenario_id"] not in test_sc and c["scenario_id"] not in val_sc),
            "test_calls": sum(1 for c, _ in rows if c["call_id"] in test_calls), "interrupt_mode": a.interrupt_mode}
    if val_sc:
        summ["val_calls"] = sum(1 for c, _ in rows if c["scenario_id"] in val_sc)
    (root / "qc" / "qc_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
