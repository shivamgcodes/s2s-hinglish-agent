"""N2 router-window extractor (CPU).

For every check_line turn in V4 calls.jsonl, cut the CUSTOMER channel (ch1) of the stereo wav for the 30 s
ending at the check-line turn's START (stereo json turn start), as DEP1's ring buffer would hold it at trigger
time. Two variants per window:
  clean : ch1 as rendered (24 kHz) -> cut -> 16 kHz with DEP1 asr_service.to16k (FFT band-limited)
  opus  : light pink noise added to the WHOLE ch1 (SNR 25-30 dB vs customer-speech power, seeded per call)
          -> Opus round trip over the WHOLE call with sphn OpusStreamWriter/Reader at 24 kHz, 1920-sample
          feeds (same path as deploy/router/bench/make_opus_inputs.py, DEP1 ring30_opus) -> cut.
          Saved as 24 kHz (exactly what the ring holds) and 16 kHz (to16k).
No codec-lag compensation (the live ring has the lag too).

Writes are mapped to check-lines by write.confirm_turn_idx - 1 (verified 1:1 on V4) with a turn-order
fallback (first check_line after the caller turn and before the confirm turn).

Usage: python extract_windows.py --v4 /workspace/hinglish/data/V4 --out /workspace/hinglish/needle_v2/windows [--jobs 32]
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import sphn

SR = 24000
SR16 = 16000
WIN_S = 30.0
SNR_LO, SNR_HI = 25.0, 30.0


def to16k(a24):  # copied from deploy/router/asr_service.py
    a = np.asarray(a24, dtype=np.float32)
    if len(a) == 0:
        return a
    n_out = int(round(len(a) * SR16 / SR))
    X = np.fft.rfft(a)
    return (np.fft.irfft(X[: n_out // 2 + 1], n_out) * (n_out / len(a))).astype(np.float32)


def seed_of(call_id):
    return int(hashlib.md5(call_id.encode()).hexdigest()[:8], 16)


def pink(n, rng):
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    X = X / np.sqrt(f)
    X[0] = 0.0
    p = np.fft.irfft(X, n)
    return (p / (np.sqrt(np.mean(p ** 2)) + 1e-12)).astype(np.float32)


def opus_roundtrip(a):
    w, r = sphn.OpusStreamWriter(SR), sphn.OpusStreamReader(SR)
    dec, nbytes = [], 0
    pad = (-len(a)) % 1920
    ap = np.pad(a, (0, pad + 1920 * 2))  # flush tail
    for i in range(0, len(ap) - 1919, 1920):
        w.append_pcm(ap[i:i + 1920])
        b = w.read_bytes()
        if b:
            nbytes += len(b)
            r.append_bytes(b)
            p = r.read_pcm()
            if p.shape[-1]:
                dec.append(p.reshape(-1))
    d = np.concatenate(dec)[: len(a)] if dec else np.zeros(0, np.float32)
    d = np.pad(d, (0, len(a) - len(d)))
    return d.astype(np.float32), nbytes


def write_wav(path, a, sr):
    sf.write(path, np.clip(a, -1, 1), sr, subtype="PCM_16")


G = {}


def init(v4, out):
    G["v4"], G["out"] = v4, out


def process(job):
    call, split, in_test_calls, record_id = job
    v4, out = G["v4"], G["out"]
    cid = call["call_id"]
    res = {"call_id": cid, "rows": [], "issues": [], "stats": {}}
    js = json.load(open(f"{v4}/stereo/{cid}.json"))
    a, sr = sf.read(f"{v4}/stereo/{cid}.wav", dtype="float32", always_2d=True)
    if sr != SR or a.shape[1] != 2:
        res["issues"].append(f"bad audio sr={sr} ch={a.shape[1]}")
        return res
    dur_wav = len(a) / SR
    if abs(dur_wav - js["duration"]) > 0.05:
        res["issues"].append(f"duration mismatch wav {dur_wav:.3f} json {js['duration']:.3f}")
    ch1 = a[:, 1].copy()
    ch0 = a[:, 0]
    sturns = {t["turn"]: t for t in js["turns"]}
    turns = call["turns"]
    # alignment check
    for i, t in enumerate(turns):
        st = sturns.get(i)
        if st is None:
            res["issues"].append(f"turn {i} missing in stereo json")
            continue
        if st["speaker"] != t["speaker"] or sorted(st["tags"]) != sorted(t["tags"]):
            res["issues"].append(f"turn {i} mismatch speaker/tags {st['speaker']}/{st['tags']} vs {t['speaker']}/{t['tags']}")
    extra = set(sturns) - set(range(len(turns)))
    if extra:
        res["issues"].append(f"stereo json has extra turns {sorted(extra)}")

    def iv(t):
        return int(round(t["start"] * SR)), int(round(t["end"] * SR))

    cust = [iv(t) for t in js["turns"] if t["speaker"] == "customer"]
    agt = [iv(t) for t in js["turns"] if t["speaker"] == "agent"]

    def rms(x, ivs):
        seg = np.concatenate([x[s:e] for s, e in ivs]) if ivs else np.zeros(1, np.float32)
        return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0

    # channel sanity: ch1 should carry customer speech, ch0 agent speech
    res["stats"]["ch1_rms_cust"] = rms(ch1, cust)
    res["stats"]["ch1_rms_agent"] = rms(ch1, agt)
    res["stats"]["ch0_rms_agent"] = rms(ch0, agt)
    res["stats"]["ch0_rms_cust"] = rms(ch0, cust)

    # noisy + opus variant over the whole call
    seed = seed_of(cid)
    rng = np.random.default_rng(seed)
    snr = float(rng.uniform(SNR_LO, SNR_HI))
    p_speech = res["stats"]["ch1_rms_cust"] ** 2
    noise = pink(len(ch1), rng) * np.float32(np.sqrt(p_speech / 10 ** (snr / 10)))
    noisy = ch1 + noise
    opus, nbytes = opus_roundtrip(noisy)
    kbps = nbytes * 8 / dur_wav / 1000
    res["stats"]["opus_kbps"] = kbps

    # writes -> check-lines
    cl_idx = [i for i, t in enumerate(turns) if "check_line" in t["tags"]]
    by_cl = defaultdict(list)
    for wi, w in enumerate(call["writes"]):
        k = w.get("confirm_turn_idx", -1) - 1
        how = "confirm_turn_idx-1"
        if k not in cl_idx:
            cands = [c for c in cl_idx if c > (w.get("caller_turn_idx") or -1) and c < w.get("confirm_turn_idx", 10 ** 9)]
            k = cands[0] if cands else None
            how = "turn_order"
        if k is None:
            res["issues"].append(f"write {wi} {w['tool']} not mappable to a check-line")
            res.setdefault("unmapped_writes", []).append({"call_id": cid, "write_idx": wi, **w})
            continue
        by_cl[k].append((wi, w, how))

    os.makedirs(f"{out}/clean_16k/{cid}", exist_ok=True)
    os.makedirs(f"{out}/opus_24k/{cid}", exist_ok=True)
    os.makedirs(f"{out}/opus_16k/{cid}", exist_ok=True)
    for ordinal, k in enumerate(cl_idx):
        st = sturns.get(k)
        if st is None:
            res["issues"].append(f"check_line turn {k} missing in stereo json; skipped")
            continue
        we = int(round(st["start"] * SR))
        ws = max(0, we - int(WIN_S * SR))
        wid = f"{cid}__cl{ordinal}"
        clean24 = ch1[ws:we]
        op24 = opus[ws:we]
        write_wav(f"{out}/clean_16k/{cid}/{wid}.wav", to16k(clean24), SR16)
        write_wav(f"{out}/opus_24k/{cid}/{wid}.wav", op24, SR)
        write_wav(f"{out}/opus_16k/{cid}/{wid}.wav", to16k(op24), SR16)
        w_s, w_e = ws / SR, we / SR
        # customer turns overlapping the window (oracle text)
        cin = []
        for t in js["turns"]:
            if t["speaker"] != "customer" or t["end"] <= w_s or t["start"] >= w_e:
                continue
            cov = "full" if (t["start"] >= w_s and t["end"] <= w_e) else "partial"
            cin.append({"turn": t["turn"], "start": t["start"], "end": t["end"], "coverage": cov,
                        "clipped_at": ("start" if t["start"] < w_s else "") + ("end" if t["end"] > w_e else ""),
                        "text_roman": t["text_roman"]})
        writes = []
        for wi, w, how in by_cl.get(k, []):
            ct = w.get("caller_turn_idx")
            cov = None
            if ct is not None and ct in sturns:
                t = sturns[ct]
                if t["end"] <= w_s or t["start"] >= w_e:
                    cov = "none"
                elif t["start"] >= w_s and t["end"] <= w_e:
                    cov = "full"
                else:
                    cov = "partial"
            writes.append({"write_idx": wi, "tool": w["tool"], "args": w["args"],
                           "caller_turn_idx": ct, "confirm_turn_idx": w.get("confirm_turn_idx"),
                           "caller_turn_speaker": sturns[ct]["speaker"] if ct in sturns else None,
                           "caller_turn_in_window": cov, "map": how})
        res["rows"].append({
            "id": wid, "call_id": cid, "scenario_id": call["scenario_id"], "split": split,
            "in_holdout_test_calls": in_test_calls, "agent_type": call["agent_type"], "record_id": record_id,
            "check_line_ordinal": ordinal, "check_line_turn_idx": k,
            "check_line_start_s": st["start"], "check_line_end_s": st["end"],
            "check_line_text": st["text_roman"],
            "window_start_s": round(w_s, 4), "window_end_s": round(w_e, 4), "window_len_s": round(w_e - w_s, 4),
            "call_duration_s": round(dur_wav, 4), "stereo_flags": js.get("flags", []),
            "writes": writes, "n_writes": len(writes),
            "customer_turns_in_window": cin,
            "noise": {"type": "pink", "seed": seed, "snr_db": round(snr, 2), "ref": "ch1 power inside customer turns"},
            "opus_kbps_call": round(kbps, 2),
            "files": {"clean_16k": f"clean_16k/{cid}/{wid}.wav", "opus_24k": f"opus_24k/{cid}/{wid}.wav",
                      "opus_16k": f"opus_16k/{cid}/{wid}.wav"},
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default=os.environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/data/V4")
    ap.add_argument("--out", default=os.environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/needle_v2/windows")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    calls = [json.loads(l) for l in open(f"{a.v4}/calls.jsonl")]
    if a.limit:
        calls = calls[:: max(1, len(calls) // a.limit)][: a.limit]
    h = json.load(open(f"{a.v4}/holdout.json"))
    recs = json.load(open(f"{a.v4}/records.json"))  # keyed by scenario_id; covers all 162 (record_map.json only has 72 old ones)
    ts, vs, trs = set(h["test_scenarios"]), set(h["val_scenarios"]), set(h["train_scenarios"])
    tcalls = set(h["test_calls"])
    jobs = []
    for c in calls:
        s = c["scenario_id"]
        split = "test" if s in ts else "val" if s in vs else "train" if s in trs else "unknown"
        jobs.append((c, split, c["call_id"] in tcalls, recs[s]["record_id"]))
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.jobs, initializer=init, initargs=(a.v4, a.out)) as p:
        results = p.map(process, jobs, chunksize=2)
    rows = [r for res in results for r in res["rows"]]
    rows.sort(key=lambda r: r["id"])
    with open(f"{a.out}/meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    issues = {res["call_id"]: res["issues"] for res in results if res["issues"]}
    unmapped = [u for res in results for u in res.get("unmapped_writes", [])]
    cstats = {res["call_id"]: res["stats"] for res in results}
    # summary
    S = {}
    S["n_calls"] = len(calls)
    S["n_calls_with_check_line"] = len({r["call_id"] for r in rows})
    S["n_windows"] = len(rows)
    S["n_windows_with_write"] = sum(r["n_writes"] > 0 for r in rows)
    S["n_windows_without_write"] = [r["id"] for r in rows if r["n_writes"] == 0]
    S["n_windows_multi_write"] = [r["id"] for r in rows if r["n_writes"] > 1]
    S["n_writes_total"] = sum(len(c["writes"]) for c in calls)
    S["n_writes_mapped"] = sum(r["n_writes"] for r in rows)
    S["unmapped_writes"] = unmapped
    S["map_methods"] = dict(Counter(w["map"] for r in rows for w in r["writes"]))
    S["zero_write_calls_with_check_line"] = sorted({r["call_id"] for r in rows if not any(len(c["writes"]) for c in calls if c["call_id"] == r["call_id"])})
    S["per_split"] = dict(Counter(r["split"] for r in rows))
    S["per_split_calls"] = dict(Counter(j[1] for j in jobs))
    S["per_agent_type"] = dict(Counter(r["agent_type"] for r in rows))
    S["per_split_agent_type"] = {sp: dict(Counter(r["agent_type"] for r in rows if r["split"] == sp)) for sp in ("train", "val", "test")}
    S["per_tool"] = dict(Counter(w["tool"] for r in rows for w in r["writes"]))
    S["per_split_tool"] = {sp: dict(Counter(w["tool"] for r in rows if r["split"] == sp for w in r["writes"])) for sp in ("train", "val", "test")}
    S["audio_minutes_per_variant"] = round(sum(r["window_len_s"] for r in rows) / 60, 2)
    S["audio_minutes_per_split"] = {sp: round(sum(r["window_len_s"] for r in rows if r["split"] == sp) / 60, 2) for sp in ("train", "val", "test")}
    wl = np.array([r["window_len_s"] for r in rows])
    S["window_len_s"] = {"min": float(wl.min()), "median": float(np.median(wl)), "max": float(wl.max()), "n_lt_30": int((wl < 29.999).sum())}
    S["caller_turn_in_window"] = dict(Counter(str(w["caller_turn_in_window"]) for r in rows for w in r["writes"]))
    S["caller_turn_in_window_per_tool"] = {t: dict(Counter(str(w["caller_turn_in_window"]) for r in rows for w in r["writes"] if w["tool"] == t)) for t in S["per_tool"]}
    S["caller_turn_speaker"] = dict(Counter(str(w["caller_turn_speaker"]) for r in rows for w in r["writes"]))
    kb = np.array([s["opus_kbps"] for s in cstats.values() if "opus_kbps" in s])
    S["opus_kbps"] = {"min": float(kb.min()), "median": float(np.median(kb)), "max": float(kb.max())}
    ratio = np.array([s["ch1_rms_cust"] / max(s["ch1_rms_agent"], 1e-9) for s in cstats.values() if "ch1_rms_cust" in s])
    S["ch1_cust_over_agent_rms_ratio"] = {"min": float(ratio.min()), "median": float(np.median(ratio))}
    ratio0 = np.array([s["ch0_rms_agent"] / max(s["ch0_rms_cust"], 1e-9) for s in cstats.values() if "ch0_rms_agent" in s])
    S["ch0_agent_over_cust_rms_ratio"] = {"min": float(ratio0.min()), "median": float(np.median(ratio0))}
    S["calls_with_issues"] = issues
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1, ensure_ascii=False)
    json.dump(cstats, open(f"{a.out}/call_stats.json", "w"), indent=1)
    brief = {k: v for k, v in S.items() if k not in ("calls_with_issues", "unmapped_writes")}
    brief["n_calls_with_issues"] = len(issues)
    brief["n_unmapped_writes"] = len(unmapped)
    print(json.dumps(brief, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
