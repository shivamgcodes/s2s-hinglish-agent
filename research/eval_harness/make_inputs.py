"""Build test INPUT wavs (customer channel only, 2 s lead, agent slots silent) + per-call test meta.

  python make_inputs.py V1 [--data-root /workspace/hinglish/data/V1] [--calls-jsonl ...] [--calls id1,id2|--all]

Reads <data_root>/stereo/<call>.wav/.json (audio/assemble.py output: ch0 = agent, ch1 = customer, 1.0 s lead)
and <calls_jsonl> (default <data_root>/calls.jsonl) for writes/record/tags. Default call list = holdout.json
test_calls. ch1 already contains only customer audio with zeros where the agent speaks (scripted durations),
so INPUT = 1.0 s extra silence + ch1 + 4 s tail. All script times are shifted by +1.0 s.
Writes tests/inputs/<V>/<call>.wav (24 kHz mono PCM16) and <call>.meta.json (everything score.py needs).
Any CPU python with numpy+soundfile (venv-tts).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import (DATA, HOLDOUT, INPUTS, LEAD_TEST, SHIFT, SR, TAIL_TEST, VOICE, ascii_prompt,  # noqa: E402
                     jdump, jload, load_calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--data-root")
    ap.add_argument("--calls-jsonl")
    ap.add_argument("--calls", help="comma list of call_ids (default: holdout test_calls)")
    ap.add_argument("--all", action="store_true", help="every call that has a stereo wav")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    root = Path(a.data_root or DATA / a.variant)
    calls = load_calls(a.calls_jsonl or root / "calls.jsonl")
    out = Path(a.out_dir or INPUTS / a.variant)
    out.mkdir(parents=True, exist_ok=True)
    if a.calls:
        ids = a.calls.split(",")
    elif a.all:
        ids = sorted(p.stem for p in (root / "stereo").glob("*.wav"))
    else:
        ids = jload(HOLDOUT)["test_calls"]
    made, missing = [], []
    for cid in ids:
        wav, js = root / "stereo" / f"{cid}.wav", root / "stereo" / f"{cid}.json"
        if not wav.exists() or not js.exists() or cid not in calls:
            missing.append(cid)
            continue
        x, sr = sf.read(str(wav), dtype="float32")
        assert sr == SR and x.ndim == 2, (cid, sr, x.shape)
        cust = x[:, 1]
        y = np.concatenate([np.zeros(int(SHIFT * SR), np.float32), cust, np.zeros(int(TAIL_TEST * SR), np.float32)])
        sf.write(str(out / f"{cid}.wav"), y, SR, subtype="PCM_16")
        al = jload(js)
        c = calls[cid]
        turns = []
        for tl, t in zip(al["turns"], c["turns"]):
            turns.append({"idx": tl["turn"], "speaker": t["speaker"], "start": round(tl["start"] + SHIFT, 3),
                          "end": round(tl["end"] + SHIFT, 3), "tags": t.get("tags") or [],
                          "text_roman": t["text_roman"], "truncated": bool(t.get("truncated"))})
        prompt = ascii_prompt(c["role_prompt"])
        meta = {"call_id": cid, "variant": a.variant, "scenario_id": c.get("scenario_id"),
                "input_wav": str((out / f"{cid}.wav").resolve()), "duration": round(len(y) / SR, 3),
                "lead_s": LEAD_TEST, "role_prompt": prompt, "role_prompt_changed": prompt != c["role_prompt"],
                "agent_gender": c.get("agent_gender", "f"), "voice": VOICE[c.get("agent_gender", "f")],
                "agent_name": c.get("agent_name"), "brand": c.get("brand"), "record": c.get("record"),
                "writes": c.get("writes", []), "turns": turns}
        jdump(out / f"{cid}.meta.json", meta)
        made.append(cid)
    print(f"[make_inputs] {a.variant}: built {len(made)} -> {out}; missing {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
