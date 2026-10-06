"""CPU unit test for vad_gate.VadFeeder with a fake model (no torch).  python3 test_vad_gate.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vad_gate import FRAME_RATE, VadFeeder, cut_customer_turns, frame_db  # noqa: E402

FS = 1920
SR = 24000


def tone(sec):
    n = int(sec * SR)
    return (0.1 * np.sin(np.arange(n) * 0.05)).astype(np.float32)


def make_call(spec):
    """spec: list of (speaker, dur). Builds an input wav with customer audio at scripted times (agent slots silent)."""
    t, turns, chunks = 2.0, [], [np.zeros(int(2.0 * SR), np.float32)]
    for i, (spk, d) in enumerate(spec):
        turns.append({"idx": i, "speaker": spk, "start": round(t, 3), "end": round(t + d, 3), "tags": [],
                      "text_roman": f"{spk} {i}"})
        chunks.append(tone(d) if spk == "customer" else np.zeros(int(d * SR), np.float32))
        chunks.append(np.zeros(int(0.5 * SR), np.float32))
        t += d + 0.5
    return np.concatenate(chunks), turns


class FakeModel:
    """Responds to customer audio: after the customer stops (input silent for `delay` frames), speaks `plan` =
    list of (n_frames_speech, n_frames_pause, sentence_end) chunks; greets first with `greet`."""

    def __init__(self, greet, reply, delay=4, forever=False, silent=False):
        self.queue = [] if silent else list(greet)
        self.reply, self.delay, self.forever, self.silent = reply, delay, forever, silent
        self.cust_quiet, self.heard_cust, self.cur = 0, False, None

    def step(self, frame):
        in_act = frame_db(frame) > -50
        if in_act:
            self.heard_cust, self.cust_quiet = True, 0
            self.queue = []  # yields when talked over
            self.cur = None
        else:
            self.cust_quiet += 1
            if self.heard_cust and self.cust_quiet == self.delay and not self.silent:
                self.queue = list(self.reply)
                self.heard_cust = False
        if self.forever and not self.silent:
            return 0.05, " word"
        if self.cur is None and self.queue:
            sp, pa, end = self.queue.pop(0)
            self.cur = ["s"] * sp + ["p"] * pa
            self.cur_end = end
            self.cur_n = sp
        if self.cur:
            k = self.cur.pop(0)
            done_speech = k == "s" and "s" not in self.cur
            if not self.cur:
                self.cur = None
            if k == "s":
                tok = " hoon." if (done_speech and self.cur_end) else " word"
                return 0.05, tok
            return 0.0, "PAD"
        return 0.0, "PAD"


def run(spec, model, **kw):
    audio, turns = make_call(spec)
    clips = cut_customer_turns(audio, SR, turns)
    f = VadFeeder(clips, turns, FS, **kw)
    for fr in f.frames():
        amp, tok = model.step(fr)
        f.observe_vad(frame_db(np.full(FS, amp, np.float32)), tok)
    return f


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, info)
        ok &= bool(cond)

    # 1 normal: greeting 2 s ending a sentence -> first customer turn 4 frames (0.32 s) after greeting end
    m = FakeModel(greet=[(25, 0, True)], reply=[(20, 0, True)])
    f = run([("agent", 3), ("customer", 2), ("agent", 3), ("customer", 2)], m)
    ev = f.timeline["customer_turns"]
    g_end = f.timeline["model_speech"][0][1]
    check("greeting end -> customer after q_sent", ev[0]["released_by"] == "model_end"
          and ev[0]["start_frame"] - g_end == 4, (g_end, ev[0]))
    check("second turn model_end", ev[1]["released_by"] == "model_end", ev[1])
    check("all played, tail_quiet", f.timeline["unplayed_turns"] == [] and f.timeline["stop_reason"] == "tail_quiet",
          f.timeline["stop_reason"])
    check("no overlap", f.timeline["overlap_s"] == 0.0, f.timeline["overlap_s"])
    check("fed == obs", len(f.fed) == len(f.obs))
    eff = f.effective_turns()
    check("effective turns", eff[1]["start"] == ev[0]["start_s"] and eff[2]["start"] == ev[0]["end_s"]
          and eff[2]["end"] == ev[1]["start_s"], [(t["speaker"], t["start"], t["end"]) for t in eff])
    # 2 mid-sentence pause (6 frames, no sentence end) must not release; q_mid = 8
    m = FakeModel(greet=[(10, 6, False), (10, 0, True)], reply=[(10, 0, True)])
    f = run([("agent", 3), ("customer", 2)], m)
    ev = f.timeline["customer_turns"]
    check("mid-sentence 0.48 s pause not a stop", ev[0]["start_frame"] == 10 + 6 + 10 + 4, ev[0])
    # 2b mid-sentence pause of 10 frames (> q_mid) does release
    m = FakeModel(greet=[(10, 10, False), (10, 0, True)], reply=[(10, 0, True)])
    f = run([("agent", 3), ("customer", 2)], m)
    check("mid-sentence 0.8 s pause releases at q_mid", f.timeline["customer_turns"][0]["start_frame"] == 10 + 8,
          f.timeline["customer_turns"][0])
    # 3 silent model: opening timeout 3 s, then silence timeout 4 s
    m = FakeModel(greet=[], reply=[], silent=True)
    f = run([("agent", 3), ("customer", 2), ("agent", 3), ("customer", 2)], m)
    ev = f.timeline["customer_turns"]
    check("open timeout 3 s", ev[0]["released_by"] == "timeout_silent" and ev[0]["start_frame"] == 38, ev[0])
    check("silence timeout 4 s", ev[1]["released_by"] == "timeout_silent"
          and abs((ev[1]["start_frame"] - ev[0]["end_frame"]) / FRAME_RATE - 4.0) < 0.1, ev[1])
    check("timeouts logged", f.timeline["n_timeout_silent"] == 2)
    check("tail stops at 3 s quiet", f.timeline["stop_reason"] == "tail_quiet"
          and abs((f.timeline["frames"] - ev[1]["end_frame"]) / FRAME_RATE - 3.0) < 0.1, f.timeline["frames"])
    # 4 model never stops: barge-in at 12 s, tail cap 15 s
    m = FakeModel(greet=[], reply=[], forever=True)
    f = run([("agent", 3), ("customer", 2)], m)
    ev = f.timeline["customer_turns"]
    check("barge-in after 12 s", ev[0]["released_by"] == "barge_in" and ev[0]["start_frame"] == 150, ev[0])
    check("overlap counted", f.timeline["overlap_s"] > 1.9, f.timeline["overlap_s"])
    check("tail cap", f.timeline["stop_reason"] == "tail_cap")
    # 5 back-to-back customer turns: 0.4 s gap when the model is silent
    m = FakeModel(greet=[(10, 0, True)], reply=[], delay=100)
    f = run([("agent", 2), ("customer", 2), ("customer", 2)], m)
    ev = f.timeline["customer_turns"]
    check("b2b gap 0.4 s", ev[1]["released_by"] == "b2b" and ev[1]["start_frame"] - ev[0]["end_frame"] == 5, ev[1])
    # 6 hard cap
    m = FakeModel(greet=[], reply=[], forever=True)
    f = run([("agent", 3), ("customer", 2), ("agent", 3), ("customer", 2)], m, max_s=14.0)
    check("hard cap, unplayed logged", f.timeline["stop_reason"] == "hard_cap" and f.timeline["unplayed_turns"] == [3]
          and len(f.fed) == 175, (f.timeline["stop_reason"], f.timeline["unplayed_turns"], len(f.fed)))
    eff = f.effective_turns()
    check("unplayed flagged in effective turns", eff[3].get("unplayed") is True)
    # 7 customer clip content = the input audio of the turn (+ pad)
    audio, turns = make_call([("agent", 3), ("customer", 2)])
    clips = cut_customer_turns(audio, SR, turns)
    check("clip length = turn + 2 pads", abs(len(clips[0][1]) / SR - 2.1) < 1e-3, len(clips[0][1]) / SR)
    # 8 hold (v2): check-line, 1.5 s pause, then confirmation -> customer 0.32 s after the confirmation
    class HoldModel(FakeModel):
        pass
    seq = [(" Ek", 0.05), (" minute", 0.05), (" rukiye.", 0.05)] + [("PAD", 0.0)] * 19 + \
          [(" Ho", 0.05), (" gaya", 0.05), (" hai.", 0.05)] + [("PAD", 0.0)] * 400
    audio, turns = make_call([("agent", 3), ("customer", 2)])
    for hold, want in ((True, 3 + 19 + 3 + 4), (False, 3 + 4)):
        f = VadFeeder(cut_customer_turns(audio, SR, turns), turns, FS, hold=hold)
        for i, fr in enumerate(f.frames()):
            tok, amp = seq[i] if i < len(seq) else ("PAD", 0.0)
            f.observe_vad(frame_db(np.full(FS, amp, np.float32)), tok)
        ev = f.timeline["customer_turns"][0]
        check(f"hold={hold}: release frame", ev["start_frame"] == want and ev["released_by"] == "model_end",
              (ev, f.timeline["events"]))
    # 9 hold timeout: check-line then silence -> release after hold_timeout
    f = VadFeeder(cut_customer_turns(audio, SR, turns), turns, FS, hold=True, hold_timeout=5.0)
    for i, fr in enumerate(f.frames()):
        tok, amp = seq[i] if i < 3 else ("PAD", 0.0)
        f.observe_vad(frame_db(np.full(FS, amp, np.float32)), tok)
    ev = f.timeline["customer_turns"][0]
    check("hold timeout 5 s", ev["released_by"] == "hold_timeout" and ev["start_frame"] == 3 + 63, ev)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
