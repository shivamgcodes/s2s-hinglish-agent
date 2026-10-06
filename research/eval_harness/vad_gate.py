"""VAD-gated (reactive) customer for the PersonaPlex test driver (added 2026-10-05; used only with
driver.py --customer-mode vad; the fixed-timing path does not import this module).

The scripted customer turns are cut out of the test input wav (tests/inputs/<V>/<call>.wav, times from its
.meta.json, already in input time) and played in script order, reactively, inside the 80 ms frame loop:

  model "active" in a frame  = output frame level > ACT_DB dBFS (RMS of the decoded 1920-sample frame; PersonaPlex
                               silence floor is ~-74 dBFS, speech -40..-15) OR the frame's text token is a word piece
                               or EPAD (EPAD = the model is about to emit a word).
  model "stopped speaking"   = after some activity, Q consecutive inactive frames, where Q = q_sent frames if the last
                               word piece ends a sentence (. ? !) and q_mid frames otherwise (mid-sentence pauses are
                               longer than 0.3 s ~26% of the time; calibration in VAD_GATING.md).
  Next customer turn starts at the frame where "stopped" is detected (gap = Q * 80 ms, 0.32 s with q_sent = 4).

Release rules while the next customer turn waits (wait starts at call start or at the end of the previous customer
turn; only model activity after the wait start counts, except that a model already speaking at wait start counts):
  * script has customer, customer back to back -> start after b2b_gap s, unless the model is speaking: then wait for
    it to stop (same rule as below);
  * the model spoke and stopped -> start ("model_end");
  * the model never spoke for silence_timeout s (first turn: open_timeout s) -> start ("timeout_silent");
  * the model has been speaking (without a detected stop) for max_speech s since its first onset in this wait ->
    start anyway ("barge_in").
After the last customer turn: stop when the model has been silent tail_quiet s after the turn end (and after any
speech it started), or after tail_max s ("tail_cap"). Hard cap: max_s s of input frames in total ("hard_cap"; the
unplayed customer turns are logged).

v2 options (both off by default, so the v1 runs V4_A2_vad / V4_A_vad are reproducible with the defaults):
  hold=True          when the utterance the model just finished is a hold / check-line phrase (HOLD_RE = score.py's
                     CHECK_RE, regex part only: "ek minute", "rukiye", "let me check", "update kar deti hoon"...), the
                     stop does not release: the caller waits for the model to resume and stop again ("model_end"
                     after "hold"), or for hold_timeout s of silence ("hold_timeout"). v1 put the caller back in
                     0.32 s into every "ek minute, main check karti hoon" pause, so the model never confirmed writes.
  tail_match_fixed   (driver only) tail_max = the fixed input's tail after its last customer turn (input duration -
                     last scripted customer end), so the VAD run gets the same post-call time as the fixed run.

Pure numpy; unit test: tests/test_vad_gate.py (CPU, fake model).
"""
import re

import numpy as np

FRAME_RATE = 12.5
DEFAULTS = dict(act_db=-45.0, q_sent=4, q_mid=8, b2b_gap=0.4, open_timeout=3.0, silence_timeout=4.0, max_speech=12.0,
                tail_quiet=3.0, tail_max=15.0, max_s=200.0, pad_s=0.05, hold=False, hold_timeout=5.0)
# = tests/score.py CHECK_RE (metric 3 check-line regex), copied so this module stays dependency-free
HOLD_RE = re.compile(r"\b(ek (minute|min|second|sec|pal)|just a (sec|second|moment|minute)|one (sec|second|moment|"
                     r"minute)|let me (check|see|look|update|quickly)|check(ing)?\b.*\b(karke|kar)|dekh (leti|leta|"
                     r"lete|lu|loon)|rukiye|hold on|bas ek|update kar (deti|deta|dete)|give me a (sec|second|moment)|"
                     r"(i'll|i will) (check|look|update))\b")


def is_hold(text):
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()
    return bool(HOLD_RE.search(t))


def frame_db(pcm):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    return float(20 * np.log10(np.sqrt(np.mean(pcm ** 2)) + 1e-9)) if pcm.size else -200.0


def cut_customer_turns(audio, sr, turns, pad_s=0.05):
    """audio: 1-D customer channel in input time. turns: inputs meta turns. Returns [(turn, clip)] for customer turns,
    edges padded by pad_s but never past the midpoint to the neighbouring customer turn."""
    cust = [t for t in turns if t["speaker"] == "customer"]
    out = []
    for k, t in enumerate(cust):
        a, b = t["start"] - pad_s, t["end"] + pad_s
        if k > 0:
            a = max(a, (cust[k - 1]["end"] + t["start"]) / 2)
        if k + 1 < len(cust):
            b = min(b, (t["end"] + cust[k + 1]["start"]) / 2)
        i0, i1 = max(0, int(round(a * sr))), min(len(audio), int(round(b * sr)))
        out.append((t, np.asarray(audio[i0:i1], dtype=np.float32)))
    return out


class VadFeeder:
    """Same interface as the driver's feeders: frames() yields (1, frame_size) float32 frames; observe_vad() is called
    once per model step (in order) with that step's output. fed / events as in the other feeders."""

    def __init__(self, clips, turns, frame_size, **kw):
        p = dict(DEFAULTS)
        p.update({k: v for k, v in kw.items() if v is not None})
        self.p = p
        self.clips = clips                      # [(turn, clip)] customer turns in script order
        self.turns = turns
        self.frame_size = frame_size
        idx_pos = {t["idx"]: i for i, t in enumerate(turns)}
        # back-to-back: the script item right before this customer turn is a customer turn
        self.b2b = []
        for t, _ in clips:
            i = idx_pos[t["idx"]]
            self.b2b.append(i > 0 and turns[i - 1]["speaker"] == "customer")
        self.fed, self.events = [], []
        self.obs = []                            # per step: (db, token)
        self.timeline = {"customer_turns": [], "model_speech": [], "events": []}
        # model activity state
        self.quiet = 10 ** 6                     # inactive frames since last active frame
        self.last_sent_end = False               # last word piece ended a sentence
        self.cur_speech_start = None             # frame index where current model speech run began
        # wait state
        self.k = 0                               # next customer turn
        self.pos = None                          # sample position in the playing clip
        self.wait_start = 0
        self.heard = False                       # model activity seen since wait start
        self.onset = None                        # first onset frame in this wait
        self.stop_reason = None
        self.utt = ""                            # text of the model utterance in progress (since the last stop)
        self.last_utt = ""                       # text of the last finished utterance
        self.stop_frame = None
        self.hold_logged = None

    # ---- model side
    def _q(self):
        return self.p["q_sent"] if self.last_sent_end else self.p["q_mid"]

    def speaking(self):
        """Model considered speaking now: active within the last Q frames."""
        return self.quiet < self._q()

    def observe_vad(self, db, token):
        """token: text piece string, 'PAD', 'EPAD' or None (skipped step)."""
        f = len(self.obs)
        self.obs.append((db, token))
        is_piece = token is not None and token not in ("PAD", "EPAD")
        active = db > self.p["act_db"] or is_piece or token == "EPAD"
        if is_piece:
            self.last_sent_end = token.rstrip().endswith((".", "?", "!"))
            self.utt += token
        was = self.speaking()
        if active:
            self.quiet = 0
            if not was:
                self.cur_speech_start = f
            if not self.heard:
                self.heard = True
                self.onset = f
        else:
            self.quiet += 1
            if was and not self.speaking() and self.cur_speech_start is not None:
                # speech ended at the last active frame (f - quiet + 1 is the first quiet frame)
                self.timeline["model_speech"].append([self.cur_speech_start, f - self.quiet + 1])
                self.cur_speech_start = None
                self.last_utt, self.utt, self.stop_frame = self.utt, "", f - self.quiet + 1

    # ---- release logic
    def _release(self, n):
        p = self.p
        waited = (n - self.wait_start) / FRAME_RATE
        first = self.k == 0
        if self.b2b[self.k] and not self.speaking():
            return "b2b" if waited >= p["b2b_gap"] else None
        if self.heard and not self.speaking():
            if p["hold"] and self.stop_frame is not None and is_hold(self.last_utt):
                if (n - self.stop_frame) / FRAME_RATE < p["hold_timeout"]:
                    if self.hold_logged != self.stop_frame:
                        self.hold_logged = self.stop_frame
                        self.timeline["events"].append({"type": "hold", "frame": n, "text": self.last_utt.strip(),
                                                        "turn_idx": self.clips[self.k][0]["idx"]})
                    return None
                return "hold_timeout"
            return "model_end"
        if not self.heard and waited >= (p["open_timeout"] if first else p["silence_timeout"]):
            return "timeout_silent"
        if self.heard and self.speaking() and (n - self.onset) / FRAME_RATE >= p["max_speech"]:
            return "barge_in"
        return None

    def _tail_done(self, n):
        p = self.p
        waited = (n - self.wait_start) / FRAME_RATE
        if waited >= p["tail_max"]:
            return "tail_cap"
        if waited >= p["tail_quiet"] and self.quiet / FRAME_RATE >= p["tail_quiet"]:
            return "tail_quiet"
        return None

    def _start_wait(self, n):
        self.wait_start = n
        self.stop_frame = None
        self.last_utt = ""
        if not self.speaking():
            self.utt = ""
        self.heard = self.speaking()
        self.onset = n if self.heard else None

    def frames(self):
        zeros = np.zeros((1, self.frame_size), dtype=np.float32)
        max_frames = int(self.p["max_s"] * FRAME_RATE)
        while True:
            n = len(self.fed)
            assert len(self.obs) == n, "observe_vad must be called once per yielded frame"
            if n >= max_frames:
                self.stop_reason = "hard_cap"
                break
            if self.pos is None:
                if self.k < len(self.clips):
                    reason = self._release(n)
                    if reason is not None:
                        t = self.clips[self.k][0]
                        ev = {"turn_idx": t["idx"], "start_frame": n, "start_s": round(n / FRAME_RATE, 3),
                              "released_by": reason, "waited_s": round((n - self.wait_start) / FRAME_RATE, 3),
                              "model_speaking_at_start": self.speaking(),
                              "script_start_s": t["start"], "script_end_s": t["end"]}
                        self.events.append(ev)
                        self.timeline["customer_turns"].append(ev)
                        if reason in ("barge_in", "timeout_silent", "hold_timeout"):
                            self.timeline["events"].append({"type": reason, "frame": n, "turn_idx": t["idx"]})
                        self.pos = 0
                else:
                    r = self._tail_done(n)
                    if r:
                        self.stop_reason = r
                        break
            if self.pos is None:
                frame = zeros
            else:
                clip = self.clips[self.k][1]
                chunk = clip[self.pos:self.pos + self.frame_size]
                frame = np.zeros((1, self.frame_size), dtype=np.float32)
                frame[0, :len(chunk)] = chunk
                self.pos += self.frame_size
                if self.pos >= len(clip):
                    ev = self.events[-1]
                    ev["end_frame"] = n + 1
                    ev["end_s"] = round((n + 1) / FRAME_RATE, 3)
                    self.pos, self.k = None, self.k + 1
                    self._start_wait(n + 1)
            self.fed.append(frame[0].copy())
            yield frame
        self._finish()

    def _finish(self):
        n = len(self.obs)
        if self.cur_speech_start is not None:
            self.timeline["model_speech"].append([self.cur_speech_start, n - min(self.quiet, n)])
            self.cur_speech_start = None
        if self.pos is not None:  # cap hit mid-turn
            self.events[-1]["end_frame"] = n
            self.events[-1]["end_s"] = round(n / FRAME_RATE, 3)
            self.events[-1]["cut_by_cap"] = True
        played = {e["turn_idx"] for e in self.events}
        self.timeline["unplayed_turns"] = [t["idx"] for t, _ in self.clips if t["idx"] not in played]
        self.timeline["stop_reason"] = self.stop_reason
        self.timeline["frames"] = n
        self.timeline["params"] = self.p
        # overlap: customer audio frames (clip playing) while the model frame is active (energy/text rule, raw,
        # no hangover)
        cust = np.zeros(n, bool)
        for e in self.events:
            cust[e["start_frame"]:e.get("end_frame", n)] = True
        act = np.array([db > self.p["act_db"] or (tok is not None and tok != "PAD") for db, tok in self.obs], bool)
        self.timeline["overlap_s"] = round(float((cust & act).sum()) / FRAME_RATE, 2)
        self.timeline["model_active_s"] = round(float(act.sum()) / FRAME_RATE, 2)
        self.timeline["model_speech"] = [[a, b, round(a / FRAME_RATE, 2), round(b / FRAME_RATE, 2)]
                                         for a, b in self.timeline["model_speech"]]
        self.timeline["n_barge_in"] = sum(e["released_by"] == "barge_in" for e in self.events)
        self.timeline["n_timeout_silent"] = sum(e["released_by"] == "timeout_silent" for e in self.events)
        self.timeline["n_hold"] = sum(e["type"] == "hold" for e in self.timeline["events"])
        self.timeline["n_hold_timeout"] = sum(e["released_by"] == "hold_timeout" for e in self.events)

    def effective_turns(self):
        """inputs-meta turns with ACTUAL times: customer turns = played start/end (unplayed -> dropped, flagged);
        agent turn = [end of the previous played customer turn (0 if none), start of the next played customer turn
        (end of input if none)]. Used by score.py / v4_eval.py / judge.py via <run>.tmeta.json."""
        played = {e["turn_idx"]: e for e in self.events}
        end_s = round(len(self.fed) / FRAME_RATE, 3)
        out, prev_end = [], 0.0
        for i, t in enumerate(self.turns):
            t2 = dict(t)
            if t["speaker"] == "customer":
                e = played.get(t["idx"])
                if e is None:
                    t2.update({"start": end_s, "end": end_s, "unplayed": True})
                else:
                    t2.update({"start": e["start_s"], "end": e.get("end_s", end_s)})
                    prev_end = t2["end"]
            else:
                nxt = next((played[u["idx"]]["start_s"] for u in self.turns[i + 1:]
                            if u["speaker"] == "customer" and u["idx"] in played), end_s)
                t2.update({"start": prev_end, "end": nxt})
            t2["script_start"], t2["script_end"] = t["start"], t["end"]
            out.append(t2)
        return out
