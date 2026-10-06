"""DEP1 Track 3: check-line trigger on the model's text stream (INTERFACE.md section 9).

Pure python (no deps), importable from every venv.

Segmenter: incremental copy of the hinglish repo's tests/tcommon.py segments(): a new segment starts after
>= 15 PAD/EPAD frames (i.e. frame - last_piece_frame > 15) or after a piece ending in . ? !

Trigger rule (spec Track 3): on every new piece, the CURRENT (possibly partial) segment is tested
  1. regex  : any of REGEX_PHRASES (case-insensitive, word-bounded)
  2. difflib: SequenceMatcher ratio >= 0.6 between the normalised segment and one of the 10 check-line
              templates (gen/common.py CHECK_LINES, female + male forms)
First rule that hits fires. Debounce: no trigger within DEBOUNCE_S (3 s of stream time) after the last
trigger, and at most one trigger per segment.
"""
import difflib
import re

GAP_FRAMES = 15
FRAME_S = 0.08
DEBOUNCE_S = 3.0
DIFFLIB_MIN = 0.6

# spec Track 3 list; "dekh leta/leti" and "update kar deta/deti" expanded
REGEX_PHRASES = [
    r"ek minute", r"rukiye", r"ek second", r"dekh let[ai]", r"update kar det[ai]", r"just a sec",
    r"let me check", r"checking", r"hold on",
]
_REGEX = [(p, re.compile(r"\b" + p + r"\b", re.I)) for p in REGEX_PHRASES]

# hinglish repo gen/common.py CHECK_LINES (pod1, read 2026-10-04): 5 phrasings x (female, male)
CHECK_LINES = [
    ("Ek minute rukiye, main check karke batati hoon.", "Ek minute rukiye, main check karke batata hoon."),
    ("Theek hai, ek second, main dekh leti hoon.", "Theek hai, ek second, main dekh leta hoon."),
    ("Haan ji, abhi update kar deti hoon, ek minute.", "Haan ji, abhi update kar deta hoon, ek minute."),
    ("Sure, let me check that for you, bas ek second.", "Sure, let me check that for you, bas ek second."),
    ("Checking, just a sec, ek minute.", "Checking, just a sec, ek minute."),
]
TEMPLATES = [t for pair in CHECK_LINES for t in pair]   # 10 strings (2 identical pairs, kept as in source)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


_TEMPLATES_N = [norm(t) for t in TEMPLATES]


def match(text):
    """-> (phrase, rule, score) or None for one segment text."""
    for p, rx in _REGEX:
        m = rx.search(text)
        if m:
            return m.group(0), "regex", 1.0
    n = norm(text)
    if not n:
        return None
    best, best_t = 0.0, None
    for t, tn in zip(TEMPLATES, _TEMPLATES_N):
        r = difflib.SequenceMatcher(None, n, tn).ratio()
        if r > best:
            best, best_t = r, t
    if best >= DIFFLIB_MIN:
        return best_t, "difflib", round(best, 4)
    return None


class Segmenter:
    """Feed (frame, piece) for non-PAD/EPAD tokens in frame order."""

    def __init__(self, gap_frames=GAP_FRAMES):
        self.gap = gap_frames
        self.done = []          # closed segments {start_f, end_f, text}
        self.cur = None
        self.last = -10**9

    def push(self, frame, piece):
        """Returns the current segment dict after adding the piece (same object until it closes)."""
        if self.cur is not None and frame - self.last > self.gap:
            self._close()
        if self.cur is None:
            self.cur = {"start_f": frame, "end_f": frame, "text": "", "id": len(self.done)}
        self.cur["text"] += piece
        self.cur["end_f"] = frame
        self.last = frame
        seg = self.cur
        if piece.rstrip().endswith((".", "?", "!")):
            self._close()
        return seg

    def _close(self):
        if self.cur is not None:
            self.cur["text"] = self.cur["text"].strip()
            self.done.append(self.cur)
        self.cur = None
        self.last = -10**9

    def last_segments(self, n=2):
        segs = list(self.done)
        if self.cur is not None:
            segs.append(dict(self.cur, text=self.cur["text"].strip()))
        return [s["text"].strip() for s in segs[-n:]]


class TriggerDetector:
    def __init__(self, debounce_s=DEBOUNCE_S):
        self.seg = Segmenter()
        self.debounce_s = debounce_s
        self.last_trigger_t = -1e9
        self.fired_segments = set()

    def push(self, frame, piece):
        """-> trigger dict or None. Call for every non-PAD/EPAD text event."""
        s = self.seg.push(frame, piece)
        t = frame * FRAME_S
        if s["id"] in self.fired_segments or t - self.last_trigger_t < self.debounce_s:
            return None
        m = match(s["text"])
        if not m:
            return None
        self.fired_segments.add(s["id"])
        self.last_trigger_t = t
        phrase, rule, score = m
        return {"phrase": phrase, "rule": rule, "score": score, "segment": s["text"].strip(),
                "frame": frame, "t": round(t, 2), "segment_start_f": s["start_f"]}


def triggers_from_tokens(tokens):
    """tokens = tests/out/... json list ('PAD'/'EPAD'/piece per frame) -> list of trigger dicts."""
    d = TriggerDetector()
    out = []
    for f, tok in enumerate(tokens):
        if tok in ("PAD", "EPAD"):
            continue
        tr = d.push(f, tok)
        if tr:
            out.append(tr)
    return out
