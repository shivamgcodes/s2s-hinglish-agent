"""Shared helpers for /workspace/hinglish/tests (pure python; no GPU)."""
import json
import re
import unicodedata
import os
import sys
from pathlib import Path

_REPO = __import__('pathlib').Path(__file__).resolve().parents[2]  # monorepo root (holds packages/ and research/)
__import__('sys').path.insert(0, __import__('os').environ.get('PP_LORA_PKG') or str(_REPO / 'packages/personaplex_lora'))
from role_prompt import _ASCII_MAP, ascii_prompt  # noqa: E402,F401  single source (packages/personaplex_lora)

ROOT = Path(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish"))
TESTS = ROOT / "tests"
DATA = ROOT / "data"
OUT = TESTS / "out"
INPUTS = TESTS / "inputs"
HOLDOUT = DATA / "holdout.json"
SR = 24000
FRAME_RATE = 12.5
LEAD_DATA = 1.0      # assemble.py lead silence
LEAD_TEST = 2.0      # spec section 6: 2 s leading silence in the test input
TAIL_TEST = 4.0      # extra silence at the end so the model can sign off
SHIFT = LEAD_TEST - LEAD_DATA  # add to every assemble.py timeline time
SEEDS = [1001, 1002, 1003]
VOICE = {"f": "NATF2.pt", "m": "NATM1.pt"}


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.rename(p)


def test_meta_path(variant, call_id, run_meta_path=None):
    """Test meta for scoring one run: <run>.tmeta.json (written by driver.py --customer-mode vad: the inputs meta with
    the ACTUAL played turn times) when it exists, else tests/inputs/<variant>/<call_id>.meta.json (fixed timing,
    unchanged)."""
    if run_meta_path is not None:
        p = Path(str(run_meta_path)[:-len(".meta.json")] + ".tmeta.json")
        if p.exists():
            return p
    return INPUTS / variant / f"{call_id}.meta.json"


def load_calls(path):
    return {c["call_id"]: c for c in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def text_from_tokens(tokens):
    return "".join(t for t in tokens if t not in ("PAD", "EPAD")).strip()


def segments(tokens, gap_frames=15):
    """Group text pieces into utterance segments: a new segment starts after >= gap_frames PAD/EPAD frames
    or after a piece ending in . ? !  Returns [{start_f, end_f, text}] (end_f = index of last piece)."""
    segs, cur, last = [], None, -10**9
    for i, t in enumerate(tokens):
        if t in ("PAD", "EPAD"):
            continue
        if cur is None or i - last > gap_frames:
            if cur:
                segs.append(cur)
            cur = {"start_f": i, "end_f": i, "text": ""}
        cur["text"] += t
        cur["end_f"] = i
        last = i
        if t.rstrip().endswith((".", "?", "!")):
            segs.append(cur)
            cur = None
            last = -10**9
    if cur:
        segs.append(cur)
    for s in segs:
        s["text"] = s["text"].strip()
    return segs
