"""Locate the N1 code (resolver.py, build_data.py, romanise.py) and put it on sys.path.

Search order: $N1_DIR, the sibling ../n1 (packages/needle_router layout), ../../needle (HF repo layout).
The N1 modules are pure python (no GPU). Single source: packages/needle_router (D-SINGLE-SOURCE / D-LEAN-HF 2026-10-07; used in place by deploy/worker +
research/needle, pip-installable as `needle_router`; the HF model repo holds no code). Origin: hinglish/needle_v2/schema/n1path.py (path
edits only: the dev-machine locations runpod2 /root/n2/needle_n1, pod1 /workspace/hinglish/needle and the laptop are gone).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.environ.get("N1_DIR"), os.path.join(os.path.dirname(_HERE), "n1"),
          os.path.join(os.path.dirname(os.path.dirname(_HERE)), "needle")]


def n1_dir():
    for d in _CANDS:
        if d and os.path.exists(os.path.join(d, "resolver.py")):
            return d
    raise FileNotFoundError("N1 code not found; set N1_DIR to the dir holding resolver.py")


N1_DIR = n1_dir()
if N1_DIR not in sys.path:
    sys.path.insert(0, N1_DIR)
