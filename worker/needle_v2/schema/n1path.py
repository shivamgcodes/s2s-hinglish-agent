"""S2S serverless worker: copy of needle_v2/schema/n1path.py (path edits only: the search list is $N1_DIR, then the
worker's own N1 runtime subset worker/needle; the dev-machine locations are gone).
Locate the N1 code (resolver.py, build_data.py, romanise.py) and put it on sys.path.
"""
import os
import sys

_CANDS = [os.environ.get("N1_DIR"),
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "needle")]


def n1_dir():
    for d in _CANDS:
        if d and os.path.exists(os.path.join(d, "resolver.py")):
            return d
    raise FileNotFoundError("N1 code not found; set N1_DIR to the dir holding resolver.py")


N1_DIR = n1_dir()
if N1_DIR not in sys.path:
    sys.path.insert(0, N1_DIR)
