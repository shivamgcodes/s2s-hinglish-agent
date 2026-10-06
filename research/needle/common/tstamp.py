"""Prefix each stdin line with seconds since start (for per-step timing)."""
import sys, time
t0 = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[{time.time() - t0:8.1f}s] {line}"); sys.stdout.flush()
