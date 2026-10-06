"""Static allowlist check of deploy/worker/assets_manifest.json (D-ASSET-SWITCH 2026-10-07; no network, no token).

  python3 deploy/worker/tests/check_manifest.py [--manifest PATH]

The image bakes ONLY the manifest's files (worker/fetch_assets.py enforces it at build time; this check runs in CI before
the build and in run_all_cpu_tests.sh). Rules:
  - every entry's repo is one of ALLOWED_REPOS (the two model repos + the public Needle engine repo); the retired
    private bundle shivamgupta/s2s-v4-assets is refused
  - revision = a full 40-hex commit (no branch names), md5 = 32 hex, size = positive int, required = true
  - path: relative, no '..', not under any forbidden prefix; the manifest's forbidden_prefixes contain at least
    REQUIRED_FORBIDDEN (training checkpoints / sweep folders must never be baked)
  - dest: unique, relative, no '..'; an "extract" entry also pins download_size + download_md5
  - the expected set of (repo, path) pairs is exactly EXPECTED (a new file must be added here on purpose)
Exit 1 on any problem.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOWED_REPOS = {"shivamgupta/personaplex-hinglish-v4-lora", "shivamgupta/needle-hinglish-router-v2", "Cactus-Compute/needle3"}
REQUIRED_FORBIDDEN = {"V4_A/", "V4_A2/", "V3_A/", "checkpoints/"}
EXPECTED = {
    ("shivamgupta/personaplex-hinglish-v4-lora", "config.json"),
    ("shivamgupta/personaplex-hinglish-v4-lora", "lora.safetensors"),
    ("shivamgupta/needle-hinglish-router-v2", "tuned_full.cact"),
    ("shivamgupta/needle-hinglish-router-v2", "n1/tuned_full.cact"),
    ("Cactus-Compute/needle3", "python/cactus_needle-3.0.2-py3-none-manylinux2014_x86_64.whl"),
}
HEX40, HEX32 = re.compile(r"^[0-9a-f]{40}$"), re.compile(r"^[0-9a-f]{32}$")


def check(m):
    problems = []
    forb = m.get("forbidden_prefixes", [])
    if not REQUIRED_FORBIDDEN <= set(forb):
        problems.append(f"forbidden_prefixes must include {sorted(REQUIRED_FORBIDDEN - set(forb))}")
    dests, pairs = set(), set()
    for e in m.get("files", []):
        n = f"{e.get('repo')}:{e.get('path')}"
        if e.get("repo") not in ALLOWED_REPOS:
            problems.append(f"{n}: repo not allowed")
        if not HEX40.match(str(e.get("revision", ""))):
            problems.append(f"{n}: revision must be a 40-hex commit")
        p = str(e.get("path", ""))
        if not p or p.startswith("/") or ".." in p.split("/"):
            problems.append(f"{n}: bad path")
        if any(p.startswith(f) for f in forb):
            problems.append(f"{n}: path under a forbidden prefix")
        if not HEX32.match(str(e.get("md5", ""))) or not isinstance(e.get("size"), int) or e["size"] <= 0:
            problems.append(f"{n}: md5/size missing or malformed")
        if e.get("required") is not True:
            problems.append(f"{n}: must be required=true (an optional entry's mismatch would pass --check silently)")
        d = str(e.get("dest", ""))
        if not d or d.startswith("/") or ".." in d.split("/") or d in dests:
            problems.append(f"{n}: bad or duplicate dest {d!r}")
        dests.add(d)
        if e.get("extract") and (not HEX32.match(str(e.get("download_md5", ""))) or not isinstance(e.get("download_size"), int)):
            problems.append(f"{n}: extract entry needs download_size + download_md5")
        pairs.add((e.get("repo"), p))
    if pairs != EXPECTED:
        problems.append(f"file set differs from EXPECTED: extra {sorted(pairs - EXPECTED)}, missing {sorted(EXPECTED - pairs)}")
    if "s2s-v4-assets" in json.dumps(m.get("files", [])):
        problems.append("the retired private repo shivamgupta/s2s-v4-assets is referenced by an entry")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "assets_manifest.json"))
    a = ap.parse_args()
    m = json.loads(Path(a.manifest).read_text())
    problems = check(m)
    for f in m.get("files", []):
        print(f"  {f['repo']}@{f['revision'][:8]}:{f['path']}  {f['size']} B  md5 {f['md5']}")
    for p in problems:
        print("PROBLEM:", p)
    print("manifest allowlist OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
