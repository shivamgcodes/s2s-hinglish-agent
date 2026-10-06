"""S2S serverless worker: static check of worker/Dockerfile against a build context (no docker needed).

  python3 deploy/worker/tests/check_dockerfile.py [--root <context root, default repo root>]

Checks: every COPY source exists under the context root and is not excluded by worker/Dockerfile.dockerignore
(Docker semantics approximated: patterns in order, last match wins, '!' re-includes, a pattern matching a parent dir
excludes its files); ENTRYPOINT/stack scripts are copied; no secret-looking ENV/ARG; the base image tag; and the
expected worker/build layout (DESIGN Amendment W2). 2026-10-06 D-BAKE (GitHub Actions + BuildKit): every RUN that
reads /run/secrets/hf_token must mount it as a secret and must not use `set -x`; HF_TOKEN never in ARG/ENV; the only
--mount allowed is type=secret; the weights land where paths.py / resolve_models.py look (S2S_PP_DIR=/opt/pp,
S2S_ASSETS=/opt/s2s/assets). 2026-10-07 (D-MONOREPO-LAYOUT): the context is the repo root with deploy/ + packages/;
research/ and the rest of deploy/ must stay out; the shared packages must be COPY'd. Exit 1 on any problem.
"""
import argparse
import fnmatch
import re
import sys
from pathlib import Path

W = Path(__file__).resolve().parent.parent          # deploy/worker
REPO = W.parent.parent                               # repo root (the build context)
# 2026-10-06: no worker/build/ any more (code-only GitHub build; assets are fetched at start by fetch_assets.py)
BUILD_FILES = ["deploy/common/data/records_v4.json", "deploy/common/s2s_token.py", "deploy/common/session.py",
               "deploy/worker/vendor/moshi/pyproject.toml", "deploy/worker/fetch_assets.py", "deploy/worker/assets_manifest.json",
               "packages/personaplex_lora/role_prompt.py", "packages/needle_router/v2/router_v2.py",
               "packages/needle_router/n1/router.py", "packages/needle_router/numconv/numconv.py",
               "packages/personaplex_lora/trainer/merge_lora.py", "packages/personaplex_lora/infer/lora_merge.py"]
MAX_REPO_FILE = 95 * 2**20      # GitHub rejects files > 100 MB


def load_ignore(p):
    pats = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        neg = s.startswith("!")
        s = s[1:] if neg else s
        pats.append((neg, s.rstrip("/")))
    return pats


def _match(pat, path):
    if "**" in pat:
        rx = re.escape(pat).replace(r"\*\*/", "(.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        return re.fullmatch(rx, path) is not None
    return fnmatch.fnmatchcase(path, pat) and path.count("/") == pat.count("/")


def ignored(pats, rel):
    parts = rel.split("/")
    excluded = False
    for neg, pat in pats:
        for i in range(1, len(parts) + 1):     # the path itself or any parent dir
            if _match(pat, "/".join(parts[:i])):
                excluded = not neg
                break
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--dockerfile", default=str(W / "Dockerfile"))
    ap.add_argument("--ignore", default=str(W / "Dockerfile.dockerignore"))
    a = ap.parse_args()
    root = Path(a.root)
    df = Path(a.dockerfile).read_text()
    pats = load_ignore(Path(a.ignore))
    problems = []
    lines = re.sub(r"\\\n", " ", df).splitlines()
    copies = [ln.split()[1:-1] for ln in lines if ln.startswith("COPY ") and "--from" not in ln]
    for srcs in copies:
        for s in srcs:
            p = root / s
            if not p.exists():
                problems.append(f"COPY source missing in context: {s}")
                continue
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            kept = [f for f in files if not ignored(pats, str(f.relative_to(root)))]
            if not kept:
                problems.append(f"COPY source fully excluded by dockerignore: {s}")
            bad = [str(f.relative_to(root)) for f in kept if "__pycache__" in f.parts]
            if bad:
                problems.append(f"pycache would be copied: {bad[:3]}")
    for f in BUILD_FILES:
        if not (root / f).exists():
            problems.append(f"build context file missing: {f}")
        elif ignored(pats, f):
            problems.append(f"build context file excluded by dockerignore: {f}")
    for f in ("deploy/worker/local/runpod2.env", "deploy/worker/tests/test_worker_mock.py", "deploy/client/package.json",
              "deploy/space/app/main.py", "deploy/run_local.sh", "docs/DECISIONS.md", "deploy/ops/push_space.sh",
              "deploy/tests/test_token.py", ".github/workflows/build-worker.yml", "deploy/common/data/scripts_v4.json",
              "packages/hinglish_text/hindi_share.py", "packages/selftest.py", "research/README.md",
              "research/eval_harness/driver.py", "research/needle/README.md"):   # D-MONOREPO(-LAYOUT)
        if (root / f).exists() and not ignored(pats, f):
            problems.append(f"should be excluded from the context: {f}")
    if re.search(r"(?im)^\s*(ENV|ARG)\s+[^\n]*(TOKEN|SECRET|API_KEY|PASSWORD)\s*=", df):
        problems.append("a secret-looking ENV/ARG is set in the Dockerfile")
    m = re.search(r"ARG CUDA_BASE=(\S+)", df)
    print("base image:", m.group(1) if m else "?")
    # D-BAKE (2026-10-06): built by GitHub Actions with BuildKit; the HF token is a build SECRET
    runs = [ln for ln in lines if ln.startswith("RUN")]
    for ln in runs:
        mounts = re.findall(r"--mount=(\S+)", ln)
        for m in mounts:
            if not m.startswith("type=secret,"):
                problems.append(f"RUN --mount other than type=secret: {m}")
        if "/run/secrets/hf_token" in ln:
            if not any("id=hf_token" in m for m in mounts):
                problems.append("a RUN reads /run/secrets/hf_token without --mount=type=secret,id=hf_token")
            if re.search(r"set -[a-z]*x", ln) or "set -o xtrace" in ln:
                problems.append("a RUN that reads the HF token uses set -x (would echo the token into the build log)")
    if not any("/run/secrets/hf_token" in ln for ln in runs):
        problems.append("no RUN reads the hf_token build secret (weights would not be baked)")
    if any(re.search(r"--mount=type=secret", ln) for ln in runs) and not df.startswith("# syntax=docker/dockerfile:1"):
        problems.append("RUN --mount=type=secret used but the first line is not '# syntax=docker/dockerfile:1.x'")
    if re.search(r"(?im)^\s*(ARG|ENV)\b[^\n]*HF_TOKEN", df):
        problems.append("HF_TOKEN appears in an ARG/ENV line (must be a BuildKit secret only)")
    for k, v in (("S2S_PP_DIR", "/opt/pp"), ("S2S_ASSETS", "/opt/s2s/assets"), ("HF_HUB_OFFLINE", "1")):
        if not re.search(rf"(?m)^\s*(ENV\s+)?{k}={re.escape(v)}\b", df):
            problems.append(f"runtime ENV {k}={v} missing")
    if "worker/build" in "\n".join(ln for ln in lines if ln.startswith("COPY ")):
        problems.append("COPY from worker/build: assets must not be baked (code-only repo)")
    for f in root.rglob("*"):
        if f.is_file() and ".git" not in f.parts and "node_modules" not in f.parts and ".venv" not in str(f) \
                and f.stat().st_size > MAX_REPO_FILE:
            problems.append(f"file too big for GitHub: {f.relative_to(root)} ({f.stat().st_size / 2**20:.0f} MiB)")
    for s in ("entrypoint.sh", "stack.sh", "rp_handler.py", "resolve_models.py", "paths.py", "fetch_assets.py",
              "assets_manifest.json", "packages/needle_router", "packages/personaplex_lora"):
        if not any(s in " ".join(c) for c in copies):
            problems.append(f"{s} is not copied into the image")
    print("COPY lines:", len(copies))
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
