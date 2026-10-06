"""N2 think-block fix: add a short `reasoning` line to v2 rows so the fine-tune trains the path the engine decodes
(<think>\n{reasoning}\n</think>\n<tool_call>...). Guide format (GUIDE_NOTES §1): one short line deriving each argument
from its span in the query, e.g. "'kitchen' -> room; 'to 10' -> brightness". Here:
    "<tool>: '<query span>' -> <arg>; ..."        arg value found (exactly or fuzzily) in the query
    "<tool>: <arg> from record"                   REF arg the customer did not say (canonical = record primary)
Span search: compact match first (value with spaces removed inside the space-removed query, e.g. phone / ID), else
the query token window with the best difflib ratio to the value (window length len(value)±3); ratio < 0.55 ->
"from record" for REF args, else the value itself is quoted (target stays the gold value either way).
Usage: python add_reasoning.py IN.jsonl IN_meta.jsonl OUT.jsonl
"""
import difflib, json, os, re, sys

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[4]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
import tools_v2  # noqa: E402

MAXW = 12


def toks(s):
    return re.findall(r"\S+", s)


def norm(s):
    return re.sub(r"[^a-z0-9@.]+", " ", str(s).lower()).strip()


def find_span(value, query):
    v, q = norm(value), toks(query)
    if not v:
        return None, 0.0
    vc = v.replace(" ", "")
    qn = [norm(t) for t in q]
    # compact (digits / IDs / emails): smallest token window whose compact form contains the value
    for L in range(1, min(MAXW, len(q)) + 1):
        for i in range(len(q) - L + 1):
            if vc and vc in "".join(qn[i:i + L]).replace(" ", ""):
                return " ".join(q[i:i + L]).strip(" ,.?!"), 1.0
    n = len(v.split())
    best, bspan = 0.0, None
    for L in range(max(1, n - 3), min(len(q), n + 3) + 1):
        for i in range(len(q) - L + 1):
            r = difflib.SequenceMatcher(None, v, " ".join(qn[i:i + L])).ratio()
            if r > best:
                best, bspan = r, " ".join(q[i:i + L]).strip(" ,.?!")
    return bspan, best


def reasoning_for(row, meta):
    at = meta["agent_type"]
    (ans,) = row["answers"]
    kinds = {m: k for m, k, _ in tools_v2.arg_specs(at, ans["name"])}
    parts = []
    for arg, val in ans["arguments"].items():
        kind = kinds.get(arg)
        span, r = find_span(val, row["query"])
        if span and r >= 0.55:
            parts.append(f"'{span}' -> {arg}")
        elif kind in tools_v2.REF_KINDS:
            parts.append(f"{arg} from record")
        else:
            parts.append(f"{arg} '{val}'")
    return f"{ans['name']}: " + "; ".join(parts) if parts else ans["name"]


def main():
    src, msrc, dst = sys.argv[1:4]
    rows = [json.loads(l) for l in open(src)]
    metas = [json.loads(l) for l in open(msrc)]
    assert len(rows) == len(metas)
    with open(dst, "w") as f:
        for r, m in zip(rows, metas):
            f.write(json.dumps({**r, "reasoning": reasoning_for(r, m)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
