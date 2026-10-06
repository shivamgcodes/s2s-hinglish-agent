"""Evaluate numconv on mined V4 Trelis chunk transcripts (data/chunks.jsonl; built from pod1
/workspace/hinglish/data/V4/work/chunks/*/*.asr.json, highest try per chunk, speaker from calls.jsonl).

Split: 'heldout' = V4 holdout test_calls + val_calls (90 calls; rules were NOT tuned on these), 'dev' = the rest.
Metrics per split, on customer chunks (primary) and all chunks:
  number-level precision / recall: multiset of digit strings extracted (numconv.extract_numbers) from the converted
      Trelis text vs from the reference text_roman;
  chunk exact: chunks with numbers in ref whose digit multiset matches exactly;
  false-conversion: chunks with NO digits in ref whose converted text contains digits;
  phone recall (10-digit ref numbers), ID recall (ref tokens like FD4124 matched as the exact token string);
  same metrics on the raw Trelis text (baseline, no conversion);
  ASR-ok subset: chunks where V4's trelis_m1 num_ok is True (Trelis heard the spoken number words right) ->
      the converter's own accuracy with ASR mishears removed.
Usage: python3 eval_numconv.py [--dump failures.jsonl]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent  # data/, holdout_calls.json, eval_results.json are read/written here (as before)
import os  # noqa: E402  (monorepo shim: numconv.py lives in packages/needle_router/numconv, $NUMCONV_DIR)
sys.path.insert(0, os.environ.get("NUMCONV_DIR", str(HERE.parents[3] / "packages" / "needle_router" / "numconv")))
from numconv import convert, extract_numbers  # noqa: E402

ID_RE = re.compile(r"\b([A-Z]{1,4})-?(\d{3,})\b")


def ids(text):
    return Counter(a + b for a, b in ID_RE.findall(text))


def stats(rows, conv):
    tp = fp = fn = 0
    ch_num = ch_ok = ch_none = ch_false = 0
    ph_tot = ph_hit = id_tot = id_hit = 0
    for r in rows:
        hyp = conv(r["text"] or "")
        ref = r["ref"] or ""
        h, g = Counter(extract_numbers(hyp)), Counter(extract_numbers(ref))
        inter = sum((h & g).values())
        tp += inter
        fp += sum(h.values()) - inter
        fn += sum(g.values()) - inter
        if g:
            ch_num += 1
            ch_ok += h == g
        else:
            ch_none += 1
            ch_false += bool(h)
        for num, c in g.items():
            if len(num) == 10:
                ph_tot += c
                ph_hit += min(c, h.get(num, 0))
        gi, hi = ids(ref), ids(hyp)
        id_tot += sum(gi.values())
        id_hit += sum((gi & hi).values())
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return {"chunks": len(rows), "num_chunks": ch_num, "P": round(p, 4), "R": round(rc, 4),
            "F1": round(2 * p * rc / (p + rc), 4) if p + rc else 0,
            "chunk_exact": f"{ch_ok}/{ch_num} ({ch_ok / max(ch_num, 1):.1%})",
            "false_conv": f"{ch_false}/{ch_none} ({ch_false / max(ch_none, 1):.2%})",
            "phone_R": f"{ph_hit}/{ph_tot}", "id_R": f"{id_hit}/{id_tot}"}


def main():
    rows = [json.loads(l) for l in open(HERE / "data" / "chunks.jsonl")]
    hold = json.load(open(HERE / "holdout_calls.json"))
    held = set(hold["test_calls"]) | set(hold["val_calls"])
    res = {}
    for split in ("heldout", "dev"):
        sr = [r for r in rows if (r["call_id"] in held) == (split == "heldout")]
        for who in ("customer", "all"):
            wr = [r for r in sr if who == "all" or r["speaker"] == who]
            res[f"{split}/{who}/numconv"] = stats(wr, convert)
            res[f"{split}/{who}/raw_baseline"] = stats(wr, lambda x: x)
            ok = [r for r in wr if r.get("num_ok") is True or not re.search(r"\d", r["ref"] or "")]
            res[f"{split}/{who}/numconv_asr_ok"] = stats(ok, convert)
    for k, v in res.items():
        print(f"{k:38s}", json.dumps(v, ensure_ascii=False))
    json.dump(res, open(HERE / "eval_results.json", "w"), indent=1, ensure_ascii=False)
    if "--dump" in sys.argv:
        out = open(sys.argv[sys.argv.index("--dump") + 1], "w")
        for r in rows:
            hyp = convert(r["text"] or "")
            h, g = Counter(extract_numbers(hyp)), Counter(extract_numbers(r["ref"] or ""))
            if h != g:
                out.write(json.dumps({"call_id": r["call_id"], "speaker": r["speaker"], "heldout": r["call_id"] in held,
                                      "num_ok": r.get("num_ok"), "ref": r["ref"], "trelis": r["text"], "conv": hyp,
                                      "ref_nums": sorted(g.elements()), "hyp_nums": sorted(h.elements())},
                                     ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
