"""Pick the best held-out checkpoint of a V1 run and write <run>/BEST_CKPT.
  python pick_best.py <run_dir>
Reads <run>/eval_metrics.csv, or, if absent, <run>/val_metrics.csv (split=heldout; V1_A post-hoc eval).
Split used from eval_metrics.csv: "val" if present (multi-set runs, EVAL_DATA="val=...,control=..."), else "eval"
(single-manifest runs, V1). Any other split (e.g. "control", the forgetting probe) is NEVER used for selection;
its values at the chosen step are only reported (<split>_total_pooled_at_best). Only control rows -> exit 1.
Criterion: lowest total_pooled among steps that have a checkpoint dir (checkpoints/checkpoint_XXXXXX/consolidated
with lora.safetensors). Step 0 (no checkpoint) is excluded. Also logs the minima of text_pooled and cb1_pooled
(over the same rows) and the window-mean total minimum, for reference only.
BEST_CKPT format (key=value lines): step, path, criterion, total_pooled, then the reference minima.
Exit 1 if no usable row.
"""
import csv, sys
from pathlib import Path

run = Path(sys.argv[1])
src = run / "eval_metrics.csv"
if not src.exists():
    src = run / "val_metrics.csv"
if not src.exists():
    print(f"[pick_best] {run}: no eval_metrics.csv / val_metrics.csv"); sys.exit(1)
all_rows = list(csv.DictReader(open(src)))
splits = list(dict.fromkeys(r["split"] for r in all_rows))
if src.name == "val_metrics.csv":
    split = "heldout"
elif "val" in splits:
    split = "val"
elif "eval" in splits:
    split = "eval"
else:
    print(f"[pick_best] {run}: no 'val' or 'eval' split in {src} (found {splits}); "
          "refusing to select on an observation-only split"); sys.exit(1)

rows = []
for r in all_rows:
    if r["split"] != split:
        continue
    step = int(r["step"])
    ck = run / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
    if not (ck / "lora.safetensors").exists():
        continue
    rows.append({"step": step, "path": str(ck), **{k: float(r[k]) for k in
                 ("total", "text", "cb1", "total_pooled", "text_pooled", "cb1_pooled")}})
if not rows:
    print(f"[pick_best] {run}: no eval row with a checkpoint ({src})"); sys.exit(1)

best = min(rows, key=lambda r: r["total_pooled"])
mt = min(rows, key=lambda r: r["text_pooled"])
mc = min(rows, key=lambda r: r["cb1_pooled"])
mw = min(rows, key=lambda r: r["total"])
lines = [
    f"step={best['step']}",
    f"path={best['path']}",
    f"criterion=lowest held-out total_pooled ({src.name}, split={split}, {len(rows)} checkpointed eval rows)",
    f"total_pooled={best['total_pooled']:.5f}",
    f"text_pooled_at_best={best['text_pooled']:.5f}",
    f"cb1_pooled_at_best={best['cb1_pooled']:.5f}",
    f"min_text_pooled={mt['text_pooled']:.5f}@{mt['step']}",
    f"min_cb1_pooled={mc['cb1_pooled']:.5f}@{mc['step']}",
    f"min_total_windowmean={mw['total']:.5f}@{mw['step']}",
]
# Observation-only splits (e.g. control): value at the chosen step, and their own minimum, for reference.
for o in splits:
    if o == split or src.name == "val_metrics.csv":
        continue
    orows = {int(r["step"]): float(r["total_pooled"]) for r in all_rows if r["split"] == o}
    if orows:
        ms = min(orows, key=orows.get)
        at = f"{orows[best['step']]:.5f}" if best["step"] in orows else "NA"
        lines.append(f"{o}_total_pooled_at_best={at}")
        lines.append(f"{o}_min_total_pooled={orows[ms]:.5f}@{ms} (not used for selection)")
        if 0 in orows:
            lines.append(f"{o}_total_pooled_step0={orows[0]:.5f}")
(run / "BEST_CKPT").write_text("\n".join(lines) + "\n")
print(f"[pick_best] {run.name}: " + " ".join(lines[:1] + lines[3:]))
