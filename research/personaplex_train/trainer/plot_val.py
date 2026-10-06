"""plot_val.py <run_dir>: val_loss.png = 4 panels (total/text/cb1/cb2-8), log-y: training curve
(metrics.csv, per-10-step rows), train-subset eval points and held-out eval points (val_metrics.csv)."""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
run = Path(sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("RUNS_ROOT", "/workspace/runs") + "/V1_A")
tr = list(csv.DictReader(open(run / "metrics.csv")))
va = list(csv.DictReader(open(run / "val_metrics.csv")))
C = {"curve": "#2a78d6", "train_subset": "#1baf7a", "heldout": "#eb6834"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
for ax, (key, title) in zip(axes.flat, [("total", "total loss"), ("text", "text loss"),
                                       ("cb1", "codebook 1 loss"), ("cb2_8_mean", "codebooks 2-8 (mean)")]):
    ax.plot([int(r["step"]) for r in tr], [float(r[key]) for r in tr], color=C["curve"], lw=1.5,
            label="training curve (metrics.csv, batch mean per 10 steps)")
    for split, lab, mk in [("train_subset", "eval: 25 train calls", "s"), ("heldout", "eval: held-out 100 calls", "o")]:
        rs = [r for r in va if r["split"] == split]
        xs, ys = [int(r["step"]) for r in rs], [float(r[key]) for r in rs]
        ax.plot(xs, ys, color=C[split], lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=lab)
        if split == "heldout":
            i = min(range(len(ys)), key=ys.__getitem__)
            ax.annotate(f"min {ys[i]:.3g} @ {xs[i]}", (xs[i], ys[i]), textcoords="offset points",
                        xytext=(8, -14), fontsize=9, color="#333333")
    ax.set_yscale("log"); ax.set_title(title, fontsize=11, color="#222222")
    ax.grid(True, which="major", alpha=0.25); ax.grid(True, which="minor", alpha=0.08)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[1]: ax.set_xlabel("step")
axes[0, 0].legend(fontsize=8.5, frameon=False, loc="lower left")
fig.suptitle(f"{run.name}: teacher-forced loss, training vs held-out (step 0 = base, zero-B)", fontsize=12)
fig.tight_layout(); fig.savefig(run / "val_loss.png", dpi=120)
print("wrote", run / "val_loss.png")
