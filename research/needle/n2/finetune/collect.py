"""N2 selection table: engine-decoded val (evals/engine_val_<T>.json) + val loss (sweep/val_losses*.json) + train
time / VRAM (sweep/train_<T>.log TIMED, smi_sweep.csv) -> evals/selection.json and a markdown table on stdout."""
import glob, json, os, re

F = os.environ.get("N2_ROOT", "/root/n2") + "/finetune"  # work dir (evals/, sweep/)
tags = [t for t in ["A_e3", "A_e6", "A_e10", "B_e3", "B_e6", "B_e10", "R_e3", "R_e6", "R_e10", "R_e15", "tuned_l8"]
        if os.path.exists(f"{F}/evals/engine_val_{t}.json")]
vl = {}
for f, render in ((f"{F}/sweep/val_losses_val.json", "plain"), (f"{F}/sweep/val_losses_R_val.json", "reasoning"),
                 (f"{F}/sweep/val_losses_R_val_e15.json", "reasoning")):
    if os.path.exists(f):
        for k, v in json.load(open(f))["models"].items():
            vl[(render, k.replace("adapter_", "").replace(".safetensors", ""))] = v["token_mean"]
rows = []
for t in tags:
    s = json.load(open(f"{F}/evals/engine_val_{t}.json"))["summary"]
    a = s["all"]
    tl = f"{F}/sweep/train_{t}.log"
    tim = None
    if os.path.exists(tl):
        m = re.search(r"TIMED (\{.*\})", open(tl).read())
        tim = json.loads(m.group(1)) if m else None
    render = "reasoning" if t.startswith("R") or t == "tuned_l8" else "plain"
    rows.append({"tag": t, "train_set": {"A": "train", "B": "train_grounded", "R": "train_grounded+reasoning", "t": "R_e15 at 8 layers"}.get(t[0], "-"),
                 "val_loss": vl.get((render, t)), "val_loss_render": render,
                 **{k: a[k] for k in ("n", "tool_match", "correct", "ask", "silent_wrong", "silent_wrong_phone",
                                      "silent_wrong_email", "missed", "multi_call")},
                 "correct_clean": s["clean"]["correct"], "correct_opus": s["opus"]["correct"],
                 "fail_groups": a["fail_groups"], "train_wall_s": tim and tim["wall_s"],
                 "train_peak_rss_gb": tim and tim["peak_rss_gb"], "engine_eval_wall_s": s["wall_s"]})
json.dump(rows, open(f"{F}/evals/selection.json", "w"), indent=1)
hdr = ["tag", "train_set", "val_loss", "tool_match", "correct", "correct_clean", "correct_opus", "ask", "silent_wrong",
       "silent_wrong_phone", "silent_wrong_email", "missed", "train_wall_s"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(str(round(r[h], 4) if isinstance(r[h], float) else r[h]) for h in hdr) + " |")
