"""Held-out (and train-subset) teacher-forced loss for a PersonaPlex LoRA run, using EXACTLY the
training loss code of the patched moshi-finetune (train.py / finetune/pp_lora.py / interleaver.py).

Standalone: train.py is not modified. Run under torchrun nproc 1 (same env as the run's launch.sh)
from /workspace/moshi-finetune, under flock /workspace/hinglish/gpu.lock:

  torchrun --nproc-per-node 1 /workspace/hinglish/trainer/eval_heldout.py \
      --run_dir /workspace/runs/V1_A --steps 0,250,500,750,1000,1250,1500

What is reused from training (imported, not re-implemented):
  - TrainArgs.load(run_dir/args.yaml)  -> duration_sec, text_padding_weight, cb1 multiplier,
    dep_train_steps, voice_codes_dir, param_dtype, lora rank/scaling
  - loaders.get_mimi / wrapped_model.get_fsdp_model (PP loaders + weight-level LoRA parametrization)
  - Interleaver(keep_main_only=True), PersonaPlexPrefix, InterleavedTokenizer (prefix + prefix_len)
  - dataset.get_dataset_iterator (sphn dataset_jsonl, duration_sec windows, pad_last_segment) in its
    finite, non-shuffled mode (seq) -> deterministic windows at 0, 100, ... s
  - Batch.collate, pp_lora.forward_train_agent, pp_lora.masked_losses (prefix mask, agent rows 1-8,
    text PAD/EPAD weight, cb1 x100, per-term weight-sum normalisation)
Differences from training: torch.no_grad, model.eval() (PersonaPlex has no dropout and no
self.training branches), no activation checkpointing (it is a no-op without grad), batch of 1 window.

Adapters: the model is built once (LoRA parametrizations on 253 weights); each checkpoint's
lora_A/lora_B (bf16, as saved = the bf16 params the training forward used) is copy_'d in.
Step 0 = all lora_B zero -> W + 0 = base weights exactly.

Aggregation per (step, split):
  window-mean: mean over windows of each window's masked_losses (total = text + audio as returned
               by masked_losses, cb1 = stats['cb1'], cb2_8 = stats['cb2_8'])   [CSV main columns]
  pooled:      token-weighted over all windows (each term's numerator and weight-sum summed over
               windows, as one giant batch would)                                [CSV *_pooled]
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("MOSHI_FINETUNE", "/workspace/moshi-finetune"))

import safetensors.torch
import sentencepiece
import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from torch.nn.utils import parametrize

from finetune import pp_lora
from finetune.args import TrainArgs
from finetune.data.dataset import DataFile, get_dataset_iterator
from finetune.data.interleaver import Batch, InterleavedTokenizer, Interleaver, PersonaPlexPrefix, Sample
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

DATA = Path(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")) / "data"


def sid_of(path: str) -> str:
    stem = Path(path).stem  # e.g. food_03_g1
    return stem.rsplit("_", 1)[0]


def read_manifest(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_splits(variant: str, out_dir: Path, n_train: int) -> dict[str, Path]:
    held = DATA / variant / "heldout" / "heldout_all.jsonl"
    train = DATA / variant / "train" / "train.jsonl"
    hold = json.loads((DATA / "holdout.json").read_text())
    H, T = read_manifest(held), read_manifest(train)
    hp, tp = {r["path"] for r in H}, {r["path"] for r in T}
    hs, ts = {sid_of(p) for p in hp}, {sid_of(p) for p in tp}
    assert not (hp & tp), f"held-out/train path overlap: {sorted(hp & tp)[:5]}"
    assert not (hs & ts), f"held-out/train scenario overlap: {sorted(hs & ts)[:5]}"
    assert hs == set(hold["test_scenarios"]), "held-out scenarios != holdout.json test_scenarios"
    assert len(hs) == 25 and len(H) == 100, (len(hs), len(H))
    # Train subset: fixed rule = sorted train paths, evenly spaced indices round(i*N/n), i=0..n-1.
    Ts = sorted(T, key=lambda r: r["path"])
    idx = sorted({round(i * len(Ts) / n_train) for i in range(n_train)})
    sub = [Ts[i] for i in idx]
    assert all(r["path"] in tp for r in sub)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "train_subset.jsonl"
    sub_path.write_text("".join(json.dumps(r) + "\n" for r in sub))
    print(f"[splits] heldout {len(H)} calls / {len(hs)} scenarios; train {len(T)} calls; "
          f"overlap paths 0, scenarios 0; train_subset {len(sub)} calls -> {sub_path}", flush=True)
    return {"heldout": held, "train_subset": sub_path}


def expected_windows(manifest: Path, duration: float) -> int:
    n = 0
    for r in read_manifest(manifest):
        s = 0.0
        while s < r["duration"]:  # same rule as sphn dataset_jsonl / maybe_load_local_dataset
            n += 1
            s += duration
    return n


def cache_windows(manifest: Path, tok: InterleavedTokenizer) -> list:
    """Encode every window once with the training data path (finite, unshuffled). CPU cache."""
    out = []
    it = get_dataset_iterator(DataFile(manifest), instruct_tokenizer=tok, rank=0, world_size=1,
                              is_finite=True, seed=None, shuffle_at_epoch=False)
    for s in it:
        out.append((s.codes.cpu(), s.prefix_len))
    return out


def lora_params(model) -> dict[str, torch.nn.Module]:
    m = {}
    for mname, mod in model.named_modules():
        if parametrize.is_parametrized(mod):
            for attr, plist in mod.parametrizations.items():
                m[f"{mname}.{attr}" if mname else attr] = plist[0]
    return m


@torch.no_grad()
def load_adapter(model, lw: dict, path: str | None):
    if path is None:
        for p in lw.values():
            p.lora_B.zero_()
        return "zero-B (base)"
    sd = safetensors.torch.load_file(path)
    keys = set()
    for k, p in lw.items():
        a, b = sd[k + ".lora_A"], sd[k + ".lora_B"]
        assert a.shape == p.lora_A.shape and b.shape == p.lora_B.shape, k
        p.lora_A.copy_(a.to(p.lora_A.device, p.lora_A.dtype))
        p.lora_B.copy_(b.to(p.lora_B.device, p.lora_B.dtype))
        keys |= {k + ".lora_A", k + ".lora_B"}
    extra = set(sd) - keys
    assert not extra, f"adapter has keys not loaded (ft_embed adapters unsupported here): {sorted(extra)[:5]}"
    assert len(keys) == 2 * len(lw)
    return f"{len(lw)} LoRA pairs"


@torch.no_grad()
def eval_windows(model, windows, args, text_pad_ids):
    a0 = model.audio_offset
    tot = {k: 0.0 for k in ("total", "text", "cb1", "cb2_8")}
    pool = {"t_num": 0.0, "t_w": 0.0, "a_num": 0.0, "a_w": 0.0, "cb_num": [0.0, 0.0], "cb_n": [0.0, 0.0]}
    for codes, P in windows:
        b = Batch.collate([Sample(codes.cuda(), None, P)])
        out = pp_lora.forward_train_agent(model, b.codes, args.dep_train_steps)
        loss, text_loss, audio_loss, st = pp_lora.masked_losses(
            out, b.codes, b.prefix_len,
            first_cb_mult=args.first_codebook_weight_multiplier,
            text_padding_weight=args.text_padding_weight,
            text_pad_ids=text_pad_ids,
            dep_steps=args.dep_train_steps,
        )
        tot["total"] += loss.item()
        tot["text"] += text_loss.item()
        tot["cb1"] += st["cb1"].item()
        tot["cb2_8"] += st["cb2_8"].item()
        # pooled weights: same masks/weights masked_losses builds (for the cross-window sums only)
        T = b.codes.shape[-1]
        keep = pp_lora.prefix_keep_mask(b.prefix_len, T)
        tmask, amask = out.text_mask & keep, out.mask & keep
        tw = tmask.float()
        tgt = torch.where(tmask, b.codes[:, :1], torch.zeros_like(b.codes[:, :1]))
        for i in text_pad_ids:
            tw[tgt == i] *= args.text_padding_weight
        aw = amask.float()
        aw[:, 0] *= args.first_codebook_weight_multiplier
        pool["t_num"] += text_loss.item() * tw.sum().item(); pool["t_w"] += tw.sum().item()
        pool["a_num"] += audio_loss.item() * aw.sum().item(); pool["a_w"] += aw.sum().item()
        # pooled cb1 / cb2-8: per-window means x their token counts (each agent cb has the same count
        # up to the delay edge, so cb2_8 mean x count(cb2..8) is the pooled sum within 1 frame)
        n_cb = amask.float().sum(dim=(0, 2))
        c1, c28 = n_cb[0].item(), n_cb[1:].sum().item()
        del out
        pool["cb_num"][0] += st["cb1"].item() * c1; pool["cb_n"][0] += c1
        pool["cb_num"][1] += st["cb2_8"].item() * c28; pool["cb_n"][1] += c28
    n = len(windows)
    res = {k: v / n for k, v in tot.items()}
    tp, ap = pool["t_num"] / pool["t_w"], pool["a_num"] / pool["a_w"]
    res.update({"total_pooled": tp + ap, "text_pooled": tp,
                "cb1_pooled": pool["cb_num"][0] / pool["cb_n"][0],
                "cb2_8_pooled": pool["cb_num"][1] / pool["cb_n"][1], "n_windows": n})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.environ.get("RUNS_ROOT", "/workspace/runs") + "/V1_A")
    ap.add_argument("--variant", default="V1")
    ap.add_argument("--steps", default="0,250,500,750,1000,1250,1500")
    ap.add_argument("--splits", default="train_subset,heldout")
    ap.add_argument("--n_train", type=int, default=25)
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    out_csv = Path(a.out_csv) if a.out_csv else run_dir / "val_metrics.csv"
    t0 = time.time()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)

    args = TrainArgs.load(run_dir / "args.yaml", drop_extra_fields=False)
    assert not args.lora.ft_embed, "ft_embed adapters: embedding tables would also need loading"
    torch.manual_seed(args.seed)

    repo = args.moshi_paths.hf_repo_id
    moshi_weight = args.moshi_paths.moshi_path or hf_hub_download(repo, loaders.MOSHI_NAME)
    mimi_weight = args.moshi_paths.mimi_path or hf_hub_download(repo, loaders.MIMI_NAME)
    tok_path = args.moshi_paths.tokenizer_path or hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)

    mimi = loaders.get_mimi(mimi_weight, device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    args.gradient_checkpointing = False  # no-op under no_grad anyway
    model = get_fsdp_model(args, moshi_weight)
    model.eval()
    lw = lora_params(model)
    print(f"[model] ready {time.time() - t0:.0f} s; LoRA parametrized weights {len(lw)}", flush=True)

    spm = sentencepiece.SentencePieceProcessor(tok_path)
    interleaver = Interleaver(spm, mimi.frame_rate, model.text_padding_token_id,
                              model.end_of_text_padding_id, model.zero_token_id, keep_main_only=True,
                              keep_and_shift=getattr(args, "keep_and_shift", False))
    prefixer = PersonaPlexPrefix(spm, args.voice_codes_dir, text_padding=model.text_padding_token_id,
                                 silence_frames=int(0.5 * mimi.frame_rate))
    tok = InterleavedTokenizer(mimi, interleaver, duration_sec=args.duration_sec, prefixer=prefixer)
    text_pad_ids = {model.text_padding_token_id, model.end_of_text_padding_id}

    split_paths = build_splits(a.variant, run_dir / "val", a.n_train)
    windows = {}
    for s in a.splits.split(","):
        t = time.time()
        windows[s] = cache_windows(split_paths[s], tok)
        exp = expected_windows(split_paths[s], args.duration_sec)
        assert len(windows[s]) == exp, (s, len(windows[s]), exp)
        print(f"[data] {s}: {len(windows[s])} windows (expected {exp}), "
              f"P range {min(w[1] for w in windows[s])}-{max(w[1] for w in windows[s])}, "
              f"encoded in {time.time() - t:.0f} s", flush=True)

    final = run_dir / "lora.safetensors"
    fields = ["step", "split", "total", "text", "cb1", "cb2_8_mean", "n_windows",
              "total_pooled", "text_pooled", "cb1_pooled", "cb2_8_pooled", "adapter"]
    rows = []
    for step in [int(x) for x in a.steps.split(",")]:
        if step == 0:
            path = None
        else:
            path = str(run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated" / "lora.safetensors")
            assert os.path.exists(path), path
        desc = load_adapter(model, lw, path)
        for s in a.splits.split(","):
            t = time.time()
            r = eval_windows(model, windows[s], args, text_pad_ids)
            row = {"step": step, "split": s, "total": r["total"], "text": r["text"], "cb1": r["cb1"],
                   "cb2_8_mean": r["cb2_8"], "n_windows": r["n_windows"],
                   "total_pooled": r["total_pooled"], "text_pooled": r["text_pooled"],
                   "cb1_pooled": r["cb1_pooled"], "cb2_8_pooled": r["cb2_8_pooled"],
                   "adapter": path or "none(zero-B)"}
            rows.append(row)
            print(f"[eval] step {step:5d} {s:12s} total {r['total']:.5g} text {r['text']:.4g} "
                  f"cb1 {r['cb1']:.4g} cb2-8 {r['cb2_8']:.4g} | pooled total {r['total_pooled']:.5g} "
                  f"| {r['n_windows']} win, {time.time() - t:.0f} s ({desc})", flush=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(fields)
                for rr in rows:
                    w.writerow([f"{rr[k]:.6g}" if isinstance(rr[k], float) else rr[k] for k in fields])

    # final lora.safetensors == last checkpoint?
    last = run_dir / "checkpoints" / f"checkpoint_{args.max_steps:06d}" / "consolidated" / "lora.safetensors"
    if final.exists() and last.exists():
        h = [hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in (final, last)]
        print(f"[final] lora.safetensors sha256 {h[0]} vs checkpoint_{args.max_steps:06d} {h[1]} "
              f"-> {'IDENTICAL' if h[0] == h[1] else 'DIFFERENT'}", flush=True)
    print(f"[done] {out_csv}; wall {time.time() - t0:.0f} s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
