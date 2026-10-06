"""Val loss of base Needle 3 and of LoRA adapters on val.jsonl (by-scenario val, F-section).

Uses the exact loss of needle.model.finetune.finetune_local: CQ W4 STE weights + A8
activations (configure_deploy), masked next-token cross-entropy on the answer span.
Reports two aggregates:
  batch_mean  -- mean over batches of 16 of the per-batch token-mean (what the CLI prints as `val`)
  token_mean  -- total masked CE / total masked tokens over all val rows
Usage: python val_loss.py --checkpoint CKPT --data val.jsonl [--adapter a.safetensors ...] [--base]
"""
import argparse, json, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", nargs="*", default=[])
    ap.add_argument("--base", action="store_true", help="also score the base model (no LoRA)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax, jax.numpy as jnp, optax
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS
    from needle.model.tokenizer import get_tokenizer
    from needle.model.finetune import fit_max_len, load_jsonl, merge_lora
    from needle.model.checkpoints import read_adapter

    params, config = load_checkpoint(args.checkpoint)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params))
    tok = get_tokenizer(config.vocab_size)
    max_len = fit_max_len(args.data, tok, args.max_len)
    seqs, masks = load_jsonl(args.data, tok, max_len)
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))

    def sums(lora, scale, ids, mask):
        merged = cq_ste_params(merge_lora(params, lora, scale) if lora else params, WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum(), mask.sum()

    step = jax.jit(sums, static_argnums=(1,))
    results = {"data": args.data, "rows": int(len(seqs)), "seq_len": int(max_len),
               "batch_size": args.batch_size, "numerics": f"CQ W{WEIGHT_BITS} STE + A8", "models": {}}

    def score(name, lora, scale):
        t0 = time.time()
        batch_means, tot, n = [], 0.0, 0.0
        for i in range(0, len(seqs), args.batch_size):
            s, c = step(lora, scale, jnp.asarray(seqs[i:i + args.batch_size]),
                        jnp.asarray(masks[i:i + args.batch_size]))
            s, c = float(s), float(c)
            batch_means.append(s / max(c, 1.0)); tot += s; n += c
        r = {"batch_mean": float(np.mean(batch_means)), "token_mean": tot / n,
             "tokens": n, "seconds": round(time.time() - t0, 1)}
        results["models"][name] = r
        print(name, json.dumps(r), flush=True)

    if args.base:
        score("base", {}, 1.0)
    for path in args.adapter:
        ad = read_adapter(path)
        lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                for k, v in ad["lora"].items()}
        score(path, lora, float(ad["scale"]))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
