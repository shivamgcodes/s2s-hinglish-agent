"""Diagnostic (REPORT §open questions): does the e10 LoRA say [] on negatives when decoded with the
TRAINING render (no <think>), and does it call a tool when the assistant turn starts with <think>?

Greedy JAX decode, CQ W4 numerics (same as finetune._score_quantised), on the first N negatives and
first N positives of test_heldout_a. Two prompts per row:
  plain : render_example(row) prompt, as in training (target begins with <tool_call>)
  think : the same prompt + "<think>\n" (what the engine's output shape suggests: every engine response
          carries a reasoning string)
Usage (pod2): python diag_think.py --checkpoint CKPT --adapter adapter_e10.safetensors --n 30 --out diag_think.json
"""
import argparse, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", default=os.environ.get("N1_ROOT", "/root/needle") + "/test_heldout_a")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import jax
    from needle.model.run import load_checkpoint, generate
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS
    from needle.model.tokenizer import get_tokenizer, THINK_START
    from needle.model.finetune import render_example, merge_lora, _calls_of
    from needle.model.checkpoints import read_adapter
    import inspect

    params, config = load_checkpoint(a.checkpoint)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda x: np.asarray(x).astype(np.float32), params))
    tok = get_tokenizer(config.vocab_size)
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    ad = read_adapter(a.adapter)
    import jax.numpy as jnp
    lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])} for k, v in ad["lora"].items()}
    scale = float(ad["scale"])  # same loading as val_loss.py
    merged = cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)

    rows = [json.loads(l) for l in open(a.test + ".jsonl")]
    meta = [json.loads(l) for l in open(a.test + "_meta.jsonl")]
    neg = [i for i, m in enumerate(meta) if m["type"] == "negative"][:a.n]
    pos = [i for i, m in enumerate(meta) if m["type"] != "negative"][:a.n]
    out = []
    t0 = time.time()
    for i in neg + pos:
        r = rows[i]
        prompt, _ = render_example({**r, "answers": []})
        rec = {"i": i, "type": meta[i]["type"], "query": r["query"], "gold": r["answers"]}
        for name, p in (("plain", prompt), ("think", prompt + THINK_START + "\n")):
            text = generate(model, merged, tok, p, max_new_tokens=160, temperature=0.0, stream=False)
            text = text if isinstance(text, str) else tok.decode(text)
            rec[name] = text
            rec[name + "_calls"] = _calls_of(text)
        out.append(rec)
        print(i, rec["type"], "plain:", rec["plain"][:90].replace("\n", " "), "| think:", rec["think"][-90:].replace("\n", " "), flush=True)
    summ = {}
    for typ in ("negative", "positive"):
        rs = [o for o in out if o["type"] == typ]
        for name in ("plain", "think"):
            summ[f"{typ}/{name}: empty list"] = sum(1 for o in rs if o[name + "_calls"] == [])
            summ[f"{typ}/{name}: starts with <think>"] = sum(1 for o in rs if o[name].lstrip().startswith("<think>"))
            want = lambda o: _calls_of("<tool_call>" + json.dumps(o["gold"], separators=(",", ":")) + "</tool_call>")
            summ[f"{typ}/{name}: exact calls"] = sum(1 for o in rs if o[name + "_calls"] == want(o))
        summ[f"{typ}: n"] = len(rs)
    print(json.dumps(summ, indent=1), "wall", round(time.time() - t0))
    json.dump({"summary": summ, "rows": out}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
