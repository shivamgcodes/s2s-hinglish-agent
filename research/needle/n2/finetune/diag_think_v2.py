"""N2 think-block diagnostic on v2 val (adapted from N1 finetune/diag_think.py; positives only, since every v2 row is
a positive). Greedy JAX decode with the adapter merged, CQ W4 numerics (as finetune._score_quantised). Two prompts:
  plain : training render prompt (target starts at <tool_call>)
  think : the same prompt + "<think>\n" (the engine always opens a think block)
Each output -> finetune._calls_of (as dicts) -> router_v2.resolve_calls -> score_map.score vs meta.gold_canonical.
Usage: python diag_think_v2.py --checkpoint CKPT --adapter A.safetensors [--rows /root/n2/data/rows/val] --out X.json
"""
import argparse, json, os, sys, time
import numpy as np

from pathlib import Path  # noqa: E402  (monorepo shim, below)
REPO = Path(__file__).resolve().parents[4]  # monorepo root (holds packages/ and research/)
N2_ROOT = os.environ.get("N2_ROOT", "/root/n2")  # runpod2 work dir (data/, windows/, V4/, finetune/, eval/)
sys.path.insert(0, os.environ.get("NEEDLE_V2_DIR", str(REPO / "packages" / "needle_router" / "v2")))  # router_v2, tools_v2, ...
sys.path.insert(1, str(REPO / "research" / "needle" / "n2" / "schema"))  # score_map.py
import router_v2, score_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", default=f"{N2_ROOT}/data/rows/val")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    from needle.model.run import load_checkpoint, generate
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS
    from needle.model.tokenizer import get_tokenizer, THINK_START
    from needle.model.finetune import render_example, merge_lora, _calls_of
    from needle.model.checkpoints import read_adapter

    params, config = load_checkpoint(a.checkpoint)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda x: np.asarray(x).astype(np.float32), params))
    tok = get_tokenizer(config.vocab_size)
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    ad = read_adapter(a.adapter)
    lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])} for k, v in ad["lora"].items()}
    merged = cq_ste_params(merge_lora(params, lora, float(ad["scale"])), WEIGHT_BITS)

    calls = {}
    for l in open(f"{N2_ROOT}/V4/calls.jsonl"):
        c = json.loads(l); calls[c["call_id"]] = c
    rows = [json.loads(l) for l in open(a.rows + ".jsonl")]
    metas = [json.loads(l) for l in open(a.rows + "_meta.jsonl")]
    out, t0 = [], time.time()
    for i, (r, m) in enumerate(zip(rows, metas)):
        prompt, _ = render_example({**r, "answers": []})
        rec, at = calls[m["call_id"]]["record"], m["agent_type"]
        o = {"i": i, "example_id": m["example_id"], "variant": m["variant"], "gold": m["gold_canonical"]}
        for name, p in (("plain", prompt), ("think", prompt + THINK_START + "\n")):
            text = generate(model, merged, tok, p, max_new_tokens=a.max_new, temperature=0.0, stream=False)
            text = text if isinstance(text, str) else tok.decode(text)
            fc = [{"name": n, "arguments": json.loads(aj)} for n, aj in (_calls_of(text) or [])]
            s = score_map.score([m["gold_canonical"]], score_map.v2_to_server(router_v2.resolve_calls(fc, rec, at)), at)
            o[name] = text; o[name + "_calls"] = fc
            o[name + "_tool_match"] = s["tool_match"]; o[name + "_correct"] = s["correct"]
        out.append(o)
        print(i, m["example_id"], "plain", o["plain_correct"], "think", o["think_correct"], flush=True)
    summ = {"adapter": a.adapter, "n": len(out), "wall_s": round(time.time() - t0)}
    for name in ("plain", "think"):
        for v in ("all", "clean", "opus"):
            rs = [o for o in out if v == "all" or o["variant"] == v]
            summ[f"{name}/{v}"] = {"tool_match": sum(o[name + "_tool_match"] for o in rs),
                                   "correct": sum(o[name + "_correct"] for o in rs), "n": len(rs)}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": out}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
