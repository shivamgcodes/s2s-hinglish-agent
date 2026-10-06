"""Verify the PersonaPlex prefix patch on one synthetic call (venv-ft; GPU; run under flock).

  python verify_sample.py build      # CPU: synthetic stereo call + alignment json + manifest
  python verify_sample.py check      # GPU: data layout, prefix, loss mask, forward equivalences,
                                     #      zero-B / random adapters for the inference tests
Outputs in /workspace/hinglish/trainer/verify/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_H = Path(__import__("os").environ.get("HINGLISH_ROOT", "/workspace/hinglish"))
_REPO = Path(__file__).resolve().parents[3]  # monorepo root
VOICE_CODES = __import__("os").environ.get("PP_VOICE_CODES_DIR") or str(_REPO / "packages" / "personaplex_lora" / "trainer" / "voice_codes")
OUT = _H / "trainer/verify"
CALLS = _H / "audio/testdata/calls.jsonl"
CHUNKS = _H / "audio/testdata/out/work/chunks"
SR = 24000


def build():
    import sphn
    OUT.mkdir(parents=True, exist_ok=True)
    call = [json.loads(l) for l in CALLS.open() if l.strip()][0]
    pool = []
    for w in sorted(CHUNKS.rglob("*.wav")):
        pcm, sr = sphn.read(str(w))
        pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=SR)[0]
        pool.append(pcm.astype(np.float32))
    pool = np.concatenate(pool)
    total_s = 112.0  # > 100 s so the call yields a second (mid-call) chunk
    wav = np.zeros((2, int(total_s * SR)), dtype=np.float32)
    t, pos, align = 1.0, 0, []
    turns = call["turns"]
    k = 0
    while True:
        turn = turns[k % len(turns)]
        k += 1
        words = turn["text_roman"].split()
        dur = max(1.0, len(words) / 2.6)
        if t + dur > total_s - 0.5:
            break
        ch = 0 if turn["speaker"] == "agent" else 1
        n = int(dur * SR)
        seg = np.take(pool, np.arange(pos, pos + n), mode="wrap")
        pos += n
        a = int(t * SR)
        wav[ch, a:a + n] = seg[: wav.shape[1] - a]
        spk = "SPEAKER_MAIN" if ch == 0 else "SPEAKER_USER"
        for i, wd in enumerate(words):
            s = t + i * dur / len(words)
            align.append([wd, [round(s, 3), round(s + 0.8 * dur / len(words), 3)], spk])
        t += dur + float(turn.get("pause_after_s") or 0.4)
    align.sort(key=lambda x: x[1][0])
    sphn.write_wav(str(OUT / "synth_call.wav"), wav, SR)
    doc = {"alignments": align, "call_id": "synth_" + call["call_id"], "role_prompt": call["role_prompt"],
           "agent_gender": call["agent_gender"], "voice_prompt": "NATF2" if call["agent_gender"] == "f" else "NATM1"}
    (OUT / "synth_call.json").write_text(json.dumps(doc, ensure_ascii=False))
    (OUT / "manifest.jsonl").write_text(json.dumps({"path": str(OUT / "synth_call.wav"), "duration": total_s}) + "\n")
    print(f"built {OUT/'synth_call.wav'}: {total_s} s, {len(align)} words "
          f"({sum(a[2]=='SPEAKER_MAIN' for a in align)} agent), turns placed {k-1}")


def check():
    import torch
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from torch.nn.utils import parametrize
    from safetensors.torch import save_file
    from moshi.models import loaders
    from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS
    from finetune import pp_lora
    from finetune.data.args import DataArgs
    from finetune.data.data_loader import build_data_loader
    from finetune.data.interleaver import InterleavedTokenizer, Interleaver, PersonaPlexPrefix
    sys.path.insert(0, __import__("os").environ.get("PP_LORA_INFER_DIR") or str(_REPO / "packages" / "personaplex_lora" / "infer"))
    import lora_merge

    torch.manual_seed(0)
    repo = loaders.DEFAULT_REPO
    mimi = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME), device="cuda")
    spm = sentencepiece.SentencePieceProcessor(hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME))
    PAD, EPAD = 3, 0
    inter = Interleaver(spm, mimi.frame_rate, PAD, EPAD, -1, keep_main_only=True)
    prefixer = PersonaPlexPrefix(spm, VOICE_CODES, text_padding=PAD,
                                 silence_frames=int(0.5 * mimi.frame_rate))
    itok = InterleavedTokenizer(mimi, inter, duration_sec=100, prefixer=prefixer)
    dl = build_data_loader(itok, DataArgs(eval_data=str(OUT / "manifest.jsonl")), batch_size=2, seed=None,
                           rank=0, world_size=1, is_eval=True)
    batch = next(dl)
    codes, plen = batch.codes, batch.prefix_len
    doc = json.loads((OUT / "synth_call.json").read_text())
    print(f"batch codes {tuple(codes.shape)} prefix_len {plen.tolist()}")

    # ---- prefix layout
    P = int(plen[0])
    ids = spm.encode("<system> " + doc["role_prompt"].strip() + " <system>")
    nv, ns = 51, 6
    pre = codes[0, :, :P].cpu()
    vf = torch.load(f"{VOICE_CODES}/NATF2.frames.pt")["vf"][:, 1:52]
    sil, sine = torch.tensor(SILENCE_TOKENS), torch.tensor(SINE_TOKENS)
    checks = {
        "P == 51+6+len(ids)+6": P == nv + ns + len(ids) + ns,
        "voice frames: text=PAD": bool((pre[0, :nv] == PAD).all()),
        "voice frames: agent rows = NATF2 vf[:,1:52]": bool((pre[1:9, :nv] == vf).all()),
        "all prefix: user rows = SINE": bool((pre[9:17] == sine[:, None]).all()),
        "silence frames: agent = SILENCE, text=PAD": bool((pre[1:9, nv:nv + ns] == sil[:, None]).all()
                                                         and (pre[0, nv:nv + ns] == PAD).all()),
        "prompt frames: text = ids": pre[0, nv + ns:nv + ns + len(ids)].tolist() == ids,
        "prompt+tail frames: agent = SILENCE": bool((pre[1:9, nv + ns:] == sil[:, None]).all()),
        "tail frames: text=PAD": bool((pre[0, P - ns:] == PAD).all()),
        "no unk(0) in prompt ids": 0 not in ids,
        "chunk 2 (mid-call) has same prefix": bool((codes[1, :, :P] == codes[0, :, :P]).all()) and int(plen[1]) == P,
    }
    print(f"\n== PREFIX (P={P} frames = 51 voice + 6 sil + {len(ids)} prompt + 6 sil; voice {doc['voice_prompt']})")
    for f in [0, 1, 50, 51, 56, 57, 58, P - 7, P - 6, P - 1]:
        print(f"  frame {f:3d}: text={pre[0,f].item():5d} agent={pre[1:9,f].tolist()} user={pre[9:17,f].tolist()}")
    print("  prompt text row decoded:", repr(spm.decode(pre[0, nv + ns:nv + ns + len(ids)].tolist())))
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    # ---- text stream after prefix (Gate 0 dump style), chunk 0
    def piece(t):
        return {PAD: "PAD", EPAD: "EPAD", -1: "ZERO"}.get(t, None) or spm.id_to_piece(t).replace("▁", " ")
    txt = codes[0, 0, P:].tolist()
    dump = [piece(t) for t in txt]
    (OUT / "text_stream_chunk0.json").write_text(json.dumps(dump, ensure_ascii=False))
    print(f"\n== TEXT STREAM chunk 0 ({len(txt)} frames @12.5 fps), first 100 frames:")
    print(json.dumps(dump[:100], ensure_ascii=False))
    print("  frame view (word starts):")
    shown = 0
    for i, t in enumerate(txt):
        if t not in (PAD, EPAD, -1) and spm.id_to_piece(t).startswith("▁") and shown < 12:
            print(f"   f{i:4d} t={i/12.5:6.2f}s  prev={dump[i-1] if i else '-':5s} {dump[i]!r}")
            shown += 1
    main = [a for a in doc["alignments"] if a[2] == "SPEAKER_MAIN" and a[1][0] < 100]
    rec = spm.decode([t for t in txt if t not in (PAD, EPAD, -1)])
    ref = " ".join(a[0] for a in main)
    print(f"  decoded agent text == SPEAKER_MAIN words in [0,100): {rec == ref}")
    print(f"  decoded[:160]: {rec[:160]!r}")
    print(f"  EPAD count {txt.count(EPAD)}, PAD {txt.count(PAD)}, word-start frames {sum(1 for t in txt if t not in (PAD,EPAD,-1) and spm.id_to_piece(t).startswith(chr(9601)))}, agent words {len(main)}")
    cust_in_text = any(a[0] in rec.split() for a in doc["alignments"] if a[2] != "SPEAKER_MAIN" and a[0] not in ref.split())
    print(f"  customer-only words present in text stream: {cust_in_text}")
    txt1 = codes[1, 0, P:].tolist()
    main1 = [a for a in doc["alignments"] if a[2] == "SPEAKER_MAIN" and 100 <= a[1][0] < 200]
    rec1 = spm.decode([t for t in txt1 if t not in (PAD, EPAD, -1)])
    print(f"  chunk 1: {sum(t != -1 for t in codes[1, 1].tolist()) - P} real audio frames, text == words[100,112): "
          f"{rec1 == ' '.join(a[0] for a in main1)}  {rec1[:80]!r}")

    # ---- model: forward equivalence, loss mask
    lm = loaders.get_moshi_lm(hf_hub_download(repo, loaders.MOSHI_NAME), device="cuda")
    lm.eval()
    c0 = codes[:1]
    with torch.no_grad():
        ref_out = lm.forward_train(c0)
        out8 = pp_lora.forward_train_agent(lm, c0, 8)
        m = ref_out.mask[:, :8]
        d_audio = (ref_out.logits[:, :8][m] - out8.logits[m]).abs().max().item()
        d_text = (ref_out.text_logits[ref_out.text_mask] - out8.text_logits[out8.text_mask]).abs().max().item()
        print(f"\n== forward_train_agent(8 steps) vs PP forward_train[:, :8]: max|dlogits| audio {d_audio:.3g}, text {d_text:.3g}")
        tot, tl, al, st = pp_lora.masked_losses(out8, c0, plen[:1], 100.0, 0.5, {PAD, EPAD})
        keep = pp_lora.prefix_keep_mask(plen[:1], c0.shape[-1])
        tm, am = out8.text_mask & keep, out8.mask & keep
        print(f"== LOSS MASK: text mask over prefix [0,{P}) sum={tm[..., :P].sum().item()}, audio mask over prefix sum="
              f"{am[..., :P].sum().item()}; after prefix: text {tm[..., P:].sum().item()}/{c0.shape[-1]-P}, "
              f"audio {am[..., P:].sum().item()}/{8*(c0.shape[-1]-P)} (delay-1 rows lose the last frame)")
        print(f"== base losses on sample: total {tot.item():.3f} text {tl.item():.3f} audio {al.item():.3f} "
              f"cb1 {st['cb1'].item():.3f} cb2-8 {st['cb2_8'].item():.3f}")
        del ref_out, out8
        # config-B voice embedding regeneration check (base tables must reproduce shipped .pt)
        vdir = Path(hf_hub_download(repo, "voices/NATF2.pt")).parent
        for v in ("NATF2", "NATM1"):
            e = lora_merge.voice_embeddings_from_codes(lm, v)
            st_ = torch.load(vdir / f"{v}.pt")["embeddings"].to(e.device)
            print(f"== voice {v}: recomputed embeddings {tuple(e.shape)} vs shipped {tuple(st_.shape)} "
                  f"bit-exact: {torch.equal(e, st_)}")

    # ---- LoRA: zero-B adapter (for driver test), random adapter merge equivalence
    names = pp_lora.apply_lora(lm, 64, 2.0, dep_steps=8)
    cfg = {"format": pp_lora.ADAPTER_FORMAT, "lora_rank": 64, "lora_scaling": 2.0, "ft_embed": False,
           "dep_train_steps": 8, "n_lora_weights": len(names)}
    zdir = OUT / "zeroB"
    zdir.mkdir(exist_ok=True)
    save_file(pp_lora.adapter_state_dict(lm, False, 8), str(zdir / "lora.safetensors"))
    (zdir / "config.json").write_text(json.dumps(cfg, indent=1))
    n_train = sum(p.numel() for n, p in lm.named_parameters() if "lora_" in n)
    print(f"\n== LoRA: {len(names)} target weights, {n_train/1e6:.1f} M LoRA params; zero-B adapter -> {zdir}")
    with torch.no_grad():
        for n, p in lm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
        cs = c0[..., : P + 300]
        o_par = pp_lora.forward_train_agent(lm, cs, 8)
        sd = pp_lora.adapter_state_dict(lm, False, 8)
        rdir = OUT / "randomB"
        rdir.mkdir(exist_ok=True)
        save_file(sd, str(rdir / "lora.safetensors"))
        (rdir / "config.json").write_text(json.dumps(cfg, indent=1))
        for mod in list(lm.modules()):
            if parametrize.is_parametrized(mod):
                for attr in list(mod.parametrizations.keys()):
                    parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)
        o_base = pp_lora.forward_train_agent(lm, cs, 8)
        info = lora_merge.merge_into(lm, sd, cfg)
        o_mrg = pp_lora.forward_train_agent(lm, cs, 8)
        m = o_par.mask
        print(f"== random-B adapter: merged {info}")
        print(f"   parametrized vs merged: max|d| audio {(o_par.logits[m]-o_mrg.logits[m]).abs().max().item():.3g} "
              f"text {(o_par.text_logits[o_par.text_mask]-o_mrg.text_logits[o_mrg.text_mask]).abs().max().item():.3g}")
        print(f"   (base vs merged, should be large): audio {(o_base.logits[m]-o_mrg.logits[m]).abs().max().item():.3g}")
    print("VERIFY DONE")


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()
