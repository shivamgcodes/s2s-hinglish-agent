"""venv-ft smoke: load PersonaPlex LM via PP loaders, forward_train + backward on random codes (B=1, T=300)."""
import os, time, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import hf_hub_download
from moshi.models import loaders
t0 = time.time()
w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
lm = loaders.get_moshi_lm(w, device="cuda")
print("loaded in %.1fs" % (time.time() - t0), "n_q", lm.n_q, "dep_q", lm.dep_q, "delays", lm.delays,
      "text_pad", lm.text_padding_token_id, "epad", lm.end_of_text_padding_id, "zero", lm.zero_token_id,
      "params %.2fB" % (sum(p.numel() for p in lm.parameters()) / 1e9), flush=True)
lm.train()
for n, p in lm.named_parameters():
    p.requires_grad = n.startswith("transformer.layers.31.") or n.startswith("depformer.layers.5.")
B, T = 1, 300
codes = torch.cat([torch.randint(0, 32000, (B, 1, T)), torch.randint(0, 2048, (B, 16, T))], 1).cuda()
torch.cuda.reset_peak_memory_stats()
out = lm.forward_train(codes)
print("logits", tuple(out.logits.shape), "mask", tuple(out.mask.shape), "text_logits", tuple(out.text_logits.shape), flush=True)
loss = torch.nn.functional.cross_entropy(out.text_logits[out.text_mask].float(), codes[:, :1][out.text_mask]) + \
       torch.nn.functional.cross_entropy(out.logits[:, :8][out.mask[:, :8]].float(), codes[:, 1:9][out.mask[:, :8]])
loss.backward()
g = sum(p.grad.float().norm().item() for p in lm.parameters() if p.grad is not None)
print("loss %.3f gradnorm-sum %.3f peak VRAM %.1f GB" % (loss.item(), g, torch.cuda.max_memory_allocated() / 1e9), flush=True)
print("SMOKE_OK")
