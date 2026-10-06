# PersonaPlex V4 Hinglish LoRA: offline run (= the HF card's example)

`run_offline.py` does the following:
1. Downloads the 4 base files of `nvidia/personaplex-7b-v1` (gated, pinned revision) and the adapter
   ([`shivamgupta/personaplex-hinglish-v4-lora`](https://huggingface.co/shivamgupta/personaplex-hinglish-v4-lora):
   `lora.safetensors` + `config.json`).
2. Merges the adapter in memory with the `personaplex_lora` package (`packages/personaplex_lora`).
3. Runs one customer-side wav through the full-duplex loop with a role prompt and a voice prompt.

Like the stock PersonaPlex `moshi/offline.py`, it writes the agent audio and the per-frame text. Since D-LEAN-HF
(2026-10-07) this is the one copy of that example; the HF repo holds only the adapter, its config, the examples and
the model card.

```bash
# 1. torch + PersonaPlex's moshi (see requirements.txt for the tested order), 2. the package:
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora"
python run_offline.py --output out.wav --max-seconds 10      # HF examples/food_23_g1_customer_30s.wav + food_23.json, g1
```
Needs access to the gated base repo (accept the NVIDIA Open Model License; `HF_TOKEN` or `huggingface-cli login`)
and one CUDA GPU with about 20 GB free. From a monorepo checkout, it also runs without installing the package: it
falls back to `../../../packages`.
