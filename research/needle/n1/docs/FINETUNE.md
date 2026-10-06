# N1 fine-tune (spec §4): pod2 RTX 3090, 2026-10-04

## Setup
- pod2 venv `/root/venv-needle`: cactus-needle 3.0.6, jax 0.10.2 + jax[cuda12], backend gpu float32 (H2).
- Data: `train.jsonl` (1,620 rows) and `val.jsonl` (360 rows), unchanged (md5 train bfde8315…, val 04881f2d…).
- Base checkpoint: `needle3.safetensors` (md5 37b800e5…).
- Data choices:
  - `--max-len 1536` has no effect. The longest row is 976 tokens (train) / 970 (val), and `fit_max_len` gives seq_len 1024 either way, with no truncation (DECISIONS I1).
  - Excluding the ungrounded positives was a no-op: 0 such rows in train, all 3 are in test (A6/I2).
  - order_ref: train uses `answers_omit_default` (A1). 3 (b) rows carry "headphones" (I3).
- Validation: `--val-split 0`. Each adapter is then scored on the by-scenario `val.jsonl` with `finetune/val_loss.py`: the same CQ W4 STE + A8 masked answer CE as training, batch 16, token-weighted mean.

## Commands (pod2, tmux `n1sweep` then `n1build`; scripts `finetune/sweep/run_sweep.sh`, `build.sh`)
```
cd /root/needle/finetune && source env_pod2.sh && cd sweep
for N in 3 6 10; do
  python ../timed.py needle finetune /root/needle/train.jsonl --checkpoint $CKPT --epochs $N --max-len 1536 \
    --val-split 0 --seed 0 --checkpoint-dir ckpt_e$N --out $PWD/adapter_e$N.safetensors 2>&1 | python ../tstamp.py > train_e$N.log
done
python ../val_loss.py --checkpoint $CKPT --data /root/needle/val.jsonl --base \
  --adapter adapter_e3.safetensors adapter_e6.safetensors adapter_e10.safetensors --out val_losses.json
needle build $CKPT --lora adapter_e10.safetensors --layers 8 --out tuned_l8.cact
needle build $CKPT --lora adapter_e10.safetensors --out tuned_full.cact
```
- CLI defaults: batch 16, lr 1e-4, LoRA r16 / alpha 32 (5 weight groups), AdamW, clip 1.0, warmup 5% + cosine decay, seed 0.
- Each run is separate and has its own full schedule: 306 / 612 / 1,020 steps (102 steps per epoch).

## Train loss per epoch (CLI line; **loss of the epoch's last batch**, which has 4 rows because 1620 = 101×16 + 4)
| epoch | e3 run | e6 run | e10 run |
|---|---|---|---|
| 1 | 1.0607 | 1.0594 | 1.0834 |
| 2 | 0.1445 | 0.0745 | 0.0696 |
| 3 | 0.1256 | 0.0305 | 0.0223 |
| 4 | | 0.3308 | 0.2945 |
| 5 | | 0.0532 | 0.0212 |
| 6 | | 0.0087 | 0.0033 |
| 7 | | | 0.0727 |
| 8 | | | 0.0030 |
| 9 | | | 0.0048 |
| 10 | | | 0.0029 |

The per-step losses (every 6/12/20 steps) are in `finetune/sweep/train_e{3,6,10}.log`. The epoch-4 spike appears in both runs and comes from the same seeded 4-row last batch.

## Val loss (`val.jsonl`, 360 rows, 5,179 answer tokens; `finetune/sweep/val_losses.json`)
| model | token_mean | batch_mean |
|---|---|---|
| base (no LoRA) | 1.1817 | 1.2783 |
| adapter_e3 | 0.3210 | 0.3285 |
| adapter_e6 | 0.1951 | 0.1974 |
| **adapter_e10** | **0.1855** | **0.1881** |

**Chosen: e10** (`finetune/sweep/adapter_e10.safetensors`, md5 d6205037…), with the lowest token_mean.
- Val loss still falls from e6 to e10, so the sweep has not reached a minimum.
- Base is a reference, not a candidate.
- Selection was done at full depth (20 layers). The 8-layer build is not scored on val.
- Val has 0 positives for 8 of 13 tools (A7), so it does not measure those tools.

## Builds (`finetune/sweep/`)
- `tuned_l8.cact`: 27.39 MB, 238 tensors, W4A8, 8 layers, md5 53649254…; build took 15.2 s.
- `tuned_full.cact`: 63.44 MB, 574 tensors, W4A8, 20 layers, md5 92ddd0d7…; build took 63.5 s.
- `needle build` exported from the package's base archive `~/.cache/cactus-needle/v3/3.0.2/needle3.cact` (35.34 MB) with the e10 LoRA merged in. It dropped the confidence head, so tuned confidence reports None, and gating should be on "a call shipped" (spec §4).
- No platform fine-tune was run, because `NEEDLE_API_KEY` is not set (F1).

## Times and resources (pod2)
| stage | wall |
|---|---|
| e3 (306 steps) | 575 s |
| e6 (612 steps) | 1,038 s |
| e10 (1,020 steps) | 1,649 s |
| val_loss (base + 3) | 201 s |
| builds | 15 s + 64 s |
| **total** | **~60 min** (07:34:56 → 08:34 UTC) |

- Steady state is ~1.51 s/step (~154 s per epoch).
- Epoch 1 takes ~260 s, because it includes the XLA compile and a second compile for the 4-row final batch.
- Peak GPU memory 14.5 GB of 24 GB (`smi_sweep.csv`, 2 s sampling). Peak host RSS 10.1 GB for training and 12.5 GB for val_loss.

## Not done here
The tuned evaluation (`evaluate.py --weights tuned_*.cact` on test_n0 / heldout a / b, plus laptop latency) is the next step. Use the same `auto_date=False` system pin and check the test md5s (G5).
