# Trainer recon: moshi-finetune (2acc879) vs PersonaPlex moshi

Written 2026-10-03 by the trainer-recon worker. Paths are on the pod. File:line refs are per file.
- MF = /workspace/moshi-finetune (commit 2acc879)
- PP = /workspace/personaplex/moshi/moshi (PersonaPlex package; never modified)
- UP = /workspace/hinglish/trainer/upstream_moshi/moshi/moshi (kyutai moshi @061cc4c, the commit pinned in MF/uv.lock:812; fetched here for reference only)

## 0. Summary
| Item | Finding |
|---|---|
| Stereo channels | **ch0 (left) = model/agent**, ch1 (right) = user/customer (MF README "Prepare dataset"; MF interleaver.py:257-265: `mimi.encode(wav[:,None])` gives [2,8,T], then `view(1,16,T)`, so ch0 lands in code rows 1-8 and ch1 in rows 9-16). In PP, rows 1-8 are the "moshi" (agent) streams and rows 9-16 are the user "input" streams (PP lm.py:754-774). |
| Alignment JSON | Same basename as the wav, `.json` (MF interleaver.py:267). Format: `{"alignments": [["word", [start_s, end_s], "SPEAKER_MAIN"], ...]}` (MF annotate.py:139-146). Times are seconds from the start of the wav. The list must be sorted by start (binary search, interleaver.py `dicho`). |
| Manifest | jsonl, one line per wav: `{"path": "/abs/path/a.wav", "duration": 83.2}` (MF README; dataset.py:202-218 via `sphn.dataset_jsonl`). Use absolute paths, because the json is opened as `splitext(path)+".json"` relative to the cwd. |
| Base model | nvidia/personaplex-7b-v1 loads with `PP models/loaders.get_moshi_lm` (loaders.py:166-260). It has 8.37 B params, n_q=16, **dep_q=16** (loaders.py:187), delays `[0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]` (loaders.py:121). Text PAD=3, EPAD=0, zero_token=-1, text initial=32000, audio initial=2048. |
| MF vs PP | MF does not run against PP as-is. PP has no `moshi.conditioners`, no `moshi.modules.lora`, no `loaders.CheckpointInfo`, no `LMModel.forward(codes, condition_tensors)` (PP has `forward_train(codes)`, lm.py:531), and no lora/gradient_checkpointing kwargs or activation checkpointing. A patch is required (see section 5). |
| Voice prompt | NATF2/NATM1 exist only as `.pt` replays: 51 pre-computed input embeddings plus a 4-column cache. I recovered the exact Mimi codes behind them (bf16 re-embedding is exact on 51/51 columns for both voices). Files are in `voice_codes/` (section 3). |
| Adapter inference | Copy the Gate 0 driver into /workspace/hinglish. It loads the base model with PP loaders, then merges the LoRA in place (`W += scale·B@A` by PP parameter name), then runs the same loop. No change to PP and no 16 GB merged files. |
| venv-ft | Built (/workspace/venv-ft): torch 2.14.1+cu130, GPU visible (RTX PRO 6000, sm_120). PP moshi imports. A forward_train+backward smoke run on the GPU passed. MF modules beyond args/loss fail to import until patched. |

## 1. moshi-finetune: how it works
### Data
- Manifest and chunking: `sphn.dataset_jsonl(jsonl, duration_sec, sample_rate=24000, pad_last_segment=True)` (dataset.py:202-208). It yields `duration_sec` chunks of each wav, with `start_time_sec`. A call longer than duration_sec becomes several independent samples, and the later ones start mid-call. Batches are fixed-size (data_loader.py:29-38).
- `InterleavedTokenizer.__call__` (interleaver.py:254-289) does the following:
  1. Mimi-encodes the stereo chunk to 16 rows at 12.5 fps.
  2. Truncates or pads to `ceil(duration_sec*12.5)` frames. The pad value is zero_token -1, which gives a zero input embedding and is masked in the loss.
  3. Reads `<wav>.json["alignments"]`, keeps the words in [start, start+duration), and shifts their times to the chunk start.
  4. Builds the text row with `Interleaver.prepare_item` and pads it with -1.
  5. Sets `codes = cat([text(1), audio(16)])` = [1,17,T]. `data.get("text_conditions")` is passed as condition_attributes. **Do not put a `text_conditions` key in our JSONs**: it routes into `model.condition_provider` (train.py:249-252), which PP lacks.
- Interleaver config (train.py:155-162): `Interleaver(spm, 12.5, text_padding=3, end_of_text_padding=0, zero_padding=-1, keep_main_only=True)`. The defaults apply for the rest: `audio_delay=0.0`, `keep_and_shift=False`, `use_bos_eos=False`, `in_word_padding=None→PAD`, `main_speaker_label="SPEAKER_MAIN"` (interleaver.py:76-89). **Only words labelled `SPEAKER_MAIN` reach the text stream.** Every other label is dropped (`_keep_main_only`). No BOS/EOS are inserted.
- Text stream construction (`build_token_stream`, interleaver.py:171-210):
  - T = ceil(dur·12.5) frames, all PAD(3) to start.
  - Each word is tokenised alone with `sp.encode(word.strip())` (no BOS; SentencePiece adds the `▁` prefix).
  - The word's first token goes at frame `floor(start·12.5)`, and its remaining tokens go in the following frames, one per frame.
  - If the frame before the first token is PAD, it becomes **EPAD(0)**.
  - Frames after the tokens, up to `int(end·12.5)`, are PAD (in-word padding).
  - Because `keep_and_shift=False`, a new word that starts while the previous word's tokens are still being emitted **replaces** the remaining tokens. Long multi-token words in fast speech lose tokens. The Hinglish data should be checked for this (the patch-builder's sample decode will show it).
- Delays are not applied in the data. `LMModel.forward_train` does it: it prepends the initial token column and `_delay_sequence`s with the model delays (PP lm.py:531-537). Logits are un-delayed back to code positions and masked where the code is -1 (lm.py:546-552).

### Loss (train.py:254-276, loss.py:4-30)
- text loss = weighted CE on row 0. Targets PAD(3) and EPAD(0) get weight `text_padding_weight` (example yaml: 0.5).
- audio loss = CE on `codes[:, 1:1+dep_q]` with codebook 0 weighted by `first_codebook_weight_multiplier` (example yaml: 100). Each term is normalised by its weight sum, and total = text + audio.
- **PP has dep_q=16, so as written the audio loss would also cover the 8 user rows.** The patch must slice `output.logits[:, :8]`, `output.mask[:, :8]` and `codes[:, 1:9]`.

### LoRA / embeddings / checkpointing (UP versions; PP lacks all of it)
- UP `modules/lora.py:5-22` `replace_all_linear_with_lora` swaps every `nn.Linear` in the model for `LoRALinear(frozen_W, lora_A, lora_B, scaling)`. That includes the depformer, `depformer_in`, `linears` and `text_linear`.
- Scaling: forward = W x + scaling·B(A x) (UP lora.py:116-118). Rank 64 with alpha 128 means `scaling=2.0`, which is the MF default (args.py:11-15). The example yaml uses rank 128, scaling 2.
- UP reaches the depformer per step only because UP split attention into `in_projs/out_projs` ModuleLists (UP transformer.py:351-362) and gating calls Linear modules. **In PP:**
  - `self_attn.in_proj_weight` is a raw Parameter used via `F.linear`/`multi_linear` (PP transformer.py:362-365, 412-417).
  - The depformer `out_proj.weight` goes through `multi_linear` (transformer.py:440-441).
  - Gating calls `gating_forward_kernel(self.linear_in.weight, self.linear_out.weight, …)` (PP gating.py:69-71).
  - So a module-swap LoRA would miss `in_proj_weight`, and it would crash wherever `.weight` is read.
  - **The patch needs weight-level LoRA**, for example `torch.nn.utils.parametrize` on each weight (W + s·B@A). For the stacked per-step depformer weights (in_proj [16·3072,1024], out_proj [16·1024,1024]), use per-step blocks (A:[16,r,in], B:[16,out,r]) to match UP semantics.
- Freezing (wrapped_model.py:174-184): params with "lora" in their name are trained. If `ft_embed`, params with "emb" in their name are also trained: `emb.*`, `text_emb`, `depformer_emb.*`, `depformer_text_emb`.
- LoRA init: A kaiming-uniform, B zeros (wrapped_model.py:75-95).
- Base weights are loaded with `safetensors.load_file` then `load_state_dict(strict=False, assign=True)` (wrapped_model.py:133-145) after building through `CheckpointInfo.get_moshi(lm_kwargs_overrides={gradient_checkpointing, lora, lora_rank, lora_scaling})` (wrapped_model.py:122-131). PP has none of this, so replace it with `loaders.get_moshi_lm(path, device, dtype=bf16)` and then apply LoRA.
- Single GPU: `get_world_size()==1` gives a plain `.cuda()` model with no FSDP (wrapped_model.py:186-187). torchrun with nproc 1 is still required for `dist.barrier()`.
- Checkpointing (checkpointing.py):
  - `save_adapters=true` saves `checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors` with only fully-trainable leaf modules (lines 111-161), plus `config.json` = the lm_config dict (82-85).
  - `num_ckpt_keep` defaults to 3 (args.py:92).
  - It needs `moshi.modules.lora.LoRALinear` (line 9), so it must be patched to save the parametrization tensors, and the `emb` tables too when ft_embed.
  - The trainer's own `metrics.csv` and `loss.png` are not in MF (it has tensorboard plus a metrics jsonl), so they have to be added.
- Gradient checkpointing: neither PP's `StreamingTransformer` nor its layers have a flag. Wrap each `transformer.layers[i]` forward with `torch.utils.checkpoint` in the trainer. `NO_TORCH_COMPILE=1` disables PP's lazy compile of the gating kernel (PP utils/compile.py:62).

### Config A source values (MF example/moshi_7B.yaml)
`lr 2e-6`, `weight_decay 0.1`, `pct_start 0.05` (OneCycle), `first_codebook_weight_multiplier 100`, `text_padding_weight 0.5`, `duration_sec 100`, `batch_size 16`, `gradient_checkpointing true`, `max_norm 1.0` (default), `ckpt_freq 100`, `lora rank 128, scaling 2, ft_embed false`. The README "quick training" settings are rank 128, scaling 2, duration 100, batch 16, 2000 steps. The TrainArgs defaults (lr 1e-4, duration 10) are NOT the recommended values.

## 2. PersonaPlex prompt mechanics (inference; the patch must replicate this)
Entry points: offline.py:236-253, and server.py:164-170 plus 283 (identical logic). `LMGen` is built with `audio_silence_frame_cnt=int(0.5*12.5)=6` (offline.py:214, server.py:107). `step_system_prompts` (lm.py:1123-1127) runs, in order:
1. **Voice prompt** (lm.py:1025-1062).
   - `.pt` path (what we use, and what the Gate 0 driver uses): `load_voice_prompt_embeddings` (lm.py:977-983) loads `embeddings` [51,1,1,4096] bf16 and `cache` [1,17,4]. Each embedding is replayed through `step_embeddings` (lm.py:853-873), which feeds the embedding straight into the transformer. The cache is then copied into LMGen state (lm.py:1038).
   - wav path (lm.py:960-975, 1010-1023, 1041-1048): load the wav, resample to 24 kHz, normalise loudness to −24 LUFS, Mimi-encode frame by frame, then `step(moshi_tokens=voice codes, text_token=3, input_tokens=SINE_TOKENS)`.
   - The `.pt` files are exactly the saved `embed_codes(input column)` of that wav path (lm.py:1052-1060). Per frame: agent rows = voice codes, **text = 3 (PAD)**, **user rows = SINE_TOKENS [430,1268,381,1611,1095,1495,56,472]** (lm.py:57, 1015-1020). It is not silence and not a zero token.
2. **Silence**, 6 frames (lm.py:1074-1084): agent = SILENCE_TOKENS [948,243,1178,546,1736,1030,1978,2008] (lm.py:56), text = 3, user = SINE_TOKENS.
3. **Text prompt**, one frame per token (lm.py:1096-1104): text = token id, agent = SILENCE_TOKENS, user = SINE_TOKENS.
   - Token ids = `sp.encode(wrap_with_system_tags(prompt))` as one string (offline.py:83-90, 241-242). No BOS. The wrapping is `"<system> " + prompt.strip() + " <system>"`. Each `<system>` tokenises to `[607 '▁<', 4831 'system', 578 '>']`, so the prompt starts and ends with those 3 ids.
   - Example: the 34-token prompt "You work for Zomato … Rs 540." gives ids `[607,4831,578,493,…,263,607,4831,578]`.
   - SentencePiece unk id = 0 = EPAD. Any character outside the vocabulary (Devanagari, '₹') becomes 0, so keep prompts ASCII.
4. **Silence**, 6 frames again.
5. Conversation: user rows = Mimi codes of the user wav. Text and agent rows are sampled (offline.py:267-279).

All of these are teacher-forced through the delay cache (lm.py:746-786). That is the same delay pattern as `_delay_sequence` in training.

**Frame alignment (verified on the cache):**
- At LMGen offset 0 the cache column 0 is overwritten with the initial token (lm.py:791-796). At offset ≤ delay, the delay-1 rows of column 1 are also overwritten (lm.py:781-786). LMGen frame 0 is therefore never seen.
- Inference input column c≥1 holds frame c on the delay-0 rows and frame c−1 on the delay-1 rows. Training column c (after forward_train prepends `initial`) holds `codes[c−1]` and `codes[c−2]`.
- Hence **training `codes[t]` = LMGen frame t+1**. The training prefix is the LMGen prompt timeline with its first frame dropped, which reproduces inference columns 0..P exactly (including RoPE positions).

**Training prefix for one sample (P frames, before conversation frame 0):**
```
rows        | 51 frames (voice)     | 6 frames     | len(prompt_ids) frames | 6 frames
text (0)    | 3                     | 3            | prompt_ids             | 3
agent (1-8) | vf[:,1:52] (voice)    | SILENCE_TOK  | SILENCE_TOK            | SILENCE_TOK
user (9-16) | SINE_TOK              | SINE_TOK     | SINE_TOK               | SINE_TOK
```
- P = 63 + len(prompt_ids), about 200-230 frames (16-18 s) for ~150-token prompts. Context is 3000 frames (loaders.py:104), so 1250 + P fits.
- Mask the loss over the prefix explicitly after `forward_train`: `text_mask[..., :P]=False` and `mask[..., :P]=False`. **Do not use -1 codes for the prefix**, because -1 gives a zero input embedding (ScaledEmbedding, lm.py:207-215), which differs from inference.

**Recovered voice codes** (/workspace/hinglish/trainer/voice_codes/):
- `NATF2.frames.pt`, `NATM1.frames.pt`: `{"vf": int64 [8,52]}` are the agent Mimi codes per LMGen voice frame. `vf[:,0]` = −1 (unrecoverable and unused). **Use `vf[:,1:52]`.**
- `*.columns.pt` hold the 17×51 input columns.
- How they were made:
  - `invert_voice.py` did greedy matching pursuit over the 16 audio tables and text_emb.
  - An intermediate coordinate-descent pass (script since deleted) rewrote `columns.pt`; its result was not yet exact.
  - `final_voice.py` fixed the text and user rows to their known values, then ran coordinate descent with top-k combination search on the 8 agent tables. This produced the final, exact files. Rerun `invert_voice.py` then `final_voice.py` to regenerate them.
- Checks, all passed for both voices:
  - bf16 re-embedding in `embed_codes` order is bit-exact on 51/51 columns.
  - col0 = [32000, 2048×16].
  - The text row is 3 for c≥1.
  - User rows are SINE, with initial on the delay-1 rows at col1.
  - The cache ring columns for cols 49 and 50 equal the recovered cols.
- vf[:,51] and vf[1:,50] come from the cache (indices 3 and 0).
- `*_vf1-51_decoded.wav` (4.08 s each) is a Mimi decode for listening.
- Equivalence: with frozen embeddings (configs A and C), feeding these codes in training equals the `.pt` replay at inference.

**Config B (ft_embed=true) caveat:** inference replays `.pt` embeddings computed with the base tables, while training embeds the prefix with the trained tables. To be exact at test time, regenerate the `.pt` with the merged model by stepping the recovered codes through `step()` with `save_voice_prompt_embeddings`. Alternatively, accept the mismatch and note it in the report.

## 3. Adapter inference without touching PP
- `grep -rni lora /workspace/personaplex/moshi` returns no hits. PP has no `--lora-weight`; it has only `--moshi-weight` in offline.py:352 and server.py.
- **Recommended:** `/workspace/hinglish/tests/driver_lora.py`, a copy of /workspace/personaplex-gate0/exp2/driver.py (the original stays unedited).
  1. Load the base with `loaders.get_moshi_lm(...)` exactly as driver.py:147 does.
  2. Read `lora.safetensors` and for each target weight do `W.data += scaling·(B@A)` in fp32, then cast to bf16. Per-step blocks are reshaped back to the stacked layout. Embedding tables are copied over when config B is used.
  3. Build `LMGen` as before.
  - Parameter names and model code are PP's own, so CUDA graphs and the functional `.weight` uses all see merged weights.
  - The base load takes ~40 s and the merge a few seconds. No 16 GB file per adapter.
- Alternative: save a merged `model.safetensors` and pass `--moshi-weight` to offline.py. It works with stock PP but costs ~16.7 GB per adapter and needs an extra write.
- Gate 1's "moshi.server --lora-weight" is not available in PP. Use the wrapper (or the merged file with `moshi.offline --moshi-weight`) for Gate 1.

## 4. PP vs UP moshi model differences relevant to training
- PP LMModel (lm.py:218-552):
  - No conditioners/fuser.
  - Has `forward_train(codes)` returning `LMOutput(logits [B,dep_q,T,2048], mask, text_logits [B,1,T,32000], text_mask)`.
  - `forward_depformer_training` loops over all dep_q=16 steps.
  - Embeddings are a plain sum (`embed_codes`, lm.py:425-439).
  - `text_linear` outputs 32000 (no extra pad class, existing_text_padding_id=3).
- Checkpoint (`model.safetensors`, 475 tensors) already has 16 depformer steps: `depformer_in.0-15`, `linears.0-15`, `depformer_emb.0-14`, depformer `self_attn.in_proj_weight` [49152,1024], gating.0-15. PP loader "copy missing weights" patches are no-ops for this file.
- UP has `in_projs/out_projs` per step with a load hook from `in_proj_weight` (UP transformer.py:351-380), activation checkpointing (UP transformer.py:696-759), LoRA, conditioners, and `forward(codes, condition_tensors)` (UP lm.py:273). Training with UP would need its forward to be proven numerically equal to PP's. Using PP's own `forward_train` with weight-level LoRA avoids that.

## 5. What the patch (trainer/personaplex_prefix.patch) has to change, minimum
1. `interleaver.py`: drop the `moshi.conditioners` import. In `InterleavedTokenizer.__call__`:
   - read `role_prompt` and `voice_prompt` (NATF2/NATM1) from the per-wav JSON;
   - build the prefix above;
   - return codes [1,17,P+T] plus P, for example as a `prefix_len` field on Sample/Batch.
   - P varies per call. Prefixes must start at column 0, because left-padding would shift inference columns. So right-pad every sample with -1 to a fixed length P_max+T (collate needs equal lengths), and store a per-sample P for the loss mask.
2. `wrapped_model.py`: load via `loaders.get_moshi_lm` and add the weight-level LoRA (temporal transformer in_proj/out_proj/gating linear_in/linear_out; depformer the same per step; optionally depformer_in, linears, text_linear to match UP "all linear"). Apply the `requires_grad` rules and the layer checkpoint wrapper.
3. `train.py`:
   - `model.forward_train(codes)`;
   - mask the prefix;
   - slice dep to 8;
   - add metrics.csv (text, cb1, mean cb2-8), loss.png, tqdm.
4. `checkpointing.py`: save LoRA tensors under PP param names, plus emb tables if ft_embed, plus config.json with rank and scaling.

## 6. Data contract for the audio/alignment worker (patch reads these)
`<call>.json` next to `<call>.wav` (stereo 24 kHz, ch0 = agent, ch1 = customer):
`{"alignments":[["Hello",[1.02,1.31],"SPEAKER_MAIN"],["haan",[5.10,5.30],"SPEAKER_OTHER"]],"role_prompt":"You work for ...","voice_prompt":"NATF2"}`
- Times are seconds from the wav start, including the 1.0 s lead silence. Words are sorted by start.
- Agent words are labelled `SPEAKER_MAIN` and carry the `text_roman` words. Customer words use any other label and are dropped from the text stream.
- `role_prompt` and `voice_prompt` are added by the patch; they are not standard moshi-finetune keys. Never write a `text_conditions` key.
- The manifest jsonl uses absolute wav paths.
- Calls longer than duration_sec (100 s) split into a second, mid-call chunk. The patch decides whether that chunk also gets the prefix.

## 7. venv-ft
- `/workspace/venv-ft`, built by `trainer/build_venv_ft.sh` (log `build_venv_ft.log`) with uv.
- Contents:
  - python 3.11.13 (same as venv-pp);
  - torch 2.14.1+cu130, torchaudio 2.11.0+cu130 and triton 3.8.0, the same versions as venv-pp;
  - numpy 2.1.3, safetensors 0.4.5, huggingface-hub 0.24.7, einops 0.7.0, sentencepiece 0.2.0, sphn 0.1.12;
  - fire, simple-parsing, pyyaml, tqdm, tensorboard, matplotlib, pyloudnorm, pandas, accelerate 1.15.0, aiohttp.
- PP moshi and moshi-finetune are on the path via `.pth` files (`personaplex_moshi.pth`, `moshi_finetune.pth`), with no install into either tree. `00_pycache_prefix.pth` sets `sys.pycache_prefix=/workspace/.pycache-ft` so nothing is written into the PP tree. `git status` there is clean, and I removed the 3 `__pycache__` dirs my first import had created.
- Not installed (unused): MF's upstream `moshi` dependency, whisper_timestamped/submitit/auditok (only annotate.py needs them).
- `venv_check.py` (flock'd), log in `venv_check.log`:
  - torch 2.14.1+cu130 sees the RTX PRO 6000 Blackwell (12,0); the bf16 matmul is OK.
  - `moshi.models.loaders`, `moshi.models.lm`, `moshi.offline`, `finetune.args` and `finetune.loss` import.
  - FAIL until patched: `finetune.data.interleaver` (moshi.conditioners), `finetune.wrapped_model` (CheckpointInfo), `finetune.checkpointing` and `train` (moshi.modules.lora).
- `ft_smoke.py` (flock'd), log in `ft_smoke.log`: the PP LM loads in 39.8 s. `forward_train` on B=1, T=300 random codes gives logits (1,16,300,2048) and text_logits (1,1,300,32000). Backward through the last temporal and depformer layers works. Peak VRAM 18.4 GB. SMOKE_OK.
