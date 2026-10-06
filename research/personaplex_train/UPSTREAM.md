# Upstream code this training depends on

## moshi-finetune (patched)

- Repo: https://github.com/kyutai-labs/moshi-finetune
- Commit: `2acc879fe7c48f885a18f6cc9548bccb2674d87b` (short `2acc879`, "format")
- Patch: `trainer/personaplex_prefix.patch`. It covers 7 files: modified `train.py`, `finetune/{args,checkpointing,eval,wrapped_model}.py` and `finetune/data/interleaver.py`, plus the new file `finetune/pp_lora.py`. It adds the PersonaPlex prefix (system prompt and voice prompt), weight-level LoRA on PersonaPlex parameter names, an agent-rows-only loss (first 8 depformer steps), a ported in-loop eval with named eval sets, and the `keep_and_shift` interleaver option.
- Apply:

  ```bash
  git clone https://github.com/kyutai-labs/moshi-finetune && cd moshi-finetune
  git checkout 2acc879fe7c48f885a18f6cc9548bccb2674d87b
  git apply /path/to/research/personaplex_train/trainer/personaplex_prefix.patch
  ```

  `trainer/make_patch.sh` regenerated the patch from the patched pod1 working tree. NOTES record that it applies to a clean `git archive` of 2acc879 and reproduces all 7 files byte for byte.
- Verified on 2026-10-07: the patch was regenerated read-only from pod1 `/workspace/moshi-finetune` with the same commands as `make_patch.sh` (`git diff 2acc879` plus `git diff --no-index` for the untracked `finetune/pp_lora.py`). It is byte-identical to the shipped patch, md5 `88ee53d9dff4ab29142c3647c4fa6804`.
- Environment: `trainer/build_venv_ft.sh` builds venv-ft. moshi-finetune and the PersonaPlex `moshi` package are both on the path through a `.pth` file.

## PersonaPlex (moshi package): unmodified

- Repo: https://github.com/NVIDIA/personaplex
- Commit: `3428dfd95309a7f3c84fd93259ded0f810d1ff91`. `git status` on pod1 `/workspace/personaplex` showed no modified or untracked files, so no patch is needed. venv-ft imports `/workspace/personaplex/moshi` through its `.pth`.
- Inference adds LoRA by merging weights at load time: `packages/personaplex_lora/infer/lora_merge.py` computes `W += s * B @ A` and recomputes voice-prompt embeddings at runtime. The PersonaPlex tree is never edited.
- Base weights: HF `nvidia/personaplex-7b-v1`. The scripts reference snapshot `fdaf4090a61cb315c138a1faee287ffd6c716309` (`voice_codes/*.py`).
