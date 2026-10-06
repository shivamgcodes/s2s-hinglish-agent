#!/bin/bash
# Regenerate trainer/personaplex_prefix.patch from the working tree of /workspace/moshi-finetune (no commit, no index change).
cd /workspace/moshi-finetune
OUT=/workspace/hinglish/trainer/personaplex_prefix.patch
{ echo "# moshi-finetune @ $(git rev-parse --short HEAD): PersonaPlex prefix/LoRA patch. Apply: git apply personaplex_prefix.patch"
  git diff
  for f in $(git ls-files --others --exclude-standard); do git diff --no-index /dev/null "$f"; done; } > $OUT
echo "$OUT: $(wc -l < $OUT) lines"; git diff --stat; git ls-files --others --exclude-standard
