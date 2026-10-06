#!/bin/bash
A=/workspace/hinglish/audio
S=$A/f5_e2e/status
until grep -q "E2E DONE" $S; do sleep 20; done
rm -rf $A/f5_e2e/v3
echo "$(date -u +%FT%T) e2e v3 (final cfg: batch 1, rate cap 4.6) START" >> $S
TTS_BACKEND=f5cs bash $A/run_variant.sh $A/f5_test/calls_v3.jsonl $A/f5_e2e/v3 2 > $A/f5_e2e/v3.log 2>&1
echo "$(date -u +%FT%T) e2e v3 final rc=$?" >> $S
echo "E2E2 DONE" >> $S
