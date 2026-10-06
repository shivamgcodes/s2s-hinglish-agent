#!/bin/bash
# E2E: 2 handmade V3 calls (f5cs) + 1 English control call (kokoro_en) through the real run_variant.sh path.
set -uo pipefail
A=/workspace/hinglish/audio
S=$A/f5_e2e/status
until grep -q "TEST DONE" $A/f5_test/status 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%T) e2e v3 START" >> $S
TTS_BACKEND=f5cs bash $A/run_variant.sh $A/f5_test/calls_v3.jsonl $A/f5_e2e/v3 2 > $A/f5_e2e/v3.log 2>&1
echo "$(date -u +%FT%T) e2e v3 rc=$?" >> $S
TTS_BACKEND=kokoro_en bash $A/run_variant.sh $A/f5_test/calls_ctl.jsonl $A/f5_e2e/ctl 2 > $A/f5_e2e/ctl.log 2>&1
echo "$(date -u +%FT%T) e2e ctl rc=$?" >> $S
echo "$(date -u +%FT%T) E2E DONE" >> $S
