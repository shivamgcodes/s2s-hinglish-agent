#!/bin/bash
A=/workspace/hinglish/audio
echo "$(date -u +%FT%T) e2e ctl (loudnorm) START" >> $A/f5_e2e/status
TTS_BACKEND=kokoro_en bash $A/run_variant.sh $A/f5_test/calls_ctl.jsonl $A/f5_e2e/ctl 2 > $A/f5_e2e/ctl.log 2>&1
echo "$(date -u +%FT%T) e2e ctl rc=$?" >> $A/f5_e2e/status
echo "E2E3 DONE" >> $A/f5_e2e/status
