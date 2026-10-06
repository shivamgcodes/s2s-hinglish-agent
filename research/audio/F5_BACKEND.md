# F5 code-switch backend + English control voices (V3 audio)

Status: FINAL. The script decision is confirmed by the duration-fix re-test (2026-10-03 14:28 UTC); numbers are below.

## Decision for the prompts/generator worker: text_tts script for V3 = ALL-DEVANAGARI (same as V1)

- `text_tts`: Hindi words AND English words both written in Devanagari, exactly as in V1.
  - Example: `जी, ड्राइवर संदीप इज़ ऑन द वे, और ईटीए अभी 8 मिनट्स है।`
  - Numbers, IDs, amounts, times, phones and emails may stay as digits/Latin inside `text_tts`, as in V1. `tts_norm` already turns them into spoken English words written in Devanagari.
  - Leftover Latin letters or digits are spelled by `tts_backends.f5_text`, so raw digits never reach F5.
  - Do NOT write ordinary English words in Latin inside `text_tts`. A leftover Latin word would be spelled out letter by letter.
- Numbers in Hindi words: data values such as IDs, amounts, times and phone numbers should be digits in `text_tts`/`text_roman`, as in V1. They are spoken as English number words.
  - Hindi number words ("बीस", "पाँच") are fine in the audio, but Whisper writes them as digits, which inflates CER.
  - In the e2e test, "बीस" scored 0.267 and "पाँच" 0.081 on clean renders.
  - Many of them per line would cause false rejects and retries.
- `text_roman`: unchanged, same as V1. It is the romanised Hinglish transcript, and MMS alignment and the training transcript use it.
- Why this choice: in the Kokoro-vs-F5 comparison (`hinglish/tts_compare/compare_table.txt`), the code-switch model was tried with both scripts.
  - All-Devanagari input gave clean Whisper transcripts on every voice. English words came back correctly, often in Latin script.
  - Mixed-script input (Hindi in Devanagari, English in Latin) was garbled for both Orato voices and for ritu: e.g. "Reformed Reasons Fair Hazard", and S3/S6 lost whole phrases.
  - Mixed-script input was also rushed: mean 3.70 s vs 6.25 s.
  - Part of the rush is the model's byte-length duration rule, now fixed here.
  - Re-test after the fix: all-Devanagari still wins on every voice, with lower CER and English words intact. Mixed-script still garbles "flight"->"flats", "departure 7:15 PM"->"डिपार्टर 715 इनी", "refund"->"रिफॉन्ड". See "Measurements" below.
- All-Devanagari also reuses `tts_norm` (number words, CER reference, `map_roman` chunk split) unchanged.

## Voices (keys go in calls.jsonl `voices`)

| role | V3 Hinglish (TTS_BACKEND=f5cs) | CONTROL English (TTS_BACKEND=kokoro_en) |
|---|---|---|
| agent female | `ritu_hinglish` | `af_heart` |
| agent male | `orato_male` | `am_michael` |
| customer female | `orato_female` | `af_bella` |
| customer male | `fleurs_hi_m1559` | `am_fenrir` |

Group mapping (same g1..g4 meaning as V1):

| group | customer | agent | agent_gender | f5cs voices {customer, agent} | kokoro_en voices {customer, agent} |
|---|---|---|---|---|---|
| g1 | male | female | f | fleurs_hi_m1559, ritu_hinglish | am_fenrir, af_heart |
| g2 | male | male | m | fleurs_hi_m1559, orato_male | am_fenrir, am_michael |
| g3 | female | female | f | orato_female, ritu_hinglish | af_bella, af_heart |
| g4 | female | male | m | orato_female, orato_male | af_bella, am_michael |

`voice_prompt` stays NATF2 for agent_gender f and NATM1 for m (assemble.py, unchanged).

Reference clips: `/workspace/hinglish/audio/voices_f5/`, registry `voices_f5.json` (wav, exact transcript, source, license).
- ritu_hinglish: from Tharshan/indicf5_hindi-english_code_switch `voices/` (rev f6781208).
  - The repo license is apache-2.0. The repo states the clip was generated with Sarvam bulbul:v3.
  - Its transcript is mixed-script (from the repo's voices.json).
- orato_male, orato_female: from tryorato/orato-tts-hindi-v1 `voices/` (repo license MIT), transcripts from its voices.json.
- fleurs_hi_m1559: a real Hindi male speaker.
  - Source: google/fleurs, config hi_in, validation split, id 1559 (datasets-server row 10). Gender label = male; median F0 129 Hz.
  - License: CC-BY-4.0 (FLEURS dataset card). Attribution: Conneau et al., FLEURS, Google.
  - Processing: the original 16 kHz clip, trimmed to 0.89-8.52 s (7.63 s) and resampled to 24 kHz PCM16.
  - Transcript: the FLEURS raw_transcription with one change, "शब्दों का" -> "शब्दों को". Whisper large-v3 hears "को", and the speaker deviated from the prompt.
  - Only this one file was put on the pod. 5 other candidate clips stayed local and are unused.
- Kokoro English voices: af_heart was already cached. am_michael, am_fenrir, af_bella and am_adam were downloaded into `/workspace/hf` (hexgrad/Kokoro-82M voices/*.pt; Apache-2.0).

## Usage

```bash
# V3 Hinglish (IndicF5 code-switch; tts stage runs in venv-f5, everything else as V1)
TTS_BACKEND=f5cs bash /workspace/hinglish/audio/run_variant.sh /workspace/hinglish/data/V3/calls.jsonl /workspace/hinglish/data/V3
# CONTROL English (Kokoro lang 'a')
TTS_BACKEND=kokoro_en bash /workspace/hinglish/audio/run_variant.sh /workspace/hinglish/data/V3ctl/calls.jsonl /workspace/hinglish/data/V3ctl
# V1 (unchanged; default backend)
bash /workspace/hinglish/audio/run_variant.sh /workspace/hinglish/data/V1/calls.jsonl /workspace/hinglish/data/V1
```

- The backend is chosen by env `TTS_BACKEND` at the plan stage. It is recorded per chunk in `work/plan.json` (`backend`, `lang`).
  - `synth.py tts` dispatches on it: f5cs goes to `f5cs.cmd_tts`.
  - `run_variant.sh` picks venv-f5 for the f5cs tts stage.
  - A non-kokoro backend is added to the stale-output `.sig`, so switching backend wipes that call's chunks.
- Control-call schema: same `calls.jsonl` schema. Each turn has `text_tts` = `text_roman` = English text, with numbers/IDs as in V1, e.g. "FD1000", "Rs 540", "6:40 PM".
  - `tts_backends.en_text` makes the spoken form, e.g. "F D one zero zero zero", "five hundred forty rupees", "six forty P M".
  - ASR for these chunks uses language `en` with `cer_en`, an English-normalised a-z0-9 CER. MMS_FA aligns `text_roman` exactly as in V1.
- CONTROL must never be trained on. This is the user's decision: train only on the V3 Hinglish split.
  - `assemble.py` writes `ROOT/train/train.jsonl` for every call whose `scenario_id` is not in `holdout.json`.
  - Build the control set from the held-out scenarios/records only, and use `<ctl_root>/heldout/heldout_all.jsonl` (or `manifest/all.jsonl`) as the eval-only `EVAL_DATA`.
  - Never pass the control root's `train/train.jsonl` to the trainer.
  - The e2e test call below reports `train_calls: 1` only because its scenario is a test id that is not in holdout.json.
- Control loudness: kokoro_en chunks are RMS-normalised to -20 dBFS, the same as f5cs. The V1 kokoro path is not normalised and is unchanged. Without this, the Kokoro English agent channel measured -24.5 dBFS vs -19 dBFS for V3.
- Unchanged path: build, align, assemble and qc are untouched. `voice_prompt` is NATF2/NATM1 from `agent_gender`.
- Reject loop: the same as V1.
  - Whisper large-v3 CER threshold 0.35, up to 3 tries. Try 0 is speed 1.0; tries 1-2 use speed U(0.95, 1.05) seeded per chunk, plus a new sampler seed derived from the chunk stem and the try.
  - Best try is kept.
- Chunking (f5cs): <= 20 words, <= 120 Devanagari chars, <= 170 spoken chars per chunk, split at punctuation/conjunction/postposition as in V1. kokoro_en: <= 20 words.
- Files changed:
  - `synth.py`, `asr.py`, `run_variant.sh`. Backups are `*.bak_pre_f5cs`.
  - New: `tts_backends.py` (stdlib text side), `f5cs.py` (venv-f5 engine), `voices_f5/`.
  - `tts_norm.py` is NOT modified.
- V1 reproducibility check: the patched `synth.py plan` with the default backend on `data/V1/calls.jsonl` gives a plan.json byte-identical to `data/V1/work/plan.json` (cmp). All per-call `.sig` files are identical too.

## f5cs engine (f5cs.py)

- Loading: the model loads once per process by importing the HF snapshot as a package (`pkgs/codeswitch` symlink -> snapshot). `AutoModel(trust_remote_code)` fails on transformers 4.49.
- Duration fix: the repo's `generate()` sizes output as `ref_len * bytes(text)/bytes(ref_text)`. f5cs instead calls the CFM sampler with an explicit duration:
  - `frames = ref_len + (syllables(text) * frames_per_syllable + EDGE_S) / speed`, with EDGE_S 0.8 s (env F5_EDGE_S). It was 0.25 s before the review fix; see "Review".
  - `frames_per_syllable` comes from each voice's own reference rate (syllables(transcript) / voiced seconds), clamped to [3.6, 4.6] syllables/s (`F5_RATE_MIN/MAX`).
  - Measured reference rates: ritu 4.71, orato_male 5.08, orato_female 5.99, fleurs 6.91 syl/s. So all four voices run at 4.6.
  - `syllables()` counts Devanagari aksharas (with word-final schwa deletion) and Latin vowel groups.
- Sampler settings: NFE 32, CFG 2.0, sway -1 (repo defaults).
- REVIEW FIX (f5review, 2026-10-03 ~15:30 UTC): tail room EDGE_S 0.25 -> 0.8 s (env F5_EDGE_S), plus a clipped-ending guard (see "Review" section). The duration formula is now `frames = ref_len + (syllables * frames_per_syllable * F5_RATE_SCALE + EDGE_S) / speed`.
- The engine params (edge_s, rate_min/max, rate_scale, nfe, tail_db, tail_extra_s) come from `tts_backends.f5_params()` (env at PLAN time). They are written into plan.json (`"f5"`) and into the f5cs `.sig`, so a changed setting wipes and re-renders that call. `f5cs.cmd_tts` uses the plan's values, not the env at tts time.
- Input checks: every input char is checked against the model vocab, and raises if one is missing. `—`, `₹`, quotes etc. are mapped first.
- Output: trim (synth.trim), loudness-normalised to -20 dBFS RMS (peak <= 0.99), 24 kHz PCM16.
- Batching: `F5_BATCH` defaults to 1. Batch 8 was measured both slower and worse:
  - Speed: 1.6-2.5 s/chunk at batch 8 vs 1.2-1.5 s at batch 1.
  - Quality: the batched sampler cut sentence endings. Same seed, same text: ritu CER 0.21 batched vs 0.09 single; fleurs 0.34 vs 0.07. E.g. S3 batched ended at "...और ETA".

## Measurements: S1-S8 (comparison sentences) x 4 voices, batch 1, Whisper large-v3 hi, CER = tts_norm.cer_best vs the all-Devanagari spoken form

Rate cap 4.6 (chosen):

| voice | input | mean dur s | Kokoro dur s | mean CER | max CER | n>0.35 | pauses>80ms | syl/s |
|---|---|---|---|---|---|---|---|---|
| ritu_hinglish | deva | 5.22 | 5.97 | 0.117 | 0.300 | 0 | 3.88 | 4.96 |
| ritu_hinglish | mixed | 5.01 | 5.97 | 0.149 | 0.385 | 1 | 4.12 | 4.91 |
| orato_male | deva | 5.32 | 5.97 | 0.143 | 0.271 | 0 | 2.88 | 4.86 |
| orato_male | mixed | 5.15 | 5.97 | 0.161 | 0.315 | 0 | 4.12 | 4.76 |
| orato_female | deva | 5.48 | 5.97 | 0.117 | 0.247 | 0 | 6.12 | 4.75 |
| orato_female | mixed | 5.24 | 5.97 | 0.199 | 0.400 | 1 | 6.00 | 4.68 |
| fleurs_hi_m1559 | deva | 5.28 | 5.97 | 0.147 | 0.326 | 0 | 2.62 | 4.89 |
| fleurs_hi_m1559 | mixed | 5.07 | 5.97 | 0.179 | 0.339 | 0 | 1.75 | 4.84 |

Rate cap 5.0 gave deva CER 0.141 / 0.186 / 0.120 / 0.116 and mixed 0.180 / 0.201 / 0.198 / 0.240, in the same voice order as the table.
- Durations were 3-6% shorter than at cap 4.6. Cap 4.6 was kept so speech is less rushed; CER is about equal.
- "Kokoro dur" is the V1 dataset utterance length, which includes 0.12-0.2 s inter-chunk gaps.
- Before the fix, the comparison's mixed-script mean duration was 3.70 s.
- Data: `/workspace/hinglish/audio/f5_test_b1r46/` and `f5_test_b1r50/` (scored.json, wavs); batched run in `f5_test/`.
- The CER here is on whole sentences. Whisper writes English words in either script; cer_best takes the min over the Devanagari and folded-Latin keys.

## End-to-end test (task item 6): /workspace/hinglish/audio/f5_e2e/

> SUPERSEDED: these numbers use the old EDGE_S 0.25. For current numbers see the "Review" section and `/workspace/hinglish/review_f5/e2e_v3`. Projection at 1.43 s/chunk: about 2.4 h TTS for 6,000 chunks, about 2.8-3 h GPU with Whisper and retries.

- Inputs: `f5_test/calls_v3.jsonl` has 2 handmade V3 Hinglish calls.
  - v3test_food_g1: 12 turns, voices fleurs_hi_m1559 / ritu_hinglish. Includes check_line, a read turn, ID FD1000, and a truncated agent turn with a 0.5 s customer overlap.
  - v3test_cab_g4: 10 turns, voices orato_female / orato_male. Includes ID RD5294, 6:40 PM, a plate MH 02 AB 4821, Rs 540.
  - `f5_test/calls_ctl.jsonl` has 1 English control call: ctltest_food_g1, the same scenario in English, voices am_fenrir / af_heart.
- Run: through the real `run_variant.sh` (plan -> tts/asr x3 -> build -> MMS align -> assemble -> qc plots -> mimi check), GPU stages under gpu.lock.
- Outputs:
  - Stereo 24 kHz wavs in `f5_e2e/{v3,ctl}/stereo/*.wav` (ch0 = agent).
  - Alignment JSON next to each wav, carrying role_prompt, voice_prompt NATF2/NATM1, voices and the turn timeline.
  - Manifests in `manifest/all.jsonl`; QC in `qc/qc_summary.json`.
- v3 (final config):
  - 24 chunks (2 lines split), 0 CER rejects, 0 retries.
  - Per-chunk CER: mean 0.043, max 0.267. The max is "बीस" transcribed as "20"; the audio is correct.
  - 0 align fallbacks, 0 low-score alignments, 1 interruption.
  - Call durations 59.6 s and 54.9 s. Channel RMS -18.8 to -19.5 dBFS.
- ctl, re-run after adding loudnorm: 12 chunks, mean CER 0.006, max 0.033, 0 rejects, 0 align fallbacks. Channel RMS -20.0 (agent) and -19.5 (customer) dBFS.
- Speed, measured under the lock with the GPU otherwise idle:
  - f5cs: 1.36 s per chunk (batch 1, NFE 32), model load 50 s, peak torch VRAM 1.43 GiB.
  - Whisper large-v3: ~0.75 s per chunk per shard (2 shards).
  - kokoro_en: 0.06 s per chunk.
- Projection for ~6,000 V3 chunks: TTS ~2.3 h (try 0), ASR ~0.3 h with 4 shards, plus retries, align and assemble (minutes). About 2.7-3 h of GPU in total.

## Review (f5review worker, 2026-10-03 ~15:00-15:45 UTC)

- DEFECT FIXED, clipped chunk endings. At EDGE_S 0.25, 4/24 e2e chunks and 10/64 S1-S8 renders ended while speech was still loud: the last 40 ms were -5 to -13 dB against the chunk's p95 frame RMS, and the tail envelope shows a hard stop. Example: "थैंक यू अर्जुन" was heard as "अर्जु". Kokoro V1 had 0/400 such chunks; its median end level is -31 dB. CER did not catch this.
  - A/B test on the 24 e2e texts x 3 seeds, Whisper large-v3 (`/workspace/hinglish/review_f5/edge/`):

    | edge_s | loud ends | mean CER | syl/s | mean trimmed duration |
    |---|---|---|---|---|
    | 0.25 | 15% | 0.049 | 5.05 | 4.22 s |
    | 0.5 | 10% | 0.029 | 4.88 | 4.36 s |
    | 0.8 | 1% | 0.033 | 4.84 | 4.40 s |

  - At 0.8 the tail stays at a -31.5 dB median, so the extra frames became trimmed silence plus slightly slower speech, not noise.
  - Guard: in `f5cs.cmd_tts`, if the last 40 ms is above -15 dB (`F5_TAIL_DB`), the chunk is re-rendered once with the same seed and +0.5 s, and the version with the cleaner ending is kept. timing.jsonl counts this in `tail_rerender` and `tail_fixed`. Forced test with tail_db -32: 6 re-renders, 2 fixed.
- Fresh e2e with the new default (`/workspace/hinglish/review_f5/e2e_v3`, via run_variant.sh): 24 chunks, 0 rejects, mean CER 0.039, 0 loud ends, 0 tail re-renders, 0 align fallbacks, 4.79 syl/s (V1 Kokoro sample: 4.56 syl/s), 1.43 s/chunk. Expected TTS cost is about 5% more per chunk than before.
- Verified OK, no change needed:
  - V1 plan.json is byte-identical after the patch (cmp), and 287/287 .sig files match. The ctl (kokoro_en) plan and sig are also identical to the builder's.
  - f5cs plan over all V1 calls (4311 chunks): no exception, 0 chars outside vocab.txt, 0 leftover Latin or digits. Longest chunk is about 10.5 s, plus a 7.6-10.6 s reference.
  - 24 kHz stereo. ch0 = agent: on every turn the speaker's own channel is louder (34/34).
  - Median F0: ritu 223 Hz (f), orato_male 102 (m), orato_female 249 (f), fleurs 119 (m). The g1/g4 voices and NATF2/NATM1 match agent_gender.
  - Alignment JSON has role_prompt, voice_prompt, voices, turns and alignments.
  - Retry loop (scratch copy, 3 try0 CERs faked to 0.9): only those 3 rendered at try1, with new audio at speeds 1.004/0.973/0.981. asr scored them, try2 had nothing to do, and build picked try1 (`retried: 3`).
  - Resume: 2 deleted try0 wavs regenerated and came out byte-identical (md5), so rendering is deterministic. A no-op tts stage exits in 8 s without loading the model.
- Caveats (not fixed):
  - f5_text spells any leftover Latin letter by letter, so an email becomes "आर ओ एच ए एन डॉट ...". There are 0 `f5_latin_spelled` flags on V1 data. If V3 text has emails, check that flag count.
  - Hindi number words ("बीस") still give CER 0.2-0.27 from Whisper writing digits. This is under the 0.35 threshold.
  - Control `train/train.jsonl` must never reach the trainer (unchanged; documented above).
- Backups: `f5cs.py|synth.py|tts_backends.py.bak_pre_f5review`.

## D2 long band (2026-10-04)
- Chunking for V4 long-band calls: `LIMITS["f5cs_long"]` = <= 24 words, <= 140 Devanagari chars, <= 200 spoken chars. Chosen per call at plan time when `length_band` is in env `F5_LONG_BANDS` (default `long`; e.g. `F5_LONG_BANDS=long,mixed`). Calls without `length_band` (V1/V3/CONTROL) keep the limits above; their plan.json and .sig are byte-identical. Long-band chunks carry `"limits": "f5cs_long"` in plan.json.
- Verified on 40 synthetic 21-24-word lines (tests/out/d2_f5/RESULTS.json): pre-guard loud ends 1/40, post-guard 0/40, CER mean 0.080 (0 > 0.35), 1.88 s/chunk; the same lines split with the old limits: 2/80, 1/80, CER 0.094.
- Parallel f5cs processes (`synth.py tts ROOT --try K --shard I --nshards N`) do NOT raise throughput: aggregate 1.50 s/chunk at N=1 vs 1.58-1.59 at N=2-4 (GPU already ~99% busy); run_variant.sh therefore runs one process. Output is byte-identical for any N.
