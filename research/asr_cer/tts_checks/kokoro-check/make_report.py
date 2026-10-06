"""Assemble REPORT.md from summary.json, synth_info.json, report_table.md and the hand-written observations."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
S = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
I = json.loads((HERE / "synth_info.json").read_text(encoding="utf-8"))
TABLE = (HERE / "report_table.md").read_text(encoding="utf-8")
sys.path.insert(0, str(HERE))
from analyze import LISTEN_FIRST  # noqa: E402

PPS = [float(row["pps_vs_voice_median"]) for row in csv.DictReader(open(HERE / "metrics.csv", encoding="utf-8"))]

r, c, L, sil = S["rtf"], S["mean_cer"], S["long_input"], S["silence"]
v = I["versions"]

listen = "\n".join(f"{n}. `{stem}` — {why}" for n, (stem, why) in enumerate(LISTEN_FIRST, 1))

report = f"""# Kokoro Hindi TTS check

Run on 2026-10-02. Facts only; no verdict. Nothing here was checked by ear: the transcript-back check and the
signal measurements are automatic, and the pronunciation notes come from Kokoro's own phoneme strings.

## Result summary

- All 40 clips (20 sentences x 2 voices) and the 2 long-input clips were produced. 24 kHz, mono, 16-bit.
- Kokoro's native output rate is 24 kHz, so nothing was resampled.
- RTF on CPU: hf_beta median {r['hf_beta']['median']}, max {r['hf_beta']['max']}; hm_omega median {r['hm_omega']['median']}, max {r['hm_omega']['max']}.
- No clip has clipping or an internal silence longer than 0.7 s. Every clip starts and ends with about 0.5 s of silence.
- Transcripts (Whisper large-v3) match the text closely for pure Hindi (group A) and for Devanagari-transliterated
  English (group D, hf_beta). The mismatches are concentrated in Latin-script English words (group B) and in
  proper nouns (group C); they are listed below.

## Versions and setup

| Item | Value |
|---|---|
| kokoro / misaki | {v['kokoro']} / {v['misaki']} |
| torch | {v['torch']} (CPU; `CUDA_VISIBLE_DEVICES=""`) |
| soundfile / numpy | {v['soundfile']} / {v['numpy']} |
| Python | {I['python']} |
| espeak-ng | 1.51 (apt, already installed) |
| Pipeline | `KPipeline(lang_code='h')`, G2P class `EspeakG2P`; loads without error |
| Voices | `hf_beta` (female), `hm_omega` (male); downloaded unauthenticated from hexgrad/Kokoro-82M |
| Native sample rate | {I['native_sample_rate']} Hz; resampled: {'yes' if I['resampled'] else 'no'} |
| CPU | AMD EPYC 7702 64-Core, 128 logical CPUs; torch used {I['torch_threads']} threads |
| ASR | faster-whisper 1.2.1, ctranslate2 4.8.2, model `large-v3`, CPU int8, beam 5 |

No extra pip package was needed for the Hindi pipeline; the existing `/workspace/venv-tts` was used unchanged.

## Speed (CPU, not GPU)

RTF = synthesis wall time / audio duration, per sentence, model and voice already loaded.

| Voice | Median RTF | Max RTF | Min RTF | Sentences |
|---|---|---|---|---|
| hf_beta | {r['hf_beta']['median']} | {r['hf_beta']['max']} | {r['hf_beta']['min']} | {r['hf_beta']['n']} |
| hm_omega | {r['hm_omega']['median']} | {r['hm_omega']['max']} | {r['hm_omega']['min']} | {r['hm_omega']['n']} |

- Reported separately, not in the RTF: pipeline load {I['pipeline_load_s']} s; first (warm-up) call {I['warmup_s']['hf_beta']} s for hf_beta and {I['warmup_s']['hm_omega']} s for hm_omega.
- The highest RTFs are the shortest clips (B2, D3, about 3.1 s of audio).
- The pod was shared at the time: a PersonaPlex job was using the GPU and other processes were running, so these
  numbers are for this CPU under that load. The GPU was not used because it was occupied.

## Long input (all five group-A sentences in one call)

| | hf_beta | hm_omega |
|---|---|---|
| Chunks produced by Kokoro | {L['hf_beta']['chunks']} | {L['hm_omega']['chunks']} |
| Duration, one call | {L['hf_beta']['audio_s']} s | {L['hm_omega']['audio_s']} s |
| Duration, sum of the five clips | {L['hf_beta']['sum_of_five_clips_s']} s | {L['hm_omega']['sum_of_five_clips_s']} s |
| Speech time (without leading/trailing silence), one call | {L['hf_beta']['speech_s_long']} s | {L['hm_omega']['speech_s_long']} s |
| Speech time, five clips | {L['hf_beta']['speech_s_five_clips']} s | {L['hm_omega']['speech_s_five_clips']} s |
| Ratio one call / five clips | {L['hf_beta']['speech_ratio']} | {L['hm_omega']['speech_ratio']} |
| Longest internal silence | {L['hf_beta']['max_internal_sil_s']} s | {L['hm_omega']['max_internal_sil_s']} s |
| Phoneme characters, one call / five clips | {L['hf_beta']['phoneme_chars_long']} / {L['hf_beta']['phoneme_chars_five_clips']} | {L['hm_omega']['phoneme_chars_long']} / {L['hm_omega']['phoneme_chars_five_clips']} |

- Truncation: the phoneme string of the single call contains all five sentences (408 characters, under Kokoro's
  510 limit), and a separate Whisper pass over the last 8 s of each clip returns the whole last sentence
  ("...असुविधा के लिए हमें खेद है आपका पैसा तीन दिन में वापस आ जाएगा"). The end of the text is present in the audio.
- The first whole-clip Whisper transcript stops early (hf_beta at "असुविधा के", hm_omega at "पहुँच जाएग"). That is
  Whisper's output ending, not the audio: the word-timestamp pass places the last word at 17.48 s of 18.12 s (hf_beta).
- Pauses: the sentence-final "।" does not appear in Kokoro's phoneme string (commas and "?" do), so in one call the
  sentences follow each other with at most {L['hf_beta']['max_internal_sil_s']} s / {L['hm_omega']['max_internal_sil_s']} s of silence.
- Rate: speech time in one call is {round((1 - L['hf_beta']['speech_ratio']) * 100)}% (hf_beta) and {round((1 - L['hm_omega']['speech_ratio']) * 100)}% (hm_omega) shorter than the five separate clips.
- Details: `long_tail_check.json`.

## Signal facts (all 40 clips)

| | hf_beta | hm_omega |
|---|---|---|
| Leading silence | {sil['hf_beta']['lead_min']}–{sil['hf_beta']['lead_max']} s | {sil['hm_omega']['lead_min']}–{sil['hm_omega']['lead_max']} s |
| Trailing silence | {sil['hf_beta']['trail_min']}–{sil['hf_beta']['trail_max']} s | {sil['hm_omega']['trail_min']}–{sil['hm_omega']['trail_max']} s |
| Longest internal silence in any clip | {sil['hf_beta']['max_internal']} s | {sil['hm_omega']['max_internal']} s |
| Highest peak (full scale = 1.0) | {sil['hf_beta']['peak_max']} | {sil['hm_omega']['peak_max']} |
| Clips with clipped samples | {sil['hf_beta']['clipped_clips']} | {sil['hm_omega']['clipped_clips']} |
| RMS level range | {sil['hf_beta']['rms_dbfs_min']} to {sil['hf_beta']['rms_dbfs_max']} dBFS | {sil['hm_omega']['rms_dbfs_min']} to {sil['hm_omega']['rms_dbfs_max']} dBFS |

- Silence = 20 ms windows below -45 dBFS.
- Speaking rate (phoneme characters per second against the voice's median over all 20 sentences) stays between
  x{min(PPS)} and x{max(PPS)} for every clip; no clip is flagged as short or long for its text.
- Per-clip values are in `metrics.csv`.

## Transcript-back check

Mean rough CER (character error rate of the Whisper transcript against the sentence):

| Group | hf_beta | hm_omega |
|---|---|---|
| A, pure Devanagari Hindi | {c['A_hf_beta']['cer_raw']} | {c['A_hm_omega']['cer_raw']} |
| B, against the sentence as written (Latin words) | {c['B_hf_beta']['cer_raw']} | {c['B_hm_omega']['cer_raw']} |
| B, against the hand-transliterated reference | {c['B_hf_beta']['cer_translit']} | {c['B_hm_omega']['cer_translit']} |
| B, lower of the two per clip | {c['B_hf_beta']['cer_best']} | {c['B_hm_omega']['cer_best']} |
| C, pure English | {c['C_hf_beta']['cer_raw']} | {c['C_hm_omega']['cer_raw']} |
| D, English words in Devanagari | {c['D_hf_beta']['cer_raw']} | {c['D_hm_omega']['cer_raw']} |

How to read the CER:
- It is rough. Both sides are normalised (punctuation and symbols removed, whitespace collapsed, Latin lowercased,
  chandrabindu folded to anusvara, nukta removed). Whisper's own errors and spelling choices are included.
- Number format counts as error: "पंद्रह" against Whisper's "15" (A4), "fifteen" against "15" (C3).
- For group B, Whisper wrote the English words in Latin in some clips (B4, B5, B7 for hf_beta; B5 for hm_omega)
  and in Devanagari in the others. "CER raw" is against the sentence as written; "CER translit" is against a
  reference with the Latin words transliterated to Devanagari by hand. The lower of the two is the meaningful one
  per clip, and the flags use it.
- Flags: "CER x" when the lower CER is above 0.15; signal flags if any; and hand-written notes from reading the
  transcript and Kokoro's phoneme string.

{TABLE}
## What the phoneme strings show about numbers and symbols

These are facts from Kokoro's phoneme output (`timings.csv`, column `phonemes`), not from listening.

- Numbers are read in Hindi: "20" as बीस, "15" as पंद्रह, "6" as छह, "250" as दो सौ पचास.
- "₹250" becomes दो सौ पचास only; no word for the rupee sign appears. "250 rupees" keeps the word "rupees".
- "A1234" becomes "A" followed by एक हज़ार दो सौ चौंतीस, a cardinal number, not the digits one by one.
- In the English sentence C1, "Sector 21" has "21" phonemised as Hindi इक्कीस. C3 spells "fifteen" in letters
  and is phonemised as English.
- Latin-script words go through espeak's English phonemes, which are non-rhotic: "order" is /ˈɔːdə/,
  "partner" /ˈpɑːtnə/, "deliver" /dɪˈlɪvə/. The Devanagari spellings go through Hindi phonemes with the r kept
  (ऑर्डर /ˈɔɾɖəɾ/, डिलीवर /ɖɪˈliːʋəɾ/).
- The sentence-final "।" is not passed on as punctuation; "," and "?" are.

## Group B versus group D, per pair

From the transcripts and phoneme strings only; not verified by ear. In no clip is an English word spelled out
letter by letter (apart from "ID", which is naturally read as letters).

| Pair | English words | Latin script (group B) | Devanagari (group D) |
|---|---|---|---|
| B3 / D1 | order, deliver | Pronounced through English phonemes. Both voices transcribed as "ओदर", "डिलिवर": present, "order" without an r. | Hindi phonemes. Both voices transcribed as "ओर्डर", "डिलीवर": present, r kept. |
| B1 / D2 | address, Sector, Faridabad, update | English phonemes. hf_beta: "अड्रेस सेक्टर 15 फाविदबाद ... अपडेट". hm_omega: "अड्रेस 115 फाविदबाद ... अपदेट"; "Sector" is absent from the transcript. "Faridabad" comes back as "फाविदबाद" in both voices. | Hindi phonemes. hf_beta transcript is exact. hm_omega: "एड्रेस" comes back as "Aris"; "Sector 15 फरीदाबाद ... अपडेट" present. |
| B2 / D3 | check | English phonemes /tʃɛk/. Transcribed "चेक" in both voices. | Hindi phonemes /ceːk/. Transcribed "चेक" in both voices. |

Other Latin-script words in group B whose transcript differs from the word (same caveat):
"ready" → "वेदी", "restaurant" → "रेस्टरंट", "order" → "आउडर" (B4, hm_omega); "partner" → "पातन" (B7, hm_omega);
"refund" → "रिफंद" (B6, both voices). In the hf_beta versions of B4 and B7 Whisper wrote these words in Latin,
spelled correctly.

## Clips to listen to first

{listen}

`listen.html` has these at the top with players, then every sentence with the two voices side by side.
`ratings.csv` is an empty sheet (id, voice, sentence, rating_1to5, notes) for your ratings.

## Deviations from the spec

1. Ran on the RunPod pod in `/workspace/kokoro-check/`, not in a local `./kokoro-check/`, because Kokoro is kept on
   the pod only. Everything was then copied to `<laptop>/kokoro-check/`.
2. CPU only. The spec says to use the GPU if present; the pod's one GPU was occupied by a PersonaPlex job, so the
   RTF numbers are CPU numbers.
3. `pip install kokoro soundfile` and `apt install espeak-ng` were not run again: kokoro 0.9.4, soundfile and
   espeak-ng 1.51 were already installed in `/workspace/venv-tts`.
4. All sentences, including the pure-English group C, were synthesised with the Hindi pipeline (`lang_code='h'`)
   and the two Hindi voices, as the spec's voice list implies.
5. Whisper: the local machine's `large-v3` cache is incomplete (no model file), so faster-whisper was installed in
   a new venv on the pod (`/workspace/venv-asr`, 1.1 GB). Downloading the model to `/workspace/hf` failed with
   "Disk quota exceeded", so the model (2.9 GB) is cached on the pod's container disk at `/root/hf-asr`, which is
   lost when the pod restarts.
6. Two CER columns for group B instead of one, for the reason given above.
7. The spec's "Checks" step (you listening and rating) is prepared, not done: `listen.html` and `ratings.csv`.

## Files

| What | Pod | Local |
|---|---|---|
| Clips (40 + 2 long) | `/workspace/kokoro-check/out/` | `<laptop>/kokoro-check/out/` |
| Sentences | `sentences.txt`, `sentences.tsv` | same names |
| Scripts | `synth.py`, `asr.py`, `asr_long_tail.py`, `asr_install.sh`, `analyze.py`, `make_report.py` | same names |
| Raw data | `timings.csv`, `synth_info.json`, `transcripts.json`, `long_tail_check.json`, `metrics.csv`, `summary.json` | same names |
| Listening page and ratings sheet | | `listen.html`, `ratings.csv` |
| Logs | `logs/` | `logs/` |
"""
(HERE / "REPORT.md").write_text(report, encoding="utf-8")
print("REPORT.md written,", len(report.splitlines()), "lines")
