# Trelis Whisper-Hinglish transcription (done by the orchestrator, 2026-10-04 ~04:20 IST)
- Model: Trelis/whisper-hinglish-preview (Whisper-large-v3 fine-tune from ARTPARK-IISc/whisper-large-v3-vaani-hindi; Apache-2.0), bf16 on the pod GPU (venv-tts, transformers>=5), under gpu.lock.
- Decoding per the model card: decoder prompt <|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>, greedy, max_new_tokens 440.
- Audio: the same model-output wavs large-v3 transcribed (/workspace/hinglish/tests/out/<tag>/V1/<call>_s1001.wav, 24 kHz mono), FFT-resampled to 16 kHz, split into <=28 s chunks at the lowest-energy 20 ms frame in the 18-28 s window (4-5 chunks per file), chunk texts joined with a space.
- Output script: mixed — Hindi in Devanagari, English in Latin, numbers written as English words (e.g. "four hundred fifty", "twenty october"), unlike large-v3 (digits) and the reference (digits).
- Script: trelis_tx.py; raw output trelis_raw.json; per-pair transcripts inputs/<pair>.whisper.txt.
