"""Opus round-trip for audio degradation (ffmpeg + libopus).
opus_roundtrip(a, sr=24000, bitrate="24k") -> float32 mono at sr, same length as input.
Encodes mono PCM -> Ogg/Opus (libopus, application voip, 20 ms frames) -> decodes back to sr.
"""
import subprocess
import numpy as np


def opus_roundtrip(a, sr=24000, bitrate="24k", application="voip", frame_ms=20):
    a = np.asarray(a, dtype=np.float32)
    enc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", bitrate, "-application", application, "-frame_duration", str(frame_ms),
         "-f", "ogg", "pipe:1"], input=a.tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(sr), "-ac", "1",
         "pipe:1"], input=enc, capture_output=True, check=True).stdout
    b = np.frombuffer(dec, dtype=np.float32).copy()
    # libopus adds pre-skip handled by ogg demux; pad/trim to input length
    if len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)))
    return b[: len(a)], len(enc)
