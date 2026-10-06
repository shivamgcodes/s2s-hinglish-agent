"""Kokoro Hindi TTS check: per-clip signal facts, rough CER against the Whisper transcript, report tables,
a local listening page and an empty ratings sheet.

Run locally from this folder after copying out/*.wav, timings.csv, synth_info.json and transcripts.json:
  python3 analyze.py
Writes metrics.csv, summary.json, report_table.md, listen.html, ratings.csv.
"""
import csv
import html
import json
import statistics
import unicodedata
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
VOICES = ["hf_beta", "hm_omega"]
SIL_DBFS = -45.0      # a 20 ms window below this level counts as silence
WIN_S = 0.02

# Group B sentences with the Latin-script words transliterated to Devanagari by hand. Whisper (language=hi)
# usually writes English words in Devanagari, so CER against this reference is the fairer of the two for group B.
B_TRANSLIT = {
    "B1": "ठीक है, मैं आपका एड्रेस सेक्टर 15 फरीदाबाद पर अपडेट कर देती हूँ।",
    "B2": "एक मिनट, मैं चेक करके बताती हूँ।",
    "B3": "आपका ऑर्डर 20 मिनट में डिलीवर हो जाएगा।",
    "B4": "आपका ऑर्डर आईडी ए1234 है, और वह अभी रेस्टोरेंट में रेडी हो रहा है।",
    "B5": "इस ऑर्डर का टोटल ₹250 है, और पेमेंट हो चुका है।",
    "B6": "आपको 250 रुपीज़ का रिफंड दो दिन में मिल जाएगा।",
    "B7": "डिलीवरी पार्टनर शाम 6 बजे तक आपके लोकेशन पर पहुँच जाएगा।",
    "B8": "क्या आप अपना फोन नंबर कन्फर्म कर सकते हैं?",
}
# Group D sentence -> its group B counterpart.
PAIRS = {"D1": "B3", "D2": "B1", "D3": "B2"}

# Written by hand after reading the transcripts and the phoneme strings. clip stem -> list of flags.
HAND_FLAGS = {
    "C1_hf_beta": ["'21' phonemised as Hindi इक्कीस inside an English sentence (phoneme string)"],
    "C1_hm_omega": ["'21' phonemised as Hindi इक्कीस inside an English sentence (phoneme string)"],
    "B4_hf_beta": ["'A1234' phonemised as 'A' + Hindi cardinal (एक हज़ार दो सौ चौंतीस), not digit by digit"],
    "B4_hm_omega": ["'A1234' phonemised as 'A' + Hindi cardinal (एक हज़ार दो सौ चौंतीस), not digit by digit",
                    "transcript: order→आउडर, restaurant→रेस्टरंट, ready→वेदी"],
    "B5_hf_beta": ["'₹' not spoken: phonemes are दो सौ पचास only"],
    "B5_hm_omega": ["'₹' not spoken: phonemes are दो सौ पचास only"],
    "B1_hf_beta": ["transcript: Faridabad→फाविदबाद"],
    "B1_hm_omega": ["transcript has '115' where 'Sector 15' is expected (no 'Sector')",
                    "transcript: Faridabad→फाविदबाद, update→अपदेट"],
    "B3_hf_beta": ["transcript: order→ओदर"],
    "B3_hm_omega": ["transcript: order→ओदर"],
    "B6_hf_beta": ["transcript: refund→रिफंद"],
    "B6_hm_omega": ["transcript: refund→रिफंद"],
    "B7_hm_omega": ["transcript: partner→पातन"],
    "D2_hm_omega": ["transcript: एड्रेस→'Aris'"],
    "C3_hf_beta": ["transcript: Connaught→Kanot"],
    "C3_hm_omega": ["transcript: Connaught→Kanawha"],
    "C4_hf_beta": ["transcript: Haldiram's→Haldaram's"],
    "C4_hm_omega": ["transcript: Haldiram's→Helder Arms"],
}
# Written by hand: clips to listen to first, in order. (clip stem, reason)
LISTEN_FIRST = [
    ("B1_hm_omega", "Transcript has '115' where 'Sector 15' is expected; Faridabad→फाविदबाद; update→अपदेट."),
    ("B4_hm_omega", "Transcript: order→आउडर, restaurant→रेस्टरंट, ready→वेदी. 'A1234' is phonemised as 'A' + एक हज़ार दो सौ चौंतीस."),
    ("B4_hf_beta", "'A1234' is phonemised as 'A' + एक हज़ार दो सौ चौंतीस (a cardinal number, not digit by digit)."),
    ("B5_hf_beta", "'₹250' is phonemised as दो सौ पचास with no currency word."),
    ("B5_hm_omega", "'₹250' is phonemised as दो सौ पचास with no currency word."),
    ("C1_hf_beta", "English sentence, but '21' is phonemised as Hindi इक्कीस."),
    ("C1_hm_omega", "English sentence, but '21' is phonemised as Hindi इक्कीस."),
    ("B3_hf_beta", "Transcript: order→ओदर (Latin-script 'order' goes through English phonemes with no r)."),
    ("B3_hm_omega", "Transcript: order→ओदर."),
    ("B7_hm_omega", "Transcript: partner→पातन."),
    ("B1_hf_beta", "Transcript: Faridabad→फाविदबाद."),
    ("D2_hm_omega", "Transcript: एड्रेस→'Aris' (the same clip in hf_beta transcribes exactly)."),
    ("C3_hm_omega", "Transcript: Connaught→Kanawha."),
    ("C3_hf_beta", "Transcript: Connaught→Kanot."),
    ("C4_hm_omega", "Transcript: Haldiram's→Helder Arms."),
    ("B6_hf_beta", "Transcript: refund→रिफंद, rupees→रूपीज."),
    ("long_A_hf_beta", "Five sentences in one call: 18.1 s against 24.1 s as five clips; longest pause inside is 0.34 s."),
    ("long_A_hm_omega", "Five sentences in one call: 21.6 s against 27.6 s as five clips; longest pause inside is 0.46 s."),
]


def norm(text):
    """Rough normalisation for CER: NFC, drop punctuation/symbols, fold chandrabindu to anusvara, drop nukta,
    lowercase Latin, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).lower().replace("ँ", "ं").replace("़", "")
    out = []
    for ch in unicodedata.normalize("NFD", text):
        cat = unicodedata.category(ch)
        if ch == "़":
            continue
        out.append(" " if cat[0] in "PSZ" else ch)
    return " ".join(unicodedata.normalize("NFC", "".join(out)).split())


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def signal_facts(path):
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    win = int(sr * WIN_S)
    n = len(x) // win
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1)) / 32768.0
    silent = 20 * np.log10(np.maximum(rms, 1e-9)) < SIL_DBFS
    voiced = np.flatnonzero(~silent)
    lead = voiced[0] * WIN_S if len(voiced) else len(x) / sr
    trail = (n - 1 - voiced[-1]) * WIN_S if len(voiced) else 0.0
    longest = run = 0
    if len(voiced):
        for s in silent[voiced[0]:voiced[-1] + 1]:
            run = run + 1 if s else 0
            longest = max(longest, run)
    total_rms = np.sqrt((x ** 2).mean()) / 32768.0
    return {
        "sample_rate": sr, "channels": ch, "bits": width * 8,
        "duration_s": round(len(x) / sr, 3),
        "peak": round(float(np.abs(x).max()) / 32768.0, 3),
        "rms_dbfs": round(float(20 * np.log10(max(total_rms, 1e-9))), 1),
        "lead_sil_s": round(float(lead), 2),
        "trail_sil_s": round(float(trail), 2),
        "max_internal_sil_s": round(longest * WIN_S, 2),
        "clipped_samples": int((np.abs(x) >= 32767).sum()),
    }


def main():
    sentences = {r["id"]: r for r in csv.DictReader(open(HERE / "sentences.tsv", encoding="utf-8"), delimiter="\t")}
    timings = {(r["id"], r["voice"]): r for r in csv.DictReader(open(HERE / "timings.csv", encoding="utf-8"))}
    transcripts = json.loads((HERE / "transcripts.json").read_text(encoding="utf-8"))
    info = json.loads((HERE / "synth_info.json").read_text(encoding="utf-8"))

    rows = []
    for sid, s in sentences.items():
        for voice in VOICES:
            stem = f"{sid}_{voice}"
            t = timings[(sid, voice)]
            hyp = transcripts[stem]["text"]
            row = {"clip": stem, "id": sid, "group": s["group"], "voice": voice, "text": s["text"]}
            row.update(signal_facts(HERE / "out" / f"{stem}.wav"))
            row["chars"] = len(norm(s["text"]).replace(" ", ""))
            row["chars_per_s"] = round(row["chars"] / row["duration_s"], 2)
            row["phoneme_chars"] = len(t["phonemes"])
            row["phonemes_per_s"] = round(row["phoneme_chars"] / row["duration_s"], 2)
            row["synth_s"], row["rtf"] = float(t["synth_s"]), float(t["rtf"])
            row["transcript"] = hyp
            row["cer_raw"] = round(cer(s["text"], hyp), 3)
            row["cer_translit"] = round(cer(B_TRANSLIT[sid], hyp), 3) if sid in B_TRANSLIT else ""
            rows.append(row)

    # Speaking-rate ratios: characters per second against the median of the same group and voice, and
    # phoneme characters per second against the median of the same voice over all 20 sentences.
    for voice in VOICES:
        pps_med = statistics.median(r["phonemes_per_s"] for r in rows if r["voice"] == voice)
        for group in "ABCD":
            sel = [r for r in rows if r["voice"] == voice and r["group"] == group]
            cps_med = statistics.median(r["chars_per_s"] for r in sel)
            for r in sel:
                r["cps_vs_group_median"] = round(r["chars_per_s"] / cps_med, 2)
                r["pps_vs_voice_median"] = round(r["phonemes_per_s"] / pps_med, 2)

    for r in rows:
        flags = []
        if r["clipped_samples"]:
            flags.append(f"clipping ({r['clipped_samples']} samples)")
        if r["max_internal_sil_s"] > 0.7:
            flags.append(f"internal silence {r['max_internal_sil_s']} s")
        if r["lead_sil_s"] > 0.7:
            flags.append(f"leading silence {r['lead_sil_s']} s")
        if r["trail_sil_s"] > 0.8:
            flags.append(f"trailing silence {r['trail_sil_s']} s")
        if r["pps_vs_voice_median"] > 1.25:
            flags.append(f"fast for its text (x{r['pps_vs_voice_median']} voice median)")
        if r["pps_vs_voice_median"] < 0.75:
            flags.append(f"slow for its text (x{r['pps_vs_voice_median']} voice median)")
        # Whisper writes the Latin-script words of group B sometimes in Latin and sometimes in Devanagari,
        # so the lower of the two CERs is the one that reflects the audio.
        best = min(r["cer_raw"], r["cer_translit"]) if r["cer_translit"] != "" else r["cer_raw"]
        r["cer_best"] = best
        if best > 0.15:
            flags.append(f"CER {best}")
        flags += HAND_FLAGS.get(r["clip"], [])
        r["flags"] = "; ".join(flags)

    with open(HERE / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"rtf": {}, "mean_cer": {}, "long_input": {}}
    for voice in VOICES:
        rtfs = [r["rtf"] for r in rows if r["voice"] == voice]
        summary["rtf"][voice] = {"median": round(statistics.median(rtfs), 3), "max": round(max(rtfs), 3),
                                 "min": round(min(rtfs), 3), "n": len(rtfs)}
        for group in "ABCD":
            sel = [r for r in rows if r["voice"] == voice and r["group"] == group]
            entry = {"cer_raw": round(statistics.mean(r["cer_raw"] for r in sel), 3)}
            if group == "B":
                entry["cer_translit"] = round(statistics.mean(r["cer_translit"] for r in sel), 3)
                entry["cer_best"] = round(statistics.mean(r["cer_best"] for r in sel), 3)
            summary["mean_cer"][f"{group}_{voice}"] = entry
        long_text = info["long_input"]["text"]
        stem = f"long_A_{voice}"
        facts = signal_facts(HERE / "out" / f"{stem}.wav")
        summary["long_input"][voice] = dict(
            info["long_input"][voice], **facts, transcript=transcripts[stem]["text"],
            cer_raw=round(cer(long_text, transcripts[stem]["text"]), 3))
        summary["long_input"][voice].pop("phonemes", None)
        # Speech time = duration minus leading and trailing silence; compare the long clip with the five clips.
        five = [r for r in rows if r["voice"] == voice and r["group"] == "A"]
        speech_five = sum(r["duration_s"] - r["lead_sil_s"] - r["trail_sil_s"] for r in five)
        speech_long = facts["duration_s"] - facts["lead_sil_s"] - facts["trail_sil_s"]
        summary["long_input"][voice].update(
            speech_s_long=round(speech_long, 2), speech_s_five_clips=round(speech_five, 2),
            speech_ratio=round(speech_long / speech_five, 3),
            phoneme_chars_long=len(" | ".join(info["long_input"][voice]["phonemes"])),
            phoneme_chars_five_clips=sum(r["phoneme_chars"] for r in five))
    for voice in VOICES:
        sel = [r for r in rows if r["voice"] == voice]
        summary.setdefault("silence", {})[voice] = {
            "lead_min": min(r["lead_sil_s"] for r in sel), "lead_max": max(r["lead_sil_s"] for r in sel),
            "trail_min": min(r["trail_sil_s"] for r in sel), "trail_max": max(r["trail_sil_s"] for r in sel),
            "max_internal": max(r["max_internal_sil_s"] for r in sel),
            "peak_max": max(r["peak"] for r in sel), "clipped_clips": sum(1 for r in sel if r["clipped_samples"]),
            "rms_dbfs_min": min(r["rms_dbfs"] for r in sel), "rms_dbfs_max": max(r["rms_dbfs"] for r in sel)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    with open(HERE / "report_table.md", "w", encoding="utf-8") as f:
        f.write("| Clip | Sentence | Whisper transcript | CER raw | CER translit | Dur s | Flags |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['clip']} | {r['text']} | {r['transcript']} | {r['cer_raw']} | {r['cer_translit']} | "
                    f"{r['duration_s']} | {r['flags']} |\n")

    with open(HERE / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "voice", "sentence", "rating_1to5", "notes"])
        for r in rows:
            w.writerow([r["id"], r["voice"], r["text"], "", ""])
        for voice in VOICES:
            w.writerow(["long_A", voice, info["long_input"]["text"], "", ""])

    by_clip = {r["clip"]: r for r in rows}
    esc = html.escape

    def player(stem):
        return f'<audio controls preload="none" src="out/{stem}.wav"></audio>'

    page = ["<!doctype html><html lang='hi'><head><meta charset='utf-8'><title>Kokoro Hindi check</title><style>",
            "body{font-family:sans-serif;margin:24px;max-width:1500px} table{border-collapse:collapse;width:100%;margin-bottom:32px}",
            "td,th{border:1px solid #bbb;padding:6px 8px;vertical-align:top;text-align:left} th{background:#eee}",
            ".t{color:#444;font-size:0.92em} .f{color:#a40000;font-size:0.9em} audio{width:260px;height:32px}",
            "</style></head><body><h1>Kokoro Hindi TTS check</h1>",
            "<p>Voices: hf_beta (female), hm_omega (male). 24 kHz mono. Transcripts are faster-whisper large-v3 "
            "(language hi for groups A, B, D and the long input; en for group C). Fill ratings in ratings.csv.</p>"]
    if LISTEN_FIRST:
        page.append("<h2>Listen first</h2><table><tr><th>Clip</th><th>Group</th><th>Voice</th><th>Sentence</th>"
                    "<th>Audio</th><th>Whisper transcript</th><th>Why</th></tr>")
        for stem, why in LISTEN_FIRST:
            if stem in by_clip:
                r = by_clip[stem]
                gid, group, voice, text, hyp = r["id"], r["group"], r["voice"], r["text"], r["transcript"]
            else:
                gid, group, voice = "long_A", "A (long)", stem.replace("long_A_", "")
                text, hyp = info["long_input"]["text"], transcripts[stem]["text"]
            page.append(f"<tr><td>{esc(gid)}</td><td>{esc(group)}</td><td>{esc(voice)}</td><td>{esc(text)}</td>"
                        f"<td>{player(stem)}</td><td class='t'>{esc(hyp)}</td><td class='f'>{esc(why)}</td></tr>")
        page.append("</table>")
    page.append("<h2>All sentences, both voices side by side</h2><table><tr><th>Id</th><th>Group</th><th>Sentence</th>"
                "<th>hf_beta (female)</th><th>hm_omega (male)</th></tr>")
    for sid, s in sentences.items():
        cells = []
        for voice in VOICES:
            r = by_clip[f"{sid}_{voice}"]
            flags = f"<div class='f'>{esc(r['flags'])}</div>" if r["flags"] else ""
            cells.append(f"<td>{player(r['clip'])}<div class='t'>{esc(r['transcript'])}</div>{flags}</td>")
        page.append(f"<tr><td>{sid}</td><td>{s['group']}</td><td>{esc(s['text'])}</td>{''.join(cells)}</tr>")
    cells = "".join(f"<td>{player('long_A_' + v)}<div class='t'>{esc(transcripts['long_A_' + v]['text'])}</div></td>"
                    for v in VOICES)
    page.append(f"<tr><td>long_A</td><td>A (long)</td><td>{esc(info['long_input']['text'])}</td>{cells}</tr>")
    page.append("</table></body></html>")
    (HERE / "listen.html").write_text("\n".join(page), encoding="utf-8")

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for r in rows:
        print(f"{r['clip']:13s} cer {r['cer_raw']:<5} tr {str(r['cer_translit']):<5} dur {r['duration_s']:<5} "
              f"lead {r['lead_sil_s']} trail {r['trail_sil_s']} int {r['max_internal_sil_s']} peak {r['peak']} "
              f"rms {r['rms_dbfs']} pps x{r['pps_vs_voice_median']} cps x{r['cps_vs_group_median']} | {r['flags']}")
        print(f"   REF: {r['text']}\n   HYP: {r['transcript']}")


if __name__ == "__main__":
    main()
