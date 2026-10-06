"""IndicF5 TTS check: per-clip signal facts, rough CER, comparison with the Kokoro check, report tables, a local
listening page and an empty ratings sheet. Normalisation, CER and the group-B transliterated reference are the
Kokoro check's own (kokoro-check/analyze.py), imported unchanged.

Run locally from the local indicf5-check/ folder after copying out/, timings.csv, synth_info.json and
transcripts.json from the pod:
  python3 scripts/analyze.py
Writes metrics.csv, summary.json, report_table.md, listen.html, ratings.csv.
"""
import csv
import html
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KOKORO = HERE.parent / "kokoro-check"
sys.path.insert(0, str(KOKORO))
from analyze import B_TRANSLIT, PAIRS, cer, norm, signal_facts  # noqa: E402  (Kokoro check's helpers)

sys.path.insert(0, str(HERE / "scripts"))
from marks import HAND_FLAGS, LISTEN_FIRST  # noqa: E402  (written by hand after reading the transcripts)

REFS = ["kokoro_hf_beta", "kokoro_hm_omega", "MAR_F_WIKI", "MAR_M_WIKI"]


def main():
    sentences = {r["id"]: r for r in csv.DictReader(open(KOKORO / "sentences.tsv", encoding="utf-8"), delimiter="\t")}
    timings = {(r["id"], r["ref"]): r for r in csv.DictReader(open(HERE / "timings.csv", encoding="utf-8"))}
    tr = json.loads((HERE / "transcripts.json").read_text(encoding="utf-8"))
    info = json.loads((HERE / "synth_info.json").read_text(encoding="utf-8"))

    rows = []
    for sid, s in sentences.items():
        for ref in REFS:
            stem = f"{sid}_{ref}"
            t = timings[(sid, ref)]
            hyp = tr[stem]["text"]
            row = {"clip": stem, "id": sid, "group": s["group"], "ref": ref, "text": s["text"]}
            row.update(signal_facts(HERE / "out" / f"{stem}.wav"))
            row["chars"] = len(norm(s["text"]).replace(" ", ""))
            row["chars_per_s"] = round(row["chars"] / max(row["duration_s"], 1e-9), 2)
            row["synth_s"], row["rtf"] = float(t["synth_s"]), float(t["rtf"])
            row["transcript"] = hyp
            row["cer_raw"] = round(cer(s["text"], hyp), 3)
            row["cer_translit"] = round(cer(B_TRANSLIT[sid], hyp), 3) if sid in B_TRANSLIT else ""
            row["cer_best"] = min(row["cer_raw"], row["cer_translit"]) if row["cer_translit"] != "" else row["cer_raw"]
            rows.append(row)

    for ref in REFS:
        for group in "ABCD":
            sel = [r for r in rows if r["ref"] == ref and r["group"] == group]
            med = statistics.median(r["chars_per_s"] for r in sel)
            for r in sel:
                r["cps_vs_group_median"] = round(r["chars_per_s"] / med, 2)

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
        if r["cps_vs_group_median"] > 1.25:
            flags.append(f"fast/short for its text (x{r['cps_vs_group_median']} group median chars/s)")
        if r["cps_vs_group_median"] < 0.75:
            flags.append(f"slow/long for its text (x{r['cps_vs_group_median']} group median chars/s)")
        if r["cer_best"] > 0.15:
            flags.append(f"CER {r['cer_best']}")
        flags += HAND_FLAGS.get(r["clip"], [])
        r["flags"] = "; ".join(flags)

    with open(HERE / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    kok = json.loads((KOKORO / "summary.json").read_text(encoding="utf-8"))
    summary = {"rtf": {}, "mean_cer": {}, "kokoro_mean_cer": kok["mean_cer"], "long_input": {}, "silence": {},
               "load_s": info.get("load_s"), "warmup_s": info.get("warmup_s"),
               "peak_vram_gib": info.get("peak_vram_gib")}
    for ref in REFS:
        sel = [r for r in rows if r["ref"] == ref]
        rtfs = [r["rtf"] for r in sel]
        summary["rtf"][ref] = {"median": round(statistics.median(rtfs), 3), "max": round(max(rtfs), 3),
                               "min": round(min(rtfs), 3), "n": len(rtfs)}
        for group in "ABCD":
            g = [r for r in sel if r["group"] == group]
            e = {"cer_raw": round(statistics.mean(r["cer_raw"] for r in g), 3)}
            if group == "B":
                e["cer_translit"] = round(statistics.mean(r["cer_translit"] for r in g), 3)
                e["cer_best"] = round(statistics.mean(r["cer_best"] for r in g), 3)
            summary["mean_cer"][f"{group}_{ref}"] = e
        summary["silence"][ref] = {
            "lead_min": min(r["lead_sil_s"] for r in sel), "lead_max": max(r["lead_sil_s"] for r in sel),
            "trail_min": min(r["trail_sil_s"] for r in sel), "trail_max": max(r["trail_sil_s"] for r in sel),
            "max_internal": max(r["max_internal_sil_s"] for r in sel), "peak_max": max(r["peak"] for r in sel),
            "clipped_clips": sum(1 for r in sel if r["clipped_samples"]),
            "rms_dbfs_min": min(r["rms_dbfs"] for r in sel), "rms_dbfs_max": max(r["rms_dbfs"] for r in sel)}
        for group in ("A", "B"):
            stem = f"long_{group}_{ref}"
            text = info["long_input"][group]
            facts = signal_facts(HERE / "out" / f"{stem}.wav")
            parts = [r for r in sel if r["group"] == group]
            speech_parts = sum(r["duration_s"] - r["lead_sil_s"] - r["trail_sil_s"] for r in parts)
            speech_long = facts["duration_s"] - facts["lead_sil_s"] - facts["trail_sil_s"]
            last = [s for s in sentences.values() if s["group"] == group][-1]
            t = timings[(f"long_{group}", ref)]
            summary["long_input"][stem] = dict(
                facts, synth_s=float(t["synth_s"]), rtf=float(t["rtf"]),
                sum_of_clips_s=round(sum(r["duration_s"] for r in parts), 2),
                speech_s_long=round(speech_long, 2), speech_s_clips=round(speech_parts, 2),
                speech_ratio=round(speech_long / speech_parts, 3), transcript=tr[stem]["text"],
                tail_8s_transcript=tr[stem].get("tail_8s_text", ""),
                cer_raw=round(cer(text, tr[stem]["text"]), 3),
                last_sentence=last["text"],
                cer_last_sentence_vs_tail=round(cer(last["text"], tr[stem].get("tail_8s_text", "")), 3))
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    with open(HERE / "report_table.md", "w", encoding="utf-8") as f:
        f.write("| Clip | Sentence | Whisper transcript | CER raw | CER translit | Dur s | RTF | Flags |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['clip']} | {r['text']} | {r['transcript']} | {r['cer_raw']} | {r['cer_translit']} | "
                    f"{r['duration_s']} | {r['rtf']} | {r['flags']} |\n")

    with open(HERE / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ref_voice", "sentence", "rating_1to5", "notes"])
        for r in rows:
            w.writerow([r["id"], r["ref"], r["text"], "", ""])
        for group in ("A", "B"):
            for ref in REFS:
                w.writerow([f"long_{group}", ref, info["long_input"][group], "", ""])

    by_clip = {r["clip"]: r for r in rows}
    esc = html.escape

    def player(src):
        return f'<audio controls preload="none" src="{esc(src)}"></audio>'

    page = ["<!doctype html><html lang='hi'><head><meta charset='utf-8'><title>IndicF5 Hindi check</title><style>",
            "body{font-family:sans-serif;margin:24px;max-width:1700px} table{border-collapse:collapse;width:100%;"
            "margin-bottom:32px} td,th{border:1px solid #bbb;padding:6px 8px;vertical-align:top;text-align:left}"
            " th{background:#eee} .t{color:#444;font-size:0.9em} .f{color:#a40000;font-size:0.88em}"
            " audio{width:230px;height:32px}</style></head><body><h1>IndicF5 Hindi TTS check</h1>",
            "<p>Four reference voices (IndicF5 copies the voice of the reference clip): Kokoro hf_beta and hm_omega "
            "(our own synthetic Hindi clips), and the IndicF5 repo's Marathi MAR_F_WIKI and MAR_M_WIKI. The last "
            "column is the Kokoro hf_beta clip of the same sentence for comparison. Transcripts are faster-whisper "
            "large-v3. Fill ratings in ratings.csv.</p>",
            "<h2>Reference clips</h2><table><tr>" + "".join(f"<th>{r}</th>" for r in REFS) + "</tr><tr>" +
            "".join(f"<td>{player(info['references'][r]['path'])}<div class='t'>"
                    f"{esc(info['references'][r]['text'])}</div></td>" for r in REFS) + "</tr></table>"]
    if LISTEN_FIRST:
        page.append("<h2>Listen first</h2><table><tr><th>Clip</th><th>Sentence</th><th>Audio</th>"
                    "<th>Whisper transcript</th><th>Why</th></tr>")
        for stem, why in LISTEN_FIRST:
            if stem in by_clip:
                text, hyp = by_clip[stem]["text"], by_clip[stem]["transcript"]
            else:
                text, hyp = info["long_input"][stem[5]], tr[stem]["text"]
            page.append(f"<tr><td>{esc(stem)}</td><td>{esc(text)}</td><td>{player('out/' + stem + '.wav')}</td>"
                        f"<td class='t'>{esc(hyp)}</td><td class='f'>{esc(why)}</td></tr>")
        page.append("</table>")
    page.append("<h2>All sentences</h2><table><tr><th>Id</th><th>Sentence</th>" +
                "".join(f"<th>IndicF5 / {r}</th>" for r in REFS) + "<th>Kokoro hf_beta</th></tr>")
    for sid, s in sentences.items():
        cells = []
        for ref in REFS:
            r = by_clip[f"{sid}_{ref}"]
            fl = f"<div class='f'>{esc(r['flags'])}</div>" if r["flags"] else ""
            cells.append(f"<td>{player('out/' + r['clip'] + '.wav')}<div class='t'>{esc(r['transcript'])}</div>"
                         f"<div class='t'>CER {r['cer_best']}</div>{fl}</td>")
        cells.append(f"<td>{player('../kokoro-check/out/' + sid + '_hf_beta.wav')}</td>")
        page.append(f"<tr><td>{sid}</td><td>{esc(s['text'])}</td>{''.join(cells)}</tr>")
    for group in ("A", "B"):
        cells = "".join(f"<td>{player(f'out/long_{group}_{ref}.wav')}<div class='t'>"
                        f"{esc(tr[f'long_{group}_{ref}']['text'])}</div></td>" for ref in REFS)
        kok_long = f"../kokoro-check/out/long_{group}_hf_beta.wav" if group == "A" else "../kokoro-check/out2/long_B_hf_alpha.wav"
        page.append(f"<tr><td>long_{group}</td><td>{esc(info['long_input'][group])}</td>{cells}"
                    f"<td>{player(kok_long)}<div class='t'>{'hf_beta' if group == 'A' else 'hf_alpha (no hf_beta long_B)'}"
                    f"</div></td></tr>")
    page.append("</table></body></html>")
    (HERE / "listen.html").write_text("\n".join(page), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("rtf", "mean_cer", "load_s", "warmup_s", "peak_vram_gib")},
                     indent=1, ensure_ascii=False))
    for k, v in summary["long_input"].items():
        print(k, {x: v[x] for x in ("duration_s", "sum_of_clips_s", "speech_ratio", "cer_raw",
                                    "cer_last_sentence_vs_tail", "max_internal_sil_s")})
        print("   TAIL:", v["tail_8s_transcript"])
    for r in rows:
        print(f"{r['clip']:22s} cer {r['cer_raw']:<5} tr {str(r['cer_translit']):<5} dur {r['duration_s']:<5} "
              f"lead {r['lead_sil_s']} trail {r['trail_sil_s']} int {r['max_internal_sil_s']} peak {r['peak']} "
              f"cps x{r['cps_vs_group_median']} rtf {r['rtf']} | {r['flags']}")
        print(f"   REF: {r['text']}\n   HYP: {r['transcript']}")


if __name__ == "__main__":
    main()
