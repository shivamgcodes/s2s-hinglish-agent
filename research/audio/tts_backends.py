"""Text side of the non-Kokoro-Hindi TTS backends (stdlib only; imported from venv-tts, venv-f5, venv-asr).

Backends (env TTS_BACKEND at `synth.py plan` time; recorded per chunk in plan.json):
  kokoro     -- V1 default. Untouched: synth.py uses tts_norm.tts_text exactly as before.
  f5cs       -- IndicF5 code-switch (Tharshan/indicf5_hindi-english_code_switch), venv-f5, see f5cs.py.
                Input = ALL-DEVANAGARI spoken form: tts_norm.tts_text (numbers/IDs/letters -> Devanagari English
                words) + any leftover Latin letters / digits spelled in Devanagari (see F5_BACKEND.md).
  kokoro_en  -- English CONTROL calls: Kokoro lang 'a'; input = English spoken form (en_text), ASR language 'en'.
tts_norm.py is NOT modified (V1 reproducibility); this module only adds functions on top of it.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_norm  # noqa: E402

BACKENDS = ("kokoro", "f5cs", "kokoro_en")
# chunk limits per backend (kokoro values = synth.py's original constants, unchanged)
LIMITS = {
    "kokoro": {"max_words": 12, "max_deva": 70, "spoken_max": 100},
    "f5cs": {"max_words": 20, "max_deva": 120, "spoken_max": 170},
    "kokoro_en": {"max_words": 20, "max_deva": 10**6, "spoken_max": 10**6},
    # D2/V4 (2026-10-04): long length band under f5cs. Only calls whose `length_band` is in F5_LONG_BANDS get it;
    # calls without length_band (V1/V3/CONTROL) keep the limits above, so their plan.json is byte-identical.
    # spoken_max 200 = synth.py's long_chunk flag threshold (170 would still split most 24-word lines with numbers).
    "f5cs_long": {"max_words": 24, "max_deva": 140, "spoken_max": 200},
}


def limits_key(backend, call):
    """Chunk-limit key for one call: 'f5cs_long' for f5cs calls whose length_band is in env F5_LONG_BANDS
    (comma list, default 'long', read at plan time), else the backend's own key."""
    import os
    bands = {b.strip() for b in os.environ.get("F5_LONG_BANDS", "long").split(",") if b.strip()}
    if backend == "f5cs" and call.get("length_band") in bands:
        return "f5cs_long"
    return backend
LANG = {"kokoro": "hi", "f5cs": "hi", "kokoro_en": "en"}

# Per-group voices for the non-V1 backends (F5_BACKEND.md "Voices"; g1..g4 = (customer, agent) genders as in V1).
# v3-data fix 2026-10-03: generate.py writes the V1 Kokoro Hindi keys (hm_psi/hf_beta/...) into calls.jsonl `voices` for
# every variant; synth.py plan maps them here for f5cs / kokoro_en (without this, kokoro_en would silently render the
# English control calls with Hindi Kokoro voices, and f5cs would KeyError).
BACKEND_VOICES = {
    "f5cs": {"g1": {"customer": "fleurs_hi_m1559", "agent": "ritu_hinglish"},
             "g2": {"customer": "fleurs_hi_m1559", "agent": "orato_male"},
             "g3": {"customer": "orato_female", "agent": "ritu_hinglish"},
             "g4": {"customer": "orato_female", "agent": "orato_male"}},
    "kokoro_en": {"g1": {"customer": "am_fenrir", "agent": "af_heart"},
                  "g2": {"customer": "am_fenrir", "agent": "am_michael"},
                  "g3": {"customer": "af_bella", "agent": "af_heart"},
                  "g4": {"customer": "af_bella", "agent": "am_michael"}},
}
VOICE_GENDER = {"fleurs_hi_m1559": "m", "orato_male": "m", "orato_female": "f", "ritu_hinglish": "f",
                "am_fenrir": "m", "am_michael": "m", "af_bella": "f", "af_heart": "f"}


def backend_voice(backend, call, role, voice):
    """Voice key for `role` of `call` under `backend` (non-kokoro only). A key already valid for the backend is kept;
    otherwise the group (call_id suffix g1..g4) decides. The voice gender must match the call's gender field."""
    valid = {v for g in BACKEND_VOICES[backend].values() for v in g.values()}
    if voice not in valid:
        voice = BACKEND_VOICES[backend][call["call_id"].rsplit("_", 1)[1]][role]
    want = call.get(role + "_gender")
    if want and VOICE_GENDER[voice] != want:
        raise SystemExit(f"{call['call_id']} {role}: voice {voice} gender != {want}")
    return voice

def f5_params():
    """f5cs engine parameters (env at `synth.py plan` time). Fixed into plan.json ("f5") and the f5cs .sig, so a
    changed setting re-renders that call instead of reusing stale audio; f5cs.cmd_tts reads them from plan.json.
    edge_s 0.8 (review fix 2026-10-03): at 0.25, ~15% of chunks hit the end of the generation buffer mid-phoneme
    (clipped final syllable, e.g. 'अर्जुन'->'अर्जु'); 0.8 -> ~1%, CER 0.049->0.033, +0.18 s/chunk after trim."""
    import os
    return {"edge_s": float(os.environ.get("F5_EDGE_S", "0.8")),
            "rate_min": float(os.environ.get("F5_RATE_MIN", "3.6")),
            "rate_max": float(os.environ.get("F5_RATE_MAX", "4.6")),
            "rate_scale": float(os.environ.get("F5_RATE_SCALE", "1.0")),
            "nfe": int(os.environ.get("F5_NFE", "32")),
            "tail_db": float(os.environ.get("F5_TAIL_DB", "-15")),  # last-40ms level vs p95 above this = clipped end
            "tail_extra_s": float(os.environ.get("F5_TAIL_EXTRA_S", "0.5"))}  # re-render once with this much more


DIGIT_DEVA = ["ज़ीरो", "वन", "टू", "थ्री", "फ़ोर", "फ़ाइव", "सिक्स", "सेवन", "एट", "नाइन"]
# characters outside the F5 vocab that can survive tts_text -> spoken equivalent
F5_PUNCT_MAP = {"—": ", ", "–": ", ", "…": ", ", "‘": "", "’": "", "“": "", "”": "", "॥": "।", "‍": "",
                "‌": "", "&": " एंड ", "+": " प्लस ", "@": " ऐट ", "/": " ", "_": " ", "#": " ", "*": " ",
                "%": " परसेंट ", "₹": " रुपये ", "\"": "", "(": ", ", ")": ", ", "[": ", ", "]": ", ", ":": ",",
                ";": ","}


def f5_text(text_tts):
    """All-Devanagari spoken form for IndicF5: tts_norm.tts_text, then leftover digits (digit by digit) and Latin
    letters (letter names) spelled in Devanagari, odd punctuation mapped. Raises if a digit/Latin char remains."""
    t = tts_norm.tts_text(text_tts)
    t = t.translate(tts_norm.DEVA_DIGITS)
    t = re.sub(r"\d", lambda m: " " + DIGIT_DEVA[int(m.group())] + " ", t)
    t = re.sub(r"[A-Za-z]", lambda m: " " + tts_norm.LETTERS[m.group().upper()][0] + " ", t)
    for a, b in F5_PUNCT_MAP.items():
        t = t.replace(a, b)
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\s+([,।?!.])", r"\1", t)
    t = re.sub(r"[,\s]*([।?!.])[,।?!.\s]*(?=\s|$)", r"\1 ", t)  # ", ।" -> "।"
    t = re.sub(r"(,\s*)+,", ",", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    if re.search(r"[A-Za-z0-9]", t):
        raise ValueError(f"f5_text left Latin/digits: {t!r}")
    return t


# ---------------- syllable count (duration model for F5) ----------------
_DEVA_IND_VOW = set("अआइईउऊऋएऐओऔऑऍ")
_DEVA_CONS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ") | set("क़ख़ग़ज़ड़ढ़फ़य़")
_VIRAMA = "्"
_MATRAS = set("ािीुूृेैोौॉॅ")


def _deva_word_syl(w):
    w = unicodedata.normalize("NFD", w).replace("़", "")
    n, last_bare = 0, False
    for i, c in enumerate(w):
        if c in _DEVA_IND_VOW:
            n += 1
            last_bare = False
        elif c in _DEVA_CONS:
            nx = w[i + 1] if i + 1 < len(w) else ""
            if nx == _VIRAMA:
                continue
            n += 1
            last_bare = nx not in _MATRAS
    if n > 1 and last_bare:
        n -= 1  # word-final schwa deletion (कमल = ka-mal)
    return n


def _latin_word_syl(w):
    if re.fullmatch(r"[A-Z]{1,5}", w):
        return sum(3 if c == "W" else 1 for c in w)  # acronym: letter names
    lw = w.lower()
    groups = re.findall(r"[aeiouy]+", lw)
    n = len(groups)
    if n > 1 and lw.endswith("e") and not lw.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def syllables(text):
    """Approximate spoken syllable count of mixed Devanagari/Latin text (digits: ~1.5 per digit)."""
    n = 0.0
    for w in re.findall(r"[ऀ-ॿ]+|[A-Za-z']+|\d", text):
        if re.match(r"[ऀ-ॿ]", w):
            n += _deva_word_syl(w)
        elif w.isdigit():
            n += 1.5
        else:
            n += _latin_word_syl(w)
    return n


def pauses(text):
    """Number of internal prosodic breaks (commas/dandas/?/! not at the end)."""
    return len(re.findall(r"[,।?!.;:]\s+\S", text))


# ---------------- English control ----------------
def _render_en(item):
    if isinstance(item, str):
        return {"rupaye": "rupees", "oh": "oh"}.get(item, item)
    kind, val = item
    if kind == "L":
        return val
    if kind == "MONTH":
        return tts_norm.MONTH_FULL[val].capitalize()
    return val


def en_text(text):
    """English spoken form for Kokoro lang 'a' (numbers -> English words, IDs -> letters/digits as words,
    punctuation kept). Uses tts_norm.groups (same tokenisation as the Hindi path and as MMS alignment)."""
    out = []
    for g in tts_norm.groups(text):
        body = " ".join(x for x in (_render_en(w) for w in g["words"]) if x)
        if body or g["lead"] or g["trail"]:
            out.append(g["lead"] + body + g["trail"])
    t = re.sub(r"\s+", " ", " ".join(out)).strip()
    return t.replace("₹", "rupees ")


def _en_key(text):
    t = en_text(text).lower()
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


def cer_en(ref, hyp):
    """CER for English chunks: both sides through en_text (Whisper digits -> words), lowercase a-z only."""
    hyp = tts_norm.clean_hyp(hyp)
    r, h = _en_key(ref), _en_key(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return tts_norm.levenshtein(r, h) / len(r)


if __name__ == "__main__":
    for t in ["अरे यार, आई थिंक मेरा पिकअप और ड्रॉप गलत है। प्लीज चेक करो।",
              "माय ऑर्डर आईडी इज़ FD1000, और मैं कब से वेट कर रहा हूँ यार।",
              "जी, ड्राइवर संदीप इज़ ऑन द वे — ETA अभी 8 मिनट्स है, ₹540 (approx)।"]:
        f = f5_text(t)
        print(f, "| syl", syllables(f), "| pauses", pauses(f))
    for t in ["Your order FD1000 of Rs 540 is out for delivery, ETA 20 minutes, call 98765 43210 at 6:40 PM on 18 Oct."]:
        print(en_text(t))
        print(round(cer_en(en_text(t), "Your order FD1000 of rupees 540 is out for delivery, ETA 20 minutes, call 9876543210 at 6.40 PM on 18 October."), 3))
    print(syllables("नमस्ते, मैं एक software engineer हूँ और आजकल machine learning projects पर काम कर रही हूँ।"))
