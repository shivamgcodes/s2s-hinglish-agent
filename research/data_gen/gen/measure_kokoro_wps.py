"""Measures Kokoro 'h' speaking rate (words/sec) on Devanagari Hinglish lines, and checks lang 'a' loads offline.
Run: CUDA_VISIBLE_DEVICES="" HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 flock /workspace/hinglish/gpu.lock \
     /workspace/venv-tts/bin/python measure_kokoro_wps.py  -> writes lexicon/../kokoro_wps.json
"""
import json
import os
import time

import numpy as np
from kokoro import KPipeline

LINES = [
    "हाँ जी, आपका डोमिनोज़ वाला ऑर्डर अभी प्रिपेयर हो रहा है, अराउंड पंद्रह मिनट्स और लगेंगे।",
    "हेलो, थैंक यू फ़ॉर कॉलिंग क्विकबाइट, दिस इज़ प्रिया। मैं आपकी कैसे हेल्प कर सकती हूँ?",
    "मेरा ऑर्डर बहुत लेट हो गया है यार, मैं कब से वेट कर रहा हूँ।",
    "एक मिनट रुकिए, मैं चेक करके बताती हूँ।",
    "जी, आपकी डिलीवरी ऐड्रेस सेक्टर इक्कीस, फ़रीदाबाद है, और पेमेंट यूपीआई से हो चुकी है।",
    "ठीक है, मैंने आपका नया ऐड्रेस अपडेट कर दिया है, टावर बी, फ़्लैट चार सौ बारह।",
    "अच्छा, तो रिफ़ंड रिक्वेस्ट सबमिट हो गई है, आपको एसएमएस पर अपडेट मिल जाएगा।",
    "नहीं नहीं, बेल मत बजाना, बस कॉल कर देना, मैं मीटिंग में हूँ।",
]
VOICES = ["hf_beta", "hm_omega", "hf_alpha", "hm_psi"]


def main():
    res = {"per_voice": {}, "lines": len(LINES)}
    p = KPipeline(lang_code="h", device="cpu")
    for v in VOICES:
        p("नमस्ते।", voice=v).__next__()
        words, secs = 0, 0.0
        for ln in LINES:
            audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in p(ln, voice=v)])
            # trim leading/trailing near-silence
            nz = np.where(np.abs(audio) > 0.01)[0]
            dur = (nz[-1] - nz[0]) / 24000 if len(nz) else len(audio) / 24000
            words += len(ln.split())
            secs += dur
        res["per_voice"][v] = {"words": words, "speech_s": round(secs, 2), "wps": round(words / secs, 3)}
        print(v, res["per_voice"][v], flush=True)
    res["wps_mean"] = round(float(np.mean([x["wps"] for x in res["per_voice"].values()])), 3)
    try:
        t0 = time.time()
        pa = KPipeline(lang_code="a", device="cpu")
        a = np.concatenate([np.asarray(r.audio) for r in pa("Hello, thank you for calling.", voice="af_heart")])
        res["lang_a"] = f"ok ({len(a) / 24000:.2f}s audio, load {time.time() - t0:.1f}s)"
    except Exception as e:  # noqa
        res["lang_a"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
    print("lang_a:", res["lang_a"])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro_wps.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
