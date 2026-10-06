"""Hand-written after reading every Whisper transcript in transcripts.json (not by pattern matching)."""

_REFS = ["kokoro_hf_beta", "kokoro_hm_omega", "MAR_F_WIKI", "MAR_M_WIKI"]


def _all(sid, flag):
    return {f"{sid}_{r}": [flag] for r in _REFS}


HAND_FLAGS = {}
# Group C (pure English): every clip is about 2 s and the transcript has no words of the sentence.
for _sid in ("C1", "C2", "C3", "C4"):
    HAND_FLAGS.update(_all(_sid, "English sentence not reproduced: transcript has none of its words; clip ~2 s"))
# Group B: Latin-script English words and digits.
HAND_FLAGS.update(_all("B1", "address/Sector/15/Faridabad/update not recognisable in transcript"))
HAND_FLAGS.update(_all("B2", "'check' not recognisable (ऐ / यह / एया / ऐस)"))
HAND_FLAGS.update(_all("B3", "'order', '20' and 'deliver' not recognisable"))
HAND_FLAGS.update(_all("B4", "'order ID A1234' not recognisable; restaurant/ready garbled"))
HAND_FLAGS.update(_all("B5", "'order', 'total ₹250', 'payment' not recognisable"))
HAND_FLAGS.update(_all("B6", "'250 rupees' and 'refund' not recognisable"))
HAND_FLAGS.update(_all("B7", "'Delivery partner', '6', 'location' not recognisable"))
HAND_FLAGS.update(_all("B8", "'phone number confirm' not recognisable"))
HAND_FLAGS["B1_MAR_F_WIKI"].append("'ठीक है' missing at the start")
HAND_FLAGS["B1_MAR_M_WIKI"].append("'ठीक है' missing at the start")
# Group D and A: digits.
HAND_FLAGS.update(_all("D1", "'20' missing or garbled (ओडर मिनट / एं / ओर्डर्स / एंड)"))
HAND_FLAGS["D2_kokoro_hf_beta"] = ["'15' → पपे"]
HAND_FLAGS["D2_kokoro_hm_omega"] = ["'15' → तरतिनस"]
HAND_FLAGS["D2_MAR_F_WIKI"] = ["'15' → तूसर"]
HAND_FLAGS["D2_MAR_M_WIKI"] = ["'15' missing (सेक्टतर)"]
HAND_FLAGS["A3_kokoro_hf_beta"] = ["कृपया → रुपया in transcript"]
HAND_FLAGS["C2_MAR_F_WIKI"].append("7 clipped samples")

LISTEN_FIRST = [
    ("C1_kokoro_hf_beta", "Pure English sentence: transcript 'He and our eyes were in grandeur.' (1.7 s). Same in all 16 group-C clips."),
    ("C2_MAR_F_WIKI", "Pure English sentence: transcript 'And I hope you have a great day.'; the only clip with clipping (7 samples)."),
    ("B4_kokoro_hf_beta", "'order ID A1234' gone: transcript 'आपका मोने तैतु मनौर्वा अभी रसडा में रही हो रहा है'."),
    ("B1_kokoro_hm_omega", "address/Sector 15/Faridabad/update → 'एंडरसेसर पर छड़ियायन परश्चय'."),
    ("D2_kokoro_hm_omega", "Same sentence as B1 in Devanagari: all English words fine, but '15' → 'तरतिनस'."),
    ("B5_MAR_M_WIKI", "'total ₹250' and 'payment' → 'सोण का चौंग न ... हैनस'."),
    ("B3_MAR_M_WIKI", "Best group-B case for 'deliver' (इलीवर), but 'order' and '20' gone."),
    ("D1_kokoro_hf_beta", "Devanagari 'ऑर्डर/डिलीवर' fine; '20' missing."),
    ("B2_kokoro_hf_beta", "'check' → 'ऐ'; compare D3 (चेक) which is exact in all four voices."),
    ("A3_kokoro_hf_beta", "Pure Hindi; transcript has रुपया for कृपया (the other three voices: कुरुपया)."),
    ("long_B_kokoro_hf_beta", "Eight mixed-script sentences in one call: the first sentence is not in the transcript."),
    ("long_A_kokoro_hm_omega", "Five Hindi sentences in one call, 21.1 s; longest internal pause 0.64 s."),
]
