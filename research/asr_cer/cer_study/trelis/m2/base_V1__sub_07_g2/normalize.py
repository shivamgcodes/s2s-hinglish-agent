import re, unicodedata
def norm(s):
    s = unicodedata.normalize('NFC', s).lower()
    s = re.sub(r"[’'`]", "", s)               # apostrophes deleted (you're -> youre)
    # other punctuation/symbols -> space (12,2024 -> 12 2024); Devanagari matras (M*) are kept
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()
if __name__ == "__main__":
    for f in ("reference.txt", "whisper.txt"):
        print(repr(norm(open(f, encoding="utf-8").read())))
