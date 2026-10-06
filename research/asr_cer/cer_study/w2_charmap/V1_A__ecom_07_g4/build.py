#!/usr/bin/env python3
"""Build char_map.jsonl: manual segment/word pairing (judgement), char DP per word pair (align.py),
explicit overrides for cross-script cases the table can't decide."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import align as A
REF = A.norm_ref(open(f"{D}/reference.txt").read())
HYP = A.norm_hyp(open(f"{D}/whisper.txt").read())
rw, hw = REF.split(" "), HYP.split(" ")

def S(r, h, o, w=""): return [r, h, o, w]
def dels(text, why): return [S(c, "", "del", why) for c in text]
# Overrides: (ref word, hyp word) -> steps. Judgement calls on sound.
OV = {
 ("for", "पर"): [S("f","प","sub","f vs p: hyp says 'par'"), S("o","","del","vowel o; hyp has inherent a of प (different vowel)"), S("r","र","match")],
 ("boat", "बुथ"): [S("b","ब","match"), S("o","ु","sub","o (boAt /o:/) vs u matra"), S("a","","del","no second vowel"), S("t","थ","match","t ~ aspirated th (English t)")],
 ("headphones", "हेडपून्स"): [S("h","ह","match"), S("e","े","match"), S("a","","match","ea spelling of e vowel"), S("d","ड","match"),
     S("p","प","match"), S("h","","del","ph=/f/ rendered as plain p: aspiration/fricative lost"), S("o","ू","sub","o vs uu: 'phones' heard as 'poons'"),
     S("n","न्","match","n with halant"), S("e","","match","silent e"), S("s","स","match")],
 ("is", "इज"): [S("i","इ","match"), S("s","ज","match","English 'is' = /iz/; ज = z")],
 ("out", "आउट"): [S("o","आ","match","'ou' /au/ ~ आउ"), S("u","उ","match"), S("t","ट","match")],
 ("date", "डियेट"): [S("d","ड","match"), S("a","िये","sub","/eɪ/ heard as 'iye' (diyet)"), S("t","ट","match"), S("e","","match","silent e")],
 ("friday", "फ्राइडे"): [S("f","फ्","match","f with halant (conjunct फ्र)"), S("r","र","match"), S("i","ाइ","match","i /aɪ/ ~ ाइ"),
     S("d","ड","match"), S("a","े","match","'day' /eɪ/ ~ डे"), S("y","","match","ay digraph")],
 ("october", "अक्टूबर"): [S("o","अ","match","conventional Hindi spelling अक्टूबर"), S("c","क्","match"), S("t","ट","match"),
     S("o","ू","match","conventional Hindi spelling -टूबर"), S("b","ब","match"), S("e","","match","schwa -er ~ inherent vowel"), S("r","र","match")],
 ("october", "अक्तुबर"): [S("o","अ","match","conventional Hindi spelling अक्तूबर"), S("c","क्","match"), S("t","त","match","t ~ dental त (variant spelling)"),
     S("o","ु","match","short u variant of conventional -तूबर"), S("b","ब","match"), S("e","","match","schwa -er ~ inherent vowel"), S("r","र","match")],
 ("just", "जस्ट"): [S("j","ज","match"), S("u","","match","/ʌ/ ~ inherent vowel of ज"), S("s","स्","match"), S("t","ट","match")],
 ("a sec", "आरसेक"): [S("a","आ","match"), S(" ","र","sub","word gap heard as 'r' (aar-sec)"), S("s","स","match"), S("e","े","match"), S("c","क","match","c=/k/")],
 ("minute", "मिना"): [S("m","म","match"), S("i","ि","match"), S("n","न","match"), S("u","ा","sub","u vs aa"), S("t","","del","t missing"), S("e","","del","word truncated (no t, so not counted as silent e)")],
 ("25", "20"): [S("2","2","match"), S("5","0","sub","5 vs 0")],
 ("delivery", "दिलीवरी"): None,  # DP is fine (d ~ द)
 ("registered", "register"): [*[S(c,c,"match") for c in "register"], S("e","","del","past-tense -ed missing"), S("d","","del","past-tense -ed missing")],
 ("shopkart", "शापकार"): [S("s","श","match"), S("h","","match","sh ~ श"), S("o","ा","sub","o vs aa (shaap)"), S("p","प","match"),
     S("k","क","match"), S("a","ा","match"), S("r","र","match"), S("t","","del","final t missing")],
}
def pair(r, h):
    k = (r, h)
    if k in OV and OV[k] is not None: return [list(x) for x in OV[k]]
    if h == "" : return dels(r, "not in hypothesis")
    return A.align(r, A.to_units(h))

def word_pairs(rws, hws, extra=None):
    """1:1 word pairing joined by single-space matches; extra = list of (ref_text, hyp_text) explicit pairs."""
    steps = []
    for k, (r, h) in enumerate(extra):
        if k: steps.append(S(" ", " ", "match"))
        steps += pair(r, h)
    return steps

R = lambda a, b: rw[a:b]
def seg_words(rlist, hlist):
    assert len(rlist) == len(hlist), (rlist, hlist)
    return list(zip(rlist, hlist))

# ---- segment pairing (judgement: what Whisper heard vs what was dropped) ----
ri = rw.index("sec")  # 'a sec' merged
s1r = rw[:ri-1] + ["a sec"] + rw[ri+1:rw.index("minute")+1]
hmin = hw.index("मिना")
s1h = hw[:hmin+1]
steps = word_pairs(None, None, seg_words(s1r, s1h))
# seg2: dropped
i_done = rw.index("minute") + 1
i_ji2 = rw.index("ji", i_done + 2)  # 'ji' before 'aapka delivery instruction'
assert rw[i_ji2+1:i_ji2+3] == ["aapka", "delivery"]
steps += dels(" " + " ".join(rw[i_done:i_ji2]), "segment not transcribed by Whisper")
# seg3
i_email_is = rw.index("email") + 2
h_ji = hmin + 1; h_email_is = hw.index("email") + 2
steps.append(S(" ", " ", "match"))
steps += word_pairs(None, None, seg_words(rw[i_ji2:i_email_is], hw[h_ji:h_email_is]))
# seg4: dropped 'the instruction is added aapka street number bataiye'
i_del5 = rw.index("delivery", i_email_is)
steps += dels(" " + " ".join(rw[i_email_is:i_del5]), "segment not transcribed by Whisper")
# seg5
i_hai5 = rw.index("hai", i_del5) + 1
h_hai5 = hw.index("है", h_email_is) + 1
steps.append(S(" ", " ", "match"))
steps += word_pairs(None, None, seg_words(rw[i_del5:i_hai5], hw[h_email_is:h_hai5]))
# seg6: dropped until the last 'delivery'
i_del7 = len(rw) - 1 - rw[::-1].index("delivery")
steps += dels(" " + " ".join(rw[i_hai5:i_del7]), "segment not transcribed by Whisper")
# seg7: 'delivery date 25 october ko hai' vs 'दिलीवरी देट फ्राइडे 20 अक्तुबर को है' (फ्राइडे extra)
i_hai7 = rw.index("hai", i_del7) + 1
h7 = hw[h_hai5:]
assert len(h7) == 7 and rw[i_del7:i_hai7] == ["delivery","date","25","october","ko","hai"]
steps.append(S(" ", " ", "match"))
r7 = rw[i_del7:i_hai7]
steps += pair(r7[0], h7[0]); steps.append(S(" "," ","match")); steps += pair(r7[1], h7[1])
steps.append(S(""," ","ins","repeated 'friday' not in this reference sentence"))
steps += [S("", u, "ins", "repeated 'friday' not in this reference sentence") for u in A.to_units(h7[2])]
for rr, hh in zip(r7[2:], h7[3:]):
    steps.append(S(" "," ","match")); steps += pair(rr, hh)
# seg8
steps += dels(" " + " ".join(rw[i_hai7:]), "segment not transcribed by Whisper (cut off)")

# ---- verify & write ----
assert "".join(s[0] for s in steps) == REF, "ref concat mismatch"
assert "".join(s[1] for s in steps) == HYP, "hyp concat mismatch"
for s_ in steps:
    if s_[2] == "sub" and not s_[3]: s_[3] = f"different sound: {s_[0]} vs {s_[1]}"
    if s_[2] == "match" and not s_[3] and s_[0] != s_[1] and s_[1].isascii(): s_[3] = "case only"
with open(f"{D}/char_map.jsonl", "w") as f:
    for k, (r, h, o, w) in enumerate(steps):
        f.write(json.dumps({"i": k, "ref": r, "hyp": h, "op": o, "why": w}, ensure_ascii=False) + "\n")
from collections import Counter
c = Counter(s_[2] for s_ in steps); N = len(REF)
met = {"ref_chars": N, "hyp_chars": len(HYP), "match": c["match"], "sub": c["sub"], "del": c["del"], "ins": c["ins"],
       "cer": round((c["sub"] + c["del"] + c["ins"]) / N, 4), "steps": len(steps),
       "normalised_reference": REF, "normalised_hypothesis": HYP}
json.dump(met, open(f"{D}/char_metrics.json", "w"), ensure_ascii=False, indent=2)
# human-readable: chunks of ~80 ref chars
lines = []; chunk = []; rc = 0
def flush():
    if not chunk: return
    top = bot = mk = ""
    import unicodedata as U
    dw = lambda t: sum(1 for ch in t if U.category(ch) not in ("Mn", "Mc"))  # display columns
    pad = lambda t, w: t + " " * (w - dw(t))
    for r, h, o, _ in chunk:
        r, h = (r or "-"), (h or ("-" if o != "match" else ""))
        w = max(dw(r), dw(h), 1)
        top += pad(r, w); bot += pad(h, w); mk += pad({"match":" ","sub":"S","del":"D","ins":"I"}[o], w)
    lines.extend(["REF: " + top, "HYP: " + bot, "OP:  " + mk, ""])
for st in steps:
    chunk.append(st); rc += len(st[0])
    if rc >= 80 and st[0] == " ": flush(); chunk = []; rc = 0
flush()
hdr = f"CER = (S {c['sub']} + D {c['del']} + I {c['ins']}) / N {N} = {met['cer']}\n('-' = nothing on that side (del/ins); blank under a ref char with op ' ' = matched to an inherent/doubled vowel or digraph letter. Devanagari matras may render with slightly off widths.)\n\n"
open(f"{D}/char_map.txt", "w").write(hdr + "\n".join(lines))
print(json.dumps({k: v for k, v in met.items() if not k.startswith("normalised")}))
