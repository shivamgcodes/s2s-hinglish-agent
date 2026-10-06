import re,unicodedata
def ref_words(t):
    t=t.lower(); t=re.sub(r"[^\w\s]"," ",t); return t.split()
def hyp_words(t):
    # drop punctuation but keep combining marks (\w misses some Devanagari marks)
    out=[]
    for ch in t:
        c=unicodedata.category(ch)
        out.append(' ' if c.startswith('P') or c.startswith('S') else ch)
    return ''.join(out).split()
