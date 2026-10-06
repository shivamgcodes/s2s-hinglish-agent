import re,unicodedata
def strip_punct(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
def ref_words(): return strip_punct(open('reference.txt',encoding='utf-8').read().lower()).split()
def hyp_words(): return strip_punct(open('whisper.txt',encoding='utf-8').read()).split()
if __name__=='__main__':
    r=ref_words();h=hyp_words()
    print(len(r),' '.join(f'{i}:{w}' for i,w in enumerate(r)))
    print(len(h),' '.join(f'{i}:{w}' for i,w in enumerate(h)))
