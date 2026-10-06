import re,unicodedata,sys
def strip_punct(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('P') and c not in '|।')
def ref_words(t): return strip_punct(t.lower()).split()
def hyp_words(t): return strip_punct(t).split()
if __name__=='__main__':
    D=sys.argv[1]
    r=ref_words(open(D+'/reference.txt').read()); h=hyp_words(open(D+'/whisper.txt').read())
    print(len(r),len(h))
    for i,w in enumerate(r): print('R',i,w)
    for i,w in enumerate(h): print('H',i,w)
