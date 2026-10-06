import re,unicodedata
def strip_punct(s):
    return ''.join(c if not unicodedata.category(c).startswith('P') else ' ' if c in '-' else '' for c in s)
def ref_words():
    t=open('reference.txt',encoding='utf-8').read().lower()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P')).split()
def hyp_words():
    t=open('whisper.txt',encoding='utf-8').read()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P')).split()
if __name__=='__main__':
    r=ref_words();h=hyp_words()
    print(len(r),len(h))
    for i,w in enumerate(r):print('R',i,w)
    for i,w in enumerate(h):print('H',i,w)
