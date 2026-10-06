import re,unicodedata,json
def norm_ref(s):
    s=s.lower()
    s=''.join(c if not unicodedata.category(c).startswith('P') else ' ' for c in s)
    return s.split()
def norm_hyp(s):
    s=''.join(c if not unicodedata.category(c).startswith('P') else ' ' for c in s)
    return s.split()
if __name__=='__main__':
    r=norm_ref(open('reference.txt').read()); h=norm_hyp(open('whisper.txt').read())
    print(len(r),len(h))
    for i,w in enumerate(r): print('R',i,w)
    for i,w in enumerate(h): print('H',i,w)
