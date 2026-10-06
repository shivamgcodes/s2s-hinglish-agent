import re,unicodedata
def norm_tokens(s,lower=True):
    if lower: s=s.lower()
    s=''.join(ch if not unicodedata.category(ch).startswith('P') else ' ' for ch in s)
    return s.split()
if __name__=='__main__':
    r=norm_tokens(open('reference.txt').read());h=norm_tokens(open('whisper.txt').read(),False)
    print(len(r),len(h))
    for i,w in enumerate(r): print('R',i,w)
    for i,w in enumerate(h): print('H',i,w)
