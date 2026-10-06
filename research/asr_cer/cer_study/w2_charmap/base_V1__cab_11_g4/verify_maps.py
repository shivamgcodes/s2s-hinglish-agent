#!/usr/bin/env python3
"""Independent verifier: rebuild normalisation, check concatenations, recount ops."""
import json, re, unicodedata, os, collections
H=os.path.dirname(os.path.abspath(__file__))
def P(s): return ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
ref=open(f'{H}/reference.txt',encoding='utf-8').read(); hyp=open(f'{H}/whisper.txt',encoding='utf-8').read()
refn=re.sub(r'\s+',' ',P(ref.lower())).strip(); hypn=re.sub(r'\s+',' ',P(hyp)).strip()
cm=[json.loads(l) for l in open(f'{H}/char_map.jsonl',encoding='utf-8')]
wm=[json.loads(l) for l in open(f'{H}/word_map.jsonl',encoding='utf-8')]
out={}
cr=''.join(s['ref'] for s in cm); ch=''.join(s['hyp'] for s in cm)
out['char_ref_ok']=cr==refn; out['char_hyp_ok']=ch==hypn
# op consistency
bad=[]
for s in cm:
    o=s['op']
    if o=='ins' and s['ref']: bad.append(s['i'])
    if o=='del' and s['hyp']: bad.append(s['i'])
    if o=='match' and not s['ref']: bad.append(s['i'])
    if o=='sub' and not(s['ref'] and s['hyp']): bad.append(s['i'])
    if len(s['ref'])>1: bad.append(('multi-ref',s['i']))
out['char_op_shape_bad']=bad
c=collections.Counter(s['op'] for s in cm); out['char_counts']=dict(c)
N=len(refn); out['ref_chars']=N; out['hyp_chars']=len(hypn)
out['cer']=round((c['sub']+c['del']+c['ins'])/N,4)
out['match_empty_hyp']=sum(1 for s in cm if s['op']=='match' and s['hyp']=='')
Rw=P(ref.lower()).split(); Hw=P(hyp).split()
wr=[s['ref'] for s in wm if s['ref']]; wh=[s['hyp'] for s in wm if s['hyp']]
out['word_ref_ok']=wr==Rw; out['word_hyp_ok']=wh==Hw
w=collections.Counter(s['op'] for s in wm); out['word_counts']=dict(w)
out['ref_words']=len(Rw); out['hyp_words']=len(Hw)
out['wer']=round((w['sub']+w['del']+w['ins'])/len(Rw),4)
print(json.dumps(out,ensure_ascii=False,indent=1))
