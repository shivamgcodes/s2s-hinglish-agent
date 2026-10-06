import json,collections
d='.'
cm=[json.loads(l) for l in open('char_map.jsonl')]
wm=[json.loads(l) for l in open('word_map.jsonl')]
cmet=json.load(open('char_metrics.json')); wmet=json.load(open('word_metrics.json'))
R=cmet['ref_normalised']; H=cmet['hyp_normalised']
r=''.join(x['ref'] or '' for x in cm); h=''.join(x['hyp'] or '' for x in cm)
print('char ref ok',r==R,'hyp ok',h==H)
c=collections.Counter(x['op'] for x in cm); print(c)
bad=[x for x in cm if (x['op']=='match' or x['op']=='sub') and (not x['ref'] or not x['hyp']) or x['op']=='del' and x['hyp'] or x['op']=='ins' and x['ref']]
print('inconsistent char ops',len(bad),bad[:5])
print('lens per step', collections.Counter((len(x['ref'] or ''),len(x['hyp'] or '')) for x in cm))
cer=(c['sub']+c['del']+c['ins'])/len(R); print('cer',cer,len(R),len(H))
rw=[x['ref'] for x in wm if x['ref']]; hw=[x['hyp'] for x in wm if x['hyp']]
Rw=R.split(); Hw=H.split()
print('word ref ok',rw==Rw,'hyp ok',hw==Hw,len(Rw),len(Hw))
cw=collections.Counter(x['op'] for x in wm); print(cw)
print('wer',(cw['sub']+cw['del']+cw['ins'])/len(Rw))
for x in wm: print(x['i'],x['op'],x['ref'],'|',x['hyp'],'|',x.get('why',''))
