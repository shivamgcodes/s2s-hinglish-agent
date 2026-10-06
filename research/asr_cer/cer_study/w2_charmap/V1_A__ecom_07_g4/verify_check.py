import json
d=os.path if False else None
import os
F=os.path.dirname(os.path.abspath(__file__))
cm=json.load(open(f'{F}/char_metrics.json')); wm=json.load(open(f'{F}/word_metrics.json'))
ch=[json.loads(l) for l in open(f'{F}/char_map.jsonl')]
wd=[json.loads(l) for l in open(f'{F}/word_map.jsonl')]
R=cm['normalised_reference']; H=cm['normalised_hypothesis']
from collections import Counter
cr=''.join(s['ref'] or '' for s in ch); chh=''.join(s['hyp'] or '' for s in ch)
print('char ref ok',cr==R,'hyp ok',chh==H)
c=Counter(s['op'] for s in ch); print(c)
# op consistency
bad=[s for s in ch if (s['op']=='del' and s['hyp']) or (s['op']=='ins' and s['ref']) or (s['op'] in('match','sub') and not(s['ref'] and s['hyp']))]
print('char op/field inconsistencies',len(bad),bad[:5])
print('cer',(c['sub']+c['del']+c['ins'])/len(R), len(R), len(H))
wr=' '.join(s['ref'] for s in wd if s['ref']); wh=' '.join(s['hyp'] for s in wd if s['hyp'])
print('word ref ok',wr.split()==R.split(),'hyp ok',wh.split()==H.split())
w=Counter(s['op'] for s in wd); print(w, len(R.split()), len(H.split()))
print('wer',(w['sub']+w['del']+w['ins'])/len(R.split()))
print('multitoken fields',[s for s in wd if (s['ref'] and ' ' in s['ref']) or (s['hyp'] and ' ' in s['hyp'])])
