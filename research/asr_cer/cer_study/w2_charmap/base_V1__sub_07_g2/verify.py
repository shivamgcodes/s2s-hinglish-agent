import json, collections
cm=json.load(open('char_metrics.json')); wm=json.load(open('word_metrics.json'))
C=[json.loads(l) for l in open('char_map.jsonl')]
W=[json.loads(l) for l in open('word_map.jsonl')]
nr=cm['normalised_reference']; nh=cm['normalised_hypothesis']
cref=''.join(s['ref'] or '' for s in C); chyp=''.join(s['hyp'] or '' for s in C)
print('char ref ok',cref==nr,'hyp ok',chyp==nh)
wref=[s['ref'] for s in W if s['ref']]; whyp=[s['hyp'] for s in W if s['hyp']]
print('word ref ok',wref==nr.split(),'hyp ok',whyp==nh.split())
cc=collections.Counter(s['op'] for s in C); wc=collections.Counter(s['op'] for s in W)
print(cc,wc)
N=len(nr); print('cer',(cc['sub']+cc['del']+cc['ins'])/N, N)
print('wer',(wc['sub']+wc['del']+wc['ins'])/len(nr.split()), len(nr.split()))
# consistency checks
for s in C:
  if s['op']=='match' and (s['ref'] or '').lower()!=(s['hyp'] or '').lower(): print('nontrivial match',s)
  if s['op']!='match': print(s)
for s in W:
  if s['op']!='match' or s['ref'].lower()!=s['hyp'].lower(): print(s)
