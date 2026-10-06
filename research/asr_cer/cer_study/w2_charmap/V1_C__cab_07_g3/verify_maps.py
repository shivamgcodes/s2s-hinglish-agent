import json, collections
cm=json.load(open('char_metrics.json')); wm=json.load(open('word_metrics.json'))
C=[json.loads(l) for l in open('char_map.jsonl')]
W=[json.loads(l) for l in open('word_map.jsonl')]
nr,nh=cm['normalised_reference'],cm['normalised_hypothesis']
cr=''.join(x['ref'] for x in C); ch=''.join(x['hyp'] for x in C)
print('char ref ok',cr==nr,'hyp ok',ch==nh)
cc=collections.Counter(x['op'] for x in C); print(cc)
# op consistency
bad=[x for x in C if (x['op']=='ins' and x['ref']!='') or (x['op']=='del' and x['hyp']!='') or (x['op'] in('match','sub') and (x['ref']=='' or x['hyp']==''))]
print('char op-shape bad',len(bad))
print('ref_chars',len(nr),'hyp_chars',len(nh))
cer=(cc['sub']+cc['del']+cc['ins'])/len(nr); print('cer',round(cer,4))
wr=[x['ref'] for x in W if x['ref']]; wh=[x['hyp'] for x in W if x['hyp']]
print('word ref ok', wr==nr.split(), 'hyp ok', wh==nh.split())
wc=collections.Counter(x['op'] for x in W); print(wc)
badw=[x for x in W if (x['op']=='ins' and x['ref']) or (x['op']=='del' and x['hyp']) or (x['op'] in('match','sub') and (not x['ref'] or not x['hyp']))]
print('word op-shape bad',len(badw))
wer=(wc['sub']+wc['del']+wc['ins'])/len(nr.split()); print('wer',round(wer,4))
# multi-char fields
print('char fields len>1:',[(x['i'],x['ref'],x['hyp']) for x in C if len(x['ref'])>1 or len(x['hyp'])>1][:20])
