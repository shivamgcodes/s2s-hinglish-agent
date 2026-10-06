import json, unicodedata, collections
def sp(s): return ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
R=open('reference.txt',encoding='utf-8').read(); H=open('whisper.txt',encoding='utf-8').read()
rn=' '.join(sp(R.lower()).split()); hn=' '.join(sp(H).split())
cm=[json.loads(l) for l in open('char_map.jsonl',encoding='utf-8')]
wm=[json.loads(l) for l in open('word_map.jsonl',encoding='utf-8')]
out={}
cr=''.join(s['ref'] for s in cm); ch=''.join(s['hyp'] for s in cm)
out['char_ref_ok']=cr==rn; out['char_hyp_ok']=ch==hn
out['rn_len']=len(rn); out['hn_len']=len(hn)
c=collections.Counter(s['op'] for s in cm); out['char_ops']=dict(c)
# consistency of op vs fields
bad=[s['i'] for s in cm if (s['op']=='del' and (s['hyp'] or not s['ref'])) or (s['op']=='ins' and (s['ref'] or not s['hyp'])) or (s['op'] in('match','sub') and (not s['ref'] or not s['hyp']))]
out['char_bad_shape']=bad
out['ref_multi']=[ (s['i'],s['ref']) for s in cm if len(s['ref'])>1]
out['cer']=(c['sub']+c['del']+c['ins'])/len(rn)
out['cer_rawhyp_steps']=None
wr=' '.join(s['ref'] for s in wm if s['ref']); wh=' '.join(s['hyp'] for s in wm if s['hyp'])
out['word_ref_ok']=wr.split()==rn.split() and wr==rn; out['word_hyp_ok']=wh==hn
w=collections.Counter(s['op'] for s in wm); out['word_ops']=dict(w)
out['nref_words']=len(rn.split()); out['nhyp_words']=len(hn.split())
out['wer']=(w['sub']+w['del']+w['ins'])/len(rn.split())
wbad=[s['i'] for s in wm if (s['op']=='del' and (s['hyp'] or not s['ref'])) or (s['op']=='ins' and (s['ref'] or not s['hyp'])) or (s['op'] in('match','sub') and (not s['ref'] or not s['hyp'])) or (' ' in s['ref'] or ' ' in s['hyp'])]
out['word_bad_shape']=wbad
print(json.dumps(out,ensure_ascii=False,indent=1))
