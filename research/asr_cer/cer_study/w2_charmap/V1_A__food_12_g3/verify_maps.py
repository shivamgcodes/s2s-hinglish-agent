import json,collections
D='.'
cm=json.load(open('char_metrics.json'));wm=json.load(open('word_metrics.json'))
C=[json.loads(l) for l in open('char_map.jsonl',encoding='utf-8')]
W=[json.loads(l) for l in open('word_map.jsonl',encoding='utf-8')]
nr,nh=cm['normalised_reference'],cm['normalised_hypothesis']
# independent normalisation
import unicodedata,re
def norm(s,lower):
    s=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)
    if lower:s=s.lower()
    return re.sub(r'\s+',' ',s).strip()
R=norm(open('reference.txt',encoding='utf-8').read(),True)
H=norm(open('whisper.txt',encoding='utf-8').read(),False)
out={}
out['norm_ref_indep_eq']=R==nr; out['norm_hyp_indep_eq']=H==nh
cr=''.join(s['ref'] for s in C); ch=''.join(s['hyp'] for s in C)
out['char_ref_concat']=cr==R; out['char_hyp_concat']=ch==H
c=collections.Counter(s['op'] for s in C); out['char_ops']=dict(c)
# consistency of op vs fields
bad=[]
for s in C:
    o=s['op']
    if o=='del' and (s['hyp']!='' or s['ref']==''): bad.append(s['i'])
    if o=='ins' and (s['ref']!='' or s['hyp']==''): bad.append(s['i'])
    if o=='sub' and (s['ref']=='' or s['hyp']==''): bad.append(s['i'])
out['char_op_field_inconsistent']=bad
nref=len(R)
out['char_cer_recount']=(c['sub']+c['del']+c['ins'])/nref
out['char_ref_len']=nref
# word
wr=[s['ref'] for s in W if s['ref']]; wh=[s['hyp'] for s in W if s['hyp']]
out['word_ref_concat']=wr==R.split(); out['word_hyp_concat']=wh==H.split()
w=collections.Counter(s['op'] for s in W); out['word_ops']=dict(w)
out['wer_recount']=(w['sub']+w['del']+w['ins'])/len(R.split())
wb=[s['i'] for s in W if (s['op']=='match' and s['ref'].lower()==s['hyp'].lower()) is False and s['op']=='match' and False]
out['metrics_char']={k:cm[k] for k in ['match','sub','del','ins','cer','ref_chars']}
out['metrics_word']=wm
print(json.dumps(out,ensure_ascii=False,indent=1))
