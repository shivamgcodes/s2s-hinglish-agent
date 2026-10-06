import json,collections
N=json.load(open('normalised.json'))
REF,HYP=N['REF'],N['HYP']
cm=[json.loads(l) for l in open('char_map.jsonl')]
wm=[json.loads(l) for l in open('word_map.jsonl')]
out={}
r=''.join(s['ref'] for s in cm); h=''.join(s['hyp'] for s in cm)
out['char_ref_ok']=r==REF; out['char_hyp_ok']=h==HYP
# op consistency
bad=[s['i'] for s in cm if (s['op']=='del' and s['hyp']) or (s['op']=='ins' and s['ref']) or (s['op'] in('match','sub') and not(s['ref'] and s['hyp'])) or len(s['ref'])>1 or (s['op']=='match' and s['ref'].isascii() and s['hyp'].isascii() and s['ref']!=s['hyp'].lower())]
out['char_op_shape_bad']=bad
c=collections.Counter(s['op'] for s in cm); out['char_counts']=dict(c)
out['ref_chars']=len(REF); out['cer']=round((c['sub']+c['del']+c['ins'])/len(REF),4)
rw=REF.split(); hw=HYP.split()
wr=[s['ref'] for s in wm if s['ref']]; wh=[s['hyp'] for s in wm if s['hyp']]
out['word_ref_ok']=wr==rw; out['word_hyp_ok']=wh==hw
out['word_hyp_tokens']=len(hw)
wc=collections.Counter(s['op'] for s in wm); out['word_counts']=dict(wc)
out['wer']=round((wc['sub']+wc['del']+wc['ins'])/len(rw),4); out['ref_words']=len(rw)
out['multi_tok']=[s for s in wm if len(s['ref'].split())>1 or len(s['hyp'].split())>1]
print(json.dumps(out,ensure_ascii=False,indent=1))
