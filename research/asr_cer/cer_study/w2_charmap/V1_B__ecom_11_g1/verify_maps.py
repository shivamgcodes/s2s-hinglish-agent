import json,unicodedata,re
from collections import Counter
cm=[json.loads(l) for l in open('char_map.jsonl')]
wm=[json.loads(l) for l in open('word_map.jsonl')]
cmet=json.load(open('char_metrics.json')); wmet=json.load(open('word_metrics.json'))
NR,NH=cmet['normalised_reference'],cmet['normalised_hypothesis']
out={}
cr=''.join(x['ref'] for x in cm); ch=''.join(x['hyp'] for x in cm)
out['char_ref_concat_ok']=cr==NR; out['char_hyp_concat_ok']=ch==NH
wr=' '.join(x['ref'] for x in wm if x['ref']); wh=' '.join(x['hyp'] for x in wm if x['hyp'])
out['word_ref_concat_ok']=wr==NR; out['word_hyp_concat_ok']=wh==NH
# independent normalisation from raw files
def norm(s,lower=True):
    if lower: s=s.lower()
    s=''.join(c if not unicodedata.category(c).startswith('P') else ' ' for c in s)
    return ' '.join(s.split())
rawR=norm(open('reference.txt').read()); rawH=norm(open('whisper.txt').read(),False)
out['NR_equals_raw_norm_punct_to_space']=rawR==NR
out['NR_equals_raw_norm_punct_deleted']=' '.join(re.sub(r'[^\w\s]','',open('reference.txt').read().lower()).split())==NR
out['NH_equals_raw_norm']=rawH==NH
c=Counter(x['op'] for x in cm); out['char_counts']=dict(c)
# consistency checks per step
bad=[]
for x in cm:
    o=x['op']
    if o=='ins' and x['ref']: bad.append(('ins_with_ref',x['i']))
    if o=='del' and x['hyp']: bad.append(('del_with_hyp',x['i']))
    if o=='sub' and (not x['ref'] or not x['hyp']): bad.append(('sub_empty_side',x['i']))
    if o=='match' and not x['ref']: bad.append(('match_no_ref',x['i']))
out['char_step_anomalies']=bad
out['char_match_with_empty_hyp']=sum(1 for x in cm if x['op']=='match' and x['ref'] and not x['hyp'])
out['ref_chars_in_map']=sum(len(x['ref']) for x in cm)
out['ref_chars_len_NR']=len(NR); out['hyp_chars_len_NH']=len(NH)
out['ins_hyp_codepoints']=sum(len(x['hyp']) for x in cm if x['op']=='ins')
E=c['sub']+c['del']+c['ins']
out['cer_recount']=round(E/len(NR),4)
out['cer_recount_ref_consumed_check']= c['match']+c['sub']+c['del']
w=Counter(x['op'] for x in wm); out['word_counts']=dict(w)
nrw=len(NR.split()); out['ref_words']=nrw; out['hyp_words']=len(NH.split())
out['wer_recount']=round((w['sub']+w['del']+w['ins'])/nrw,4)
wbad=[]
for x in wm:
    o=x['op']
    if o=='ins' and x['ref']: wbad.append(('ins_with_ref',x['i']))
    if o=='del' and x['hyp']: wbad.append(('del_with_hyp',x['i']))
    if o in('sub','match') and (not x['ref'] or not x['hyp']): wbad.append((o+'_empty',x['i']))
    if x['ref'] and ' ' in x['ref']: wbad.append(('multiword_ref',x['i']))
    if x['hyp'] and ' ' in x['hyp']: wbad.append(('multiword_hyp',x['i']))
out['word_step_anomalies']=wbad
out['metrics_char']={k:cmet[k] for k in ['ref_chars','hyp_chars','match','sub','del','ins','errors','cer']}
out['metrics_word']={k:wmet[k] for k in ['ref_words','hyp_words','match','sub','del','ins','errors','wer']}
print(json.dumps(out,ensure_ascii=False,indent=1))
json.dump(out,open('verify_mech.json','w'),ensure_ascii=False,indent=1)
