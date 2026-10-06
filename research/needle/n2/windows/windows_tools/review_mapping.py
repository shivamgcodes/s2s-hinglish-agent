import json,os,random,collections
R=os.environ.get('N2_ROOT','/root/n2'); V=R+'/V4'
meta=[json.loads(l) for l in open(R+'/windows/meta.jsonl')]
calls={json.loads(l)['call_id']:json.loads(l) for l in open(V+'/calls.jsonl')}
recs=json.load(open(V+'/records.json'))
print('records.json type',type(recs).__name__, list(recs)[:3] if isinstance(recs,dict) else '')
# record consistency
diff=0;nk=0
for m in meta:
    cr=calls[m['call_id']]['record']; rr=recs.get(m['scenario_id']) if isinstance(recs,dict) else None
    if rr is None: nk+=1; continue
    if rr!=cr and rr.get('record',rr)!=cr: diff+=1; 
    if diff==1 and rr!=cr and rr.get('record',rr)!=cr: print('DIFF example', m['id'], list(rr.keys())[:8], list(cr.keys())[:8]); diff+=0
print('record differs from calls.record:',diff,'no rec',nk)
# customer turn overlapping check-line start
clip=collections.Counter()
for m in meta:
    for t in m['customer_turns_in_window']:
        if t['clipped_at']: clip[t['clipped_at']]+=1
print('clipped customer turns',dict(clip))
endc=[m['id'] for m in meta if any(t['clipped_at']=='end' for t in m['customer_turns_in_window'])]
print('windows with customer speech cut at check-line start',len(endc),endc[:5])
# 10 random human-readable mapping
random.seed(11)
for m in random.sample(meta,10):
    c=calls[m['call_id']]; w=m['writes'][0]
    print('---',m['id'],w['tool'],w['args'])
    print('  CL :',c['turns'][m['check_line_turn_idx']]['text_roman'][:80])
    print('  CF :',c['turns'][w['confirm_turn_idx']]['text_roman'][:110])
    print('  CT :',c['turns'][w['caller_turn_idx']]['text_roman'][:110])
# window must not contain customer audio after check-line: last customer turn end <= window_end
bad=[m['id'] for m in meta for t in m['customer_turns_in_window'] if t['start']>=m['window_end_s']]
print('turns starting after window end',len(bad))
# windows with no customer speech or tiny
print('min customer turns in window',min(len(m['customer_turns_in_window']) for m in meta))
# arg coverage of NEW args
cov=collections.Counter()
for m in meta:
    for w in m['writes']:
        for a,v in w['arg_coverage'].items(): cov[(w['tool'],a,v)]+=1
for k,v in sorted(cov.items()):
    if k[2] in ('neither','record_only') and k[1] in ('phone','email','address','location','new_address','instruction','new_number'): print('COV',k,v)
