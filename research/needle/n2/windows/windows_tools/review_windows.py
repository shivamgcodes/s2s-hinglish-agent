# Independent adversarial review of /root/n2/windows (REVIEW-windows)
import json, os, random, collections, sys
import numpy as np, soundfile as sf
R=os.environ.get('N2_ROOT','/root/n2'); W=R+'/windows'; V=R+'/V4'
meta=[json.loads(l) for l in open(W+'/meta.jsonl')]
calls={}
for l in open(V+'/calls.jsonl'):
    c=json.loads(l); calls[c['call_id']]=c
hold=json.load(open(V+'/holdout.json'))
recs=json.load(open(V+'/records.json'))
byid={m['id']:m for m in meta}
issues=collections.Counter(); ex=collections.defaultdict(list)
def bad(k,x):
    issues[k]+=1
    if len(ex[k])<5: ex[k].append(x)
# 1. coverage: every check_line turn of calls.jsonl has exactly one window
n_cl=0
for cid,c in calls.items():
    st=json.load(open(f'{V}/stereo/{cid}.json'))
    cls=[i for i,t in enumerate(c['turns']) if 'check_line' in (t.get('tags') or [])]
    n_cl+=len(cls)
    # turn alignment between calls.jsonl and stereo json
    if len(st['turns'])!=len(c['turns']): bad('turn_count_mismatch',cid)
    for i,(a,b) in enumerate(zip(c['turns'],st['turns'])):
        if a['speaker']!=b['speaker'] or a['text_roman']!=b['text_roman']: bad('turn_text_mismatch',(cid,i)); break
    for o,ti in enumerate(cls):
        m=byid.get(f'{cid}__cl{o}')
        if not m: bad('missing_window',(cid,o)); continue
        if m['check_line_turn_idx']!=ti: bad('cl_turn_idx',m['id'])
        sti=[t for t in st['turns'] if t['turn']==ti][0]
        if abs(m['window_end_s']-sti['start'])>1e-6: bad('win_end_ne_cl_start',m['id'])
        if abs(m['window_start_s']-max(0,sti['start']-30))>1e-6: bad('win_start',m['id'])
        if st['turns'][ti]['speaker']!='agent': bad('cl_not_agent',m['id'])
        # write mapping independent: writes whose caller_turn < ti < confirm_turn; choose last such check-line
    # independent write mapping: for each write, last check-line strictly between caller & confirm
    for wi,w in enumerate(c['writes']):
        cand=[ti for ti in cls if w.get('caller_turn_idx',-1)<ti<w.get('confirm_turn_idx',10**9)]
        if not cand: bad('write_no_cl_between',(cid,wi,w.get('caller_turn_idx'),w.get('confirm_turn_idx'),cls)); continue
        o=cls.index(cand[-1]); m=byid[f'{cid}__cl{o}']
        if not any(x['write_idx']==wi and x['tool']==w['tool'] and x['args']==w['args'] for x in m['writes']): bad('write_map_disagree',(cid,wi))
        if len(cand)>1: bad('multi_cl_between',(cid,wi))
    if cls and not c['writes']: bad('cl_without_write',cid)
print('check_lines',n_cl,'windows',len(meta))
# 2. split leakage
sp={}
for k in ('test_scenarios','val_scenarios'):
    for s in hold[k]: sp[s]=k
for m in meta:
    exp={'test_scenarios':'test','val_scenarios':'val'}.get(sp.get(m['scenario_id']),'train')
    if m['split']!=exp: bad('split_wrong',m['id'])
spl_recs=collections.defaultdict(set)
for m in meta: spl_recs[m['split']].add(m['record_id'])
print('record overlap test&train',len(spl_recs['test']&spl_recs['train']),'val&train',len(spl_recs['val']&spl_recs['train']),'test&val',len(spl_recs['test']&spl_recs['val']))
spl_scen=collections.defaultdict(set)
for m in meta: spl_scen[m['split']].add(m['scenario_id'])
print('scenario overlap',len(spl_scen['test']&spl_scen['train']),len(spl_scen['val']&spl_scen['train']))
# record_id correctness
for m in meta:
    c=calls[m['call_id']]
    r=c['record']; rid=r.get('record_id') if isinstance(r,dict) else None
    if rid and rid!=m['record_id']: bad('record_id_mismatch',(m['id'],rid,m['record_id']))
# 3. audio checks on all windows: durations, sr, rms; and 12 random: independent cut correlation
random.seed(7)
def to16k(x,sr):
    import numpy.fft as F
    n=len(x); m=int(round(n*16000/sr)); X=F.rfft(x); Y=X[:m//2+1]; return F.irfft(Y,m)*(m/n)
for m in meta:
    for v in ('clean_16k','opus_16k','opus_24k'):
        p=W+'/'+m['files'][v] if not m['files'][v].startswith('/') else m['files'][v]
        if not os.path.exists(p): bad('missing_wav',(m['id'],v)); continue
        info=sf.info(p); exp_sr=24000 if v.endswith('24k') else 16000
        if info.samplerate!=exp_sr or info.channels!=1: bad('sr_ch',(m['id'],v))
        if abs(info.frames/exp_sr-m['window_len_s'])>0.002: bad('dur',(m['id'],v,info.frames/exp_sr,m['window_len_s']))
samp=random.sample(meta,12)
for m in samp:
    st=json.load(open(f"{V}/stereo/{m['call_id']}.json"))
    x,sr=sf.read(f"{V}/stereo/{m['call_id']}.wav",dtype='float32')
    ch1=x[:,1]; ch0=x[:,0]
    a=int(round(m['window_start_s']*sr)); b=int(round(m['window_end_s']*sr))
    cut=ch1[a:b]
    c16,_=sf.read(W+'/'+m['files']['clean_16k'],dtype='float32')
    o24,_=sf.read(W+'/'+m['files']['opus_24k'],dtype='float32')
    ref=to16k(cut.astype(np.float64),sr)
    L=min(len(ref),len(c16)); cc=np.corrcoef(ref[:L],c16[:L])[0,1]
    L2=min(len(cut),len(o24))
    # opus lag search
    best=max(((np.corrcoef(cut[:L2-400],o24[k:k+L2-400])[0,1],k) for k in range(0,400,5)))
    # agent leakage: rms of window during agent turns vs customer turns (window-relative)
    def rms_in(sig,spk):
        segs=[]
        for t in st['turns']:
            if t['speaker']!=spk: continue
            s=max(t['start'],m['window_start_s']); e=min(t['end'],m['window_end_s'])
            if e>s: segs.append(sig[int((s-m['window_start_s'])*16000):int((e-m['window_start_s'])*16000)])
        return float(np.sqrt(np.mean(np.concatenate(segs)**2))) if segs else float('nan')
    rc=rms_in(c16,'customer'); ra=rms_in(c16,'agent')
    # does window audio contain speech right before end (customer speaking at the check-line?)
    tail=float(np.sqrt(np.mean(c16[-8000:]**2)))
    print(f"{m['id']:28s} split={m['split']:5s} len={m['window_len_s']:.2f} corr_clean={cc:.4f} opus_corr={best[0]:.3f}@{best[1]} rms_cust={rc:.4f} rms_agentturns={ra:.5f} tail0.5s={tail:.4f} tool={m['writes'][0]['tool']}")
    if cc<0.99: bad('clean_cut_mismatch',m['id'])
    if not (ra!=ra) and ra>0.1*rc: bad('agent_leak_ch1',m['id'])
# global RMS stats from call_stats
cs=json.load(open(W+'/call_stats.json'))
print('call_stats sample',list(cs.items())[0])
print('ISSUES',dict(issues)); 
for k,v in ex.items(): print(' ',k,v)
