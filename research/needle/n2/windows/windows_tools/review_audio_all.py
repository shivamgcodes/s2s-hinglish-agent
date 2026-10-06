import json,os,numpy as np,soundfile as sf
W=os.environ.get('N2_ROOT','/root/n2')+'/windows'
meta=[json.loads(l) for l in open(W+'/meta.jsonl')]
rows=[];snr=[];clip=0;nan=0
for m in meta:
    r={}
    for v in ('clean_16k','opus_16k','opus_24k'):
        x,_=sf.read(W+'/'+m['files'][v],dtype='float32'); r[v]=float(np.sqrt(np.mean(x**2))); 
        if np.isnan(x).any(): nan+=1
        if np.abs(x).max()>=0.999: clip+=1
    a,_=sf.read(W+'/'+m['files']['opus_16k'],dtype='float32'); b,_=sf.read(W+'/'+m['files']['opus_24k'],dtype='float32')
    # opus_16k must equal to16k(opus_24k)
    import numpy.fft as F
    n=len(b); k=int(round(n*16000/24000)); y=F.irfft(F.rfft(b.astype(np.float64))[:k//2+1],k)*(k/n)
    L=min(len(a),len(y)); r['o16_vs_o24']=float(np.corrcoef(a[:L],y[:L])[0,1])
    rows.append(r); snr.append(m['noise']['snr_db'])
for k in rows[0]:
    v=np.array([r[k] for r in rows]); print(k,'min %.4f median %.4f max %.4f'%(v.min(),np.median(v),v.max()))
print('snr',min(snr),max(snr),'clipped files',clip,'nan',nan)
lo=sorted(range(len(rows)),key=lambda i:rows[i]['clean_16k'])[:3]; print('lowest rms',[(meta[i]['id'],round(rows[i]['clean_16k'],4),meta[i]['window_len_s']) for i in lo])
