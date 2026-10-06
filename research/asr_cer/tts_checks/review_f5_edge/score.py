import json,sys,numpy as np,collections,wave
class sf:
    @staticmethod
    def read(f):
        w=wave.open(f); return np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768,w.getframerate()
import os  # monorepo shim: trelis_m1/synth/tts_norm/tts_backends are in research/audio ($AUDIO_DIR); data under $HINGLISH_ROOT
from pathlib import Path as _P
HINGLISH_ROOT = os.environ.get("HINGLISH_ROOT", "/workspace/hinglish")
AUDIO_DIR = os.environ.get("AUDIO_DIR", str(_P(__file__).resolve().parents[4] / "research" / "audio"))  # was /workspace/hinglish/audio
sys.path.insert(0,AUDIO_DIR)
import tts_norm, tts_backends as tb
from faster_whisper import WhisperModel
m=WhisperModel('large-v3',device='cuda',compute_type='float16')
idx=json.load(open('index.json'))
def endr(a,sr=24000):
    w=int(.01*sr); n=len(a)//w; r=np.sqrt((a[:n*w].reshape(n,w)**2).mean(1)+1e-12); return 20*np.log10(r[-4:].max()/np.percentile(r,95))
def voiced(a,sr=24000):
    w=int(.02*sr); n=len(a)//w; r=np.sqrt((a[:n*w].reshape(n,w)**2).mean(1)+1e-12); i=np.where(r>r.max()*10**(-35/20))[0]; return (i[-1]-i[0]+1)*.02
res=collections.defaultdict(list)
for d in idx:
    a,_=sf.read(d['f']); segs,_=m.transcribe(d['f'],language='hi',beam_size=5,temperature=0.0,condition_on_previous_text=False)
    t=' '.join(s.text.strip() for s in segs); c=tts_norm.cer_best(d['ref'],t)
    d.update(cer=float(c),end=float(endr(a)),dur=len(a)/24000,rate=float(tb.syllables(d['ref'])/voiced(a)),hyp=t); res[d['edge']].append(d)
for e,L in res.items():
    c=np.array([x['cer'] for x in L]); en=np.array([x['end'] for x in L]); r=np.array([x['rate'] for x in L])
    print(f"edge {e}: n {len(L)} cer mean {c.mean():.3f} n>0.35 {(c>0.35).sum()} | loud_end(>-15dB) {(en>-15).sum()} ({100*(en>-15).mean():.0f}%) | syl/s {r.mean():.2f}")
json.dump(idx,open('scored.json','w'),ensure_ascii=False,indent=0)
