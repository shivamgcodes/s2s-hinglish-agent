import json, os, statistics as st
from collections import defaultdict
R=[json.loads(l) for l in open(os.environ.get("HINGLISH_ROOT", "/workspace/hinglish") + "/tmp_trelis/calib.jsonl")]
def rank(x):
    o=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
    for j,i in enumerate(o): r[i]=j
    return r
def spear(a,b):
    ra,rb=rank(a),rank(b); n=len(a); ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); da=sum((x-ma)**2 for x in ra)**.5; db=sum((y-mb)**2 for y in rb)**.5
    return num/(da*db)
def q(v,ps=(0.5,0.9,0.95,0.99)): v=sorted(v); return [round(v[min(len(v)-1,int(p*len(v)))],3) for p in ps]
V3=[r for r in R if r["var"]=="V3"]; V4=[r for r in R if r["var"]=="V4"]
retr={r["id"] for r in V3 if r["k"]>0}
rand=[r for r in V3 if r["id"] not in retr]   # random try0, old passed (<=0.35 by construction? check)
badset=[r for r in V3 if r["id"] in retr and r["k"]==0]
print("V3 n",len(V3),"random try0",len(rand),"retried ids",len(retr))
print("random try0 old>0.35:",sum(r["old_cer"]>0.35 for r in rand))
print("spearman old vs trelis cer (all V3):",round(spear([r["old_cer"] for r in V3],[r["cer"] for r in V3]),3))
print("spearman random:",round(spear([r["old_cer"] for r in rand],[r["cer"] for r in rand]),3))
print("quantiles p50/90/95/99 new CER: random V3",q([r["cer"] for r in rand]),"V4 try0",q([r["cer"] for r in V4]),"old V3 random",q([r["old_cer"] for r in rand]))
print("retried-set try0: old",q([r["old_cer"] for r in badset]),"new",q([r["cer"] for r in badset]))
for th in (0.10,0.15,0.20,0.25,0.30,0.35):
    print(f"th {th}: rand V3 rej {sum(r['cer']>th for r in rand)}/{len(rand)}  V3 old-rejected try0 rej {sum(r['cer']>th for r in badset)}/{len(badset)}  V4 try0 rej {sum(r['cer']>th for r in V4)}/{len(V4)}  V4 digits {sum(r['cer']>th for r in V4 if r['digits'])}/{sum(r['digits'] for r in V4)} V4 nodig {sum(r['cer']>th for r in V4 if not r['digits'])}/{sum(not r['digits'] for r in V4)}  wcmis {sum(r['cer']>th for r in V4+rand if any(f.startswith('wordcount') for f in r['flags']))}/{sum(any(f.startswith('wordcount') for f in r['flags']) for r in V4+rand)}")
# best-try agreement on retried chunks
g=defaultdict(dict)
for r in V3:
    if r["id"] in retr: g[r["id"]][r["k"]]=r
agree=tot=0
for i,d in g.items():
    if len(d)<2: continue
    ob=min(d,key=lambda k:(d[k]["old_cer"],k)); nb=min(d,key=lambda k:(d[k]["cer"],k))
    tot+=1; agree+=ob==nb
print("best-try agreement",agree,"/",tot)
# old final (best of tries) still >0.35 vs new
fin_old=[min(d[k]["old_cer"] for k in d) for d in g.values()]; fin_new=[min(d[k]["cer"] for k in d) for d in g.values()]
print("retried ids final old>0.35:",sum(x>0.35 for x in fin_old),"final new >0.2/0.25/0.3:",[sum(x>t for x in fin_new) for t in (0.2,0.25,0.3)])
print("deva share in hyp: tokens",sum(r["hyp_deva_tokens"] for r in R),"oov",sum(r["hyp_oov_tokens"] for r in R),"words", sum(len(r["trelis"].split()) for r in R))
print("--- worst V4 by new cer")
for r in sorted(V4,key=lambda r:-r["cer"])[:12]:
    print(round(r["cer"],3),r["dur"], "|REF",r["ref_m1"],"\n      |HYP",r["hyp_m1"])
print("--- worst V3 random by new cer (old)")
for r in sorted(rand,key=lambda r:-r["cer"])[:8]:
    print(round(r["cer"],3),r["old_cer"],"|REF",r["ref_m1"],"\n      |HYP",r["hyp_m1"], "\n      |OLD",r["old_text"])
print("--- V3 old-rejected try0 that new passes at 0.2 (sample)")
for r in [r for r in badset if r["cer"]<=0.2][:5]:
    print(round(r["cer"],3),r["old_cer"],"|REF",r["ref_m1"],"\n      |HYP",r["hyp_m1"], "\n      |OLD",r["old_text"])
