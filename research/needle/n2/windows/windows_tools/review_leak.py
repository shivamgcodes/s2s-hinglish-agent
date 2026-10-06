import json,os,numpy as np,soundfile as sf,sys
V=os.environ.get("N2_ROOT","/root/n2")+"/V4"
for cid in sys.argv[1:]:
    st=json.load(open(f"{V}/stereo/{cid}.json")); x,sr=sf.read(f"{V}/stereo/{cid}.wav",dtype="float32")
    print(cid, st.get("interrupt_mode"), st.get("flags"))
    for t in st["turns"]:
        if t["speaker"]!="agent": continue
        a,b=int(t["start"]*sr),int(t["end"]*sr); c0,c1=x[a:b,0],x[a:b,1]
        r1=np.sqrt(np.mean(c1**2))
        if r1>0.005:
            cc=np.corrcoef(c0,c1)[0,1]
            fr=np.sqrt(np.convolve(c1**2,np.ones(2400)/2400,"same")); act=np.where(fr>0.02)[0]
            ov=[(u["turn"],u["start"],u["end"]) for u in st["turns"] if u["speaker"]=="customer" and u["start"]<t["end"] and u["end"]>t["start"]]
            span="%.2f-%.2f"%((act[0]+a)/sr,(act[-1]+a)/sr) if len(act) else ""
            print("  agent turn",t["turn"],"%.2f-%.2f"%(t["start"],t["end"]),"ch1rms %.4f corr_ch0 %.3f"%(r1,cc),"ch1 active",span,"overlap cust",ov,t.get("truncated"))
