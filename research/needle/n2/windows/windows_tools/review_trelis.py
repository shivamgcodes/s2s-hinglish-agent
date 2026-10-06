import json,os,random,sys,re
from pathlib import Path
import numpy as np,soundfile as sf,torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
sys.path.insert(0,os.environ.get('NUMCONV_DIR',str(Path(__file__).resolve().parents[5]/'packages'/'needle_router'/'numconv'))); import numconv  # monorepo shim
W=os.environ.get('N2_ROOT','/root/n2')+'/windows'
meta=[json.loads(l) for l in open(W+'/meta.jsonl')]
random.seed(3); samp=random.sample([m for m in meta if m['writes'][0]['tool'] in ('update_contact_number','update_email_address','change_delivery_address','change_pickup_location')],4)+random.sample(meta,4)
def to16k(a):
    n=int(round(len(a)*16000/24000)); X=np.fft.rfft(a); return (np.fft.irfft(X[:n//2+1],n)*(n/len(a))).astype(np.float32)
repo="Trelis/whisper-hinglish-preview"
proc=WhisperProcessor.from_pretrained(repo); model=WhisperForConditionalGeneration.from_pretrained(repo,dtype=torch.bfloat16).to("cuda").eval()
ids=proc.tokenizer.convert_tokens_to_ids; mc=proc.tokenizer("<|mixedcode|>",add_special_tokens=False).input_ids
prompt=[ids("<|startoftranscript|>"),ids("<|hi|>"),*mc,ids("<|transcribe|>"),ids("<|notimestamps|>")]
print('device',next(model.parameters()).device, torch.cuda.get_device_name(0))
for m in samp:
    a,_=sf.read(W+'/'+m['files']['opus_24k'],dtype='float32')
    f=proc.feature_extractor(to16k(a),sampling_rate=16000,return_tensors='pt').input_features.to('cuda',torch.bfloat16)
    with torch.no_grad(): out=model.generate(input_features=f,decoder_input_ids=torch.tensor([prompt],device='cuda'),max_new_tokens=440)
    t=proc.tokenizer.decode(out[0],skip_special_tokens=True).strip()
    conv=numconv.convert(t)
    w=m['writes'][0]
    print('===',m['id'],w['tool'],w['args'])
    print(' ORACLE:',' | '.join(x['text_roman'] for x in m['customer_turns_in_window'])[:400])
    print(' TRELIS:',t[:400]); print(' NUMCONV:',conv[:400])
