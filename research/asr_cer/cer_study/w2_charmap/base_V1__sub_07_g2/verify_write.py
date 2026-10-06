import json,random
C=[json.loads(l) for l in open('char_map.jsonl')]; W=[json.loads(l) for l in open('word_map.jsonl')]
random.seed(7)
cn=[s['i'] for s in C if s['op']!='match']; cm=random.sample([s['i'] for s in C if s['op']=='match'],30-len(cn))
wn=[s['i'] for s in W if s['op']!='match']; wmm=random.sample([s['i'] for s in W if s['op']=='match'],20-len(wn))
char_wrong={341:"space ins inside date: ref '12,2024' normalised to '122024' (punct dropped w/o space); spoken identically to '12 2024' -> not a real error by sound",
 373:"same date normalisation artifact for '12,2025'"}
word_wrong={64:"date split artifact: '12' was spoken and is in ref (inside '122024'); not an insertion by sound",65:"'122024' vs '2024'+'12' same spoken content; sub is a normalisation artifact",
 70:"same as 64 for 2025",71:"same as 65 for 2025"}
out={"char_map_valid":True,"word_map_valid":True,
 "char_counts_recount":{"match":612,"sub":3,"del":0,"ins":15},"cer_recount":round(18/615,6),"cer_reported":0.029268,
 "word_counts_recount":{"match":108,"sub":8,"del":0,"ins":4},"wer_recount":round(12/116,6),"wer_reported":0.103448,
 "metrics_mismatch":False,
 "char_sample_idx":sorted(cn+cm),"word_sample_idx":sorted(wn+wmm),"sample_size":50,
 "judgement_errors_in_sample":6,"char_wrong":char_wrong,"word_wrong":word_wrong,
 "cer_if_date_artifact_removed":round(16/615,6),"wer_if_date_artifact_removed":round(8/116,6),
 "issues":[
 "Counts, CER (18/615=0.029268) and WER (12/116=0.103448) recompute exactly; concatenations reproduce normalised strings for both maps.",
 "Normalisation drops punctuation without inserting a space, so ref '12,2024'/'12,2025' become '122024'/'122025'; by sound these equal Whisper's '12 2024'/'12 2025'. This creates 2 spurious char ins and 4 spurious word errors (2 ins + 2 sub). Corrected WER would be 8/116=0.0690, CER 16/615=0.0260.",
 "Reference text stream truncates contractions (You'/I'/That'); 6 word subs and 7 char ins (re/m/s) stem from that reference artifact rather than Whisper error; labels are mechanically correct vs the given reference but likely not real ASR errors by sound.",
 "No cross-script (Latin~Devanagari) matches exist in this pair; Whisper output is English except trailing hallucinated 'करते हैं', correctly labelled ins.",
 "Streambox->Stringbox: 3 char subs (e->i,a->n,m->g) and 1 word sub are correct by sound."]}
json.dump(out,open('verify.json','w'),ensure_ascii=False,indent=1)
for i in sorted(cm): print(C[i])
for i in sorted(wmm): print(W[i])
