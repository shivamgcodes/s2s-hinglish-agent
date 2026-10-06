import json, subprocess
char_sample={ # step: (label_correct, note)
 31:(1,"Tune vs Tuna: silent e vs a, sub ok"),32:(1,"extra space ins ok"),55:(1,"ai->ai matra, i absorbed"),
 73:(1,"help p vs b: sub ok"),86:(1,"hoon oo ~ uu"),97:(1,"please s=/z/ ~ ज"),114:(1,"ID letter name I = आई"),
 122:(1,"phone o vs uu: sub ok"),128:(1,"anusvara before b = m"),137:(1,"bataiye i absent: del ok"),146:(1,"koi o absent: del ok"),
 154:(1,"pallavi l: del ok"),189:(1,"premium u ~ य glide (borderline, accepted)"),203:(1,"Rs not heard: del"),210:(1,"dollar ins"),
 214:(1,"and vs in: sub"),246:(1,"the ~ दे (Indian-English d)"),250:(1,"reason r vs v: sub"),257:(1,"why w vs bh: sub"),
 259:(1,"why y=/ai/"),345:(1,"please l vs r: sub"),391:(1,"abhi short a vs आ: sub (strict ok)"),420:(1,"provide i=/ai/"),
 444:(1,"english final sh missing: del"),458:(1,"thank vs थे: sub"),472:(1,"the vs बे: sub"),484:(1,"-tion ti = श"),
 575:(1,"request q ~ क्"),582:(1,"is vs इद: s vs d sub"),615:(1,"calling c vs p: sub"),627:(1,"tune e vs अ (cost-equivalent to ins)"),
 648:(1,"trailing hallucination ins")}
word_sample={5:(1,"Tuna sub"),6:(1,"Plus ins"),13:(1,"helb sub"),21:(1,"ID match"),23:(1,"phoon sub"),25:(1,"bataye spelling variant ok"),
 27:(1,"koi vs ki sub"),29:(1,"name spelling variant ok"),38:(1,"Rs del"),40:(1,"dollar ins"),41:(1,"and vs in sub"),
 48:(1,"reason vs veezan sub"),60:(0,"understand->अंडरस्टेन drops final d; char map marks del; strict = sub (english->इंग्ली was sub)"),
 71:(0,"field->फिल ('fill'): final d dropped + short vowel; char map marks del; strict = sub"),
 74:(0,"abhi->आभी labelled match but char map labels the vowel-length difference sub; strict = sub"),
 83:(1,"ingli sub"),86:(1,"thank->थे sub"),106:(1,"merged token sub"),107:(1,"del of absorbed word"),115:(1,"paling sub")}
del char_sample[648]
cm=[json.loads(l) for l in open('char_map.jsonl',encoding='utf-8')]
wm=[json.loads(l) for l in open('word_map.jsonl',encoding='utf-8')]
mech=json.loads(subprocess.check_output(['python3','verify_check.py']))
errs=sum(1-v[0] for v in char_sample.values())+sum(1-v[0] for v in word_sample.values())
out={"pair":"V1_A__sub_11_g3",
 "char_map_valid":mech['char_ref_ok'] and mech['char_hyp_ok'],
 "word_map_valid":mech['word_ref_ok'] and mech['word_hyp_ok'],
 "char_ops_recount":mech['char_ops'],"word_ops_recount":mech['word_ops'],
 "cer_recount":round(mech['cer'],4),"wer_recount":round(mech['wer'],4),
 "metrics_files_agree":True,
 "wer_if_flagged_labels_fixed":round((30+3)/119,4),
 "sample_size":len(char_sample)+len(word_sample),"char_sample_size":len(char_sample),"word_sample_size":len(word_sample),
 "judgement_errors_in_sample":errs,
 "char_sample":[{"i":k,"ref":cm[k]['ref'],"hyp":cm[k]['hyp'],"op":cm[k]['op'],"correct":bool(v[0]),"note":v[1]} for k,v in char_sample.items()],
 "word_sample":[{"i":k,"ref":wm[k]['ref'],"hyp":wm[k]['hyp'],"op":wm[k]['op'],"correct":bool(v[0]),"note":v[1]} for k,v in word_sample.items()],
 "issues":[
  "Word map labels understand->अंडरस्टेन (i=60), field->फिल (i=71), abhi->आभी (i=74) as match although the char map records del/del/sub inside them and english->इंग्ली is labelled sub; strict WER would be 33/119=0.2773 instead of 0.2521.",
  "Char map has 100 'match' steps with an empty hyp field (ref letters absorbed into an akshara, inherent schwa, digraphs, silent e); concatenation still exact, but these are not 1:1 char pairs.",
  "Hyp side is chunked by akshara/grapheme (e.g. 'डु', 'मैं', 'आई'), so ins/sub counts depend on that unit; hyp_chars=592 is in codepoints, not units. CER denominator (642 ref chars) unaffected.",
  "Steps 31/627 label silent-e vs a/अ as sub; cost-equivalent to an ins, not a count error.",
  "Mechanical checks pass: concatenations reproduce normalised ref/hyp exactly; recounts equal metrics files (CER 57/642=0.0888, WER 30/119=0.2521)."]}
json.dump(out,open('verify.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(errs,out['sample_size'],out['cer_recount'],out['wer_recount'])
