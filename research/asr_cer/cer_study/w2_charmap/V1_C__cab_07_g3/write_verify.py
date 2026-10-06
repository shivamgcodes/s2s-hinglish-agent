import json
char_sample={210:"WRONG: 'k' of pickup vs hyp space labelled sub; 'ck' is one /k/ already spoken as क (cf. steps 169/256 'ck spelled by क'); should be k absorbed + space ins",
165:"WRONG: 'ch' in checking is one phoneme heard as ट; map counts c del + h sub (2 errors for 1 sound); same at 252",
192:"label count ok but pairing dubious: 'u' del + 't' sub with ा; better u~ा sub, t del (no count change)",
379:"WRONG: 'wh' in white is one /w/; 'h' paired with ा as sub, but ा is part of /aɪ/ (cf. step 351 i~ाइ match)",
31:"WRONG: 'g' ins in 'right' -- gh is silent, no extra sound; Latin-vs-Latin compared by spelling not sound (also 446)",
32:"WRONG: 'h' ins in 'right' -- silent gh (also 447)",
490:"WRONG: 'i' of 'din' matched to 'i' in hallucinated 'milte' far away; spurious match, din~day should be d match + i/a, n/y subs",
401:"ok-lenient: 0 ~ 'जी रो' (zero, z~ज, Indian pronunciation); internal space absorbed into match",
71:"ok sub (help->हिल्प, e vs i)",247:"ok sub (delhi->दिली)",203:"ok match (the ~ द, Indian 'th')",218:"ok",219:"ok",221:"ok letter name I",223:"ok letter name G",
244:"ok 3~थ्री",337:"ok 1~वन",339:"ok sub two->दू",363:"ok",97:"ok y~ी",317:"ok c~स",351:"ok i~ाइ",402:"ok sub one->वा",417:"ok ins nasal",
128:"ok sub d/B",160:"ok del",296:"ok del",294:"ok sub",387:"ok sub u/ो",378:"ok sub w/भ"}
char_wrong=[210,165,379,31,32,490]
word_sample={0:"ok",6:"ok",11:"ok",14:"WRONG: help ~ हिल्प labelled match, but vowel heard differently (hilp); char map itself labels e/ि as sub -> should be sub",
19:"ok",26:"ok",31:"ok",34:"ok",35:"ok",37:"ok",40:"ok",44:"ok",51:"WRONG: delhi ~ दिली 'match' though vowel e->i differs (char map: sub)",
60:"ok",66:"WRONG: delhi ~ डिली 'match' though e->i (char map: sub)",74:"ok",79:"WRONG: maruti ~ मारोटी 'match' though u->o (char map step 387: sub)",83:"ok",95:"ok",99:"ok"}
word_wrong=[14,51,66,79]
v={"pair":"V1_C__cab_07_g3","char_map_valid":True,"word_map_valid":True,
"recount":{"char":{"match":417,"sub":29,"del":13,"ins":44,"ref_chars":459,"cer":0.1874},"word":{"match":65,"sub":24,"del":1,"ins":16,"ref_words":90,"wer":0.4556}},
"metrics_files_agree":True,
"char_sample":{str(k):v for k,v in char_sample.items()},"char_wrong":char_wrong,
"word_sample":{str(k):v for k,v in word_sample.items()},"word_wrong":word_wrong,
"sample_size":len(char_sample)+len(word_sample),"judgement_errors_in_sample":len(char_wrong)+len(word_wrong),
"issues":[
"Structural: concatenations reproduce normalised ref/hyp exactly for both maps; recounts match metrics files (CER 0.1874, WER 0.4556).",
"30 char steps are op='match' with empty hyp (absorbed digraphs/silent letters/inherent schwa); 0 cost by design but shape-wise they are deletions forgiven by judgement.",
"Inconsistent policy: Latin-vs-Latin spans (right/ridenow, steps 31-34, 446-449) compared by spelling (silent gh counted as 2 ins) while Latin-vs-Devanagari compared by sound.",
"Digraph handling inconsistent: 'ck' absorbed (169,256) but 'ch' in checking counted as del+sub (165-166, 252-253); 'wh' in white h paired with ा (379); pickup 'k' vs space sub (210).",
"Spurious far match: 'i' of 'din' aligned to 'i' in hallucinated 'milte' (490), under-counting by 1.",
"Word map vs char map disagree: help/हिल्प, delhi/दिली, delhi/डिली, maruti/मारोटी are word 'match' but char-level vowel 'sub'; strict WER would be 45/90=0.50.",
"Estimated strict-by-sound CER after corrections ~78/459=0.170 (rough).",
"Lenient: '0'~'जी रो' (z~ज) counted match."]}
json.dump(v,open('verify.json','w'),ensure_ascii=False,indent=1)
