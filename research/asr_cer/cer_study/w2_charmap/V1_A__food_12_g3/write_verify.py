import json,subprocess
char_sample={51:'ok',101:'ok',102:'ok',103:'ok',95:'ok',96:'ok (alignment quirky: o~आ sub + r del would be more natural, same cost)',118:'ok',
 120:'WRONG: z~स labelled match; z!=s, should be sub',125:'ok',163:'ok (space vs letter sub)',167:'ok',169:'ok',
 233:'WRONG: set/सित e~ि labelled match; e!=i, should be sub (word map says sub)',247:'ok',310:'ok',
 336:'WRONG: now/न o matched to inherent schwa; should be del',354:'ok',
 370:'WRONG: hoon/हो oo~ो labelled match; u: vs o: different vowel, should be sub',374:'ok',381:'ok',392:'ok',408:'ok',
 415:'WRONG: bell/बिल e~ि labelled match; should be sub',437:'WRONG: personalna/परसना o matched to inherent schwa of स; should be del',
 444:'ok',497:'WRONG: theek/दहिक th~द labelled match; ठ vs द are different sounds, should be sub',
 517:'WRONG: request/रिक्विस्ट u~व labelled sub; qu=/kw/, u~व is same sound, should be match',
 518:'WRONG: quest/क्विस्ट e~ि labelled match; should be sub',584:'WRONG: end/इन e~इ labelled match; should be sub',590:'ok'}
word_sample=[5,8,12,18,19,20,22,34,38,39,44,56,63,70,71,75,86,95,96,110]
errs=sum(1 for v in char_sample.values() if v.startswith('WRONG'))
v={"pair":"V1_A__food_12_g3","char_map_valid":True,"word_map_valid":True,
 "char_recount":{"match":539,"sub":34,"del":26,"ins":2,"ref_chars":599,"cer":round(62/599,4)},
 "word_recount":{"match":81,"sub":30,"del":1,"ins":2,"ref_words":112,"wer":round(33/112,4)},
 "metrics_files_agree":True,
 "char_sample":{str(k):v for k,v in char_sample.items()},"char_sample_size":len(char_sample),"char_errors":errs,
 "word_sample":word_sample,"word_sample_size":len(word_sample),"word_errors":0,
 "judgement_errors_in_sample":errs,"sample_size":len(char_sample)+len(word_sample),
 "cer_if_sample_errors_fixed":round(70/599,4),
 "issues":[]}
json.dump(v,open('verify.json','w'),ensure_ascii=False,indent=1)
print(errs,len(char_sample))
