import json, subprocess, os
D=os.path.dirname(os.path.abspath(__file__))
rec=json.loads(subprocess.check_output(['python3',f'{D}/verify_recount.py']))
char_sample={  # i: (claimed_op, verdict_correct, note)
 163:('match',True,'igh~ाइ'),164:('match',True,'silent g'),231:('match',True,'th~द'),
 233:('match',True,'the~दा e~ा: lenient but accepted'),255:('match',True,'c~स'),258:('match',True,'doubled l'),
 336:('match',True,'ea digraph'),337:('match',True,'s~ज /z/'),319:('match',False,'mittal schwa vs मिताल long ा; should be sub'),
 134:('match',True,'e~एं'),469:('match',True,'man a~े'),208:('match',True,'november e~schwa (standard नवंबर)'),
 220:('match',True,'status a~े'),157:('del',False,'your~योर: o~ो is heard; should be match (u silent)'),
 222:('del',False,'status u = schwa of ट in standard स्टेटस; should be match'),
 206:('del',False,'november o vs न schwa (standard नवंबर); sound present, not deleted'),
 517:('del',False,'minute u vs मिनट schwa; standard spelling, should be match'),
 622:('del',False,'welcome o vs वेलकम schwa; should be match'),
 421:('del',False,"details 'ai' digraph, i silent; should be match"),260:('del',True,'cancelled -d missing'),
 402:('sub',False,'update u = /ʌ/ ~ अ (standard अपडेट); should be match'),
 414:('sub',False,'new ~ न्यू: ew=/ju:/; should be match'),415:('sub',False,'new ~ न्यू; should be match'),
 431:('sub',False,'resubmit re~री: e~ी should be match, extra space is the only error'),
 161:('sub',True,'f~प'),176:('sub',True,'s~च'),151:('sub',True,'0~Z'),283:('ins',True,'पीम extra ी'),
 31:('ins',True,'in this extra t'),655:('ins',True,'hallucinated tail')}
word_sample={5:True,6:True,7:True,24:True,28:True,31:True,32:True,33:True,
 36:(False,'delhi~दिली labelled match but char map has e->ि sub; dili != delhi'),41:True,46:True,55:True,
 61:(False,'jyoti~जोती: y glide missing (char map deletes y); should be sub'),
 62:(False,'mittal~मिताल: short a/geminate vs long aa; should be sub'),85:True,86:True,
 92:(False,'for~फार labelled match but char map has o->ा sub; faar != for'),
 106:(False,'जी attributed to ji after done; char map attributes it to ji before you are welcome (ref 120); counts unchanged but alignment wrong'),
 124:True,126:True}
cerr=sum(1 for v in char_sample.values() if not v[1]); werr=sum(1 for v in word_sample.values() if v is not True)
issues=[
 "Concatenation checks pass for char and word maps; counts and CER 0.2632 / WER 0.4567 reproduce exactly.",
 "Systematic char-map bias: reference vowel letters realised as schwa (u/o/i) are labelled 'del' when Devanagari has inherent schwa (status x2, november, minute, welcome, reason, resubmit, your, details x2, from) -> CER inflated.",
 "update u~अ (x2) and new~न्यू (e,w) labelled sub though standard loan spellings; resubmit e~ी mis-labelled sub.",
 "mittal a~ा matched: romanisation lets schwa match long ा (length conflation).",
 "Word/char inconsistency: delhi~दिली and for~फार are word 'match' but char map has vowel subs.",
 "jyoti~जोती and mittal~मिताल word 'match' too lenient.",
 "word step 106 vs char map: जी attributed to different ref 'ji' (counts unaffected).",
 "Rough corrected estimate: ~15 fewer char errors -> CER ~0.24; WER +3 subs -> ~0.48.",
 "Sample was targeted at suspicious steps; error rate is not a random-sample estimate."]
out={'char_map_valid':rec['char_ref_concat_ok'] and rec['char_hyp_concat_ok'],
 'word_map_valid':rec['word_ref_tokens_join'] and rec['word_hyp_tokens_join'],
 'cer_recount':rec['cer_recount'],'wer_recount':rec['wer_recount'],'recount_detail':rec,
 'metrics_files_match_recount':True,
 'char_sample':{str(k):{'op':v[0],'correct':v[1],'note':v[2]} for k,v in char_sample.items()},
 'word_sample':{str(k):({'correct':True} if v is True else {'correct':False,'note':v[1]}) for k,v in word_sample.items()},
 'judgement_errors_in_sample':cerr+werr,'char_errors':cerr,'word_errors':werr,
 'sample_size':len(char_sample)+len(word_sample),'issues':issues}
json.dump(out,open(f'{D}/verify.json','w'),ensure_ascii=False,indent=1)
print(cerr,werr,len(char_sample),len(word_sample))
