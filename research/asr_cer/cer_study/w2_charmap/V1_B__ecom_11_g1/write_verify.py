import json
mech=json.load(open('verify_mech.json'))
char_sample=[16,17,26,30,35,38,79,80,86,88,109,116,117,120,190,201,211,217,262,297,316,318,327,359,362,371,400,411,432,544]
char_wrong={17:"for: o->ा labelled match, but 'for' vowel /ɔ/ vs 'paar' /aː/ differs (word map itself says 'different vowel'); should be sub",
 38:"main: 'ai' (मैं /ɛ̃/) vs hyp में /e/ labelled match; should be sub (word map scores it sub)",
 79:"hoon: 'oo' (ū) vs hyp ो (o) labelled match; should be sub (word map says vowel lost)",
 201:"on: o vs अ ('un') labelled match; word map scores sub 'vowel differs'; should be sub",
 297:"delivery: e->ि labelled sub, but 'delivery' is pronounced /dɪˈlɪvəri/ so ि is the same sound; should be match (word map says match)",
 400:"same as 297 (second 'delivery')"}
word_sample=[3,6,14,16,22,23,26,36,37,39,44,57,60,66,72,80,92,105,112,124]
word_wrong={}
c=mech['char_counts']
adj_err=c['sub']+c['del']+c['ins']+7+2+1+1-2
v={"pair":"V1_B__ecom_11_g1",
 "char_map_valid":mech['char_ref_concat_ok'] and mech['char_hyp_concat_ok'],
 "word_map_valid":mech['word_ref_concat_ok'] and mech['word_hyp_concat_ok'],
 "cer_recount":mech['cer_recount'],"wer_recount":mech['wer_recount'],
 "recount_matches_metrics":True,
 "char_counts_recount":c,"word_counts_recount":mech['word_counts'],
 "char_sample":char_sample,"char_sample_wrong":{str(k):v for k,v in char_wrong.items()},
 "word_sample":word_sample,"word_sample_wrong":word_wrong,
 "judgement_errors_in_sample":len(char_wrong)+len(word_wrong),
 "sample_size":len(char_sample)+len(word_sample),
 "cer_if_full_map_relabelled":round(adj_err/565,4),
 "issues":[]}
v['issues']=[
 "Mechanics OK: char and word map concatenations reproduce normalised ref/hyp exactly; recounted match/sub/del/ins (526/31/8/48; 66/41/0/26) and CER 0.154 / WER 0.6262 equal the metrics files.",
 "Normalised reference deletes punctuation (H-12 -> h12) whereas tok.py maps punctuation to space (-> 'h 12'); metrics use the deletion form - minor convention inconsistency in the folder's own scripts.",
 "Char-map/word-map label inconsistencies: 'for'->पार vowel o->ा marked match in char map (7x: steps 17,100,180,241,422,490,540) though word map says different vowel; 'main'->में (steps 38,49), 'hoon'->हो (79), 'on'->अन (201) marked char match but word sub; 'delivery' e->ि marked char sub (297,400) though word map (and pronunciation) say match. Relabelling all: errors 87->96, CER 0.154->0.1699.",
 "Step 109 is a 'sub' with empty hyp (ref 'o' vs implicit schwa of र) - defensible by sound but a sub with no hyp chars is structurally odd.",
 "Insertions are counted per akshara (48 steps) not per codepoint (61 inserted codepoints); multi-akshara hyp words mapped to single digit chars (e.g. 9->नाइन as one match) make CER units mixed.",
 "WER number convention heavily inflates errors: '98000 01299' (2 ref words) -> 11 hyp words scored 2 sub + 9 ins though only '8'->एक and one extra zero are actually wrong; '450' scored sub+2 ins each time (3x). 15~फिफ्टीन counted match under same convention. ~25 of 67 word errors come from number tokenisation/hallucinated tail, not mishearing.",
 "Borderline (not counted): 'the' e->ा ('da') marked match - schwa vs long aa."]
json.dump(v,open('verify.json','w'),ensure_ascii=False,indent=1)
print(v['judgement_errors_in_sample'],v['sample_size'],v['cer_if_full_map_relabelled'])
