import json, random, collections, re, hashlib
import os
D=os.environ.get('N1_WORKDIR','/workspace/hinglish/needle').rstrip('/')+'/'  # N1 working dir (was the laptop needle/ dir)
def L(p): return [json.loads(l) for l in open(D+p)]
print("== 1. metric recompute (independent of score fields)")
for t in ['test_n0','test_heldout_a','test_heldout_b']:
    d=json.load(open(D+f'results/base_{t}.json')); rows=d['rows']; M=d['metrics']['overall']
    pos=[r for r in rows if r['positive']]; neg=[r for r in rows if not r['positive']]
    names=lambda cs:[c['name'] for c in cs]
    gold=lambda r:[c['name'] for c in r['gold']] if r['gold'] and isinstance(r['gold'][0],dict) else r['score']['gold_names']
    tm=sum(names(r['function_calls'])==r['score']['gold_names'] for r in pos)
    el=sum(r['function_calls']==[] for r in neg)
    wh=sum(r['function_calls']==[] and len(r['suppressed_calls'])>0 for r in pos)
    # gold names from test file directly
    T=L(f'{t}.jsonl')
    assert len(T)==len(rows)
    tm2=sum(1 for r,x in zip(rows,T) if r['positive'] and names(r['function_calls'])==names(x['answers']))
    qmis=sum(r['query']!=x['query'] for r,x in zip(rows,T))
    print(t,f"tool_match_pos {tm}/{len(pos)} (vs test-file gold {tm2}) report {M['tool_exact_match_positives']['k']}/{M['tool_exact_match_positives']['n']};",
          f"empty_neg {el}/{len(neg)} report {M['empty_list_accuracy_negatives']['k']};",
          f"withheld_only {wh} report {M['positives_withheld_only']['k']}; query mismatch vs test file {qmis}")
    print("   correct_pos via score flag", sum(r['score']['correct'] for r in pos), "report", M['correct_positives']['k'])
print("== 2. disjointness")
S={}
for p in ['train','val','test_heldout_a','test_heldout_b']:
    S[p]=set(m['scenario_id'] for m in L(p+'_meta.jsonl'))
    print(p,len(S[p]),'scenarios')
h=json.load(open(D+'src/holdout.json'))
print('train&val',S['train']&S['val'],'train&test',S['train']&S['test_heldout_a'],'val&test',S['val']&S['test_heldout_a'])
print('test==holdout.test_scenarios', S['test_heldout_a']==set(h['test_scenarios']), 'a==b',S['test_heldout_a']==S['test_heldout_b'])
# query-level leakage
qtr=set(json.loads(l)['query'] for l in open(D+'train.jsonl'))
for p in ['val','test_heldout_a','test_heldout_b','test_n0']:
    q=[json.loads(l)['query'] for l in open(D+p+'.jsonl')]
    print(p,'queries also in train:',sum(x in qtr for x in q),'/',len(q))
print("== 3. positives confirm_turn_idx-2, joins")
calls={}
for V in ['V1','V3']:
    for c in L(f'src/{V}_calls.jsonl'): calls[(V,c['call_id'])]=c
ex=L('examples_raw.jsonl'); pos=[e for e in ex if e['type']=='positive']
random.seed(1); bad=0
tp=[e for e in pos if 'twopart' in e['shape'] or len(e['turn_idx'])>1]
samp=random.sample([e for e in pos if e not in tp],8)+random.sample(tp,7)
for e in samp:
    c=calls[(e['variant'],e['call_id'])]; T=c['turns']; w=c['writes'][e['write_idx']]
    ok1=e['confirm_turn_idx']==w['confirm_turn_idx'] and e['value_turn_idx']==w['confirm_turn_idx']-2
    ok2='check_line' in T[w['confirm_turn_idx']-1]['tags'] and T[e['value_turn_idx']]['speaker']=='customer'
    j=" ".join(T[x]['text_roman'].strip() for x in e['turn_idx'])
    ok3=j==e['renderings']['a']
    ok4=all(T[x]['speaker']=='customer' for x in e['turn_idx'])
    bad+=not(ok1 and ok2 and ok3 and ok4)
    print(e['example_id'],e['shape'],e['turn_idx'],'conf',w['confirm_turn_idx'],ok1,ok2,ok3,ok4)
    if len(e['turn_idx'])>1:
        for x in range(e['turn_idx'][0],e['value_turn_idx']+1): print('    t%d %s: %s'%(x,T[x]['speaker'],T[x]['text_roman'][:110]))
print('bad',bad)
# full-population check
allbad=0
for e in pos:
    c=calls[(e['variant'],e['call_id'])]; w=c['writes'][e['write_idx']]
    if e['value_turn_idx']!=w['confirm_turn_idx']-2 or " ".join(c['turns'][x]['text_roman'].strip() for x in e['turn_idx'])!=e['renderings']['a']: allbad+=1
print('all positives failing idx/join check:',allbad,'/',len(pos))
print("== 4. order_ref on 15 random positives")
random.seed(2)
for e in random.sample(pos,15):
    a=e['answers'][0]['arguments']
    print(f"[{e['order_ref_rule']}] ref={a['order_ref']!r} gold={e['gold']['entity_id']} | {e['renderings']['a'][:160]}")
print("== 5. system facts")
for p in ['test_n0','test_heldout_a','test_heldout_b','train','val']:
    sy=[json.loads(l).get('system') for l in open(D+p+'.jsonl')]
    print(p,'rows',len(sy),'distinct system',len(set(sy)),'all start with pinned date',all((s or '').startswith('date: 2026-10-04 Sun 10:00') for s in sy))
for p in ['test_n0','test_heldout_a','test_heldout_b']:
    print(p,'md5',hashlib.md5(open(D+p+'.jsonl','rb').read()).hexdigest())
