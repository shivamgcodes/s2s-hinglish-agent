#!/usr/bin/env python3
import json, os, collections, sys
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from word_norm import ref_words, hyp_words
def load(p): return [json.loads(l) for l in open(os.path.join(D,p), encoding='utf-8') if l.strip()]
nr = open(f'{D}/norm_ref.txt', encoding='utf-8').read().rstrip('\n')
nh = open(f'{D}/norm_hyp.txt', encoding='utf-8').read().rstrip('\n')
cm = load('char_map.jsonl'); wm = load('word_map.jsonl')
cr = ''.join(s['ref'] for s in cm); ch = ''.join(s['hyp'] for s in cm)
out = {}
out['char_ref_concat_ok'] = cr == nr; out['char_hyp_concat_ok'] = ch == nh
out['char_ref_len'] = len(nr); out['char_hyp_len'] = len(nh)
out['char_idx_contiguous'] = [s['i'] for s in cm] == list(range(len(cm)))
c = collections.Counter(s['op'] for s in cm); out['char_ops'] = dict(c)
# consistency of op vs empty fields
bad = [s['i'] for s in cm if (s['op']=='del' and s['hyp']) or (s['op']=='ins' and s['ref']) or (s['op'] in('match','sub') and not s['ref'])]
out['char_op_field_inconsistent'] = bad
out['char_ref_units_in_steps'] = collections.Counter(len(s['ref']) for s in cm)
out['cer_recount'] = round((c['sub']+c['del']+c['ins'])/len(nr), 4)
rw = ref_words(open(f'{D}/reference.txt',encoding='utf-8').read()); hw = hyp_words(open(f'{D}/whisper.txt',encoding='utf-8').read())
wr = [s['ref'] for s in wm if s['ref']]; wh = [s['hyp'] for s in wm if s['hyp']]
out['word_ref_tokens_join'] = ' '.join(wr) == ' '.join(rw)
out['word_hyp_tokens_join'] = ' '.join(wh) == ' '.join(hw)
out['word_ref_n'] = len(rw); out['word_hyp_n'] = len(hw)
w = collections.Counter(s['op'] for s in wm); out['word_ops'] = dict(w)
out['wer_recount'] = round((w['sub']+w['del']+w['ins'])/len(rw), 4)
out['word_idx_contiguous'] = [s['i'] for s in wm] == list(range(len(wm)))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
