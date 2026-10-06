#!/usr/bin/env python3
"""Manual sound-judgement corrections on char_map.auto.jsonl -> char_map.jsonl.
PATCH: auto step index range [a,b] (inclusive) -> replacement list of (ref,hyp,op,why).
SET: single auto step index -> (op, why) relabel keeping ref/hyp."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
A = [json.loads(l) for l in open(f'{D}/char_map.auto.jsonl', encoding='utf-8')]
SIL = 'silent / spelling letter, no separate sound'
SCH = "schwa: English unstressed vowel ~ inherent 'a' of akshara"
SET = {
 100: ('match', SCH), 113: ('match', "double 'll' = one sound"), 125: ('match', "'oo' ~ ु"),
 145: ('match', "'u' in number (ʌ) ~ inherent 'a'"), 148: ('match', SCH), 155: ('del', "Whisper बताये has no 'i'"),
 171: ('match', "silent 'gh'"), 172: ('match', "silent 'gh'"), 193: ('match', "silent 'h' in Delhi"),
 203: ('del', "'a' of 'bai' missing in मुमी"), 217: ('match', SCH), 225: ('match', "'our' ~ ो, 'u' silent"),
 229: ('match', "'a-e' in gate ~ े"), 231: ('match', "silent final 'e'"), 264: ('match', "'ea' = ee ~ ी"),
 271: ('match', SCH), 276: ('match', "silent final 'e'"), 280: ('match', "'g' in vegetarian pronounced j ~ ज"),
 298: ('match', "'oa' spelling ~ ो"), 320: ('del', "'m' of PM missing (Whisper wrote only पी)"),
 329: ('match', "silent 'gh'"), 330: ('match', "silent 'gh'"),
 338: ('match', "'t' in departure pronounced ch ~ च"), 339: ('match', "'ture' ~ चर"), 341: ('match', SCH),
 359: ('match', SCH), 385: ('match', SCH), 397: ('match', SCH), 408: ('match', "silent 'h' in which"),
 436: ('del', "'d' of record missing in एकर"), 498: ('match', "silent final 'e'"),
 503: ('match', "'a-e' in safe ~ े"), 505: ('match', "silent final 'e'"),
 508: ('sub', "'ou' (ɜː) vs inherent 'a' of च"), 509: ('del', "no separate sound for 'u'"),
 513: ('match', "'ey' = ee ~ ी"),
}
PATCH = {
 (105, 108): [('e', '', 'match', "'ea' = ee ~ ी"), ('a', '', 'match', "'ea' = ee ~ ी"),
              ('s', 'ज', 'match', "'please' pronounced pleez ~ ज"), ('e', '', 'match', "silent final 'e'")],
 (132, 134): [('i', 'आई', 'match', "letter 'I' spoken aai ~ आई"), ('d', 'री', 'sub', "letter 'D' (dee) vs री (ree)")],
 (245, 253): [('s', 'सी', 'match', ''), ('e', '', 'match', "'ea' = ee ~ ी"), ('a', '', 'match', "'ea' = ee ~ ी"),
              ('t', 'ट', 'match', ''), (' ', ' ', 'match', ''),
              ('1', 'थ्वे', 'match', "'12' spoken 'twelve' ~ थ्वेले"), ('2', 'ले', 'match', "'12' spoken 'twelve' ~ थ्वेले"),
              ('f', '', 'del', "letter F not in Whisper")],
 (259, 260): [('a', 'इड', 'sub', "'a' vs 'id' (इड)")],
 (347, 350): [('p', 'पी', 'match', "letter P ~ पी"), ('', ' ', 'ins', 'Whisper splits P M into two words'),
              ('m', 'एम', 'match', "letter M ~ एम")],
 (372, 374): [('t', 'दा', 'match', "'th' (ð) conventionally द"), ('h', '', 'match', "part of 'th' digraph"),
              ('e', '', 'sub', "'the' (də) vs दा (daa)")],
 (377, 381): [('u', '', 'match', "'u' in current (ʌ) ~ inherent 'a' of क"), ('r', '', 'match', "double 'rr' = one sound"),
              ('r', 'रं', 'match', ''), ('e', '', 'match', SCH), ('n', '', 'match', "'n' ~ anusvara ं")],
 (461, 463): [('t', 'दे', 'match', "'th' (ð) conventionally द"), ('h', '', 'match', "part of 'th' digraph"),
              ('e', '', 'match', "'the' ~ दे")],
 (473, 480): [('a', '', 'match', ''), ('t', '', 'del', "ref 't' merged into next Whisper word"), (' ', ' ', 'match', ''),
              ('h', 'थूँ', 'sub', "'h' vs थ (th)"), ('o', '', 'match', "'oo' ~ ू"), ('o', '', 'match', "'oo' ~ ू"),
              ('n', '', 'match', "'n' ~ chandrabindu ँ")],
 (529, 531): [('c', 'को', 'match', "'ca' (kɔː) ~ को"), ('a', '', 'match', "'al' (ɔː) ~ ो"), ('l', '', 'match', "double 'll' = one sound")],
 (537, 576): None,  # tail rebuilt below
}
def S(r, h, op, why): return {'ref': r, 'hyp': h, 'op': op, 'why': why}
out, k = [], 0
starts = {a: (b, v) for (a, b), v in PATCH.items()}
while k < len(A):
    if k in starts:
        b, v = starts[k]
        if v is None:  # tail: 'udaan air' + hallucinated continuation
            out += [S('u', 'उ', 'match', ''), S('d', 'दा', 'match', ''), S('a', '', 'match', "'aa' ~ ा"),
                    S('a', '', 'match', "'aa' ~ ा"), S('n', 'न', 'match', ''), S(' ', ' ', 'match', ''),
                    S('a', 'चि', 'sub', "'ai' (e) vs चि (chi)"), S('i', 'ये', 'sub', "'i' vs ये (ye)"),
                    S('r', '', 'del', "'r' of Air not in Whisper")]
            for s in A[k:b + 1]:
                if s['ref'] == '' or True:
                    pass
            # remaining hyp after 'उदान चिये' = every unit of the auto steps after 'चि','ये'
            hyp_tail = ''.join(s['hyp'] for s in A[k:b + 1])
            consumed = 'उदान चिये'
            assert hyp_tail.startswith(consumed), hyp_tail
            # reuse the auto unit segmentation for the rest
            seen = ''
            for s in A[k:b + 1]:
                if not s['hyp']: continue
                if len(seen) >= len(consumed):
                    out.append(S('', s['hyp'], 'ins', 'Whisper text after call end with no counterpart in ref (hallucinated continuation)'
                                 if s['hyp'] != ' ' else 'extra word boundary in Whisper'))
                seen += s['hyp']
        else:
            out += [S(*t) for t in v]
        k = b + 1; continue
    s = dict(A[k])
    if k in SET: s['op'], s['why'] = SET[k]
    out.append(s); k += 1
with open(f'{D}/char_map.jsonl', 'w', encoding='utf-8') as f:
    for i, s in enumerate(out):
        f.write(json.dumps({'i': i, 'ref': s['ref'], 'hyp': s['hyp'], 'op': s['op'], 'why': s['why']}, ensure_ascii=False) + '\n')
print(len(out), 'steps')
