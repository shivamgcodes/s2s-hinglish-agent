"""Build sound-aligned character map. Segments: identical stretches are char-matched
automatically; differing stretches are hand-aligned below."""
import json
from normalize import norm
REF = norm(open("reference.txt", encoding="utf-8").read())
HYP = norm(open("whisper.txt", encoding="utf-8").read())

def M(r, h, why="same sound"): return ("match", r, h, why)
def S(r, h, why): return ("sub", r, h, why)
def D(r, why): return ("del", r, "", why)
def I(h, why): return ("ins", "", h, why)

def num_12(): return [M("1", "twel", "12 spoken as ordinal 'twelfth' - same number"),
                      M("2", "fth", "12 spoken as ordinal 'twelfth' - same number")]
def year(last_digit, last_word):
    return [M("2", "twen", "20 read as 'twenty' - same number"),
            M("0", "ty ", "20 read as 'twenty' - same number"),
            M("2", "twenty ", "2x read as 'twenty ...' - same number"),
            M(last_digit, last_word, f"{last_digit} read as '{last_word}' - same number")]

# sequence of plain strings (must be identical in both) and hand-aligned step lists
SEG = [
 "hello thank you for calling str",
 [S("e", "i", "'ea' /i:/ in stream vs short /I/ in string - different vowel"),
  D("a", "second letter of the 'ea' digraph - no separate sound in hyp"),
  S("m", "ng ", "/m/ vs /ng/ (stream->string); word-break space folded in, not a sound")],
 "box this is ",
 [M("v", "व"), M("a", "ा", "a ~ aa: Hinglish romanisation of Varun's vowel"),
  M("r", "र"), M("u", "ु"), M("n", "ण", "Varun is conventionally वरुण with ण")],
 " how can i help you today sure i can check that for you can i get your email address okay i understand your concern could you share your account number thank you a",
 [M("l", "ll ", "alright vs all right - same sound, spelling/word-break variant")],
 "right let me pull up your account you",
 [I("r", "audio says you're; reference text truncated to you'"), I("e", "you're - inserted vowel")],
 " on the family annual plan it renews annually and is active last renewal was october ",
 num_12(), [M(" ", " ")], year("4", "four"),
 " next renewal is october ", num_12(), [M(" ", " ")], year("5", "five"),
 " i",
 [I("m", "audio says I'm; reference truncated to I'")],
 " sorry could you repeat that i",
 [I("m", "audio says I'm; reference truncated to I'")],
 " sorry i didnt catch that could you repeat your name please thank you but that",
 [I("s", "audio says that's; reference truncated to That'")],
 " correct your account is active renewal is confirmed for next year anything else i can help with today you",
 [I("r", "audio says you're; reference truncated to You'"), I("e", "you're - inserted vowel")],
 " welcome have a great day",
]
steps = []
for seg in SEG:
    if isinstance(seg, str):
        steps += [("match", c, c, "identical") for c in seg]
    else:
        steps += seg
with open("char_map.jsonl", "w", encoding="utf-8") as f:
    for i, (op, r, h, why) in enumerate(steps):
        f.write(json.dumps({"i": i, "ref": r, "hyp": h, "op": op, "why": why}, ensure_ascii=False) + "\n")
print(len(steps), "steps written")
