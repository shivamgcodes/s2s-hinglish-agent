"""Hand cases for numconv.convert (N2, 2026-10-05). Run: python3 test_numconv.py  (no deps; exits 1 on failure).
Mined real-data metrics are in eval_numconv.py / eval_results.json."""
import sys
from pathlib import Path

import os  # noqa: E402  (monorepo shim: numconv.py lives in packages/needle_router/numconv, $NUMCONV_DIR)
sys.path.insert(0, os.environ.get("NUMCONV_DIR", str(Path(__file__).resolve().parents[4] / "packages" / "needle_router" / "numconv")))
from numconv import convert, extract_numbers  # noqa: E402

POS = [  # (input, expected)
    # phones: digit-by-digit, double/triple, Hindi, Devanagari, digit groups
    ("mera number nau aath double zero seven three five six two one hai", "mera number 9800735621 hai"),
    ("mera number nine eight seven three zero seven five eight nine three hai", "mera number 9873075893 hai"),
    ("number hai 98730 75893", "number hai 9873075893"),
    ("mera number 9873075893 hai", "mera number 9873075893 hai"),
    ("मेरा number ९८७३० ७५८९३ है", "मेरा number 9873075893 है"),
    ("मेरा नंबर नौ आठ सात तीन शून्य सात पांच आठ नौ तीन है", "मेरा नंबर 9873075893 है"),
    ("नाइन एट सेवन थ्री जीरो सेवन फाइव एट नाइन थ्री", "9873075893"),
    ("triple five two two one one nine eight", "555221198"),
    ("double nine eight seven six five four three two", "998765432"),
    ("ek do teen chaar paanch chhe saat aath nau das", "12345678910"),
    ("shunya ek do teen", "0123"),
    # IDs
    ("FD four one two four", "FD4124"),
    ("order FD four one two four hai", "order FD4124 hai"),
    ("PNR FV seven six nine five", "PNR FV7695"),
    ("transaction ID TX six one eight zero", "transaction ID TX6180"),
    ("AC8 nine eight zero", "AC8980"),
    ("gate D four", "gate D4"),
    # amounts / compositional
    ("Rs paanch sau tees", "Rs 530"),
    ("paanch sau tees rupaye", "530 rupaye"),
    ("saat sau rupaye", "700 rupaye"),
    ("सात सौ रुपये", "700 रुपये"),
    ("seven hundred ninety seven rupees", "797 rupees"),
    ("ek lakh bees hazaar", "120000"),
    ("teen lakh", "300000"),
    ("gyarah hazaar paanch sau", "11500"),
    ("dedh sau rupaye", "150 rupaye"),
    ("ninyanve rupaye", "99 rupaye"),
    ("das rupaye", "10 rupaye"),
    ("do rupaye", "2 rupaye"),
    ("two point five GB", "2.5 GB"),
    # units / context words
    ("twenty five minutes", "25 minutes"),
    ("pachchis minute lagenge", "25 minute lagenge"),
    ("teen baje aana", "3 baje aana"),
    ("nau baje tak", "9 baje tak"),
    ("nau baj gaye abhi tak nahi aaya", "9 baj gaye abhi tak nahi aaya"),
    ("eight thirty pm", "8:30 pm"),
    ("sector baais", "sector 22"),
    ("sector das", "sector 10"),
    ("flat number teen sau chaar", "flat number 304"),
    ("terminal two", "terminal 2"),
    ("twelve", "12"),
    # emails
    ("desk dot pandey three at hotmail dot com", "desk dot pandey 3 at hotmail dot com"),
    ("sharma eighty three eight zohomail dot in", "sharma 83 at zohomail dot in"),
    # REVIEW-numconv-1: Trelis 'at' = eight inside digit strings (real V4 transcripts)
    ("naya number nine at seven three zero seven five eight nine three hai", "naya number 9873075893 hai"),
    ("woh hai at zero seven six zero nine zero seven nine four", "woh hai 8076090794"),
    ("mera number hai at at two six zero three two four two one", "mera number hai 8826032421"),
    ("mera number eight at two six zero three two four two one hai", "mera number 8826032421 hai"),
    ("order ID EC at five zero five hai", "order ID EC8505 hai"),
    ("ride RD at zero zero five cancel", "ride RD8005 cancel"),
    ("card seven at four five", "card 7845"),
    ("last four digits nine at zero one", "last four digits 9801"),
    ("main at seven baje aaunga", "main at 7 baje aaunga"),  # 'at' stays before a time
    ("at five thirty pm", "at 5:30 pm"),
    # REVIEW-numconv-2: 'X hazaar do sau'
    ("ek hazaar do sau", "1200"),
    ("Rs ek hazaar do sau", "Rs 1200"),
    ("teen hazaar do sau rupaye", "3200 rupaye"),
    ("mujhe do sau rupaye wapas chahiye", "mujhe 200 rupaye wapas chahiye"),
]
NEG = [  # must come back unchanged
    "ek minute, main check karti hoon",
    "ek second please",
    "let me check, one minute",
    "एक मिनट मैं check करती हूँ",
    "mujhe paisa wapas do",
    "ek minute do",
    "do teen din mein ho jayega",
    "mere saath aisa kyun hua",
    "ek saath dono order",
    "char log hain",
    "one of my orders",
    "I want to cancel this one",
    "aath din ho gaye",
    "haan ji do minute dijiye",
    "मुझे पैसे वापस दो",
    "आप मेरे साथ ऐसा मत कीजिए",
    "",
    # REVIEW-numconv-1 negatives: real 'at'
    "at eight",
    "at least two days",
    "sharma 83 at gmail dot com",
    "updated to at 26032421",
    # REVIEW-numconv-2/3: 'sau baar', approximate ranges
    "sau baar bola",
    "aath das din ho gaye",
    "आठ दस दिन",
    "paanch das minute",
    "two three days",
    "five ten minutes",
    "two three din lagenge",
    "teen chaar din",
    "do din mein aa jayega",
    "saat mein ho jayega",
]
EXTRACT = [
    ("9 8 7 3 0 8 4 7 6 9", ["9873084769"]),
    ("98730 75893", ["9873075893"]),
    ("Rs 1,85,000 aur FD4124", ["185000", "4124"]),
]


def main():
    fails = 0
    for inp, exp in POS:
        got = convert(inp)
        if got != exp:
            fails += 1
            print(f"FAIL pos: {inp!r}\n   got {got!r}\n   exp {exp!r}")
    for inp in NEG:
        got = convert(inp)
        if got != inp:
            fails += 1
            print(f"FAIL neg: {inp!r} -> {got!r}")
    for inp, exp in EXTRACT:
        got = extract_numbers(inp)
        if got != exp:
            fails += 1
            print(f"FAIL extract: {inp!r} -> {got!r}, exp {exp!r}")
    total = len(POS) + len(NEG) + len(EXTRACT)
    print(f"{total - fails}/{total} passed ({len(POS)} positive, {len(NEG)} negative, {len(EXTRACT)} extract)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
