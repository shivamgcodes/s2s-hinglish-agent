"""Single source: packages/hinglish_text (D-SINGLE-SOURCE 2026-10-07; used by research/data_gen and the N1 data build
in packages/needle_router/n1/build_data.py). Origin: pod1 hinglish/gen/hindi_share.py (= needle/vendor copy). Edits:
the Google word lists (google-20k-en.txt, google-10000-en.txt; github.com/first20hours/google-10000-english, no clear
licence) are NOT in the repo: put them in a dir and set HINDI_SHARE_LEXICON (default: ./lexicon); selftest guidance
path from env NEEDLE_GUIDANCE_MD. lexicon/{hindi,english}_extra.txt are project-made and stay here.

Deterministic Hindi-token share for romanised Hinglish (FROZEN once V1 is generated).

Used by gen/validate.py and tests/score.py. Pure python, no GPU, no third-party deps.

RULE (documented; do not change after V1 generation):
 1. Pre-strip ID-like spans before tokenising: Indian vehicle plates (DL 01 AB 4821), and any
    token containing a digit (A1234, 18A, 6:40, 98000 01234, 540, 2nd) -> EXCLUDED.
 2. Tokenise on letters (apostrophes kept inside words: domino's, don't). Lower-case for lookup.
 3. Per-call EXCLUDE set (names/brands/places from the record + a built-in Delhi/NCR/city list)
    -> EXCLUDED from the denominator.
 4. Token in the romanised-Hindi lexicon (HINDI, incl. generated verb conjugations) -> HINDI,
    unless it is in AMBIG.
 5. AMBIG tokens (words that are both romanised Hindi and English: main, to, is, us, me, do, hi, so, use,
    tab, man, sun, ban, bad, mile, lag, ...): take up to 2 nearest non-ambiguous known (H/E) tokens on each
    side within the line; nearest on each side weighs 1.0, second 0.5; HINDI if Hindi weight > English
    weight, ENGLISH if lower; tie -> HINDI for 'main'/'na', else ENGLISH.
 6. ALL-CAPS token of length >= 2 in the original text (UPI, PNR, ETA, PM, SOS) -> ENGLISH.
 7. Token in English wordlist (google-20k + EXTRA_EN) -> ENGLISH.
 8. Capitalised (non line-initial) unknown token -> EXCLUDED (proper noun).
 9. Anything else -> UNKNOWN: EXCLUDED from denominator and reported (OOV rate).
 share = HINDI / (HINDI + ENGLISH). Per speaker: pooled over all that speaker's lines.
"""
import os
import re
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_DIR = os.path.join(HERE, "lexicon")
GOOGLE_LEX_DIR = os.environ.get("HINDI_SHARE_LEXICON") or LEX_DIR   # google-*-en.txt (not in the repo)

# ---------------------------------------------------------------- Hindi lexicon (romanised)
_HINDI_WORDS = """
haan han haa haanji haanjee ji jee nahi nahin nai na naa mat kya kyaa kyun kyon kyunki kyonki kaise kaisa kaisi kaun kaunsa kaunsi
kaunse kahan kahaan kidhar kab kitna kitni kitne jab tab ab abhi abhie tabhi kabhi sabhi sab bhi hi toh to bas
aur ya lekin par pe per magar phir fir agar warna varna isliye kyunki jaise waise matlab yaani yani
main mai mein me mujhe mujhko mera meri mere hum ham humein hamein humko hamara hamari hamare hamaare apna apni apne
aap aapka aapki aapke aapko aapse aapne tum tumhara tumhari tumhare tumko tumhe tumhein tu tera teri tere
yeh ye yah woh wo vo voh vah is us isko usko ise use isse usse iska iski iske uska uski uske inka inki inke unka unki unke
inhe inhein unhe unhein inko unko inse unse isme usme ismein usmein yahan yahaan wahan wahaan idhar udhar jahan jahaan
koi kuch kuchh kisi kis kisne kise kiska kiski kiske jo jis jise jiska jiski jiske jinhe sabko sabse dono donon
ko se ka ki ke ne tak liye lie saath sath baad pehle pahle andar bahar baahar upar neeche niche paas pass samne saamne
peeche piche taraf wala wali wale waala waali waale wala.
hai hain hoon hun hu ho tha thi the thay hoga hogi honge hota hoti hote hua hui hue huye
gaya gayi gaye gai gayee gya gyi raha rahi rahe rha rhi rhe chuka chuki chuke saka saki sake
ek do teen char chaar paanch panch chhe chheh saat aath nau das gyarah barah pandrah bees tees chalis pachas sau hazaar hazar
pehla pehli pehle doosra doosri doosre dusra dusri dusre teesra aadha adha dedh dhai
bahut bohot bahot thoda thodi thode zyada jyada kam kaafi kafi bilkul ekdum zaroor jaroor zarur jarur shayad sirf bas
achha accha acha achchha acchha achhi acchi achhe acche theek thik sahi galat bura buri bure badiya badhiya
jaldi jald der deri late? aaj kal parson subah sham shaam raat dopahar din hafte hafta mahina mahine saal ghanta ghante baje baj
ghar dukaan dukan gaadi gadi gaddi rasta raasta sadak gali mohalla makaan makan kamra darwaza darwaaza gate? ghanti
paisa paise rupaye rupay rupees? khana khaana khane pani paani chai doodh roti sabzi sabji daal dal chawal biryani raita paneer
samosa dosa thali lassi chhole bhature halwa mithai naan kulcha parantha paratha
log logon aadmi aurat bhai bhaiya bhaiyya didi behen bhen behan maa papa mummy mammi pita mata beta beti bachcha bachche
pati patni bhabhi chacha chachi mausi mama nana nani dada dadi parivaar parivar ghar-wale gharwale sasural dost
naam number? pata jagah samay waqt vakt baat baatein cheez cheezein kaam kaam-kaaj sawaal sawal jawab jawaab madad
shukriya dhanyavaad dhanyawad dhanyavad namaste namaskar alvida maaf maafi kripya krupya
yaar yar arre are arey arrey achha-achha oho uff haaye hmm accha. chalo chaliye chalie dekhiye suniye boliye bataiye
pakka pakki ruko rukiye rukiyega isi usi yahi wahi wohi vahi abhi-to koi-baat
naya nayi naye purana purani purane sasta mehnga mehenga mahanga thanda thandi garam garm gila geela geeli
khush naraz naraaz pareshan pareshaan tension? dikkat dikkatt problem? taklif takleef pareshani
sach sachmuch waise vaise aise waisa aisa aisi waisi itna itni itne utna utni utne jitna jitni
zaroorat jarurat zarurat chahiye chaahiye chahie mangta milega milegi milenge mila mili mile? milta milti
abhi-abhi turant fauran seedha sidha seedhe sidhe ulta wapas vapas waapas dobara dubara phirse fir-se
saamaan samaan saman bag? jhola paiso dabba dibba packet? thaila
haath pair sar aankh muh dil man? tabiyat tabiyet
kiraya kiraaya bhada bhaada safar yatra udaan
sabse zyada-se-zyada kam-se-kam lagbhag kareeb karib takreeban
haina na? naa? achhaa
hogaya hogya hojayega ho-jayega karo karna karni karne karta karti karte kiya kiye kiyaa ki? kijiye kijiyega karenge karega karegi karunga karungi karoonga karoongi karoge karogi kariye kariyega karke karkar kara karaa karaya karayi karaye karwa karwana karwaya karwayi karwaiye karwa-do
diya diye di dijiye dijiyega dena dene deni deta deti dete denge dega degi dunga dungi doonga doongi doge dogi deke dekar do-na dedo de-do de-dijiye de
liya liye li lijiye lijiyega lena lene leni leta leti lete lenge lega legi lunga lungi loonga loongi loge logi leke lekar lo le
hona hone honi hoti hote hota hoke hokar hoga hogi honge hoon ho jaana jana jaane jaani jaata jata jaati jati jaate jate gaya gayi gaye jaiye jaiyega jaayega jayega jaayegi jayegi jaayenge jayenge jaunga jaungi jaoge jaogi jao jaa ja jaake jaakar jakar
aana aana aane aani aata aati aate aaya aayi aaye aaiye aayega aayegi aayenge aaunga aaungi aaoge aao aake aakar aa aaj
sakta sakti sakte saka saki sake sakega sakegi sakenge sakunga sakungi sakoge
chahta chahti chahte chaha chahi chahiye chahunga chahungi chahoge
rakha rakhi rakhe rakhna rakhiye rakh
"""

# verb roots ending in a consonant: root + suffixes
_CONS_ROOTS = """reh kar dekh bol sun mil lag rakh ruk samajh pahunch pahuch nikal badal chal baith padh likh soch bhool bhul ban bhej
maang mang bhar chhod chod pakad jod uth laut pooch puch sambhal dhoondh dhundh gir khul bach jeet jit kat kaat bik bech
pahan nibat sudhar sudhaar bigad tut toot phat phas phans ghoom ghum lauta lautt cancel? jhel sah"""
_CONS_SUFFIX = ["", "na", "ne", "ni", "ta", "ti", "te", "a", "i", "e", "o", "iye", "iyega", "enge", "egi", "ega", "unga",
                "ungi", "oonga", "oongi", "oge", "ogi", "ke", "kar", "wa", "wana", "wane", "waya", "wayi", "waye", "waiye",
                "wao", "ati", "ata", "ate", "aya", "ayi", "aye", "aiye", "ayenge", "ayega", "ayegi", "aunga", "aungi", "ao",
                "ana", "ane", "aake", "aakar"]
# verb roots ending in -a (causatives etc.): root + suffixes
_A_ROOTS = """bata dikha bula hata utha mila samjha pahuncha nipta badla chala laga sambhala lauta bitha jama bana bacha
ruka sudhra bhijwa karwa dilwa"""
_A_SUFFIX = ["", "na", "ne", "ni", "ta", "ti", "te", "ya", "yi", "ye", "iye", "iyega", "yenge", "yega", "yegi", "unga",
             "ungi", "oonga", "oongi", "oge", "ogi", "o", "ke", "kar", "enge", "ega", "egi", "ye", "i", "e"]

# Tokens that are also ordinary English words; resolved by context (rule 5).
AMBIG = {"main", "to", "is", "us", "me", "do", "hi", "so", "bus", "use", "tab", "man", "sun", "ban", "more", "bad",
         "mile", "lag", "pal", "per", "pass", "bat", "the", "mil", "chal", "bach",
         "dal", "bag", "tension", "problem", "rupees", "gate", "number", "late", "a", "i", "e", "o", "die",
         "re", "are", "pi", "bata", "bhar", "kat", "log"}
# AMBIG tokens that are really English and should simply be English (not context-resolved):
ALWAYS_EN = {"the", "a", "i", "problem", "tension", "number", "late", "rupees", "gate", "bag", "die", "are", "pass",
             "more", "bat", "per", "e", "o", "re"}
DEFAULT_HI = {"main", "na"}

EXTRA_EN = """okay ok hello hi bye thanks thank sorry please sir madam maam ma'am yeah yep nope hmm wow
app apps online offline refund refunds cancel cancelled canceled cancellation cancelling reschedule rescheduled
eta otp upi pnr sos gb tb pm am km kms kg wifi ac pin pincode email emails gmail id ids
delivery deliver delivered delivering rider riders driver drivers cab cabs ride rides pickup drop dropoff
order orders ordered ordering subscription subscriptions renewal renew renewed renews renewing plan plans premium
monthly yearly annual annually quarterly billing billed charge charged charges payment payments paid pay
booking bookings booked flight flights boarding gate terminal seat seats window aisle baggage luggage cabin check-in checkin
connecting connection onward layover itinerary passenger passengers meal meals veg vegetarian non-veg jain
address addresses landmark tower flat floor apartment block sector phase lane road street colony society building
office home parcel package packages product products item items variant colour color size quantity
status update updated updating instruction instructions contact number numbers phone mobile call called calling
confirm confirmed confirming confirmation request requested requesting submit submitted processing process
minute minutes second seconds hour hours today tomorrow yesterday morning evening night
security guard reception lobby entrance exit parking trolley suitcase suitcases
hotel airport station metro mall market headphones earphones phone case shoes laptop charger watch
pizza burger fries coke pepsi sandwich wrap momos noodles rice cold soggy wet packaging damaged broken
frustrating frustrated annoying actually basically exactly definitely obviously literally seriously
noted done sure alright right fine great perfect
don't can't won't i'm it's that's i'll you're we'll didn't isn't doesn't i've let's what's there's haven't wasn't
you'll we're they're he's she's couldn't shouldn't wouldn't aren't weren't i'd you've"""
# generated or listed Hindi forms that are really English words in this domain: forced English
FORCE_EN = {"minute", "banana", "bike", "kate", "lie", "manga", "packet", "pair", "phase", "bitter", "kare?"}


def _gen_hindi():
    words = set()
    for w in _HINDI_WORDS.split():
        w = w.strip(".").rstrip("?")
        if w and "?" not in w:
            for part in w.split("-"):
                words.add(part)
    for r in _CONS_ROOTS.split():
        if r.endswith("?"):
            continue
        for s in _CONS_SUFFIX:
            words.add(r + s)
    for r in _A_ROOTS.split():
        for s in _A_SUFFIX:
            words.add(r + s)
    return words - FORCE_EN


@lru_cache(maxsize=1)
def _lexicons():
    hindi = _gen_hindi()
    eng = set()
    for fn in ("google-20k-en.txt",):
        p = os.path.join(GOOGLE_LEX_DIR, fn)
        with open(p, encoding="utf-8") as f:
            eng.update(w.strip().lower() for w in f if w.strip())
    eng.update(EXTRA_EN.split())
    # extra lexicon files, one word per line, optional (frozen with the code)
    for fn, target in (("hindi_extra.txt", hindi), ("english_extra.txt", eng)):
        p = os.path.join(LEX_DIR, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                target.update(w.strip().lower() for w in f if w.strip() and not w.startswith("#"))
    return hindi, eng


PLACES = """delhi new-delhi ncr gurgaon gurugram noida faridabad ghaziabad dwarka saket rohini janakpuri pitampura vasant kunj
lajpat nagar karol bagh connaught chanakyapuri hauz khas malviya greater kailash mayur vihar preet laxmi shahdara
okhla nehru kalkaji defence colony rajouri punjabi bagh paschim vihar mehrauli chhattarpur munirka indirapuram vaishali
kaushambi sohna golf cyber dlf udyog vihar huda mumbai bombay bangalore bengaluru chennai kolkata hyderabad pune jaipur
lucknow chandigarh goa ahmedabad kochi patna bhopal indore amritsar varanasi srinagar leh dehradun manesar
aerocity igi palam mahipalpur rajiv chowk kashmere gate anand vihar sarai kale khan nizamuddin""".split()

PLATE_RE = re.compile(r"\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4}\b")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏ0-9]+(?:['’][A-Za-z]+)*|[0-9][0-9:.,/-]*[0-9A-Za-z]*")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def tokenize(line):
    """Returns list of (raw_token, start_index). Plates are replaced by a single digit-bearing placeholder."""
    line = PLATE_RE.sub(" PLATE0 ", line)
    return [(m.group(0), m.start()) for m in TOKEN_RE.finditer(line)]


def build_exclude(record=None, extra=()):
    """Per-call exclude set from a record dict: every token of name/brand/place-like fields, plus capitalised
    tokens found in any record string value that are not in the Hindi or common-English lists."""
    hindi, eng = _lexicons()
    ex = set(PLACES)
    for e in extra:
        ex.update(t.lower() for t, _ in tokenize(str(e)))
    if not record:
        return ex

    def walk(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif isinstance(v, str):
            k = key.lower()
            if any(s in k for s in ("name", "brand", "restaurant", "airline", "store", "seller", "city", "locality",
                                    "area", "driver", "model", "service", "app")):
                for t, _ in tokenize(v):
                    tl = t.lower()
                    if tl not in hindi and not any(c.isdigit() for c in tl):
                        ex.add(tl)
            for t, _ in tokenize(v):
                tl = t.lower()
                if t[:1].isupper() and tl not in hindi and tl not in eng:
                    ex.add(tl)
    walk(record)
    return ex


def classify_line(line, exclude=frozenset()):
    """Returns list of (token, label) with label in {'H','E','X','U'} (Hindi, English, excluded, unknown)."""
    hindi, eng = _lexicons()
    toks = tokenize(line)
    labels = []
    for i, (t, pos) in enumerate(toks):
        tl = t.lower().replace("’", "'")
        base = tl.split("'")[0] if "'" in tl and tl.endswith("'s") else tl
        if any(c.isdigit() for c in tl):
            labels.append("X")
        elif base in exclude or tl in exclude:
            labels.append("X")
        elif tl in AMBIG and tl not in ALWAYS_EN:
            labels.append("A")
        elif tl in ALWAYS_EN:
            labels.append("E")
        elif tl in hindi:
            labels.append("H")
        elif len(t) >= 2 and t.isupper():
            labels.append("E")
        elif tl in eng or base in eng:
            labels.append("E")
        elif t[:1].isupper() and i > 0:
            labels.append("X")
        else:
            labels.append("U")
    # resolve ambiguous tokens by context: nearest known neighbour on each side weight 1.0, second 0.5
    out = list(labels)
    for i, lab in enumerate(labels):
        if lab != "A":
            continue
        left = [labels[j] for j in range(i - 1, -1, -1) if labels[j] in ("H", "E")][:2]
        right = [labels[j] for j in range(i + 1, len(labels)) if labels[j] in ("H", "E")][:2]
        sh = se = 0.0
        for side in (left, right):
            for w, l in zip((1.0, 0.5), side):
                sh += w * (l == "H")
                se += w * (l == "E")
        tl = toks[i][0].lower()
        if sh > se:
            out[i] = "H"
        elif se > sh:
            out[i] = "E"
        else:
            out[i] = "H" if tl in DEFAULT_HI else "E"
    return [(t, l) for (t, _), l in zip(toks, out)]


def counts(lines, exclude=frozenset()):
    c = {"H": 0, "E": 0, "X": 0, "U": 0}
    unknown = []
    for ln in lines:
        for t, l in classify_line(ln, exclude):
            c[l] += 1
            if l == "U":
                unknown.append(t)
    return c, unknown


def share(lines, exclude=frozenset()):
    c, _ = counts(lines if isinstance(lines, (list, tuple)) else [lines], exclude)
    d = c["H"] + c["E"]
    return round(c["H"] / d, 4) if d else 0.0


def call_shares(call, exclude=None):
    """{'agent': share, 'customer': share} for a call dict with turns[].text_roman."""
    if exclude is None:
        exclude = build_exclude(call.get("record"), [call.get("agent_name", ""), call.get("brand", "")])
    res = {}
    for spk in ("agent", "customer"):
        res[spk] = share([t["text_roman"] for t in call["turns"] if t["speaker"] == spk], exclude)
    return res


def _selftest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = [p.strip() for p in ln.strip().strip("|").split("|")]
            if len(parts) == 3 and parts[0].isdigit():
                rows.append((int(parts[0]), parts[1], parts[2]))
    ex = build_exclude(None, ["Domino's", "Amit", "Gmail"])
    print(f"{'#':>2} {'R1':>6} {'R2':>6}   (H/E/X/U)")
    tot = {"R1": {"H": 0, "E": 0, "X": 0, "U": 0}, "R2": {"H": 0, "E": 0, "X": 0, "U": 0}}
    per = {"R1": {"agent": [], "customer": []}, "R2": {"agent": [], "customer": []}}
    for n, r1, r2 in rows:
        out = []
        spk = "agent" if n % 2 == 1 and n not in (19,) else "customer"
        if n in (17, 18):
            spk = "agent"
        for col, txt in (("R1", r1), ("R2", r2)):
            c, unk = counts([txt], ex)
            for k in c:
                tot[col][k] += c[k]
            per[col][spk].append(txt)
            out.append((share([txt], ex), c, unk))
        print(f"{n:>2} {out[0][0]:>6.2f} {out[1][0]:>6.2f}   R1 {out[0][1]} R2 {out[1][1]}"
              + (f"  UNK {out[0][2] + out[1][2]}" if out[0][2] or out[1][2] else ""))
    for col in ("R1", "R2"):
        c = tot[col]
        print(f"{col} pooled share {c['H'] / (c['H'] + c['E']):.3f}  counts {c}  "
              f"agent-rows {share(per[col]['agent'], ex):.3f}  customer-rows {share(per[col]['customer'], ex):.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NEEDLE_GUIDANCE_MD", os.path.join(HERE, "..", "..", "research", "data_gen", "guidance", "r1_vs_r2.md")))
    elif len(sys.argv) > 1 and sys.argv[1] == "--explain":
        print(classify_line(" ".join(sys.argv[2:])))
    elif len(sys.argv) > 1 and sys.argv[1] == "--collisions":
        hindi, eng = _lexicons()
        common = set()
        with open(os.path.join(GOOGLE_LEX_DIR, "google-10000-en.txt")) as f:
            common = {w.strip() for w in f}
        print(sorted((hindi & common) - AMBIG))
