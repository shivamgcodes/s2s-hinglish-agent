"""V4 records (handoff D2 §2): one caller record per scenario of guidance/scenario_creation_v2.json, with ENFORCED diversity.

Invoked via `python records.py --v2 [...]` (records.py default path is unchanged and still writes data/records.json).

Differences vs the V1/V3 records (records.make_records):
  * seeded random pools (seed V4_SEED) for everything that used to repeat: dates (call dates spread over 3 months, and
    per-type date bundles derived from them so they are ordered by construction), ~60 Delhi/NCR/Mumbai/Bangalore addresses,
    cab landmarks, restaurants, drivers/cars/plates, flight numbers, card last-4, 8 email domains, 20 phone prefixes,
    larger customer name pools. Allocation caps every pooled value at <= 3 records.
  * the Gemma prompt pins those values as FIXED VALUES and its domain hints carry no copyable example values.
  * extra deterministic checks (dates in Information only from the record's bundle, addresses/locations only from the
    record's assignment, emails/phones only the record's own, tokens <= 150 for BOTH customer genders).
  * a uniqueness pass over all records: any value (dates, amounts, times, minutes, plates, flights, seats, gates, names of
    restaurants/products/drivers/plans, caller instructions/reasons, ...) used by > 3 records -> the records beyond the
    first 3 are regenerated with an explicit avoid-list.
  * a Gemma self-consistency judge per record (dates ordered and consistent with the call date, amounts plausible,
    status consistent with ETA/dates, Information consistent with facts and the scenario); failures are regenerated.
Outputs: data/V4/records.json, data/V4/record_map.json, data/V4/records_stats.json, data/V4/records_log.jsonl
"""
import datetime as dt
import json
import os
import random
import re
from collections import Counter, defaultdict

from common import (AGENT_PLAIN, BRANDS, DATA, GUIDE, ID_PREFIX, PAIRINGS, TOOL_ARGS, extract_json, jdump)
import records as R1  # the V1/V3 module (n_tokens, RECORD_SCHEMA, build_role_prompt)

V4_SEED = 20261004
V4_DIR = f"{DATA}/V4"
SCEN_V2 = f"{GUIDE}/scenario_creation_v2.json"
MAX_ROUNDS = 6
INFO_MAX_TOKENS = 150
REUSE_CAP = 3

# --------------------------------------------------------------------------- scenarios (v2 file, same shape as v1)
def load_scenarios_v2(path=SCEN_V2):
    d = json.load(open(path))
    out = {}
    for a in d["agents"]:
        for s in a["scenarios"]:
            out[s["id"]] = {**s, "agent_type": a["agent_type"], "brand_context": a.get("brand_context", a["agent_type"]),
                            "static_read": a["tools"]["static_read"], "static_write_all": a["tools"]["static_write"],
                            "writes": s.get("tools") or [], "tool_args": a.get("tool_args") or {}}
    return out


# --------------------------------------------------------------------------- per-type constants for the new types
V2_AGENT_PLAIN = {**AGENT_PLAIN, "bank_card_support": "bank", "telecom_prepaid_support": "mobile network company"}
V2_BRANDS = {**BRANDS,
             "bank_card_support": [("Kosh Bank", "Ishita", "Rakesh"), ("Pragati Bank", "Nandini", "Sameer"),
                                   ("Udyam Bank", "Lavanya", "Tushar")],
             "telecom_prepaid_support": [("Tarang Mobile", "Charu", "Mayank"), ("Vaani Telecom", "Kirti", "Prateek"),
                                         ("Sampark Mobile", "Sonia", "Hitesh")]}
V2_ID_PREFIX = {**ID_PREFIX, "bank_card_support": "TX", "telecom_prepaid_support": "TL"}  # bank: ids are transaction ids
# args of the new write tools, as described to Gemma in gen/scenarios_v2.py (common.TOOL_ARGS does not have them)
NEW_TOOL_ARGS = {"block_card": ["card_last4"], "raise_transaction_dispute": ["transaction_id", "reason"],
                 "request_card_replacement": ["card_last4", "address"], "deactivate_service": ["service_name"],
                 "activate_pack": ["pack_name"], "block_sim": ["mobile_number"], "update_alternate_number": ["phone"]}
GENERIC_BRANDS = [("Sahaj Services", "Mansi", "Alok"), ("Prayas Care", "Juhi", "Kabir"), ("Nidaan Support", "Tara", "Vedant")]

# customer names: the V1 pools plus more, so every surname / first name is used by <= 3 of ~160 records
V2_LAST = ["Mehta", "Sharma", "Verma", "Gupta", "Malhotra", "Kapoor", "Arora", "Bansal", "Chopra", "Saxena",
           "Agarwal", "Bhatia", "Khanna", "Sethi", "Tandon", "Jain", "Singh", "Yadav", "Chauhan", "Rawat",
           "Mishra", "Tiwari", "Pandey", "Srivastava", "Joshi", "Nair", "Iyer", "Reddy", "Das", "Bose",
           "Ahuja", "Grover", "Kohli", "Luthra", "Mittal", "Goel", "Bhatt", "Dubey", "Kulkarni", "Desai",
           "Patil", "Shetty", "Menon", "Pillai", "Rao", "Hegde", "Kamath", "Shukla", "Tripathi", "Chaturvedi",
           "Thakur", "Rathore", "Negi", "Bisht", "Dhillon", "Gill", "Sandhu", "Bajaj", "Oberoi", "Sinha",
           "Prasad", "Banerjee", "Mukherjee", "Ghosh", "Chatterjee", "Naidu"]
V2_FIRST_M = ["Rohan", "Amit", "Saurabh", "Ankit", "Vivek", "Rajat", "Mohit", "Kunal", "Siddharth", "Pankaj",
              "Tarun", "Ashish", "Lalit", "Naveen", "Yash", "Pranav", "Dhruv", "Ritesh", "Sumit", "Manoj",
              "Hemant", "Ajay", "Vishal", "Lokesh", "Anuj", "Gautam", "Nitin", "Rakshit", "Sahil", "Kartik",
              "Ishaan", "Mayur", "Parth", "Rishabh", "Sachin", "Tejas", "Uday", "Vinay", "Jatin", "Keshav",
              "Arnav", "Devansh", "Harshit", "Kshitij", "Neeraj", "Omkar", "Puneet", "Raghav", "Shreyas", "Tanmay",
              "Ayush", "Chirag", "Darshan", "Girish", "Himanshu", "Lakshay", "Manav", "Nikunj", "Pratik"]
V2_FIRST_F = ["Riya", "Anjali", "Shreya", "Nisha", "Pallavi", "Swati", "Megha", "Komal", "Preeti", "Sakshi",
              "Jyoti", "Kritika", "Aarti", "Bhavna", "Rashmi", "Payal", "Garima", "Shalini", "Monika", "Ritika",
              "Deepika", "Sonal", "Tanya", "Neelam", "Aishwarya", "Bhumika", "Chhavi", "Diksha", "Ekta", "Falguni",
              "Gunjan", "Harshita", "Kavita", "Lata", "Madhuri", "Namrata", "Ojasvi", "Prachi", "Radhika",
              "Sanjana", "Trisha", "Urvashi", "Vandana", "Yamini", "Zoya", "Akanksha", "Charvi", "Damini", "Esha",
              "Gauri", "Ira", "Jhanvi", "Khushi", "Mitali", "Naina", "Poonam", "Rupali", "Sheetal", "Tanvee"]

# ~60 addresses (each <= 8 words): 32 Delhi/NCR, 14 Mumbai, 14 Bangalore
ADDRESSES = {
    "Delhi NCR": ["Flat 304, Tower B, Sector 50, Noida", "C-118, Malviya Nagar, New Delhi",
                  "House 27, Block D, Vasant Kunj, Delhi", "Flat 1102, Supertech Capetown, Sector 74, Noida",
                  "B-45, Greater Kailash Part 1, Delhi", "House 612, Sector 23, Gurgaon",
                  "Flat 9A, DLF Phase 4, Gurgaon", "E-21, Rajouri Garden, New Delhi",
                  "Flat 207, Shipra Suncity, Indirapuram, Ghaziabad", "House 88, Sector 16, Faridabad",
                  "A-302, Dwarka Sector 10, Delhi", "J-14, Saket, New Delhi",
                  "Flat 5C, Nirvana Country, Sector 50, Gurgaon", "H-73, Mayur Vihar Phase 1, Delhi",
                  "Flat 1504, Tower 6, Sector 137, Noida", "D-9, Hauz Khas Enclave, New Delhi",
                  "House 41, Pitampura, Delhi", "Flat 803, Sushant Lok 1, Gurgaon", "B-210, Kalkaji, New Delhi",
                  "House 15, Sector 31, Gurgaon", "Flat 402, Gaur City 2, Greater Noida West",
                  "K-34, Defence Colony, New Delhi", "Flat 12, Vaishali Sector 4, Ghaziabad", "C-7, Janakpuri, New Delhi",
                  "House 302, Sector 45, Noida", "Flat 1801, Palm Drive, Sector 66, Gurgaon",
                  "G-19, Laxmi Nagar, Delhi", "Flat 6B, Rohini Sector 9, Delhi", "A-12, Model Town, Delhi",
                  "House 7, Golf Links, New Delhi", "Flat 305, South City 2, Gurgaon", "B-88, Paschim Vihar, New Delhi"],
    "Mumbai": ["Flat 502, Hiranandani Gardens, Powai, Mumbai", "Flat 1203, Lokhandwala Complex, Andheri West, Mumbai",
               "Room 14, Shivaji Park, Dadar, Mumbai", "Flat 704, Sector 17, Vashi, Navi Mumbai",
               "Flat 9, Pali Hill, Bandra West, Mumbai", "Flat 1601, Oberoi Splendor, Jogeshwari East, Mumbai",
               "Flat 303, Thakur Village, Kandivali East, Mumbai", "Flat 21, Hill Road, Bandra, Mumbai",
               "Flat 1105, Lodha Splendora, Thane West", "Flat 6, Chembur Colony, Chembur, Mumbai",
               "Flat 408, Juhu Tara Road, Juhu, Mumbai", "Flat 2302, Raheja Vihar, Powai, Mumbai",
               "Flat 11, Colaba Causeway, Colaba, Mumbai", "Flat 607, Kharghar Sector 20, Navi Mumbai"],
    "Bangalore": ["Flat 204, Prestige Shantiniketan, Whitefield, Bangalore", "House 56, 4th Block, Koramangala, Bangalore",
                  "Flat 1102, Brigade Gateway, Rajajinagar, Bangalore", "House 23, 12th Main, Indiranagar, Bangalore",
                  "Flat 305, Sobha Dream Acres, Varthur, Bangalore", "House 9, 2nd Cross, Jayanagar, Bangalore",
                  "Flat 801, Salarpuria Sattva, HSR Layout, Bangalore", "House 140, BTM Layout 2nd Stage, Bangalore",
                  "Flat 12B, Electronic City Phase 1, Bangalore", "House 33, 8th Cross, Malleshwaram, Bangalore",
                  "Flat 606, Purva Riviera, Marathahalli, Bangalore", "House 71, JP Nagar 6th Phase, Bangalore",
                  "Flat 1504, Mantri Espana, Bellandur, Bangalore", "House 18, Basavanagudi, Bangalore"],
}
LANDMARKS = {
    "Delhi NCR": ["IGI Airport Terminal 3", "IGI Airport Terminal 1", "Cyber Hub, DLF Cyber City", "Connaught Place Inner Circle",
                  "Select Citywalk Mall, Saket", "Huda City Centre Metro Station", "New Delhi Railway Station Gate 1",
                  "Ambience Mall, Gurgaon", "DLF Mall of India, Noida", "AIIMS Main Gate, Ansari Nagar",
                  "Nehru Place Metro Station", "Hauz Khas Village", "Aerocity Worldmark", "Khan Market",
                  "Botanical Garden Metro, Noida", "Hazrat Nizamuddin Railway Station"],
    "Mumbai": ["Mumbai Airport Terminal 2", "Bandra Kurla Complex", "Phoenix Palladium, Lower Parel",
               "Chhatrapati Shivaji Terminus", "Gateway of India", "Andheri Metro Station", "Infinity Mall, Malad",
               "Powai Lake Gate", "Dadar Station West", "R City Mall, Ghatkopar", "Juhu Beach Gate 5", "Thane Station East"],
    "Bangalore": ["Kempegowda Airport Terminal 1", "MG Road Metro Station", "Manyata Tech Park Gate 2",
                  "Phoenix Marketcity, Whitefield", "Orion Mall, Rajajinagar", "Forum Mall, Koramangala",
                  "Majestic Bus Stand", "Infosys Gate 1, Electronic City", "100 Feet Road, Indiranagar",
                  "Cubbon Park Gate", "Bangalore Cantonment Station", "UB City, Vittal Mallya Road"],
}
CITY_WEIGHTS = [("Delhi NCR", 0.5), ("Mumbai", 0.25), ("Bangalore", 0.25)]

# D2 repair (2026-10-04): scenario-aware overrides for newly drawn det fields. The first run pinned FIXED values that
# contradicted some scenario flows (city named in the flow, airport/hospital pickups, office addresses, new sign-ups),
# so those records could never pass the judge. Applied only to records drawn in the repair run (see repair_records_v2).
OFFICES = {
    "Delhi NCR": ["Tower B, 6th Floor, DLF Cyber City, Gurgaon", "Floor 3, Candor TechSpace, Sector 62, Noida",
                  "Office 410, Ansal Plaza, Khel Gaon Marg, Delhi", "Floor 9, Unitech Cyber Park, Sector 39, Gurgaon",
                  "Office 214, Statesman House, Barakhamba Road, Delhi", "Floor 5, Logix Cyber Park, Sector 62, Noida"],
    "Mumbai": ["Floor 7, Delphi Building, Hiranandani, Powai, Mumbai", "Office 1204, One BKC, Bandra East, Mumbai",
               "Floor 4, Mindspace Building 9, Malad West, Mumbai", "Office 806, Peninsula Business Park, Lower Parel, Mumbai",
               "Floor 2, Nirlon Knowledge Park, Goregaon East, Mumbai", "Office 51, Kanakia Wall Street, Andheri East, Mumbai"],
    "Bangalore": ["Floor 6, Embassy Tech Village, Bellandur, Bangalore", "Block C, 3rd Floor, RMZ Ecospace, Bangalore",
                  "Floor 8, Prestige Tech Park, Marathahalli, Bangalore", "Office 302, Brigade Tech Gardens, Whitefield, Bangalore",
                  "Floor 4, Bagmane Tech Park, CV Raman Nagar, Bangalore", "Office 115, Salarpuria Hallmark, Kondapur Road, Bangalore"],
}
HOSPITALS = {"Delhi NCR": ["AIIMS Main Gate, Ansari Nagar", "Max Hospital, Saket", "Medanta Hospital, Sector 38, Gurgaon",
                           "Fortis Hospital, Sector 62, Noida"],
             "Mumbai": ["Lilavati Hospital, Bandra West", "Kokilaben Hospital, Andheri West", "Hinduja Hospital, Mahim"],
             "Bangalore": ["Manipal Hospital, Old Airport Road", "Narayana Health City, Bommasandra",
                           "St John's Hospital, Koramangala"]}
SUB_SPOT = {"airport": ["Arrival Gate 2", "Arrival Gate 4", "Arrival Gate 6", "Pillar 11", "Pillar 18", "Departure Gate 3",
                        "Pickup Zone B", "Exit 5"],
            "hospital": ["Emergency Wing Gate", "OPD Block Gate 2", "Gate 4, Discharge Lounge", "Cardiac Block Entrance",
                         "Gate 3, Maternity Wing"],
            "other": ["Gate 3", "Exit 2", "Gate 5", "East Entrance", "Parking Exit B"]}
CITY_WORDS = [("Mumbai", r"\b(mumbai|navi mumbai|thane|bandra|andheri|powai)\b"),
              ("Bangalore", r"\b(bangalore|bengaluru|whitefield|koramangala)\b"),
              ("Delhi NCR", r"\b(delhi|gurgaon|gurugram|noida|ghaziabad|faridabad)\b")]


def scenario_overrides(s):
    """Flags derived from a scenario's title + flow (lower-case keyword rules; see NOTES 'D2 / V4 records repair')."""
    t = (s["title"] + " | " + " | ".join(s["flow"])).lower()
    o = {}
    cities = [c for c, rx in CITY_WORDS if re.search(rx, t)]
    if len(cities) == 1:
        o["city"] = cities[0]
    if s["agent_type"] == "cab_ride_support":
        if re.search(r"to (the )?airport|catch (a|the|their|my) flight|miss(ing)? (a|the|their|my) flight", t):
            o["drop_kind"] = "airport"
        elif re.search(r"terminal|landed|airport gate|arrival|at the airport", t):
            o["pickup_kind"] = "airport"
        elif re.search(r"discharg|hospital wing|from (the )?hospital", t):
            o["pickup_kind"] = "hospital"
        if re.search(r"slight|different (airport )?gate|another gate|new gate|new pickup gate|different .*wing|"
                     r"other side|nearby|moved from the original|exit", t):
            o["new_pickup_near"] = True
    if s["agent_type"] in ("food_delivery_support", "ecommerce_support") and re.search(r"office|meeting room|workplace", t) \
            and re.search(r"address", t):
        o["office_address"] = True
    m = re.search(r"thought (?:it|the \w+) was (?:billed |charged )?(monthly|quarterly|annual)", t)
    if s["agent_type"] == "subscription_account_support" and m:
        o["cycle_not"] = m.group(1)
    if s["agent_type"] == "subscription_account_support" and re.search(
            r"just signed up|new user|new subscriber|first[- ]time|recently signed|signed up (yesterday|last week|recently)", t):
        o["new_signup"] = True
    return o
CITY_AIRPORT = {"Delhi NCR": "Delhi", "Mumbai": "Mumbai", "Bangalore": "Bangalore"}
PLATE_PREFIX = {"Delhi NCR": ["DL 01", "DL 3C", "DL 8C", "HR 26", "HR 55", "UP 16"], "Mumbai": ["MH 02", "MH 01", "MH 04", "MH 43"],
                "Bangalore": ["KA 01", "KA 03", "KA 05", "KA 51"]}
RESTAURANTS_CITY = {  # national chains under None, city-only places under their city
    None: ["Biryani Blues", "Domino's", "Wow! Momo", "Behrouz Biryani", "Burger Singh", "Faasos", "Chaayos", "Subway", "KFC",
           "Pizza Hut", "McDonald's", "Theobroma", "La Pino'z Pizza", "Oven Story Pizza", "The Good Bowl", "Lunchbox",
           "Rolls Mania", "Natural Ice Cream", "Paradise Biryani", "Saravana Bhavan", "Haldiram's"],
    "Delhi NCR": ["Sagar Ratna", "Moti Mahal Delux", "Bikanervala", "Karim's", "Om Sweets", "Nirula's", "Rajinder Da Dhaba",
                  "Big Chill Cafe", "Kake Di Hatti", "Pind Balluchi"],
    "Mumbai": ["Bademiya", "Cafe Madras", "Shiv Sagar", "Jumboking", "Kailash Parbat", "Britannia & Co"],
    "Bangalore": ["Meghana Foods", "Empire Restaurant", "Truffles", "MTR", "Vidyarthi Bhavan", "Nagarjuna"],
}
RESTAURANTS = [x for v in RESTAURANTS_CITY.values() for x in v]
DRIVERS = ["Ramesh", "Suresh", "Mahesh", "Dinesh", "Rajesh", "Mukesh", "Naresh", "Satish", "Pradeep", "Sanjay", "Vijay",
           "Raju", "Sunil", "Anil", "Ravi", "Manoj", "Deepak", "Ashok", "Santosh", "Gopal", "Harish", "Imran", "Salim",
           "Jagdish", "Kishore", "Prakash", "Shankar", "Venkatesh", "Murali", "Ganesh", "Balwinder", "Gurpreet", "Rafiq",
           "Iqbal", "Yogesh", "Kamal"]
CARS = ["white Maruti Dzire", "grey Hyundai Aura", "silver Toyota Etios", "black Honda Amaze", "white Tata Tigor EV",
        "blue Maruti WagonR", "red Hyundai i20", "white Toyota Innova Crysta", "silver Maruti Ertiga", "grey Tata Nexon",
        "white Hyundai Xcent", "brown Maruti Ciaz", "white Kia Carens", "silver Honda City", "blue Tata Punch",
        "white Mahindra XUV300", "grey Maruti Baleno", "white Renault Triber"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "hotmail.com", "rediffmail.com", "icloud.com",
                 "protonmail.com", "zohomail.in"]
EMAIL_WORDS = ["home", "work", "mail", "desk", "inbox", "india", "personal", "office", "family", "connect", "info", "online"]
PHONE_PREFIXES = ["98111", "99100", "97180", "88260", "70420", "81300", "95990", "96500", "90150", "73030",
                  "87500", "99580", "98730", "93120", "85270", "80760", "76780", "89200", "91490", "63970"]
AIRLINE_CODE = {"Indus Airways": "IU", "SkyIndia Airlines": "SY", "Udaan Air": "UD"}
SUB_CYCLES = [("monthly", 0.55), ("quarterly", 0.15), ("annual", 0.30)]
TELECOM_VALIDITY = [28, 56, 84]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
CALL_START = dt.date(2026, 10, 1)
CALL_DAYS = 92  # call dates spread over Oct-Dec 2026 (3 months)
BASE_YEAR = 2026


def fmt_date(d, weekday=False):
    s = f"{d.day} {MONTHS[d.month - 1]}" + (f" {d.year}" if d.year != BASE_YEAR else "")
    return (d.strftime("%A") + " " + s) if weekday else s


def dm_key(d):
    return f"{d.day} {MONTHS[d.month - 1]}"  # day-month: what the uniqueness pass sees in Information


def add_months(d, n):
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return dt.date(y, m, day)


def date_bundle(atype, call, rng, cycle=None, validity=None):
    """Ordered date facts per agent type, relative to the call date. Returns {name: date}."""
    r = rng.randint
    b = {}
    if atype == "food_delivery_support":
        b["older_order_date"] = call - dt.timedelta(r(3, 75))
    elif atype == "ecommerce_support":
        order = call - dt.timedelta(r(2, 14))
        b["order_date"] = order
        b["expected_delivery_date"] = call + dt.timedelta(r(1, 10))
        b["delivered_on"] = order + dt.timedelta(r(1, max(1, (call - order).days - 1)))
        b["older_order_date"] = order - dt.timedelta(r(15, 90))
    elif atype == "cab_ride_support":
        b["past_ride_date"] = call - dt.timedelta(r(2, 75))
    elif atype == "subscription_account_support":
        n = {"monthly": 1, "quarterly": 3, "annual": 12}[cycle]
        last = call - dt.timedelta(r(1, 26 if n == 1 else 60))
        b["last_renewal_date"] = last
        b["next_renewal_date"] = add_months(last, n)
        b["account_created_on"] = add_months(last, -n * r(2, 5)) - dt.timedelta(r(0, 9))
    elif atype == "airport_ticket_counter":
        flight = call + dt.timedelta(0 if rng.random() < 0.6 else r(1, 2))
        b["flight_date"] = flight
        b["return_flight_date"] = flight + dt.timedelta(r(3, 21))
        b["booking_date"] = call - dt.timedelta(r(6, 80))
    elif atype == "bank_card_support":
        stmt = call - dt.timedelta(r(4, 35))
        b["transaction_date"] = call - dt.timedelta(r(0, 14))
        b["other_transaction_date"] = b["transaction_date"] - dt.timedelta(r(1, 20))
        b["statement_date"] = stmt
        b["payment_due_date"] = stmt + dt.timedelta(18)
    elif atype == "telecom_prepaid_support":
        last = call - dt.timedelta(r(1, validity - 2))
        b["last_recharge_date"] = last
        b["validity_end_date"] = last + dt.timedelta(validity)
        b["previous_recharge_date"] = last - dt.timedelta(validity + r(0, 5))
    else:
        b["previous_contact_date"] = call - dt.timedelta(r(3, 40))
        b["follow_up_date"] = call + dt.timedelta(r(2, 10))
    return b


class Deck:
    """Draw from a pool so that no value is drawn more than `cap` times; random among the least-used values."""

    def __init__(self, items, rng, cap=REUSE_CAP):
        self.items, self.rng, self.cap, self.n = list(items), rng, cap, Counter()

    def draw(self, exclude=()):
        cand = [x for x in self.items if self.n[x] < self.cap and x not in exclude]
        if not cand:
            raise RuntimeError(f"deck exhausted ({len(self.items)} items x cap {self.cap})")
        lo = min(self.n[x] for x in cand)
        x = self.rng.choice([c for c in cand if self.n[c] == lo])
        self.n[x] += 1
        return x


def wchoice(rng, pairs):
    x, acc = rng.random(), 0.0
    for v, w in pairs:
        acc += w
        if x < acc:
            return v
    return pairs[-1][0]


DET_KEYS = ["record_id", "brand", "agent_name_f", "agent_name_m", "agent_type_plain", "customer_last", "customer_first_m",
            "customer_first_f", "phone", "primary_id", "secondary_id", "new_phone", "other_phone", "old_email", "new_email",
            "city", "call_date", "dates", "assigned", "misread_id"]


def det_fields_v2(scen, seed=V4_SEED, existing=None, overrides=False):
    """Deterministic, diversity-capped identity + pool fields for every scenario (in file order).
    overrides: apply scenario_overrides() to newly drawn records (D2 repair run; False = first-run behaviour).
    existing: {sid: record} already generated (e.g. data/V4/records.json); their fields are kept as they are and
    pre-loaded into the decks, so adding scenarios later (--only new ids) does not reshuffle the existing records."""
    existing = {k: v for k, v in (existing or {}).items() if k in scen}
    rng = random.Random(seed)
    last_d, fm_d, ff_d = Deck(V2_LAST, rng), Deck([x.rstrip("_") for x in V2_FIRST_M], rng), Deck([x.rstrip("_") for x in V2_FIRST_F], rng)
    addr_d = {c: Deck(v, rng) for c, v in ADDRESSES.items()}
    land_d = {c: Deck(v, rng) for c, v in LANDMARKS.items()}
    rest_d, drv_d, car_d = Deck(RESTAURANTS, rng), Deck(DRIVERS, rng), Deck(CARS, rng)
    eta_food, eta_cab = Deck([f"{m} minutes" for m in range(6, 49)], rng), Deck([f"{m} minutes" for m in range(3, 26)], rng)
    used_ids, used_phones, used_emails, used_misc = set(), set(), set(), set()
    date_n = Counter()
    per_type = Counter()
    agent_names = {n for v in V2_BRANDS.values() for b in v for n in b[1:]} | {n for b in GENERIC_BRANDS for n in b[1:]}
    out = {}
    for sid, e in existing.items():  # register existing values
        A, city = e["assigned"], e["city"]
        last_d.n[e["customer_last"]] += 1
        fm_d.n[e["customer_first_m"]] += 1
        ff_d.n[e["customer_first_f"]] += 1
        used_ids.update([e["primary_id"], e["secondary_id"]])
        used_phones.update([e["phone"], e["new_phone"], e["other_phone"]])
        used_emails.update([e["old_email"], e["new_email"]])
        for d in {DATE_RE.search(x).group(1) + " " + DATE_RE.search(x).group(2) for x in e["dates"].values()}:
            date_n[d] += 1
        for k, v in A.items():
            if k in ("current_delivery_address", "new_delivery_address", "current_address", "new_address"):
                addr_d[city].n[v] += 1
            elif k in ("pickup", "drop", "new_pickup", "new_drop"):
                land_d[city].n[v] += 1
            elif k in ("restaurant", "older_order_restaurant"):
                rest_d.n[v] += 1
            elif k == "driver_name":
                drv_d.n[v] += 1
            elif k == "car":
                car_d.n[v] += 1
            elif k == "eta_minutes":
                (eta_cab if e["agent_type"] == "cab_ride_support" else eta_food).n[v] += 1
            elif PHONE_RE.fullmatch(str(v)):
                used_phones.add(v)
            else:
                used_misc.add(v)

    def uniq(gen, used):
        for _ in range(10000):
            v = gen()
            if v not in used:
                used.add(v)
                return v
        raise RuntimeError("could not draw a unique value")

    def phone():
        return uniq(lambda: f"{rng.choice(PHONE_PREFIXES)} {rng.randint(0, 99999):05d}", used_phones)

    def email(last, avoid_dom=None):
        doms = [d for d in EMAIL_DOMAINS if d != avoid_dom]
        pats = [lambda: f"{last}{rng.randint(10, 99)}", lambda: f"{last}.{rng.choice(EMAIL_WORDS)}{rng.randint(1, 99)}",
                lambda: f"{rng.choice(EMAIL_WORDS)}.{last}{rng.randint(1, 9)}", lambda: f"{last}_{rng.randint(70, 99)}",
                lambda: f"{last}{rng.choice(EMAIL_WORDS)}{rng.randint(100, 999)}"]
        return uniq(lambda: f"{rng.choice(pats)()}@{rng.choice(doms)}", used_emails)

    for sid, s in scen.items():
        at = s["agent_type"]
        k = per_type[at]
        per_type[at] += 1
        if sid in existing:
            out[sid] = {k2: existing[sid][k2] for k2 in DET_KEYS if k2 in existing[sid]}
            continue
        brands = V2_BRANDS.get(at, GENERIC_BRANDS)
        brand, an_f, an_m = brands[k % len(brands)]
        last = last_d.draw()
        fm = fm_d.draw(exclude=agent_names)
        ff = ff_d.draw(exclude=agent_names)
        city = wchoice(rng, CITY_WEIGHTS)
        ov = scenario_overrides(s) if overrides else {}
        city = ov.get("city", city)
        cycle = wchoice(rng, SUB_CYCLES) if at == "subscription_account_support" else None
        if cycle and ov.get("cycle_not"):
            cycle = wchoice(rng, [(c, w) for c, w in SUB_CYCLES if c != ov["cycle_not"]] + [(None, 1.0)]) or "quarterly"
        validity = rng.choice(TELECOM_VALIDITY) if at == "telecom_prepaid_support" else None
        # call date + date bundle: redraw until no date in the bundle is already offered to REUSE_CAP records
        best = None
        for _ in range(20000):  # keep the bundle whose most-offered day-month is least used (0 overflow if possible)
            call = CALL_START + dt.timedelta(rng.randrange(CALL_DAYS))
            bundle = date_bundle(at, call, rng, cycle, validity)
            if ov.get("new_signup"):  # account created at the first (= last) renewal
                bundle["account_created_on"] = bundle["last_renewal_date"]
            keys = {dm_key(d) for d in bundle.values()}
            score = (max(date_n[x] for x in keys), sum(date_n[x] for x in keys))
            if best is None or score < best[0]:
                best = (score, call, bundle, keys)
            if score[0] < REUSE_CAP:
                break
        _, call, bundle, keys = best
        for d in keys:
            date_n[d] += 1
        if at == "airport_ticket_counter":
            L = "BCDFHJKLMNPQRSTVWXZ"
            pid = uniq(lambda: rng.choice(L) + rng.choice(L) + f"{rng.randint(1000, 9999)}", used_ids)
            sid2 = uniq(lambda: rng.choice(L) + rng.choice(L) + f"{rng.randint(1000, 9999)}", used_ids)
        else:
            pre = V2_ID_PREFIX.get(at, "RF")
            pid = uniq(lambda: f"{pre}{rng.randint(1000, 9999)}", used_ids)
            sid2 = uniq(lambda: f"{pre}{rng.randint(1000, 9999)}", used_ids)
        lw = last.lower()
        old_email = email(lw)
        f = {
            "record_id": f"V4-{sid}", "brand": brand, "agent_name_f": an_f, "agent_name_m": an_m,
            "agent_type_plain": V2_AGENT_PLAIN.get(at, "customer support company"),
            "customer_last": last, "customer_first_m": fm, "customer_first_f": ff,
            "phone": phone(), "primary_id": pid, "secondary_id": sid2, "new_phone": phone(), "other_phone": phone(),
            "old_email": old_email, "new_email": email(lw, avoid_dom=old_email.split("@")[1]),
            "city": city, "call_date": call.isoformat(),
            "dates": {k2: fmt_date(v, weekday=(at == "ecommerce_support" and k2 in ("expected_delivery_date", "delivered_on")))
                      for k2, v in bundle.items()},
            "assigned": {},
        }
        A = f["assigned"]
        ws = set(s["writes"])
        wargs = {a for w in ws for a in tool_args(s, w)}
        if at in ("food_delivery_support", "ecommerce_support") and ov.get("office_address"):
            offs = rng.sample(OFFICES[city], 2)
            A["current_delivery_address"] = offs[0]
            if "address" in wargs:
                A["new_delivery_address"] = offs[1]
        elif at in ("food_delivery_support", "ecommerce_support"):
            A["current_delivery_address"] = addr_d[city].draw()
            if "address" in wargs:
                A["new_delivery_address"] = addr_d[city].draw(exclude=[A["current_delivery_address"]])
        elif "address" in wargs:
            A["current_address"] = addr_d[city].draw()
            A["new_address"] = addr_d[city].draw(exclude=[A["current_address"]])
        if at == "food_delivery_support":
            other = [x for c, v in RESTAURANTS_CITY.items() if c not in (None, city) for x in v]
            A["restaurant"] = rest_d.draw(exclude=other)
            A["older_order_restaurant"] = rest_d.draw(exclude=other + [A["restaurant"]])
            A["eta_minutes"] = eta_food.draw()
        if at == "cab_ride_support":
            kind = ov.get("pickup_kind")
            if kind == "airport":
                A["pickup"] = land_d[city].draw(exclude=[x for x in LANDMARKS[city] if "Airport" not in x])
            elif kind == "hospital":
                A["pickup"] = rng.choice(HOSPITALS[city])
            else:
                A["pickup"] = land_d[city].draw()
            if ov.get("drop_kind") == "airport":
                A["drop"] = land_d[city].draw(exclude=[A["pickup"]] + [x for x in LANDMARKS[city] if "Airport" not in x])
            else:
                A["drop"] = land_d[city].draw(exclude=[A["pickup"]] + [x for x in LANDMARKS[city] if "Airport" in x and kind == "airport"])
            if any("pickup" in w for w in ws):
                if ov.get("new_pickup_near") or kind:
                    A["new_pickup"] = uniq(lambda: f"{A['pickup']}, {rng.choice(SUB_SPOT[kind or 'other'])}", used_misc)
                else:
                    A["new_pickup"] = land_d[city].draw(exclude=[A["pickup"], A["drop"]])
            if any("drop" in w for w in ws):
                A["new_drop"] = land_d[city].draw(exclude=[A["pickup"], A["drop"], A.get("new_pickup")])
            A["driver_name"] = drv_d.draw()
            A["car"] = car_d.draw()
            A["plate"] = uniq(lambda: f"{rng.choice(PLATE_PREFIX[city])} {rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}"
                                      f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')} {rng.randint(1000, 9999)}", used_misc)
            A["eta_minutes"] = eta_cab.draw()
        if at == "airport_ticket_counter":
            code = AIRLINE_CODE.get(brand, "IU")
            A["origin_city"] = CITY_AIRPORT[city]
            A["flight_number"] = uniq(lambda: f"{code} {rng.randint(101, 989)}", used_misc)
            A["return_flight_number"] = uniq(lambda: f"{code} {rng.randint(101, 989)}", used_misc)
        if at in ("subscription_account_support", "bank_card_support", "ecommerce_support", "food_delivery_support"):
            A["card_last4"] = uniq(lambda: f"{rng.randint(1000, 9999)}", used_misc)
        if cycle:
            A["billing_cycle"] = cycle
        if validity:
            A["pack_validity"] = f"{validity} days"
            A["alternate_number_on_record"] = phone()
        if sid == "air_09":
            p = pid
            f["misread_id"] = p[:2] + p[3] + p[2] + p[4:]
        if sid in ("food_08", "ecom_13") and "new_delivery_address" in A:
            na = A["new_delivery_address"]
            m = re.search(r"\d+", na)
            wrong = str(int(m.group()) + rng.choice([1, 2, 10, 100])) if m else "1"
            A["new_delivery_address_wrong"] = na[:m.start()] + wrong + na[m.end():] if m else na
        out[sid] = f
    return out


# --------------------------------------------------------------------------- Gemma prompt (no copyable example values)
TYPE_HINTS_V2 = {
    "food_delivery_support": "The order is from the FIXED restaurant: concrete menu items with quantities that this restaurant really serves, total in Rs, status (e.g. preparing / out for delivery / delivered), the FIXED ETA if not yet delivered, the FIXED current delivery address, payment method, existing delivery instruction only if the scenario needs it. Distractor: an older order (secondary ID) from the FIXED older-order restaurant on the FIXED older-order date.",
    "ecommerce_support": "Product with brand, model and variant (pick something specific and uncommon; vary categories: kitchen, fashion, books, electronics, beauty, sports, home, toys), price in Rs, status (e.g. ordered / shipped / out for delivery / delivered / return picked up), the FIXED order date and the FIXED expected delivery date (or FIXED delivered-on date if delivered), the FIXED delivery address, payment method. Distractor: an older order (secondary ID) on the FIXED older-order date.",
    "cab_ride_support": "Ride with status (e.g. driver on the way / ride in progress / completed), the FIXED ETA (only if the driver is on the way), the FIXED driver first name, car and plate, the FIXED pickup and drop, fare in Rs, payment method. For safety scenarios add: 'SOS button available in the app' and 'emergency helpline 112'. Distractor: a past ride (secondary ID) on the FIXED past-ride date.",
    "subscription_account_support": "Account ID (main reference), a plan name that fits the brand (invent a distinctive one) with the FIXED billing cycle, price in Rs per cycle, status (active/paused/cancelled), the FIXED last and next renewal dates, payment method (a bank or UPI app; cards use the FIXED card last 4 digits), registered email. Distractor: e.g. a previous plan or the account creation date.",
    "airport_ticket_counter": "PNR (main reference), the FIXED flight number, route from the FIXED origin city to a destination city, the FIXED flight date, departure time and boarding time (boarding 40-60 minutes before departure; pick unusual minutes, not :00/:30 only), terminal and gate, seat (row number + letter, window/aisle/middle), checked and cabin baggage allowance, meal preference, onward connection only if the scenario needs it, booking status, contact email (old email). Distractor: e.g. the FIXED return flight or the booking date.",
    "bank_card_support": "Card variant and network (e.g. a platinum/rewards/cashback credit or debit card), card ending in the FIXED card last 4 digits, card status (active/blocked/hotlisted), credit limit and available limit in Rs (available <= limit), the FIXED statement date and payment due date, total due and minimum due in Rs (minimum due about 5% of total), recent transactions each with its transaction ID, merchant, amount in Rs and date: transaction <main reference ID> on the FIXED transaction date is the one this scenario is about; transaction <second reference> on the FIXED other transaction date is the distractor. Registered email. Never an OTP, CVV, PIN, expiry or full card number.",
    "telecom_prepaid_support": "Prepaid number = the customer's phone on record (already given; do not repeat it); current pack (a distinctive pack name, price in Rs, the FIXED pack validity, data per day, calls/SMS), the FIXED last recharge date and validity end date, main balance and remaining data today, active value-added services / caller tune with their monthly charge if the scenario involves them, available packs with prices if the scenario involves activating one, the charge/recharge reference = main reference ID, SIM status (active/suspended), alternate number on record only if the scenario needs it, registered email. Distractor: e.g. the previous recharge on the FIXED previous recharge date.",
}
GENERIC_HINT = "Every static-read fact this scenario needs, concrete and specific, consistent with the FIXED dates."

V2_RECORD_SCHEMA = R1.RECORD_SCHEMA


def tool_args(s, w):
    return s.get("tool_args", {}).get(w) or TOOL_ARGS.get(w) or NEW_TOOL_ARGS.get(w) or ["value"]


def fixed_values_text(f):
    lines = [f"- main reference ID: {f['primary_id']}   (second/distractor reference: {f['secondary_id']})",
             f"- customer phone on record: {f['phone']}",
             f"- old email on record: {f['old_email']}",
             f"- if the caller gives a new own phone number: {f['new_phone']}; if the caller gives someone else's number (family member): {f['other_phone']}",
             f"- if the caller gives a new email: {f['new_email']}",
             f"- city: {f['city']}"]
    for k, v in f["dates"].items():
        lines.append(f"- {k.replace('_', ' ')}: {v}")
    for k, v in f["assigned"].items():
        lines.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(lines)


INFO_WORDS = {"bank_card_support": 60, "telecom_prepaid_support": 60, "airport_ticket_counter": 65}  # repair run only


def record_prompt_v2(sid, s, f, avoid=(), short=False):
    writes = s["writes"]
    maxw = INFO_WORDS.get(s["agent_type"], 85) if short else 85
    listcap = (" Any list (services, packs, transactions, items) has at most 2 entries; drop distractors before needed facts."
               if short else "")
    wtxt = "; ".join(f"{w}({', '.join(tool_args(s, w))})" for w in writes) or "none"
    extra = ""
    if sid == "air_09":
        extra = f"\n- The passenger first misreads the PNR as {f['misread_id']} then corrects it to {f['primary_id']}: put misread_pnr in caller_values."
    if "new_delivery_address_wrong" in f["assigned"]:
        extra = (f"\n- The caller first says a wrong flat/house number and corrects it: caller_values must have new_address_wrong = "
                 f"{f['assigned']['new_delivery_address_wrong']} and new_address = {f['assigned']['new_delivery_address']}.")
    av = ""
    if avoid:
        av = ("\nAlready used by other records, so do NOT use any of these values (pick clearly different ones): "
              + "; ".join(sorted(avoid)) + "\n")
    hint = TYPE_HINTS_V2.get(s["agent_type"], GENERIC_HINT)
    return f"""You are creating ONE realistic customer record for a synthetic customer-support call dataset (India).

Company: {f['brand']} ({s['brand_context']}; agent type {s['agent_type']}).
Today (the day of the call): {fmt_today(f) + " (dates written without a year are in " + f['call_date'][:4] + ")" if short else fmt_date(dt.date.fromisoformat(f['call_date']), weekday=True)}.
Scenario {sid}: {s['title']}
Flow:
""" + "\n".join(f"- {x}" for x in s["flow"]) + f"""
Static-read tools this company has (facts the agent may read out): {', '.join(s['static_read'])}
Write actions in this scenario: {wtxt}

FIXED VALUES (use exactly as written whenever you need that kind of value; never invent other IDs, phone numbers, emails,
dates, addresses, pickup/drop places, restaurants, driver names, cars, plates or flight numbers):
{fixed_values_text(f)}
Use ONLY the fixed dates above (no other calendar dates), and only those this scenario needs. Today's date itself is not
a fact to include. Customer name is added separately; do NOT mention the customer's name anywhere.
{av}
Domain guidance: {hint}
Everything you invent (items, products, amounts, times, plan names, seat/gate, bank names, instructions, reasons) must be
specific, realistic for India, and DIFFERENT from the usual defaults: avoid round or stock values; vary them.

Produce JSON:
- facts: every static-read fact this scenario's flow could need the agent to say (key/value; values short, concrete; times like 7:25 PM, money like Rs 1,349, dates exactly as fixed). Include the main reference ID. If the flow says the agent tells an existing instruction, existing address, registered email etc., include it.
- distractors: 1-2 extra facts about this customer that the call does NOT need.
- information: ONE prose paragraph containing ALL facts and the distractors, compact comma-separated style ("Order <ID> from <restaurant>: <items>, Rs <total>, status <status>, ..."). Max {maxw} words.{listcap} No customer name, no phone (added separately). Do not mention tools, checking or instructions to the agent. Money as 'Rs 540' style (never the rupee symbol). Every fact must be internally consistent: dates in order, status consistent with ETA and dates, totals plausible for the items/plan/fare.
- caller_values: the exact values the CALLER will dictate in this call for the write actions (new address / pickup / drop = the FIXED new value; new_phone, family_phone, new_email from FIXED VALUES; instruction as a SHORT specific note of at most 8 words; reason in at most 8 words; affected items). Empty list if the scenario has no write and the caller dictates nothing. Never put these in information unless the flow says they are already on record.{extra}
- caller_situation: one short sentence of the caller's situation (why they call, their mood), consistent with the facts.
Return only the JSON."""


# --------------------------------------------------------------------------- deterministic checks
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b")
DATE_RE = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r")(?: (\d{4}))?\b")
PHONE_RE = re.compile(r"\b\d{5} \d{5}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
ALL_POOL_PLACES = {x for v in ADDRESSES.values() for x in v} | {x for v in LANDMARKS.values() for x in v}
OLD_DEFAULTS = ["Lajpat Nagar II", "Sector 15, Gurgaon", "12 October", "@yahoo.com", "98000 "]


def info_tokens(f, info):
    """Max SPM tokens of the Information slot (incl. 'Customer: <name>, phone ...') over both customer genders."""
    return max(R1.n_tokens(R1.build_role_prompt(f, info, "f", cg).split("Information:", 1)[1]) for cg in ("m", "f"))


def norm(v):
    return re.sub(r"\s+", " ", re.sub(r"[^\w@.:/ +-]", " ", str(v).lower())).strip(" .,")


CYCLE_WORDS = {"monthly": r"\bmonthly\b", "quarterly": r"\bquarterly\b", "annual": r"\b(annual|annually|yearly)\b"}


CV_REF_KEY = re.compile(r"ref|transaction|txn", re.I)
CV_ID_VAL = re.compile(r"^[A-Z]{2,4}\d{3,8}$")


def cv_ref_errors(cv, info):
    """D2 repair pass 5: a reference / transaction ID the caller quotes must be on record in Information
    (repair pass 1 dropped air_18's 'Duplicate Payment Ref WS5041' while shortening)."""
    return [f"caller value {k} = {v} must also be in information (the agent must find it on record)"
            for k, v in cv.items() if CV_REF_KEY.search(k) and CV_ID_VAL.match(str(v).strip()) and str(v).strip() not in info]


def fut_year_ok(d, allowed):
    """'21 January' written for the fixed '21 January 2027' (a later year) is natural speech: accepted."""
    return any(a.startswith(d + " ") and int(a.split()[-1]) > BASE_YEAR for a in allowed)


def categorical_errors(f, facts, info):
    """D2 repair: the assigned billing cycle / pack validity must be the one used (first run: sub_10 assigned monthly,
    Gemma wrote quarterly). facts = {key: value}."""
    errs = []
    A = f["assigned"]
    cyc = A.get("billing_cycle")
    if cyc:
        for other, rx in CYCLE_WORDS.items():
            if other != cyc and any(re.search(rx, str(v), re.I) for k, v in facts.items() if re.search("cycle|billing", k, re.I)):
                errs.append(f"billing cycle must be {cyc} (FIXED), not {other}")
        if not re.search(CYCLE_WORDS[cyc], info, re.I):
            errs.append(f"information must state the FIXED billing cycle ({cyc})")
    val = A.get("pack_validity")
    if val:
        n = val.split()[0]
        for k, v in facts.items():
            if re.search("validity", k, re.I) and not re.search("end|date|till|until|expir", k, re.I) and \
                    re.search(r"\d+ ?days?", str(v)) and n not in str(v):
                errs.append(f"pack validity must be {val} (FIXED), not {v}")
    return errs


def check_record_v2(sid, s, f, r):
    errs = []
    info = r["information"]
    if f["primary_id"] not in info:
        errs.append(f"information must contain the main reference ID {f['primary_id']}")
    if "₹" in info:
        errs.append("use 'Rs', not the rupee symbol")
    for n in (f["customer_first_m"], f["customer_first_f"], f["customer_last"]):
        if re.search(r"\b" + re.escape(n) + r"\b", info):
            errs.append("do not mention the customer's name in information")
            break
    for n in (f["agent_name_f"], f["agent_name_m"]):
        if re.search(r"\b" + n + r"\b", info):
            errs.append(f"the name {n} is reserved for the support agent; use a different name")
    nt = info_tokens(f, info)
    if nt > INFO_MAX_TOKENS:
        errs.append(f"information too long ({nt} tokens incl. customer line; max {INFO_MAX_TOKENS}): shorten")
    for i in re.findall(r"\b[A-Z]{2}\d{4}\b", info):
        if i not in (f["primary_id"], f["secondary_id"], f.get("misread_id")):
            errs.append(f"unexpected ID {i}; only use {f['primary_id']} / {f['secondary_id']}")
    # dates: every calendar date must come from the record's fixed bundle; no abbreviated/other month mentions
    allowed = {re.sub(r"^[A-Z][a-z]+day ", "", d) for d in f["dates"].values()}
    allowed_dm = {" ".join(d.split()[:2]) for d in allowed}
    cv_text = " ".join(c["value"] for c in r["caller_values"])
    for txt in (info, cv_text):
        spans = []
        for m in DATE_RE.finditer(txt):
            spans.append(m.span())
            d = f"{m.group(1)} {m.group(2)}"
            full = d + (f" {m.group(3)}" if m.group(3) and m.group(3) != str(BASE_YEAR) else "")
            if d in allowed_dm and full not in allowed and not (full == d and fut_year_ok(d, allowed)):  # repair: '19 August 2025' passed in run 1
                errs.append(f"date '{m.group(0)}' has the wrong year; write it exactly as fixed ({'; '.join(sorted(allowed))})")
            if d not in allowed_dm:
                errs.append(f"date '{m.group(0)}' is not one of the fixed dates ({'; '.join(sorted(allowed))})")
        for m in MONTH_RE.finditer(txt):
            if not any(a <= m.start() < b for a, b in spans):
                errs.append(f"month '{m.group(0)}' used outside a fixed date; write dates exactly as fixed (e.g. '{sorted(allowed)[0]}')")
                break
    # phones / emails: only the record's own
    phones = {f["phone"], f["new_phone"], f["other_phone"]} | {v for v in f["assigned"].values() if PHONE_RE.fullmatch(v)}
    for txt in (info, cv_text):
        for p in PHONE_RE.findall(txt):
            if p not in phones:
                errs.append(f"phone {p} is not a fixed value")
        for e in EMAIL_RE.findall(txt):
            if e.rstrip(".") not in (f["old_email"], f["new_email"]):
                errs.append(f"email {e} is not a fixed value")
    # places: pool places must be the ones assigned to this record
    mine = {v for v in f["assigned"].values()}
    for p in ALL_POOL_PLACES:
        if p in info and not any(p in m for m in mine):  # repair: 'X, Arrival Gate 4' assigned -> X allowed
            errs.append(f"'{p}' is not this record's address/place; use the FIXED values")
    for bad in OLD_DEFAULTS:
        if bad in info and not any(bad in v for v in list(mine) + list(f["dates"].values())):
            errs.append(f"'{bad.strip()}' is an over-used default; do not use it")
    # caller values: places must be the assigned new values; word limits as V1
    cv = {c["key"]: c["value"] for c in r["caller_values"]}
    newvals = {norm(v) for k, v in f["assigned"].items() if k.startswith("new_")}
    for k, v in cv.items():
        lim = 8 if any(x in k for x in ("instruction", "reason", "address", "location", "pickup", "drop")) else 99
        if len(v.split()) > lim + 1:
            errs.append(f"caller value {k} has {len(v.split())} words; max {lim}")
        if any(x in k for x in ("address", "location", "pickup", "drop")) and newvals and norm(v) not in newvals:
            errs.append(f"caller value {k} must be exactly one of the FIXED new values: {'; '.join(v2 for k2, v2 in f['assigned'].items() if k2.startswith('new_'))}")
    errs += categorical_errors(f, {c["key"]: c["value"] for c in r["facts"]}, info)
    errs += cv_ref_errors({c["key"]: c["value"] for c in r["caller_values"]}, info)
    errs += dup_id_errors(info)  # D2 review: bank_20 listed one transaction ID twice
    for k in ("product", "older_order_product", "card_variant", "merchant", "other_merchant"):  # D2 review: pooled (family=True)
        if k in f["assigned"] and f["assigned"][k].split(",")[0].lower() not in info.lower():
            errs.append(f"use the FIXED {k.replace('_', ' ')} '{f['assigned'][k]}' exactly as written in information")
    dict_args = {"address", "location", "instruction", "phone", "email", "reason"}
    if s["writes"] and not cv and any(set(tool_args(s, w)) & dict_args for w in s["writes"]):
        errs.append("caller_values empty but scenario has writes with values")
    return list(dict.fromkeys(errs)), nt


# --------------------------------------------------------------------------- uniqueness pass
INCLUDE_KEY = re.compile(r"restaurant|item|product|driver|car|vehicle|plate|model|plan|merchant|hotel|flight|seat|gate|"
                         r"address|pickup|drop|location|landmark|instruction|reason|bank|destination|connection", re.I)
EXEMPT_VAL = re.compile(r"^\s*(terminal \d\w?|\d+ ?kg.*|\d+(\.\d+)? ?gb.*|\d+ days|monthly|annual|quarterly|yes|no|none|"
                        r"veg(etarian)?|non-veg(etarian)?|window|aisle|middle|confirmed|active|cash|upi|credit card|debit card)\s*$", re.I)
ENTITY_RES = [("date", DATE_RE), ("amount", re.compile(r"\bRs\.? ?[\d,]+(?:\.\d+)?")),
              ("time", re.compile(r"\b\d{1,2}:\d{2} ?(?:AM|PM|am|pm)?")), ("minutes", re.compile(r"\b\d+ minutes?\b")),
              ("plate", re.compile(r"\b[A-Z]{2} \d{1,2}[A-Z]? [A-Z]{1,2} \d{4}\b")),
              ("flight", re.compile(r"\b[A-Z]{2} \d{3,4}\b")), ("id", re.compile(r"\b[A-Z]{2}\d{4}\b")),
              ("card_last4", re.compile(r"ending (?:in )?(\d{4})")), ("email", EMAIL_RE), ("phone", PHONE_RE),
              ("seat", re.compile(r"\bseat (\d{1,2}[A-F])\b", re.I)), ("gate", re.compile(r"\bGate \w+", re.I))]
CATEGORICAL_ASSIGNED = {"billing_cycle", "origin_city", "pack_validity"}
EXEMPT_KEY = re.compile(r"status|type|pref|class|method|mode|cycle|meal|baggage|allowance|terminal|validity|benefit", re.I)

# --------------------------------------------------------------------------- D2 review fixes (reviewer, 2026-10-04)
# Everything below is used only with family=True (records.py --v2 --repair --family); family=False = behaviour before.
# (1) family reuse: exact-value matching let near-duplicates through ('Agaro Imperial 1200W Hair Dryer, Professional
#     Black' / '..., Matte Black' / 'Agaro Regal ...': 'Agaro' in 16/23 ecom records). Families = word bigrams of
#     non-generic words inside one Information segment + the first non-generic word of a capitalised segment.
FAMILY_EXEMPT = set("""rs order orders older previous past ride account pnr flight sim status registered email payment paid via
card cards credit debit bank ending last last4 upi wallet net banking cash delivery delivered shipped ordered expected out
preparing driver way confirmed active blocked hotlisted inactive suspended terminal gate seat window aisle middle meal veg
non non-veg hindu jain vegetarian preference baggage checked cabin boarding departure arrival return booking plan monthly
annual quarterly yearly cycle renewal next created signed balance data day validity pack packs services service alerts
unlimited calls sms main ref reference secondary distractor distractors transaction transactions txn statement due total
minimum limit available reward points rewards end date the and with from for via new delhi ncr mumbai bangalore bengaluru
noida gurgaon gurugram ghaziabad city address flat house floor tower sector phase block road nagar colony eta minutes fare
app sos button emergency helpline premium edition variant professional matte black white blue grey silver red green pay paytm phonepe gpay
gold rose set piece pieces combo january february march april june july august september october november december
monday tuesday wednesday thursday friday saturday sunday india support customer contact number alternate phone hdfc icici
axis kotak sbi indusind idfc first federal rbl canara pnb hsbc standard chartered dbs citi bandhan union idbi""".split())


def family_items(rec):
    info = rec["information"]
    for v in sorted(rec.get("assigned", {}).values(), key=lambda x: len(str(x)), reverse=True):
        if len(str(v)) > 6:
            info = info.replace(str(v), " | ")
    for v in rec.get("dates", {}).values():
        info = info.replace(v, " | ")
    brand = set(rec.get("brand", "").lower().split())
    out = set()
    for seg in re.split(r"[,;:.()|/]| - ", info):
        toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'&!-]+", seg)]
        toks = [t for t in toks if t not in FAMILY_EXEMPT and len(t) > 2 and t not in brand]
        for a, b in zip(toks, toks[1:]):
            out.add(f"{a} {b}")
        if toks and re.match(r"[A-Z]", seg.strip() or "x"):
            out.add(toks[0])
    # review pass 8: the first non-generic word of plan/pack/merchant/product/variant/transaction fact values
    # ('SuperSaver 84' / 'SuperSaver 4G' in 7/20 tel records, 'Amazon India' in 6 bank records)
    for k, v in rec.get("facts", {}).items():
        if FAMILY_FACT_KEY.search(k) and not EXEMPT_KEY.search(k):
            for part in re.split(r"[;,()]", str(v)):
                toks = [t.lower() for t in re.findall(r"\b[A-Za-z][A-Za-z'&!-]+", part)]
                toks = [t for t in toks if t not in FAMILY_EXEMPT and len(t) > 2 and t not in brand and t != "days"]
                if toks:
                    out.add(toks[0])
    return out


FAMILY_FACT_KEY = re.compile(r"plan|pack|merchant|product|variant|transaction", re.I)


# (2) a fixed product pool for ecom records redone with family=True (Gemma collapses to Agaro/Philips/Prestige).
ECOM_PRODUCTS = [
    "Borosil Vision Glass Lunch Box Set", "Wonderchef Nutri-Blend Mixer", "Milton Thermosteel Flask 1L",
    "Pigeon Stovekraft Pressure Cooker 5L", "Butterfly Rapid Electric Kettle 1.5L", "Hawkins Contura Pressure Cooker 3L",
    "Cello Opalware Dinner Set 27 pieces", "Fabindia Cotton Kurta, size L", "Bata Comfit Sandals, size 8",
    "Levi's 511 Slim Jeans, waist 32", "Woodland Leather Boots, size 9", "Biba Anarkali Kurta Set, size M",
    "Allen Solly Formal Shirt, size 40", "Puma Velocity Running Shoes, size 7", "Wings of Fire hardcover",
    "Atomic Habits paperback", "NCERT Physics Class 12 book set", "Ponniyin Selvan box set", "JBL Go 3 Bluetooth Speaker",
    "Logitech MK270 Keyboard and Mouse Combo", "Realme Narzo 70 Pro 8GB/128GB", "Sony WH-CH520 Headphones",
    "SanDisk Ultra 128GB Pen Drive", "Canon Pixma G3010 Printer", "Lenovo Tab M10 Wi-Fi", "Noise ColorFit Pro 5 Smartwatch",
    "Ambrane 20000mAh Power Bank", "Mamaearth Vitamin C Face Wash 100ml", "Lakme 9to5 Primer Lipstick, shade Red Coat",
    "Biotique Bio Kelp Shampoo 650ml", "Forest Essentials Facial Ubtan 50g", "Nivia Storm Football, size 5",
    "Cosco CB-88 Badminton Racket pair", "SG Kashmir Willow Cricket Bat", "Yonex Mavis 350 Shuttlecocks, tube of 6",
    "Strauss Adjustable Dumbbells 10kg", "Sleepwell Ortho Pillow, pack of 2", "Bajaj Majesty Room Heater",
    "Usha Maxx Air Pedestal Fan", "Wipro 9W Smart LED Bulb, pack of 2", "Kent Grand RO Water Purifier",
    "Godrej Interio Study Table", "Funskool Monopoly Classic", "Lego Classic Creative Bricks 484",
    "Hot Wheels 10-Car Gift Pack", "Smartivity Robotic Arm Kit", "Nerf Elite 2.0 Blaster", "Skybags Brat Backpack 46L",
    "American Tourister Ivy Trolley 68cm", "Fastrack Reflex Beat Smartwatch", "Titan Neo Analog Watch",
    "Havells Instanio Water Heater 3L", "Crompton Hill Briz Ceiling Fan", "Classmate Pulse Notebooks, pack of 6",
]


def ecom_product_assign(todo, det, scen, kept, seed=V4_SEED):
    """family=True: give each redone ecom record a FIXED product and older-order product from ECOM_PRODUCTS (each used once;
    products whose first word already appears in a kept record are skipped)."""
    rng = random.Random(seed + 606)
    kept_words = {w for r in kept.values() if r["agent_type"] == "ecommerce_support"
                  for w in re.findall(r"[a-z][a-z'-]+", r["information"].lower())}
    pool = [p for p in ECOM_PRODUCTS if p.split()[0].lower() not in kept_words]
    rng.shuffle(pool)
    n = 0
    for sid in todo:
        if scen[sid]["agent_type"] != "ecommerce_support" or "product" in det[sid]["assigned"]:
            continue
        if len(pool) < 2:
            raise RuntimeError("ECOM_PRODUCTS exhausted")
        det[sid]["assigned"]["product"], det[sid]["assigned"]["older_order_product"] = pool.pop(), pool.pop()
        n += 1
    return n


# bank records redone with family=True: Gemma's card variants / merchants collapse to Titanium / Millennia / Regalia /
# Sapphire / Infinite and Zomato / Amazon / BigBasket (repair pass 6 churned 8 rounds on them) -> FIXED pooled values.
BANK_CARD_VARIANTS = ["Aura Cashback Credit Card", "Voyager Travel Credit Card", "Fuel Plus Credit Card",
                      "Everyday Saver Credit Card", "Zenith Signature Credit Card", "Nova Lifetime Free Credit Card",
                      "Elevate Business Credit Card", "Prime Moneyback Credit Card", "Orbit Travel Credit Card",
                      "Smart Saver Debit Card", "Gold Plus Debit Card", "Classic Savings Debit Card"]
BANK_MERCHANTS = ["Swiggy", "Zepto", "Myntra", "Nykaa", "Croma", "Vijay Sales", "Decathlon", "MakeMyTrip", "IRCTC",
                  "BookMyShow", "Uber", "Indian Oil petrol pump", "DMart", "Lifestyle", "Shoppers Stop", "Ajio", "Lenskart",
                  "Urban Company", "PharmEasy", "Haldiram's", "Cafe Coffee Day", "PVR Cinemas", "Westside", "Pantaloons"]


def bank_assign(todo, det, scen, kept, seed=V4_SEED):
    rng = random.Random(seed + 808)
    txt = " ".join(r["information"] for r in kept.values() if r["agent_type"] == "bank_card_support")
    variants = [v for v in BANK_CARD_VARIANTS if v.split()[0] not in txt]
    merchants = [m for m in BANK_MERCHANTS if m.split()[0] not in txt]
    rng.shuffle(variants)
    rng.shuffle(merchants)
    n = 0
    for sid in todo:
        s = scen[sid]
        if s["agent_type"] != "bank_card_support" or "card_variant" in det[sid]["assigned"]:
            continue
        debit = re.search(r"\bdebit\b", (s["title"] + " " + " ".join(s["flow"])).lower())
        v = next(x for x in variants if ("Debit" in x) == bool(debit))
        variants.remove(v)
        A = det[sid]["assigned"]
        A["card_variant"], A["merchant"], A["other_merchant"] = v, merchants.pop(), merchants.pop()
        if re.search(r"duplicate|charged twice|double", (s["title"] + " " + " ".join(s["flow"])).lower()):
            # duplicate charge: the second reference is the original charge, same merchant, same day
            A["other_merchant"] = A["merchant"]
            if "transaction_date" in det[sid]["dates"] and "other_transaction_date" in det[sid]["dates"]:
                det[sid]["dates"]["other_transaction_date"] = det[sid]["dates"]["transaction_date"]
        n += 1
    return n


# (3) Information clean-up (deterministic, after the repair): prompt-scaffold words that leaked into Information
#     ('Distractor:', 'secondary reference AC..', 'main reference ID TL..', 'last4 1234') and one payment bank in
#     38/69 food/ecom/sub records ('HDFC'), re-spread over a pool with the same <= 3 records cap.
SECOND_REF_WORD = {"subscription_account_support": "old account", "food_delivery_support": "older order",
                   "ecommerce_support": "older order", "cab_ride_support": "past ride", "telecom_prepaid_support": "previous ref",
                   "bank_card_support": "other transaction", "airport_ticket_counter": "other booking"}
PAY_BANKS = ["HDFC", "ICICI", "Axis", "Kotak", "SBI", "IndusInd", "IDFC First", "Federal", "RBL", "Canara", "PNB", "HSBC",
             "Standard Chartered", "DBS", "Citi", "Bandhan", "AU", "Union", "IDBI"]
PAY_BANK_RE = re.compile(r"\b(" + "|".join(re.escape(b) for b in PAY_BANKS) + r")\b")
LEAK_RE = re.compile(r"\b(distractors?|secondary (reference|ref|id)|main reference|last ?4\b|fixed value)", re.I)


def clean_text(txt, atype):
    txt = re.sub(r"\bDistractors?:\s*(?:an?\s+)?(?:(?:older|previous)\s+)?order\b", "Older order", txt, flags=re.I)
    txt = re.sub(r"\bDistractors?:\s*", "", txt)
    txt = re.sub(r"\bsecondary (?:reference|ref|ID)\s+(?=[A-Z]{2}\d{4})", SECOND_REF_WORD.get(atype, "other ref") + " ", txt,
                 flags=re.I)
    txt = re.sub(r"\bmain reference(?: ID)?\s+(?=[A-Z]{2}\d{4})", "ref ", txt, flags=re.I)
    txt = re.sub(r"\s*\(\s*last ?4(?: digits)?\s*:?\s*(\d{4})\s*\)", r" ending \1", txt, flags=re.I)
    txt = re.sub(r"\blast ?4(?: digits)?\s*:?\s*(\d{4})", r"ending \1", txt, flags=re.I)
    txt = re.sub(r"\bending ending\b", "ending", txt)
    return txt


def _map_record_text(r, fn):
    r["information"] = fn(r["information"])
    r["facts"] = {k: fn(v) if isinstance(v, str) else v for k, v in r["facts"].items()}
    r["distractors"] = [fn(x) if isinstance(x, str) else x for x in r["distractors"]]
    r["caller_values"] = {k: fn(v) if isinstance(v, str) else v for k, v in r["caller_values"].items()}
    r["caller_situation"] = fn(r["caller_situation"])


def normalise_records(allrec, order, seed=V4_SEED):
    """-> (allrec, log). Leak clean-up on every record, then payment banks capped at REUSE_CAP records (file order;
    non-bank types). A record whose Information would exceed INFO_MAX_TOKENS keeps its old text (logged)."""
    import copy
    rng = random.Random(seed + 707)
    log = {"leak_fixed": [], "bank_swapped": {}, "reverted_tokens": []}
    used = Counter()
    for sid in order:
        r = allrec[sid]
        new = copy.deepcopy(r)
        _map_record_text(new, lambda t: clean_text(t, r["agent_type"]))
        if new["information"] != r["information"]:
            log["leak_fixed"].append(sid)
        if r["agent_type"] != "bank_card_support":
            banks = list(dict.fromkeys(PAY_BANK_RE.findall(new["information"])))
            if banks:
                b = banks[0]
                if used[b] >= REUSE_CAP:
                    cand = [x for x in PAY_BANKS if used[x] < REUSE_CAP and x not in banks and
                            not any(x in str(v) for v in allrec.values() if False)]
                    if cand:
                        lo = min(used[x] for x in cand)
                        nb = rng.choice([x for x in cand if used[x] == lo])
                        _map_record_text(new, lambda t, b=b, nb=nb: re.sub(r"\b" + re.escape(b) + r"\b", nb, t))
                        log["bank_swapped"][sid] = [b, nb]
                        b = nb
                used[b] += 1
        if new["information"] != r["information"]:
            nt = info_tokens(new, new["information"])
            if nt > INFO_MAX_TOKENS:
                log["reverted_tokens"].append(sid)
                log["bank_swapped"].pop(sid, None)
                continue
            new["information_tokens_spm"] = nt
            new["role_prompts"] = {g: R1.build_role_prompt(new, new["information"], p["agent_gender"], p["customer_gender"])
                                   for g, p in PAIRINGS.items()}
            allrec[sid] = new
    log["bank_use"] = dict(used)
    return allrec, log


def dup_id_errors(info):
    return [f"ID {i} appears {n} times in information; a second transaction/order needs the second reference ID"
            for i, n in Counter(re.findall(r"\b[A-Z]{2}\d{4}\b", info)).items() if n > 1]


def value_items(rec, family=False):
    """(type, normalized value) pairs that count towards the reuse cap for one record."""
    out = {("family", x) for x in family_items(rec)} if family else set()
    txt = rec["information"] + " | " + " | ".join(rec["caller_values"].values())
    # assigned pool values count once as whole values (below); strip them so e.g. 'Infosys Gate 1, Electronic City'
    # does not also count as gate 'gate 1' (D2 repair pass 3)
    for v in sorted(rec.get("assigned", {}).values(), key=len, reverse=True):
        if len(str(v)) > 6 and not PHONE_RE.fullmatch(str(v)):
            txt = txt.replace(str(v), " ")
    for typ, rx in ENTITY_RES:
        for m in rx.finditer(txt):
            v = {"date": lambda: f"{m.group(1)} {m.group(2)}", "seat": lambda: m.group(1),
                 "card_last4": lambda: m.group(1)}.get(typ, lambda: m.group(0))()
            out.add((typ, norm(v)))
        if typ == "plate":  # plates contain 'XX 1234' runs that would read as flight numbers
            txt = rx.sub(" ", txt)
    for k, v in list(rec["facts"].items()) + list(rec["caller_values"].items()):
        if INCLUDE_KEY.search(k) and not EXEMPT_KEY.search(k) and not EXEMPT_VAL.match(str(v)) and len(str(v)) > 2:
            out.add((re.sub(r"[^a-z]+", "_", k.lower()).strip("_"), norm(v)))
    for k, v in rec.get("assigned", {}).items():
        if k not in CATEGORICAL_ASSIGNED:
            out.add((k, norm(v)))
    return out


def reuse_violations(allrec, order, family=False):
    """Return {sid: [values to avoid]} for records beyond the first REUSE_CAP users of any value (by `order`)."""
    users = defaultdict(list)
    for sid in order:
        if sid in allrec:
            for t, v in value_items(allrec[sid], family):
                users[v].append((sid, t))
    viol = defaultdict(list)
    for v, us in users.items():
        sids = list(dict.fromkeys(s for s, _ in us))
        for sid in sids[REUSE_CAP:]:
            viol[sid].append(v)
    return dict(viol), users


# --------------------------------------------------------------------------- Gemma self-consistency judge
JUDGE_SCHEMA = {"type": "object", "properties": {"consistent": {"type": "boolean"},
                                                   "problems": {"type": "array", "items": {"type": "string"}}},
                "required": ["consistent", "problems"]}


def parse_fixed_date(txt):
    m = DATE_RE.search(txt)
    return dt.date(int(m.group(3) or BASE_YEAR), MONTHS.index(m.group(2)) + 1, int(m.group(1)))


def fmt_today(f):
    d = dt.date.fromisoformat(f["call_date"])
    return d.strftime("%A") + f" {d.day} {MONTHS[d.month - 1]} {d.year}"


def date_facts_text(f):
    """D2 repair: the FIXED dates with their offsets from today and the relations the code built them with, so the judge
    does not redo calendar arithmetic (first run: Gemma flagged correct '23 November + 84 days = 15 February 2027' etc.)."""
    call = dt.date.fromisoformat(f["call_date"])
    D = {k: parse_fixed_date(v) for k, v in f["dates"].items()}
    lines = []
    for k, d in D.items():
        n = (d - call).days
        rel = "today" if n == 0 else (f"{-n} days before today" if n < 0 else f"{n} days after today")
        lines.append(f"  {k.replace('_', ' ')} = {f['dates'][k]} [{d.isoformat()}] ({rel})")
    A = f["assigned"]
    if "last_renewal_date" in D:
        lines.append(f"  next renewal = last renewal + one {A.get('billing_cycle', '')} cycle (correct by construction)")
        if D.get("account_created_on") == D["last_renewal_date"]:
            lines.append("  account created on the first renewal date (new subscriber)")
    if "validity_end_date" in D:
        lines.append(f"  validity end = last recharge + {(D['validity_end_date'] - D['last_recharge_date']).days} days "
                     f"(= the pack validity {A.get('pack_validity')}, correct by construction)")
    if "payment_due_date" in D:
        lines.append(f"  payment due = statement date + {(D['payment_due_date'] - D['statement_date']).days} days")
    return "\n".join(lines)


def judge_prompt(sid, s, f, rec):
    return f"""You check ONE synthetic customer record for internal consistency. Be strict about real contradictions and
implausible values, but do not nitpick style or wording.

Company: {f['brand']} ({s['agent_type']}). Today (day of the call): {fmt_date(dt.date.fromisoformat(f['call_date']), weekday=True)}.
Scenario: {s['title']}
Flow: {' / '.join(s['flow'])}
Facts: {json.dumps(rec['facts'], ensure_ascii=False)}
Information (the agent's prompt): {rec['information']}
Caller will dictate: {json.dumps(rec['caller_values'], ensure_ascii=False)}
Caller situation: {rec['caller_situation']}

Check:
1. Dates: chronological and consistent with today (order date before delivery; a delivered date is not after today; an
   expected delivery is after today unless delivered; last renewal before next renewal with a gap matching the billing
   cycle; booking before flight; statement before due date; recharge before validity end with the stated validity;
   past/older items are before today). A weekday written with a date must match.
2. Status vs ETA/dates: an ETA only for something not yet delivered/arrived/completed; "delivered"/"completed" has no ETA;
   status matches the dates (e.g. "out for delivery" today, "shipped" before the expected delivery date).
3. Amounts plausible for India (item prices vs total, fare vs distance, plan price vs cycle, minimum due vs total due,
   available limit <= credit limit, boarding before departure by a sensible margin).
4. Information agrees with the facts (no value differs) and contains what the scenario's flow needs the agent to say;
   the record makes the scenario possible (e.g. a cancellable order is not already delivered unless the flow says so;
   the caller's new value differs from the current one).
Return JSON: consistent (true/false) and problems (each a short concrete sentence naming the wrong value; empty if consistent)."""


def judge_prompt_repair(sid, s, f, rec):
    """D2 repair judge: today WITH year + code-computed date facts (run-1 judge assumed 2024/2025 and redid date arithmetic)."""
    return f"""You check ONE synthetic customer record for internal consistency. Be strict about real contradictions and
implausible values, but do not nitpick style or wording.

Company: {f['brand']} ({s['agent_type']}). Today (day of the call): {fmt_today(f)} (the current year is {f['call_date'][:4]}).
Scenario: {s['title']}
Flow: {' / '.join(s['flow'])}
Facts: {json.dumps(rec['facts'], ensure_ascii=False)}
Information (the agent's prompt): {rec['information']}
Caller will dictate: {json.dumps(rec['caller_values'], ensure_ascii=False)}
Caller situation: {rec['caller_situation']}
FIXED dates (computed by code; their calendar arithmetic, gaps, weekdays and year roll-over are CORRECT - do not re-check
or recompute them, and do not flag a gap of up to 2-3 weeks as implausible):
{date_facts_text(f)}

Check:
1. Dates: each FIXED date is used in its own role (e.g. the order date is not called the delivery date), and the status
   wording agrees with the offsets above (something dated after today is not described as already happened, and the
   reverse). Do not flag correct FIXED dates.
2. Status vs ETA/dates: an ETA only for something not yet delivered/arrived/completed; "delivered"/"completed" has no ETA;
   status matches the dates (e.g. "out for delivery" today, "shipped" before the expected delivery date).
3. Amounts plausible for India (item prices vs total, fare vs distance, plan price vs cycle, minimum due vs total due,
   available limit <= credit limit, boarding before departure by a sensible margin).
4. Information agrees with the facts (no value differs) and contains what the scenario's flow needs the agent to say;
   the record makes the scenario possible (e.g. a cancellable order is not already delivered unless the flow says so;
   the caller's new value differs from the current one).
Return JSON: consistent (true/false) and problems (each a short concrete sentence naming the wrong value; empty if consistent)."""


# --------------------------------------------------------------------------- main loop
def _record(sid, s, f, r, nt, errs):
    rec = {"scenario_id": sid, "agent_type": s["agent_type"], **f,
           "facts": {c["key"]: c["value"] for c in r["facts"]} if r else {},
           "distractors": r["distractors"] if r else [],
           "caller_values": {c["key"]: c["value"] for c in r["caller_values"]} if r else {},
           "caller_situation": r["caller_situation"] if r else "",
           "information": r["information"] if r else "",
           "information_tokens_spm": nt, "record_errors": errs}
    rec["role_prompts"] = {g: R1.build_role_prompt(f, rec["information"], p["agent_gender"], p["customer_gender"])
                           for g, p in PAIRINGS.items()}
    return rec


def compress_prompt(sid, s, f, cand):
    """D2 repair: a candidate that failed ONLY on length is shortened instead of regenerated from scratch."""
    r, nt = cand
    words = len(r["information"].split())
    target = max(30, int(words * (INFO_MAX_TOKENS - 14) / nt))
    return f"""Shorten the 'information' of this customer record for a support-call dataset. It is {nt} tokens; the limit is
{INFO_MAX_TOKENS} including a customer-name line, so information must be at most {target} words.

Scenario {sid}: {s['title']}
Flow:
""" + "\n".join(f"- {x}" for x in s["flow"]) + f"""

Record JSON:
{json.dumps(r, ensure_ascii=False)}

Rules: keep every value the flow needs EXACTLY as written (IDs, dates, amounts, names, places). Shorten by: dropping
distractors (keep at most one), keeping at most 2 entries in any list, removing labels/words that are not needed, and
removing facts the flow never uses. Remove dropped facts from 'facts' and 'distractors' too, so facts == information.
Do not add anything new. caller_values and caller_situation stay exactly the same. Return the full JSON in the same shape."""


def make_records_v2(llm, scen_path=SCEN_V2, out_dir=V4_DIR, only=None, seed=V4_SEED, max_rounds=MAX_ROUNDS,
                    keep_det=(), repair=False, family=False):
    """repair=True (D2 repair run, used with only=<records to redo>): records not in `only` are kept and count first in
    the uniqueness pass; values already used by REUSE_CAP kept records go into every redo prompt's avoid-list; det fields
    of `only` records are kept when listed in keep_det, else redrawn with scenario_overrides; shorter Information budget
    for air/bank/telecom; the uniqueness pass also runs in the last round. repair=False = first-run behaviour."""
    scen = load_scenarios_v2(scen_path)
    prev = json.load(open(f"{out_dir}/records.json")) if (only is not None and os.path.exists(f"{out_dir}/records.json")) else {}
    det = det_fields_v2(scen, seed, existing={k: v for k, v in prev.items() if k not in only or k in set(keep_det)} if prev else None,
                        overrides=repair)
    order = list(scen)
    kept = {k: v for k, v in prev.items() if only is not None and k not in only} if repair else {}
    kept_order = [k for k in order if k in kept]
    os.makedirs(out_dir, exist_ok=True)
    logp = f"{out_dir}/records_log.jsonl"
    todo = [sid for sid in order if only is None or sid in only]
    if family:  # D2 review: pooled FIXED products for redone ecom records
        print(f"[records_v2] family: pooled ecom products for {ecom_product_assign(todo, det, scen, kept, seed)} records, "
              f"bank card/merchants for {bank_assign(todo, det, scen, kept, seed)}", flush=True)
    feedback = {sid: "" for sid in todo}
    avoid = defaultdict(set)
    if kept:
        _, kusers = reuse_violations(kept, kept_order, family)
        at_cap = {v for v, us in kusers.items() if len({x for x, _ in us}) >= REUSE_CAP}
        for sid in todo:
            avoid[sid] |= at_cap
        print(f"[records_v2] repair: {len(kept)} kept records, {len(todo)} to redo, {len(at_cap)} values at cap -> avoid-list", flush=True)
    accepted, last_bad, acc_round = {}, {}, {}
    shorten = {}  # repair: sid -> (raw candidate, tokens) whose ONLY error is length -> next round is a compress prompt
    ev = Counter()
    judge_failed, judge_fixed = set(), set()
    uniq_regen = set()
    for rnd in range(max_rounds):
        if not todo:
            break
        prompts = [compress_prompt(sid, scen[sid], det[sid], shorten[sid]) if (repair and sid in shorten) else
                   record_prompt_v2(sid, scen[sid], det[sid], avoid[sid], short=repair) +
                   (f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {feedback[sid]}" if feedback[sid] else "") for sid in todo]
        ev["compress_prompts"] += sum(1 for sid in todo if repair and sid in shorten)
        shorten = {}
        outs = llm.chat(prompts, schema=V2_RECORD_SCHEMA, temperature=0.7, max_tokens=2500 if repair else 1500,
                        seeds=[seed + 1000 * (rnd + 1) + order.index(sid) for sid in todo])
        cand, nxt = {}, []
        for sid, (txt, fin, _) in zip(todo, outs):
            try:
                r = extract_json(txt)
                errs, nt = check_record_v2(sid, scen[sid], det[sid], r)
            except Exception as e:  # noqa
                errs, nt, r = [f"bad JSON ({fin}): {e}"], 0, None
            if r and sid == "air_09":
                r["caller_values"] = [c for c in r["caller_values"] if "misread" not in c["key"]] + \
                    [{"key": "misread_pnr", "value": det[sid]["misread_id"]}]
            rec = _record(sid, scen[sid], det[sid], r, nt, errs) if r else None
            if errs:
                ev["det_fail"] += 1
                if repair and r and nt > INFO_MAX_TOKENS and all(e.startswith("information too long") for e in errs):
                    shorten[sid] = (r, nt)
                if repair and nt > INFO_MAX_TOKENS:
                    errs.append(f"cut at least {nt - INFO_MAX_TOKENS + 8} tokens: drop the distractors first, keep at most 2 "
                                f"entries in any list, use fewer words per fact")
                feedback[sid] = "; ".join(errs)
                if rec:
                    last_bad[sid] = rec
                nxt.append(sid)
                _log(logp, rnd, sid, "det_fail", errs)
            else:
                cand[sid] = rec
        # self-consistency judge on the deterministic passes
        if cand:
            ids = list(cand)
            jp = judge_prompt_repair if repair else judge_prompt
            jouts = llm.chat([jp(sid, scen[sid], det[sid], cand[sid]) for sid in ids], schema=JUDGE_SCHEMA,
                             temperature=0.0, max_tokens=800)
            for sid, (txt, fin, _) in zip(ids, jouts):
                try:
                    j = extract_json(txt)
                    probs = [p for p in j["problems"] if p.strip()]
                    ok = bool(j["consistent"]) and not probs  # 'consistent' with listed problems counts as a fail
                except Exception as e:  # noqa
                    ok, probs = False, [f"judge output unparsable ({fin}): {e}"]
                ev["judge_calls"] += 1
                if ok:
                    if sid in judge_failed:
                        judge_fixed.add(sid)
                    cand[sid]["consistency"] = {"ok": True, "round": rnd}
                    accepted[sid] = cand[sid]
                    acc_round[sid] = rnd
                    _log(logp, rnd, sid, "accepted", [])
                else:
                    ev["judge_fail"] += 1
                    judge_failed.add(sid)
                    feedback[sid] = "Self-consistency problems: " + "; ".join(probs)
                    cand[sid]["consistency"] = {"ok": False, "problems": probs, "round": rnd}
                    last_bad[sid] = cand[sid]
                    accepted.pop(sid, None)
                    nxt.append(sid)
                    _log(logp, rnd, sid, "judge_fail", probs)
        # uniqueness pass over everything accepted so far (records earlier in file order keep their values)
        # earlier-accepted records keep their values; ties in file order (avoids churning records accepted in round 0)
        viol, _ = reuse_violations({**kept, **accepted},
                                   kept_order + sorted(accepted, key=lambda x: (acc_round[x], order.index(x))), family)
        for sid, vals in viol.items():
            if sid in kept:  # cannot happen: kept records come first in the order
                continue
            if rnd == max_rounds - 1 and not repair:
                break
            ev["uniq_regen"] += 1
            uniq_regen.add(sid)
            avoid[sid] |= set(vals)
            feedback[sid] = "values already used by 3 other records: " + "; ".join(sorted(vals))
            last_bad[sid] = accepted.pop(sid)
            nxt.append(sid)
            _log(logp, rnd, sid, "uniq_fail", sorted(vals))
        todo = sorted(set(nxt), key=order.index)
        print(f"[records_v2] round {rnd}: accepted {len(accepted)}, retry {len(todo)} {dict(ev)}", flush=True)
    # residual: keep the last candidate with its errors
    for sid in todo:
        rec = last_bad.get(sid)
        if rec is None:
            rec = _record(sid, scen[sid], det[sid], None, 0, ["no valid candidate"])
        if not rec["record_errors"] and not rec.get("consistency", {}).get("ok", False):
            rec["record_errors"] = ["self-consistency: " + "; ".join(rec.get("consistency", {}).get("problems", []))]
        accepted[sid] = rec
    allrec = {}
    outp = f"{out_dir}/records.json"
    if only is not None and os.path.exists(outp):
        allrec = json.load(open(outp))
    allrec.update(accepted)
    allrec = {sid: allrec[sid] for sid in order if sid in allrec}
    if repair:  # final global reuse check over ALL records incl. residuals (kept first, then redone in file order)
        fviol, _ = reuse_violations(allrec, kept_order + [x for x in order if x in allrec and x not in kept], family)
        for sid, vals in fviol.items():
            msg = "over reuse cap (> 3 records): " + "; ".join(sorted(vals))
            allrec[sid]["record_errors"] = [e for e in allrec[sid]["record_errors"] if not e.startswith("over reuse cap")] + [msg]
    ids = [r["primary_id"] for r in allrec.values()] + [r["secondary_id"] for r in allrec.values()]
    assert len(ids) == len(set(ids)), "duplicate IDs across records"
    jdump(outp, allrec)
    stats = diversity_stats(allrec, order)
    stats["self_consistency"] = {"judge_calls": ev["judge_calls"], "judge_fail_events": ev["judge_fail"],
                                 "records_failed_judge_at_least_once": len(judge_failed),
                                 "of_those_fixed_by_regeneration": len(judge_fixed),
                                 "deterministic_fail_events": ev["det_fail"],
                                 "uniqueness_regenerations": ev["uniq_regen"], "compress_prompts": ev["compress_prompts"], "records_regenerated_for_uniqueness": len(uniq_regen),
                                 "residual_records_with_errors": [s for s, r in allrec.items() if r["record_errors"]]}
    import hashlib
    stats["scenario_file"] = {"path": scen_path, "md5": hashlib.md5(open(scen_path, "rb").read()).hexdigest(),
                              "n_scenarios": len(scen), "n_records": len(allrec)}
    jdump(f"{out_dir}/records_stats.json", stats)
    print(json.dumps(stats, indent=1, ensure_ascii=False)[:6000], flush=True)
    return allrec


def _log(path, rnd, sid, ev, items):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"round": rnd, "sid": sid, "event": ev, "items": items}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- stats + record map
def diversity_stats(allrec, order=None):
    order = order or list(allrec)
    _, users = reuse_violations(allrec, order)
    by_type = defaultdict(Counter)
    for v, us in users.items():
        for t in {t for _, t in us}:
            by_type[t][v] = len({s for s, tt in us if tt == t})
    maxreuse = {t: {"max": c.most_common(1)[0][1], "value": c.most_common(1)[0][0], "distinct": len(c)}
                for t, c in sorted(by_type.items())}
    over = {v: len(set(s for s, _ in us)) for v, us in users.items() if len(set(s for s, _ in us)) > REUSE_CAP}
    calls = sorted(r["call_date"] for r in allrec.values())
    alld = []
    for r in allrec.values():
        for m in DATE_RE.finditer(r["information"]):
            alld.append(f"{m.group(1)} {m.group(2)}")
    months = Counter(r["call_date"][:7] for r in allrec.values())
    dom = Counter(e.split("@")[1] for r in allrec.values() for e in (r["old_email"], r["new_email"]))
    pre = Counter(p.split()[0] for r in allrec.values() for p in (r["phone"], r["new_phone"], r["other_phone"]))
    toks = [r["information_tokens_spm"] for r in allrec.values()]
    defaults = {d: sum(d in r["information"] for r in allrec.values()) for d in OLD_DEFAULTS}
    return {"n_records": len(allrec), "max_reuse_per_value_type": maxreuse,
            "values_over_cap": dict(sorted(over.items(), key=lambda x: -x[1])),
            "call_date_range": [calls[0], calls[-1]] if calls else None, "call_dates_per_month": dict(sorted(months.items())),
            "info_dates_distinct": len(set(alld)), "info_date_mentions": len(alld),
            "info_date_max_reuse": Counter(alld).most_common(3),
            "email_domains": dict(dom.most_common()), "phone_prefixes": dict(pre.most_common()),
            "cities": dict(Counter(r["city"] for r in allrec.values())),
            "surname_max_reuse": Counter(r["customer_last"] for r in allrec.values()).most_common(1),
            "info_tokens_spm": {"max": max(toks) if toks else 0, "mean": round(sum(toks) / max(1, len(toks)), 1)},
            "old_default_hits": defaults}


def write_record_map(allrec, old_path=f"{DATA}/records.json", out_dir=V4_DIR):
    old = json.load(open(old_path))
    m = {"note": "V4 records (data/V4/records.json) replace data/records.json for V4 only. ALL old role prompts change: "
                 "every V4 record has new names/phones/emails/IDs/dates/places, so no old call_id's prompt is reused.",
         "scenarios": {}, "calls": {}}
    for sid, o in old.items():
        n = allrec.get(sid)
        m["scenarios"][sid] = {"new_record_id": n["record_id"] if n else None,
                               "old_primary_id": o["primary_id"], "new_primary_id": n["primary_id"] if n else None,
                               "prompt_changed": (n is None) or any(n["role_prompts"][g] != o["role_prompts"][g] for g in PAIRINGS)}
        for g in PAIRINGS:
            m["calls"][f"{sid}_{g}"] = {"new_record_id": n["record_id"] if n else None,
                                        "new_call_id": f"{sid}_{g}" if n else None,
                                        "prompt_changed": (n is None) or n["role_prompts"][g] != o["role_prompts"][g]}
    m["new_scenarios"] = [sid for sid in allrec if sid not in old]
    jdump(f"{out_dir}/record_map.json", m)
    print(f"[records_v2] record_map: {len(m['scenarios'])} old scenarios, {len(m['calls'])} old call_ids, "
          f"{sum(v['prompt_changed'] for v in m['calls'].values())} prompts changed, {len(m['new_scenarios'])} new scenarios",
          flush=True)
    return m


# --------------------------------------------------------------------------- D2 repair run (2026-10-04)
def override_conflicts(s, rec):
    """Ways an existing record's det fields contradict scenario_overrides(s) (-> redraw det fields)."""
    o, A, out = scenario_overrides(s), rec["assigned"], []
    if o.get("city") and o["city"] != rec["city"]:
        out.append(f"city {rec['city']} but flow names {o['city']}")
    if o.get("drop_kind") == "airport" and "Airport" not in A.get("drop", ""):
        out.append("to-airport scenario, non-airport drop")
    if o.get("pickup_kind") == "airport" and "Airport" not in A.get("pickup", ""):
        out.append("airport scenario, non-airport pickup")
    if o.get("pickup_kind") == "hospital" and not any(h in A.get("pickup", "") for v in HOSPITALS.values() for h in v):
        out.append("hospital scenario, non-hospital pickup")
    if (o.get("new_pickup_near") or o.get("pickup_kind")) and "new_pickup" in A and not A["new_pickup"].startswith(A["pickup"]):
        out.append("new pickup should be near the old one")
    if o.get("office_address") and not any(A.get("current_delivery_address") in v for v in OFFICES.values()):
        out.append("office scenario, home delivery address")
    if o.get("cycle_not") and A.get("billing_cycle") == o["cycle_not"]:
        out.append(f"caller wrongly thought {o['cycle_not']}; the real cycle must differ")
    if o.get("new_signup") and rec["dates"].get("account_created_on") != rec["dates"].get("last_renewal_date"):
        out.append("new sign-up but account created long before")
    return out


def plan_repair(scen, recs, family=False):
    """-> (redo ids in file order, keep_det ids, reasons {sid: [..]})."""
    order = list(scen)
    reasons = defaultdict(list)
    redraw = set()
    for sid in order:
        r = recs.get(sid)
        if r is None:
            reasons[sid].append("missing")
            redraw.add(sid)
            continue
        if r["record_errors"]:
            reasons[sid].append("residual error")
        c = categorical_errors(r, r["facts"], r["information"]) + cv_ref_errors(r["caller_values"], r["information"])
        if family:
            c += dup_id_errors(r["information"])
        allowed = {re.sub(r"^[A-Z][a-z]+day ", "", d) for d in r["dates"].values()}
        for m in DATE_RE.finditer(r["information"] + " " + " ".join(map(str, r["caller_values"].values()))):
            d = f"{m.group(1)} {m.group(2)}"
            full = d + (f" {m.group(3)}" if m.group(3) and m.group(3) != str(BASE_YEAR) else "")
            if full not in allowed and not (full == d and fut_year_ok(d, allowed)):
                c.append(f"date '{m.group(0)}' not exactly a fixed date")
        if c:
            reasons[sid] += c
        oc = override_conflicts(scen[sid], r)
        if oc:
            reasons[sid] += oc
            redraw.add(sid)
    clean = [s for s in order if s in recs and s not in reasons]
    viol, _ = reuse_violations(recs, clean + [s for s in order if s in recs and s not in clean], family)
    for sid, vals in viol.items():
        reasons[sid].append("over cap: " + "; ".join(sorted(vals)))
    redo = [s for s in order if s in reasons]
    keep_det = [s for s in redo if s not in redraw]
    return redo, keep_det, dict(reasons)


def repair_records_v2(llm, scen_path=SCEN_V2, out_dir=V4_DIR, max_rounds=8, dry=False, family=False):
    import shutil
    scen = load_scenarios_v2(scen_path)
    recs = json.load(open(f"{out_dir}/records.json"))
    redo, keep_det, reasons = plan_repair(scen, recs, family)
    print(f"[records_v2] repair plan: redo {len(redo)} (redraw det {len(redo) - len(keep_det)})", flush=True)
    for sid in redo:
        print(f"  {sid}: {' | '.join(reasons[sid])[:300]}", flush=True)
    if dry:
        return redo, keep_det, reasons
    for fn in ("records.json", "records_stats.json", "record_map.json", "records_log.jsonl"):
        src, dst = f"{out_dir}/{fn}", f"{out_dir}/run1_{fn}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    k = 1
    while os.path.exists(f"{out_dir}/repair_plan_{k}.json"):
        k += 1
    jdump(f"{out_dir}/repair_plan_{k}.json", {"redo": redo, "keep_det": keep_det, "reasons": reasons})
    old_stats = json.load(open(f"{out_dir}/run1_records_stats.json"))
    prev_stats = json.load(open(f"{out_dir}/records_stats.json"))
    prev_runs = prev_stats.get("self_consistency_repair_runs", [])
    allrec = make_records_v2(llm, scen_path=scen_path, out_dir=out_dir, only=redo, max_rounds=max_rounds,
                             keep_det=keep_det, repair=True, family=family)
    st = json.load(open(f"{out_dir}/records_stats.json"))
    if family:  # D2 review: deterministic Information clean-up + payment-bank spread over ALL records
        allrec, nlog = normalise_records(allrec, list(scen))
        jdump(f"{out_dir}/records.json", allrec)
        st.update({k: v for k, v in diversity_stats(allrec, list(scen)).items()})
        fv, _ = reuse_violations(allrec, list(scen), family=True)
        st["family_over_cap_after_normalise"] = {s: sorted(v) for s, v in fv.items()}
        st["normalise_log"] = nlog
        print(f"[records_v2] normalise: leak fixed {len(nlog['leak_fixed'])}, bank swapped {len(nlog['bank_swapped'])}, "
              f"reverted {nlog['reverted_tokens']}, family over cap after: {len(fv)}", flush=True)
    st["self_consistency_run1"] = old_stats.get("self_consistency")
    this = st.pop("self_consistency")
    this["redo"], this["redrawn_det"] = len(redo), len(redo) - len(keep_det)
    st["self_consistency_repair_runs"] = prev_runs + [this]
    jdump(f"{out_dir}/records_stats.json", st)
    return allrec
