"""S2S serverless: token tests (DESIGN 4.5 + 8). Stdlib only; any python3.

  python3 /root/deploy_serverless/tests/test_token.py

Checks: the frozen vector, the negative vectors, expiry/leeway, not_yet, aud, malformed inputs, tamper of every
payload field, secret length guard, new_sid shape. Replay is a worker concern (DESIGN 3.1); here we only check that
two mints for the same sid verify independently (the worker's replay set must do the refusing).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import s2s_token as T  # noqa: E402

TV = json.load(open(os.path.join(HERE, "..", "common", "test_vectors.json")))
SEC, P, TOK = TV["secret"], TV["payload"], TV["token"]
FAILS = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {info}" if info and not cond else ""))
    if not cond:
        FAILS.append(name)


def code_of(fn):
    try:
        fn()
    except T.TokenError as e:
        return e.code
    except ValueError:
        return "ValueError"
    return None


def main():
    tok = T.mint(SEC, sid=P["sid"], mode=P["mode"], aud=P["aud"], record_id=P["record_id"], pairing=P["pairing"],
                 seed=P["seed"], max_s=P["max_s"], ttl_s=120, now=P["iat"])
    check("vector token matches frozen string", tok == TOK, tok)
    check("payload json matches frozen", T.encode_payload(P).decode() == TV["payload_json"])
    check("vector verifies", T.verify(SEC, TOK, mode="lb", aud="ep-test", now=P["iat"] + 5) == P)
    for neg in TV["negative"]:
        c = code_of(lambda: T.verify(SEC, neg["token"], mode=neg["mode"], aud=P["aud"], now=neg["now"]))
        check(f"negative {neg['name']} -> {neg['code']}", c == neg["code"], c)
    # expiry and leeway (default 10 s)
    check("exp + 10 still ok (leeway)", code_of(lambda: T.verify(SEC, TOK, mode="lb", aud="ep-test", now=P["exp"] + 10)) is None)
    check("exp + 11 expired", code_of(lambda: T.verify(SEC, TOK, mode="lb", aud="ep-test", now=P["exp"] + 11)) == "expired")
    check("iat - 11 not_yet", code_of(lambda: T.verify(SEC, TOK, mode="lb", aud="ep-test", now=P["iat"] - 11)) == "not_yet")
    check("leeway 0", code_of(lambda: T.verify(SEC, TOK, mode="lb", aud="ep-test", now=P["exp"] + 1, leeway_s=0)) == "expired")
    check("aud mismatch", code_of(lambda: T.verify(SEC, TOK, mode="lb", aud="ep-other", now=P["iat"])) == "aud")
    check("other secret -> bad_sig",
          code_of(lambda: T.verify(SEC + "x", TOK, mode="lb", aud="ep-test", now=P["iat"])) == "bad_sig")
    # malformed
    for name, bad in [("empty", ""), ("two parts", "v1.abc"), ("v2 prefix", "v2" + TOK[2:]), ("None", None),
                      ("four parts", TOK + ".x"), ("huge", "v1." + "A" * 5000 + ".x")]:
        c = code_of(lambda: T.verify(SEC, bad, mode="lb", aud="ep-test", now=P["iat"]))
        check(f"malformed: {name}", c == "malformed", c)
    # tamper each payload field, re-encode, keep the old signature -> bad_sig
    h, b, s = TOK.split(".")
    for k in P:
        q = dict(P)
        q[k] = (q[k] + 1) if isinstance(q[k], int) else (str(q[k]) + "x")
        forged = h + "." + T._b64e(T.encode_payload(q)) + "." + s
        c = code_of(lambda: T.verify(SEC, forged, mode="lb", aud="ep-test", now=P["iat"]))
        check(f"tamper {k} -> bad_sig", c == "bad_sig", c)
    # signature valid but payload missing a field -> malformed (attacker cannot do this without the key; guards bugs)
    q = dict(P)
    del q["max_s"]
    si = "v1." + T._b64e(T.encode_payload(q))
    c = code_of(lambda: T.verify(SEC, si + "." + T._sign(SEC.encode(), si), mode="lb", aud="ep-test", now=P["iat"]))
    check("validly signed but missing field -> malformed", c == "malformed", c)
    # secret guard
    check("short secret refused at mint", code_of(lambda: T.mint("short", sid="x", mode="lb", aud="a", record_id=None,
                                                                 pairing=None, seed=None, max_s=300)) == "ValueError")
    check("bad mode refused at mint", code_of(lambda: T.mint(SEC, sid="x", mode="pod", aud="a", record_id=None,
                                                            pairing=None, seed=None, max_s=300)) == "ValueError")
    # nulls allowed for record_id / pairing / seed
    t2 = T.mint(SEC, sid="s", mode="queue", aud="ep", record_id=None, pairing=None, seed=None, max_s=300)
    p2 = T.verify(SEC, t2, mode="queue", aud="ep")
    check("null cfg fields roundtrip", p2["record_id"] is None and p2["seed"] is None and p2["exp"] - p2["iat"] == 120)
    # same sid minted twice: both verify (replay refusal is the worker's job)
    a = T.mint(SEC, sid="dup", mode="lb", aud="ep", record_id="r", pairing="g1", seed=1, max_s=300, now=100)
    b2 = T.mint(SEC, sid="dup", mode="lb", aud="ep", record_id="r", pairing="g1", seed=1, max_s=300, now=101)
    check("two tokens per sid both verify", T.verify(SEC, a, mode="lb", aud="ep", now=105)["sid"] ==
          T.verify(SEC, b2, mode="lb", aud="ep", now=105)["sid"] == "dup" and a != b2)
    sids = {T.new_sid() for _ in range(1000)}
    check("new_sid: 22 urlsafe chars, unique", len(sids) == 1000 and all(len(x) == 22 and set(x) <= T._B64CHARS for x in sids))
    print(f"\n{'ALL OK' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
