"""S2S serverless: HMAC session token shared by the HF Space (mints) and the RunPod worker (verifies).
DESIGN.md section 4. Stdlib only, so it imports from every venv (venv-pp, venv-asr, venv-needle) and from the Space.

  token = "v1." + b64url(payload_json) + "." + b64url(HMAC_SHA256(key, "v1." + b64url(payload_json)))
  payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)  (UTF-8); b64url without '=' padding
  key = S2S_SESSION_SECRET (UTF-8, >= 32 chars). Generate: python -c "import secrets;print(secrets.token_urlsafe(48))"

API (frozen, DESIGN 4.3):
  mint(secret, *, sid, mode, aud, record_id, pairing, seed, max_s, ttl_s=120, now=None) -> str
  verify(secret, token, *, mode, aud, now=None, leeway_s=10) -> dict     raises TokenError(code)
        code in {"malformed", "bad_sig", "expired", "not_yet", "mode", "aud"}
  new_sid() -> str

Replay protection is NOT here: the worker keeps a per-worker replay set (DESIGN 3.1).
Self-check:  python s2s_token.py   (checks common/test_vectors.json)
"""
import base64
import hashlib
import hmac
import json
import secrets
import time

VERSION = "v1"
MIN_SECRET_LEN = 32
FIELDS = ("aud", "exp", "iat", "max_s", "mode", "pairing", "record_id", "seed", "sid")
MODES = ("lb", "queue")


class TokenError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


def _b64e(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s):
    if not isinstance(s, str) or not s or any(c not in _B64CHARS for c in s):
        raise TokenError("malformed", "bad base64")
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except Exception as e:
        raise TokenError("malformed", f"bad base64: {e}") from None


_B64CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def _key(secret):
    if not isinstance(secret, str) or len(secret) < MIN_SECRET_LEN:
        raise ValueError(f"S2S_SESSION_SECRET must be a string of >= {MIN_SECRET_LEN} chars")
    return secret.encode("utf-8")


def _sign(key, signing_input):
    return _b64e(hmac.new(key, signing_input.encode("ascii"), hashlib.sha256).digest())


def new_sid():
    """22-char urlsafe random id (secrets.token_urlsafe(16)), unique per call."""
    return secrets.token_urlsafe(16)


def encode_payload(payload):
    """The exact signed payload bytes (sorted keys, compact separators, UTF-8)."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def mint(secret, *, sid, mode, aud, record_id, pairing, seed, max_s, ttl_s=120, now=None):
    key = _key(secret)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not sid or not isinstance(sid, str):
        raise ValueError("sid required")
    iat = int(time.time() if now is None else now)
    payload = {"sid": sid, "iat": iat, "exp": iat + int(ttl_s), "mode": mode, "aud": str(aud),
               "record_id": record_id, "pairing": pairing, "seed": None if seed is None else int(seed),
               "max_s": int(max_s)}
    signing_input = VERSION + "." + _b64e(encode_payload(payload))
    return signing_input + "." + _sign(key, signing_input)


def verify(secret, token, *, mode, aud, now=None, leeway_s=10):
    """Check format, signature, time window, mode and audience. Returns the payload dict."""
    key = _key(secret)
    if not isinstance(token, str) or len(token) > 4096:
        raise TokenError("malformed", "not a string or too long")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != VERSION:
        raise TokenError("malformed", "expected v1.<payload>.<sig>")
    signing_input = parts[0] + "." + parts[1]
    if not hmac.compare_digest(_sign(key, signing_input).encode("ascii"), parts[2].encode("ascii", "replace")):
        raise TokenError("bad_sig")
    try:
        payload = json.loads(_b64d(parts[1]).decode("utf-8"))
    except TokenError:
        raise
    except Exception as e:
        raise TokenError("malformed", f"payload: {e}") from None
    if not isinstance(payload, dict) or any(f not in payload for f in FIELDS):
        raise TokenError("malformed", "missing fields")
    if not isinstance(payload["iat"], int) or not isinstance(payload["exp"], int):
        raise TokenError("malformed", "iat/exp not int")
    t = time.time() if now is None else now
    if t > payload["exp"] + leeway_s:
        raise TokenError("expired")
    if t < payload["iat"] - leeway_s:
        raise TokenError("not_yet")
    if payload["mode"] != mode:
        raise TokenError("mode", f"token mode {payload['mode']!r} != {mode!r}")
    if str(payload["aud"]) != str(aud):
        raise TokenError("aud")
    return payload


def _selfcheck(path=None):
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_vectors.json")
    with open(path) as f:
        tv = json.load(f)
    p = tv["payload"]
    tok = mint(tv["secret"], sid=p["sid"], mode=p["mode"], aud=p["aud"], record_id=p["record_id"],
               pairing=p["pairing"], seed=p["seed"], max_s=p["max_s"], ttl_s=p["exp"] - p["iat"], now=p["iat"])
    assert tok == tv["token"], (tok, tv["token"])
    assert verify(tv["secret"], tok, mode=p["mode"], aud=p["aud"], now=p["iat"] + 1) == p
    for neg in tv["negative"]:
        try:
            verify(tv["secret"], neg["token"], mode=neg["mode"], aud=p["aud"], now=neg["now"])
        except TokenError as e:
            assert e.code == neg["code"], (neg["name"], e.code)
        else:
            raise AssertionError(f"negative vector {neg['name']} verified")
    print("s2s_token selfcheck OK:", path)


if __name__ == "__main__":
    _selfcheck()
