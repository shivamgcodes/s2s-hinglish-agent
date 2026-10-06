"""S2S serverless worker: copy of DEP1 router/router_core.py (path edits: needle dir and weights from paths.py) +
D-ROUTER-V2 (2026-10-06): Needle v2 is the default router; S2S_ROUTER=n1 (or router="n1") is the old N1 path, unchanged.
DEP1 Track 3: router pipeline, transport-agnostic (used by router_service.py live and offline_test.py offline).

Needle v2 (needle_v2/REPORT.md section 3, the 5 deploy changes):
  1. router = needle_v2/schema/router_v2.py; supported = agent_type in tools_v2.AGENT_TYPES
  2. Needle gets asr["text_raw"] (raw Trelis text; router_v2 applies numconv + romanise itself), not the romanised text
  3. guard = call["ask"] (resolver_v2 / NEW-arg checks), not is_unbound: phone/email updates have no record reference
     (resolved_id None) and is_unbound would drop them all. An ask call is NOT executed; it is emitted as stage
     "needs_clarification" (name, ask_reasons, refs, server_args): the agent has to ask the customer.
  4. execute call["server_args"] (resolved values named as in writes[]) on entity resolved_id, or the record's
     primary_id when the tool has no record reference (update_contact_number / update_email_address)
  5. N1_DIR = paths.NEEDLE (the worker's N1 runtime subset), weights = paths.NEEDLE_V2_WEIGHTS

    core = RouterCore(agent_type, record, asr_fn, emit_fn)
    core.prewarm()                                  # builds the Needle agent for (agent_type, record) once
    tr = core.on_text(frame, piece)                 # every non-PAD/EPAD text event -> trigger dict or None
    if tr: core.handle_trigger(tr, pcm)             # pcm = user audio window (float32, 24 kHz)

handle_trigger stages (each emitted through emit_fn as an INTERFACE.md 4.1 action event):
  trigger -> asr (asr_fn(pcm)) -> needle (router.route_raw) -> resolved (router.resolve_calls)
          -> per call: executed (StubTools) | unbound (N1 guard) | needs_clarification (v2 ask) ; error on any exception.
N1 unbound guard (DECISIONS.md D-T3-2): a call is dropped when resolved_id is None (resolver rule "ambiguous") or
when the model gave a non-empty order_ref that the resolver could not bind (rule ends with "_unmatched" and
fell back to a default entity). unbound_rule="none_only" drops only resolved_id None.

Needle runs on CPU in venv-needle: JAX_PLATFORMS=cpu is set before needle is imported.
"""
import os
import re
import sys
import time
import uuid

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # Needle must never open a CUDA context next to the live server

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402

NEEDLE_DIR = str(paths.NEEDLE)
if NEEDLE_DIR not in sys.path:
    sys.path.insert(0, NEEDLE_DIR)
os.environ["N1_DIR"] = NEEDLE_DIR   # v2: n1path must find THIS N1 subset first (never a dev-machine needle tree)
NEEDLE_V2_SCHEMA = str(paths.NEEDLE_V2 / "schema")

import trigger as trig  # noqa: E402
from stub_tools import StubTools  # noqa: E402

SILENCE_RMS = 0.005   # D-T3-8: below this the window is treated as silence (no ASR, no Needle)
WEIGHTS = paths.NEEDLE_WEIGHTS   # DEP1 INTERFACE.md section 9 (tuned_full.cact; router.DEFAULT_WEIGHTS is absent)
WEIGHTS_V2 = paths.NEEDLE_V2_WEIGHTS
ROUTER = paths.ROUTER            # v2 (default) | n1


def load_router(kind):
    """(module, weights) for 'v2' or 'n1'."""
    if kind == "n1":
        import router  # <S2S_NEEDLE>/router.py (needs venv-needle)
        return router, WEIGHTS
    if kind != "v2":
        raise ValueError(f"S2S_ROUTER must be v2 or n1, got {kind!r}")
    if NEEDLE_V2_SCHEMA not in sys.path:
        sys.path.insert(0, NEEDLE_V2_SCHEMA)
    import router_v2  # <S2S_NEEDLE_V2>/schema/router_v2.py (+ ../numconv, N1 subset via N1_DIR)
    return router_v2, WEIGHTS_V2


def is_unbound(call, rule="strict"):
    if call["resolved_id"] is None:
        return True, f"resolver rule {call['resolver_rule']}: no single entity"
    if rule == "strict" and call["resolver_rule"].endswith("_unmatched"):
        return True, f"order_ref {call['order_ref']!r} matched no entity (rule {call['resolver_rule']})"
    return False, ""


class RouterCore:
    def __init__(self, agent_type, record, asr_fn, emit_fn=None, weights=None, unbound_rule="strict",
                 debounce_s=trig.DEBOUNCE_S, router_kind=None):
        self.router_kind = (router_kind or ROUTER).strip().lower()
        router, default_weights = load_router(self.router_kind)
        self.router = router
        self.v2 = self.router_kind == "v2"
        self.agent_type = agent_type
        self.record = record
        self.asr_fn = asr_fn
        self.emit_fn = emit_fn or (lambda ev: None)
        self.weights = weights or default_weights
        self.unbound_rule = unbound_rule
        self.detector = trig.TriggerDetector(debounce_s)
        self.tools = StubTools(record)
        self.history = []          # one summary dict per handled trigger
        self.supported = agent_type in (router.tools_v2.AGENT_TYPES if self.v2 else router.tools.AGENT_TYPES)

    # ---------------------------------------------------------------- setup
    def prewarm(self):
        """Build (and cache) the Needle agent for this session so the first trigger does not pay ~1 s."""
        if not self.supported:
            return None
        t = time.perf_counter()
        self.router.route_raw(self.agent_type, self.record, "hi there", weights=self.weights)
        return round((time.perf_counter() - t) * 1000, 1)

    # ---------------------------------------------------------------- stream
    def on_text(self, frame, piece):
        if not self.supported:
            return None
        tr = self.detector.push(frame, piece)
        if tr:   # review fix D-T3-7: snapshot at trigger time (the worker runs later, on another thread)
            tr["model_segments"] = self.model_segments(2)
        return tr

    def model_segments(self, n=2):
        return self.detector.seg.last_segments(n)

    # ---------------------------------------------------------------- stages
    def _emit(self, tid, stage, payload, latency_ms, t0, frame):
        ev = {"trigger_id": tid, "stage": stage, "payload": payload, "latency_ms": round(latency_ms, 1),
              "since_trigger_ms": round((time.perf_counter() - t0) * 1000, 1), "frame": frame}
        self.emit_fn(ev)
        return ev

    def handle_trigger(self, tr, pcm, sample_rate=24000, frame=None):
        """Runs ASR -> Needle -> resolver -> guard -> stub tools for one trigger. Returns a summary dict."""
        t0 = time.perf_counter()
        frame = tr["frame"] if frame is None else frame
        tid = "t-" + uuid.uuid4().hex[:8]
        evs = []
        summ = {"trigger_id": tid, "trigger": tr, "events": evs}
        try:
            evs.append(self._emit(tid, "trigger", {k: tr[k] for k in ("phrase", "rule", "score", "segment")},
                                  0.0, t0, frame))
            # ---- ASR
            t = time.perf_counter()
            pcm = np.asarray(pcm, dtype=np.float32)
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.0
            if rms < SILENCE_RMS:
                # review fix D-T3-8: Trelis loops on silent windows ("pratibhagi ke liye ..." for 8-10 s of GPU)
                # and the non-empty garbage went on to Needle. Real trigger windows have RMS >= 0.053.
                asr = {"text": "", "backend": "skipped", "skipped": f"silence (rms {rms:.5f} < {SILENCE_RMS})"}
            else:
                asr = self.asr_fn(pcm)
            lat_asr = (time.perf_counter() - t) * 1000
            transcript = asr["text"]
            evs.append(self._emit(tid, "asr", {"text": transcript, "backend": asr.get("backend"),
                                               "skipped": asr.get("skipped"), "rms": round(rms, 5),
                                               "audio_s": round(len(pcm) / sample_rate, 2),
                                               "text_raw": asr.get("text_raw"),
                                               "service_latency_ms": asr.get("latency_ms")}, lat_asr, t0, frame))
            # ---- Needle (v2: the raw Trelis text, numconv + romanise inside router_v2; N1: the romanised text)
            needle_text = (asr.get("text_raw") or transcript) if self.v2 else transcript
            segs = tr.get("model_segments") or self.model_segments(2)
            t = time.perf_counter()
            if re.search(r"\w", needle_text or ""):
                resp = self.router.route_raw(self.agent_type, self.record, needle_text, weights=self.weights)
            else:   # review fix D-T3-7: Needle returns a write (e.g. cancel_order) on an empty transcript
                resp = {"function_calls": [], "skipped": "empty_transcript"}
            lat_needle = (time.perf_counter() - t) * 1000
            fcalls = resp.get("function_calls") or []
            evs.append(self._emit(tid, "needle", {"function_calls": fcalls, "model_segments": segs,
                                                  "reasoning": resp.get("reasoning"),
                                                  "suppressed_calls": resp.get("suppressed_calls"),
                                                  "skipped": resp.get("skipped"), "router": self.router_kind},
                                  lat_needle, t0, frame))
            # ---- resolver
            t = time.perf_counter()
            calls = self.router.resolve_calls(fcalls, self.record, self.agent_type)
            evs.append(self._emit(tid, "resolved", {"calls": calls}, (time.perf_counter() - t) * 1000, t0, frame))
            # ---- guard + stub tools
            executed, unbound, asks = [], [], []
            for c in calls:
                t = time.perf_counter()
                if self.v2:
                    if c.get("ask"):
                        asks.append(c)
                        evs.append(self._emit(tid, "needs_clarification", {
                            "name": c["name"], "ask_reasons": c.get("ask_reasons"), "refs": c.get("refs"),
                            "server_args": c.get("server_args"), "arguments": c.get("arguments"),
                            "order_ref": c.get("order_ref"), "resolver_rule": c.get("resolver_rule"),
                            "reason": "router asks the customer: " + ", ".join(c.get("ask_reasons") or [])},
                            (time.perf_counter() - t) * 1000, t0, frame))
                        continue
                    target = c["resolved_id"] or self.record.get("primary_id")
                    args = dict(c.get("server_args") or {})
                    result, diff = self.tools.execute(c["name"], target, args)
                    executed.append(c)
                    evs.append(self._emit(tid, "executed", {"name": c["name"], "resolved_id": target,
                                                            "arguments": args, "model_arguments": c.get("arguments"),
                                                            "resolver_rule": c.get("resolver_rule"),
                                                            "result": result, "record_diff": diff},
                                          (time.perf_counter() - t) * 1000, t0, frame))
                    continue
                bad, reason = is_unbound(c, self.unbound_rule)
                if bad:
                    unbound.append(dict(c, reason=reason))
                    evs.append(self._emit(tid, "unbound", {"name": c["name"], "order_ref": c["order_ref"],
                                                           "reason": reason, "resolver_rule": c["resolver_rule"],
                                                           "resolved_id": c["resolved_id"]},
                                          (time.perf_counter() - t) * 1000, t0, frame))
                    continue
                result, diff = self.tools.execute(c["name"], c["resolved_id"], c["arguments"])
                executed.append(c)
                evs.append(self._emit(tid, "executed", {"name": c["name"], "resolved_id": c["resolved_id"],
                                                        "arguments": c["arguments"], "result": result,
                                                        "record_diff": diff},
                                      (time.perf_counter() - t) * 1000, t0, frame))
            summ.update(transcript=transcript, needle_text=needle_text, asr_backend=asr.get("backend"),
                        function_calls=fcalls, router=self.router_kind,
                        calls=calls, executed=executed, unbound=unbound, asks=asks, model_segments=segs,
                        lat_ms={"asr": round(lat_asr, 1), "needle": round(lat_needle, 1),
                                "total": round((time.perf_counter() - t0) * 1000, 1)})
        except Exception as e:  # noqa: BLE001  -- every failure is an event, the stream keeps going
            evs.append(self._emit(tid, "error", {"where": "handle_trigger", "message": f"{type(e).__name__}: {e}"},
                                  0.0, t0, frame))
            summ["error"] = f"{type(e).__name__}: {e}"
        self.history.append(summ)
        return summ
