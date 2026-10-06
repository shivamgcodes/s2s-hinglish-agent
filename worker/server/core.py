"""S2S serverless worker: copy of the DEP1/pod1 demo server/core.py as of 2026-10-06 (D-AGC, D-GATE, D-FILL, D-FILL2,
D-TOGGLE), with serverless edits only (DECISIONS.md 2026-10-06 "PORT"):
  - FILL_JSON comes from env S2S_TURN_FILL_JSON (default empty = no JSON override; the demo used
    the deploy tree's turn_fill.json). The per-call mode comes from the session config (cfg["turn_fill"], set by the
    worker from the websocket query param turn_fill=ticker|off) and mid-call changes arrive as a control frame
    (worker_server.py, kind 0x08) and are applied in the GPU thread via TurnFiller.request_mode().
  - Session recording is OFF by default (S2S_RECORD_SESSIONS=1 enables it); files go to S2S_SESSIONS_DIR
    (default $S2S_LOGS/sessions). When off, nothing is buffered.
AGC / gate / filler numbers and env names are unchanged from the demo.

DEP1 Track 1: model-independent part of the Engine (INTERFACE.md section 8).

EngineBase holds everything that does not touch the GPU: frame counter, ring buffer, inject schedule, mute,
play queue, step-time statistics, /metrics assembly. `engine.Engine` (PersonaPlex on CUDA) and
`mock_server.MockEngine` (CPU replay) subclass it and implement `_load`, `_prompt_phase`, `_model_step`.

Frame convention (binding, INTERFACE.md section 8):
  frame f = index of the step_frame() call since the prompt phase (0-based). FrameOut(frame=f).token is what
  lm_gen.step returned on call f. A text token passed with lm_gen.step(text_token=x) on call j is returned
  by call j+1 (max_delay = 1). EngineBase.frame = number of completed step_frame calls = index of the NEXT
  frame; this one value is the "current frame" everywhere (/internal/session, X-End-Frame, /metrics).
  inject(start_frame) clamps start_frame to >= frame + 2: the call that is possibly in flight (index frame)
  has already read its forced token (for frame+1), so frame+2 is the earliest frame that can still be planned.
"""
import collections
import dataclasses
import subprocess
import threading
import time
import uuid

import numpy as np

from ring import RingBuffer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = 1920


# --- input AGC (user 2026-10-06: "boost my input to the server") ------------------------------------
# Live mic audio arrived at about -45..-31 dBFS on speech; the V4 training customer audio is about
# -21 dBFS (median 80 ms speech-frame RMS). Bring speech to that level before the model / ring / ASR.
# Per 80 ms frame: speech-level estimate (frames above the gate), smoothed gain, quiet frames get at
# most AGC_NOISE_GAIN_DB, soft limiter. Env: S2S_INPUT_AGC=0 disables.
import os as _os
AGC_ON = _os.environ.get("S2S_INPUT_AGC", "1") != "0"
AGC_TARGET_DB = float(_os.environ.get("S2S_AGC_TARGET_DB", "-21"))
AGC_MAX_GAIN_DB = float(_os.environ.get("S2S_AGC_MAX_GAIN_DB", "24"))
AGC_GATE_DB = float(_os.environ.get("S2S_AGC_GATE_DB", "-52"))
AGC_NOISE_GAIN_DB = float(_os.environ.get("S2S_AGC_NOISE_GAIN_DB", "6"))
# Noise gate (user 2026-10-06 debug): the model was trained with DIGITAL SILENCE between customer turns; a live
# room noise floor (~-66 dBFS) made it stay completely silent (replay test: raw mic -> no output, gated -> normal).
# Frames below GATE_OPEN_DB (raw, before AGC) become exact zeros, with a hangover so word tails are kept.
GATE_ON = _os.environ.get("S2S_INPUT_GATE", "1") != "0"
GATE_OPEN_DB = float(_os.environ.get("S2S_GATE_OPEN_DB", "-55"))
GATE_HANGOVER = int(_os.environ.get("S2S_GATE_HANGOVER_FRAMES", "4"))   # 4 x 80 ms
# D-GATE2 (serverless, 2026-10-06): ADAPTIVE open threshold. Live serverless call (browser mic, call_0851): speech
# -20..-31 dBFS, room noise -43..-57 dBFS, so the fixed -55 gate stayed open on noise and the model went silent for
# 136 s (same failure as D-GATE / deployment_quirks 6.3). Per session the noise floor = the 10th percentile of the raw
# frame dB over the last GATE_FLOOR_WIN_S; open threshold = clip(floor + GATE_MARGIN_DB, GATE_OPEN_DB, GATE_MAX_OPEN_DB).
# GATE_OPEN_DB stays the lower bound (a quiet room behaves as before; digital silence -> exactly -55); the upper bound
# keeps normal speech passing even when the window is mostly speech. The fixed threshold is used until
# GATE_WARMUP_FRAMES frames are seen. S2S_GATE_ADAPTIVE=0 = the fixed D-GATE gate, bit-identical.
GATE_ADAPTIVE = _os.environ.get("S2S_GATE_ADAPTIVE", "1") != "0"
GATE_MARGIN_DB = float(_os.environ.get("S2S_GATE_MARGIN_DB", "12"))
GATE_FLOOR_WIN_S = float(_os.environ.get("S2S_GATE_FLOOR_WIN_S", "5"))
GATE_FLOOR_PCT = float(_os.environ.get("S2S_GATE_FLOOR_PCT", "10"))
GATE_MAX_OPEN_DB = float(_os.environ.get("S2S_GATE_MAX_OPEN_DB", "-38"))
GATE_WARMUP_FRAMES = int(_os.environ.get("S2S_GATE_WARMUP_FRAMES", "12"))   # ~1 s
AGC_STEP_DB = 3.0          # max gain change per frame (no pumping)


# --- turn-taking filler (user 2026-10-06): "everytime model waits for some time, and its not a trigger wait ... for
# every turn taking wait, play a ticker, or a hmm, so that the model hears some noise, shuts up and let me speak ...
# play this artificial noise for 1s for now". The model fills every silence itself (never quiet > ~2 s, see
# writeup/deployment_quirks.md 6.4). When the model finishes an utterance (output quiet for quiet_ms) that is NOT
# a check-line, feed the MODEL'S INPUT a synthetic sound for `seconds` (only while the user is silent / gated).
# The user does not hear it; the ring buffer / ASR get the real user audio.
# D-FILL2 (user 2026-10-06: "instead of hmm ... try a ticker, or maybe keyboard noise ... take 5 noises, pick the
# best"): modes hmm | ticker | keyboard | pink | breath | noise | off. Env defaults below; TurnFiller.reset() (each
# session start) re-reads the optional JSON FILL_JSON {"mode","seconds","quiet_ms","db"} which overrides them, so the
# sound can be switched without a restart. db = RMS of the 1 s buffer in dBFS (model-input level, after the AGC whose
# speech target is -21 dBFS); peaks are clipped to 0.9. Per-mode default RMS in FILL_MODE_DB.
import json as _json
import re as _re
FILL_JSON = _os.environ.get("S2S_TURN_FILL_JSON", "")      # serverless: empty = off (demo: a fixed turn_fill.json)
FILL_MODES_PUBLIC = ("ticker", "off")                          # the user-facing toggle (D-TOGGLE)
TICKER_PRESET = {"mode": "ticker", "seconds": 1.0, "quiet_ms": 400, "db": -34.0}   # D-TOGGLE "ticker" = D-FILL2 pick
RECORD_ON = _os.environ.get("S2S_RECORD_SESSIONS", "0") == "1"
FILL_MODE = _os.environ.get("S2S_TURN_FILL", "ticker")        # D-FILL2 provisional pick; see FILL_MODE_DB keys, or off
FILL_S = float(_os.environ.get("S2S_TURN_FILL_S", "1.0"))
FILL_QUIET_MS = float(_os.environ.get("S2S_TURN_FILL_QUIET_MS", "400"))
FILL_LEVEL_DB_ENV = _os.environ.get("S2S_TURN_FILL_DB")        # None -> per-mode default
FILL_MODE_DB = {"hmm": -24.0, "ticker": -34.0, "keyboard": -33.0, "pink": -32.0, "breath": -30.0, "noise": -24.0}
_CHECK_RE = _re.compile(r"\b(ek minute|rukiye|ek second|dekh let[ai]|update kar det[ai]|just a sec|let me check|"
                        r"checking|hold on|check kar)\b", _re.I)


def _band(x, lo, hi):
    """Brick-wall FFT band-pass (offline, on the whole buffer)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _click(rng, dur_s, tau_s, lo, hi, ring_hz=None):
    """A short decaying click: band-limited noise burst (+ optional damped resonance)."""
    m = int(dur_s * SAMPLE_RATE)
    t = np.arange(m) / SAMPLE_RATE
    c = _band(rng.normal(0, 1, m * 4), lo, hi)[:m] * np.exp(-t / tau_s)
    if ring_hz:
        c = c + 0.8 * np.sin(2 * np.pi * ring_hz * t) * np.exp(-t / (tau_s * 1.5)) * np.std(c) * 3
    return c / (np.max(np.abs(c)) + 1e-9)


def _make_fill(mode, seconds, level_db):
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(7)
    env = np.ones(n)
    a, r = int(0.06 * SAMPLE_RATE), int(0.2 * SAMPLE_RATE)
    env[:a] = np.linspace(0, 1, a); env[-r:] = np.linspace(1, 0, r)
    if mode == "noise":                        # white noise
        x = rng.normal(0, 1, n) * env
    elif mode == "pink":                       # soft room noise: pink (1/f power), 40 Hz - 8 kHz
        X = np.fft.rfft(rng.normal(0, 1, n))
        f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        X = X / np.sqrt(np.maximum(f, 1.0)); X[(f < 40) | (f > 8000)] = 0
        x = np.fft.irfft(X, n) * env
    elif mode == "breath":                     # exhale: noise 300-3000 Hz, broad 1 kHz emphasis, rise-fall envelope
        X = np.fft.rfft(rng.normal(0, 1, n))
        f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        X = X * np.exp(-0.5 * ((f - 1000) / 700) ** 2); X[(f < 300) | (f > 3000)] = 0
        x = np.fft.irfft(X, n) * np.sin(np.pi * t / seconds) ** 1.5
    elif mode == "ticker":                     # metronome: 4 ms decaying clicks every 200 ms (5 per second)
        x = np.zeros(n)
        c = _click(rng, 0.005, 0.0012, 1500, 6000, ring_hz=2500)
        for s in np.arange(0.02, seconds - 0.01, 0.2):
            i = int(s * SAMPLE_RATE); x[i:i + len(c)] += c[:n - i]
    elif mode == "keyboard":                   # typing: ~10 keys/s irregular, press + softer release, varied level
        x = np.zeros(n)
        s = 0.03
        while s < seconds - 0.08:
            g = rng.uniform(0.45, 1.0)
            p = _click(rng, 0.008, 0.0015, 800, 5000, ring_hz=rng.uniform(1800, 3200))
            th = np.sin(2 * np.pi * 180 * np.arange(len(p)) / SAMPLE_RATE) * np.exp(-np.arange(len(p)) / 60.0)
            p = p + 0.3 * th
            i = int(s * SAMPLE_RATE); x[i:i + len(p)] += g * p[:n - i]
            rel = s + rng.uniform(0.04, 0.08)
            q = _click(rng, 0.005, 0.001, 1500, 6000)
            j = int(rel * SAMPLE_RATE)
            if j < n:
                x[j:j + len(q)] += 0.35 * g * q[:n - j]
            s += max(0.05, rng.gamma(4.0, 0.025))          # mean 100 ms between keys
    else:  # "hmm": voiced nasal hum, f0 ~140 Hz with slight glide, harmonics up to ~1.2 kHz
        f0 = 140.0 * (1.0 - 0.06 * t / max(seconds, 1e-3))
        ph = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        x = sum((1.0 / k ** 1.3) * np.sin(k * ph) for k in range(1, 9))
        x = (x + 0.02 * rng.normal(0, 1, n)) * env
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (level_db / 20.0)
    return np.clip(x, -0.9, 0.9).astype(np.float32)


def _fill_config():
    """Env defaults, overridden by FILL_JSON if present and valid (errors -> env, never break a session)."""
    cfg = {"mode": FILL_MODE, "seconds": FILL_S, "quiet_ms": FILL_QUIET_MS,
           "db": float(FILL_LEVEL_DB_ENV) if FILL_LEVEL_DB_ENV else None, "source": "env"}
    try:
        if FILL_JSON and _os.path.exists(FILL_JSON):
            with open(FILL_JSON) as fh:
                j = _json.load(fh)
            if "mode" in j and "db" not in j:
                cfg["db"] = None                              # per-mode default for a JSON-chosen mode
            for k in ("mode", "seconds", "quiet_ms", "db"):
                if k in j and j[k] is not None:
                    cfg[k] = j[k] if k == "mode" else float(j[k])
            cfg["source"] = "json"
    except Exception as e:  # noqa: BLE001
        print("turn_fill.json ignored:", e, flush=True)
    if cfg["mode"] != "off" and cfg["mode"] not in FILL_MODE_DB:
        cfg["mode"] = "hmm"
    if cfg["db"] is None:
        cfg["db"] = FILL_MODE_DB.get(cfg["mode"], -24.0)
    return cfg


def _session_config(session_mode):
    """_fill_config() (env defaults + optional JSON), then the per-session/toggle mode on top (serverless)."""
    c = _fill_config()
    if session_mode == "off":
        c.update({"mode": "off", "source": "session"})
    elif session_mode == "ticker":
        c.update(dict(TICKER_PRESET, source="session"))
    return c


class TurnFiller:
    def __init__(self):
        self._key, self.buf = None, None
        self.reset()

    def reset(self, session_mode=None):
        """Session start. session_mode (serverless, per call): 'ticker' -> the D-TOGGLE ticker preset, 'off' -> off,
        None -> env/JSON as in the demo."""
        c = _session_config(session_mode)
        self._pending = None
        self.mode, self.seconds, self.level_db, self.source = c["mode"], c["seconds"], c["db"], c["source"]
        self.quiet_frames = max(1, round(c["quiet_ms"] / 80.0))
        key = (self.mode, self.seconds, self.level_db)
        if key != self._key:
            self.buf = _make_fill(self.mode, self.seconds, self.level_db) if self.mode != "off" else None
            self._key = key
        if self.buf is not None:
            self.buf_rms_db = round(20 * np.log10(float(np.sqrt(np.mean(self.buf ** 2))) + 1e-12), 1)
            self.buf_peak_db = round(20 * np.log10(float(np.max(np.abs(self.buf))) + 1e-12), 1)
        else:
            self.buf_rms_db = self.buf_peak_db = None
        self.n_fill, self.n_skip_check, self.n_cancel_user = 0, 0, 0
        self.events = []        # (frame, kind) per session: fill / cancel / skip_check
        self.k = 0              # frame index (after() calls this session)
        self.text = ""
        self.armed = False
        self.quiet = 0
        self.pos = -1           # >= 0 while filling

    def request_mode(self, mode):
        """Serverless toggle (any thread): remembered here and applied by input() in the GPU thread at the next frame,
        so the buffer is never swapped while input() is slicing it."""
        if mode in FILL_MODES_PUBLIC:
            self._pending = mode

    def apply_now(self, session_mode=None):
        """User toggle (2026-10-06): re-read the config and apply it immediately, keeping the session state."""
        c = _session_config(session_mode)
        self.mode, self.seconds, self.level_db, self.source = c["mode"], c["seconds"], c["db"], c["source"]
        self.quiet_frames = max(1, round(c["quiet_ms"] / 80.0))
        key = (self.mode, self.seconds, self.level_db)
        if key != self._key:
            self.buf = _make_fill(self.mode, self.seconds, self.level_db) if self.mode != "off" else None
            self._key = key
        if self.buf is None:
            self.pos = -1
            self.buf_rms_db = self.buf_peak_db = None
        else:                    # serverless: metrics only (the demo left these None after an off -> ticker switch)
            self.buf_rms_db = round(20 * np.log10(float(np.sqrt(np.mean(self.buf ** 2))) + 1e-12), 1)
            self.buf_peak_db = round(20 * np.log10(float(np.max(np.abs(self.buf))) + 1e-12), 1)

    def input(self, x):
        """Before the model step: replace silent (gated) input with the next filler chunk."""
        p = getattr(self, "_pending", None)
        if p is not None:
            self._pending = None
            self.apply_now(p)
        if self.buf is None or self.pos < 0:
            return x
        if np.any(x != 0):      # user started speaking: stop filling
            self.pos = -1; self.n_cancel_user += 1; self.events.append((self.k, "cancel"))
            return x
        chunk = self.buf[self.pos:self.pos + len(x)]
        if len(chunk) < len(x):
            chunk = np.concatenate([chunk, np.zeros(len(x) - len(chunk), np.float32)])
        self.pos += len(x)
        if self.pos >= len(self.buf):
            self.pos = -1
        return chunk

    def after(self, piece, pcm_model, user_in):
        """After the model step: track the model's utterance and decide when to start a fill."""
        k = self.k
        self.k += 1
        if self.buf is None:
            return None
        active = 20 * np.log10(float(np.sqrt(np.mean(pcm_model ** 2))) + 1e-12) > -45
        if piece and piece.strip():
            self.text = (self.text + piece)[-160:]
            self.armed = True
        self.quiet = 0 if active else self.quiet + 1
        if np.any(user_in != 0):           # user is talking: no fill, start a new model utterance afterwards
            self.armed, self.text = False, ""
            return None
        if self.armed and self.pos < 0 and self.quiet >= self.quiet_frames:
            last = self.text
            self.armed, self.text = False, ""
            if _CHECK_RE.search(last):       # trigger wait (check-line): let the model continue to its confirmation
                self.n_skip_check += 1; self.events.append((k, "skip_check"))
                return "skip_check"
            self.pos = 0
            self.n_fill += 1; self.events.append((k + 1, "fill"))   # first filled input = next frame
            return "fill"
        return None

    def metrics(self):
        return {"mode": self.mode, "source": self.source, "seconds": self.seconds, "quiet_ms": self.quiet_frames * 80,
                "level_db": self.level_db, "buf_rms_db": self.buf_rms_db, "buf_peak_db": self.buf_peak_db,
                "fills": self.n_fill, "skipped_check_line": self.n_skip_check, "cancelled_by_user": self.n_cancel_user,
                "filling": self.pos >= 0}


class InputAGC:
    def __init__(self, adaptive=None):
        self.adaptive = GATE_ADAPTIVE if adaptive is None else bool(adaptive)
        self.reset()

    def reset(self):
        self.floor_win = collections.deque(maxlen=max(1, round(GATE_FLOOR_WIN_S * 12.5)))   # raw frame dB (D-GATE2)
        self.noise_floor_db = None
        self.open_db = GATE_OPEN_DB
        self.level_db = None   # running speech level
        self.hang = 0
        self.gated_frames = 0
        self.frames = 0
        self.gain_db = 0.0
        self.last_in_db = -120.0
        self.last_out_db = -120.0

    def __call__(self, x):
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        db = 20.0 * np.log10(rms)
        self.last_in_db = db
        self.frames += 1
        if GATE_ON:
            if self.adaptive:
                self.floor_win.append(db)
                if len(self.floor_win) >= GATE_WARMUP_FRAMES:
                    self.noise_floor_db = float(np.percentile(np.fromiter(self.floor_win, float), GATE_FLOOR_PCT))
                    self.open_db = min(max(GATE_OPEN_DB, self.noise_floor_db + GATE_MARGIN_DB), GATE_MAX_OPEN_DB)
            if db > self.open_db:
                self.hang = GATE_HANGOVER
            elif self.hang > 0:
                self.hang -= 1
            else:
                self.gated_frames += 1
                self.last_out_db = -120.0
                return np.zeros_like(x, dtype=np.float32)
        speech = db > AGC_GATE_DB
        if speech:
            if self.level_db is None:
                self.level_db = db
            else:
                a = 0.3 if db > self.level_db else 0.05     # fast attack, slow release
                self.level_db += a * (db - self.level_db)
        want = 0.0 if self.level_db is None else min(max(AGC_TARGET_DB - self.level_db, 0.0), AGC_MAX_GAIN_DB)
        if not speech:
            want = min(want, AGC_NOISE_GAIN_DB)
        if not AGC_ON:
            return x
        self.gain_db += float(np.clip(want - self.gain_db, -AGC_STEP_DB, AGC_STEP_DB))
        y = x * (10.0 ** (self.gain_db / 20.0))
        pk = float(np.max(np.abs(y))) if y.size else 0.0
        if pk > 0.9:                                         # soft limiter
            y = 0.9 * np.tanh(y / 0.9)
        self.last_out_db = 20.0 * np.log10(float(np.sqrt(np.mean(y * y)) + 1e-12))
        return y.astype(np.float32)

    def metrics(self):
        return {"on": AGC_ON, "gain_db": round(self.gain_db, 1),
                "speech_level_db": None if self.level_db is None else round(self.level_db, 1),
                "last_in_db": round(self.last_in_db, 1), "last_out_db": round(self.last_out_db, 1),
                "target_db": AGC_TARGET_DB, "gate_on": GATE_ON, "gate_open_db": GATE_OPEN_DB,
                "gated_frac": round(self.gated_frames / max(self.frames, 1), 3),
                "gate_adaptive": self.adaptive, "gate_margin_db": GATE_MARGIN_DB,
                "noise_floor_db": None if self.noise_floor_db is None else round(max(self.noise_floor_db, -120.0), 1),
                "gate_threshold_db": round(self.open_db, 1)}
PAD, EPAD = 3, 0
SPECIAL = {PAD: "PAD", EPAD: "EPAD"}
STATS_WINDOW = 500
CONTEXT_MARGIN = 4   # end the session when fewer than this many KV positions are left


@dataclasses.dataclass
class FrameOut:
    frame: int
    token: int
    piece: str | None          # None for PAD / EPAD; else id_to_piece with "▁" -> " "
    pcm: np.ndarray            # [1920] float32, after mute/play
    pcm_model: np.ndarray      # [1920] float32, model audio before mute/play
    forced: bool
    step_ms: float
    lm_step_ms: float


class InjectPlan(dict):
    """{"first_frame","last_frame","n_frames","frames":[[frame, token_id, piece], ...]} (+ "requested_start_frame")."""


def _stats(xs):
    if not xs:
        return {"last": None, "p50": None, "p95": None, "max": None, "n": 0}
    a = np.fromiter(xs, float)
    return {"last": round(float(a[-1]), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2), "n": int(len(a))}


class SmiPoller:
    """nvidia-smi memory.used, refreshed on a background timer (never per request / per frame)."""

    def __init__(self, period_s: float = 2.0, enabled: bool = True):
        self.value = None
        self.enabled = enabled
        self.period_s = period_s
        if enabled:
            threading.Thread(target=self._run, daemon=True, name="smi-poller").start()

    def _run(self):
        while True:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"], timeout=10)
                self.value = int(out.split()[0])
            except Exception:
                self.value = None
            time.sleep(self.period_s)


class EngineBase:
    sample_rate = SAMPLE_RATE
    frame_size = FRAME_SIZE

    def __init__(self, spm, context: int, smi: bool = True):
        self.spm = spm
        self.context = int(context)
        self.ring = RingBuffer(30.0, SAMPLE_RATE)
        self._lock = threading.Lock()
        self.frame = 0
        self.session_id = None
        self.cfg = None
        self.active = False
        self._sched = {}            # returned frame -> forced token id
        self._plan = None           # active InjectPlan (for inject events)
        self._plan_state = None
        self._mute = False
        self._play = collections.deque()
        self._play_samples = 0
        self.step_ms = collections.deque(maxlen=STATS_WINDOW)
        self.lm_step_ms = collections.deque(maxlen=STATS_WINDOW)
        self.session_step_ms = []   # whole session, for end_session summary
        self.session_lm_ms = []
        self.prompt_positions = None
        self.prompt_phase_s = None
        self.listeners = []         # callables(event_dict) for inject events (called on the engine thread)
        self.router = {"last_event_t_wall": None,
                       "counts": {k: 0 for k in ("trigger", "asr", "needle", "resolved", "executed", "unbound",
                                                 "needs_clarification", "error")}}
        self.extra_metrics = {}     # server-level fields (e.g. backlog), merged into metrics()
        self._smi = SmiPoller(enabled=smi)

    # ---- to implement -------------------------------------------------------------------------------
    def _prompt_phase(self, cfg, should_abort):            # -> None; runs voice + text prompt phases
        raise NotImplementedError

    def _model_step(self, pcm: np.ndarray, forced_token):  # -> (token, pcm_model[1920], lm_ms, step_ms)
        raise NotImplementedError

    def kv_used(self) -> int:                              # transformer KV positions used so far
        raise NotImplementedError

    def vram(self) -> dict:
        return {"allocated_gib": None, "reserved_gib": None, "max_allocated_gib": None,
                "nvidia_smi_used_mib": self._smi.value}

    # ---- session ------------------------------------------------------------------------------------
    def start_session(self, cfg: dict, should_abort=None) -> str:
        with self._lock:
            self.frame = 0
            self._sched.clear()
            self._plan, self._plan_state = None, None
            self._mute = False
            self._play.clear()
            self._play_samples = 0
            self.session_step_ms, self.session_lm_ms = [], []
            for k in self.router["counts"]:
                self.router["counts"][k] = 0
            self.router["last_event_t_wall"] = None
        self.ring.reset()
        if not hasattr(self, "agc"):
            self.agc = InputAGC()
        self.agc.reset()
        if not hasattr(self, "filler"):
            self.filler = TurnFiller()
        self.filler.reset(cfg.get("turn_fill"))
        self._rec_in, self._rec_agc, self._rec_out, self._rec_min = [], [], [], []
        self._rec_on = RECORD_ON
        self.cfg = dict(cfg)
        self.session_id = "s-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        t0 = time.time()
        self._prompt_phase(self.cfg, should_abort)
        self.prompt_phase_s = round(time.time() - t0, 2)
        self.prompt_positions = self.kv_used()
        self.active = True
        return self.session_id

    def context_frames_left(self) -> int:
        return int(self.context - self.kv_used())

    def context_full(self) -> bool:
        return self.context_frames_left() < CONTEXT_MARGIN

    def end_session(self) -> dict:
        self.active = False
        self._dump_recording()
        a = self.session_step_ms[25:] or self.session_step_ms
        b = self.session_lm_ms[25:] or self.session_lm_ms

        def st(x):
            if not x:
                return None
            x = np.asarray(x)
            return {"n": int(len(x)), "p50": round(float(np.percentile(x, 50)), 2),
                    "p95": round(float(np.percentile(x, 95)), 2), "p99": round(float(np.percentile(x, 99)), 2),
                    "max": round(float(x.max()), 2), "mean": round(float(x.mean()), 2)}
        return {"session_id": self.session_id, "frames": self.frame, "conv_s": round(self.frame / FRAME_RATE, 2),
                "prompt_positions": self.prompt_positions, "prompt_phase_s": self.prompt_phase_s,
                "context_frames_left": self.context_frames_left(),
                "step_ms_excl_first25": st(a), "lm_step_ms_excl_first25": st(b),
                "compute_rtf": round(float(np.sum(a)) / (len(a) * 80.0), 3) if a else None,
                "vram": self.vram()}

    # ---- per frame ----------------------------------------------------------------------------------
    def step_frame(self, pcm: np.ndarray) -> FrameOut:
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        assert len(pcm) == FRAME_SIZE, len(pcm)
        rec = getattr(self, "_rec_on", False)       # serverless: recording off unless S2S_RECORD_SESSIONS=1
        if not hasattr(self, "_rec_in"):
            self._rec_in, self._rec_agc, self._rec_out = [], [], []
        if rec:
            self._rec_in.append(pcm.copy())
        if not hasattr(self, "agc"):
            self.agc = InputAGC()
        pcm = self.agc(pcm)
        if rec:
            self._rec_agc.append(pcm.copy())
        if not hasattr(self, "filler"):
            self.filler = TurnFiller()
        pcm_user = pcm
        fl = self.filler
        fill_was, cancels_was = fl.pos >= 0, fl.n_cancel_user
        pcm = fl.input(pcm)
        fill_ev = []                                # S2S 2026-10-06 ticker indicator: real filler transitions
        if fill_was and fl.pos < 0:
            if fl.n_cancel_user > cancels_was:
                fill_ev.append(("cancel", "user_speech"))
            elif fl.buf is None:
                fill_ev.append(("stop", "off"))
            else:
                fill_ev.append(("stop", "done"))
        if rec and hasattr(self, "_rec_min"):
            self._rec_min.append(pcm.copy())        # model input incl. filler (D-FILL2 recording)
        with self._lock:
            f = self.frame
            pass_tok = self._sched.get(f + 1)          # forced on call f -> returned at frame f+1
            exp_tok = self._sched.pop(f, None)         # forced on call f-1 -> returned now
        tok, pcm_model, lm_ms, step_ms = self._model_step(pcm, pass_tok)
        forced = exp_tok is not None
        if forced and tok != exp_tok:
            raise RuntimeError(f"inject mismatch at frame {f}: expected {exp_tok}, got {tok}")
        self.ring.append(pcm_user)
        with self._lock:
            if self._play_samples > 0:
                out = np.zeros(FRAME_SIZE, np.float32)
                got = 0
                while got < FRAME_SIZE and self._play:
                    head = self._play[0]
                    take = min(FRAME_SIZE - got, len(head))
                    out[got:got + take] = head[:take]
                    got += take
                    if take == len(head):
                        self._play.popleft()
                    else:
                        self._play[0] = head[take:]
                self._play_samples -= got
            elif self._mute:
                out = np.zeros(FRAME_SIZE, np.float32)
            else:
                out = pcm_model
            self.frame = f + 1
            ev = self._inject_progress(f)
        self.step_ms.append(step_ms); self.lm_step_ms.append(lm_ms)
        self.session_step_ms.append(step_ms); self.session_lm_ms.append(lm_ms)
        for e in ev:
            for cb in list(self.listeners):
                try:
                    cb(e)
                except Exception:
                    pass
        piece = None if tok in SPECIAL else self.spm.id_to_piece(tok).replace("▁", " ")
        r = self.filler.after(piece, pcm_model, pcm_user)
        if r == "fill":
            fill_ev.append(("start", "model_paused"))     # filler chunks go to the model from the next frame on
        elif r == "skip_check":
            fill_ev.append(("skip_check", "check_line"))
        for st, why in fill_ev:
            fe = {"type": "filler", "state": st, "reason": why, "frame": f, "t": round(f / FRAME_RATE, 3),
                  "mode": fl.mode, "seconds": fl.seconds}
            for cb in list(self.listeners):
                try:
                    cb(fe)
                except Exception:
                    pass
        if rec and hasattr(self, "_rec_out"):
            self._rec_out.append(np.asarray(out, np.float32).copy())
        return FrameOut(frame=f, token=tok, piece=piece, pcm=out, pcm_model=pcm_model, forced=forced,
                        step_ms=step_ms, lm_step_ms=lm_ms)

    def _inject_progress(self, f):
        """Inject events due after frame f (list; a one-frame plan yields started + done)."""
        p = self._plan
        if p is None:
            return []
        states = []
        if self._plan_state == "queued" and f >= p["first_frame"]:
            self._plan_state = "started"
            states.append("started")
        if self._plan_state == "started" and f >= p["last_frame"]:
            self._plan_state = "done"
            states.append("done")
        if self._plan_state == "done":
            self._plan = None
        return [{"type": "inject", "state": st, "first_frame": p["first_frame"], "last_frame": p["last_frame"],
                 "n_frames": p["n_frames"]} for st in states]

    # ---- inject / mute / play -----------------------------------------------------------------------
    def encode_words(self, words):
        toks = []
        for item in words:
            w, n_pad = item[0], int(item[1]) if len(item) > 1 else 0
            if isinstance(w, bool) or not isinstance(w, (int, str)):
                raise ValueError(f"bad word entry {item!r}")
            if isinstance(w, int):
                toks.append(int(w))
            else:
                ids = self.spm.encode(w)
                if not ids:
                    raise ValueError(f"word {w!r} encodes to no tokens")
                toks.extend(int(i) for i in ids)
            toks.extend([PAD] * max(0, n_pad))
        return toks

    def inject(self, words, start_frame: int) -> InjectPlan:
        toks = self.encode_words(words)
        if not toks:
            raise ValueError("empty inject")
        with self._lock:
            first = max(int(start_frame), self.frame + 2)
            # a new plan replaces any part of an earlier plan not yet passed to the model
            for k in [k for k in self._sched if k >= self.frame + 2]:
                del self._sched[k]
            frames = []
            for i, t in enumerate(toks):
                self._sched[first + i] = t
                frames.append([first + i, t, SPECIAL.get(t) or self.spm.id_to_piece(t).replace("▁", " ")])
            plan = InjectPlan(first_frame=first, last_frame=first + len(toks) - 1, n_frames=len(toks),
                              frames=frames, requested_start_frame=int(start_frame))
            self._plan, self._plan_state = plan, "queued"
        ev = {"type": "inject", "state": "queued", "first_frame": plan["first_frame"],
              "last_frame": plan["last_frame"], "n_frames": plan["n_frames"]}
        for cb in list(self.listeners):
            try:
                cb(ev)
            except Exception:
                pass
        return plan

    def mute(self, on: bool) -> None:
        with self._lock:
            self._mute = bool(on)

    @property
    def muted(self) -> bool:
        return self._mute

    def play(self, pcm: np.ndarray) -> float:
        x = np.asarray(pcm, np.float32).reshape(-1).copy()
        with self._lock:
            if len(x):
                self._play.append(x)
                self._play_samples += len(x)
            return round(self._play_samples / SAMPLE_RATE, 3)

    # ---- metrics ------------------------------------------------------------------------------------
    def metrics(self) -> dict:
        n = len(self.step_ms)
        sess = {"active": bool(self.active), "session_id": self.session_id if self.active else None,
                "record_id": (self.cfg or {}).get("record_id") if self.active else None,
                "agent_type": (self.cfg or {}).get("agent_type") if self.active else None,
                "frame": self.frame if self.active else None,
                "conv_s": round(self.frame / FRAME_RATE, 2) if self.active else None,
                "context_frames_left": self.context_frames_left() if self.active else None}
        with self._lock:
            inj = {"active": self._plan is not None, "queued_frames": len([k for k in self._sched if k >= self.frame])}
            play_s = round(self._play_samples / SAMPLE_RATE, 3)
        m = {"t_wall": time.time(), "session": sess,
             "step_ms": _stats(self.step_ms), "lm_step_ms": _stats(self.lm_step_ms),
             "rtf": round(float(np.sum(self.step_ms)) / (n * 80.0), 3) if n else None,
             "vram": self.vram(),
             "ring": {"seconds": round(min(self.ring.end_sample, self.ring.capacity) / SAMPLE_RATE, 2),
                      "end_sample": self.ring.end_sample},
             "inject": inj, "mute": bool(self._mute), "play_queued_s": play_s,
             "router": {"last_event_t_wall": self.router["last_event_t_wall"],
                        "counts": dict(self.router["counts"])}}
        if hasattr(self, "agc"):
            m["input_agc"] = self.agc.metrics()
        if hasattr(self, "filler"):
            m["turn_fill"] = self.filler.metrics()
        m.update(self.extra_metrics)
        return m

    def note_action(self, stage: str):
        if stage in self.router["counts"]:
            self.router["counts"][stage] += 1
        self.router["last_event_t_wall"] = time.time()


def _dump_recording_impl(self):
    """User 2026-10-06 debug: full-session recording, stereo L = model out, R = user in (raw, before AGC),
    plus the post-AGC user channel, to <S2S_SESSIONS_DIR>/<sid>.wav / <sid>_user_agc.wav (serverless: only with
    S2S_RECORD_SESSIONS=1; default dir $S2S_LOGS/sessions, /tmp/s2s/logs/sessions in the image)."""
    try:
        import os, wave
        if not getattr(self, "_rec_in", None):
            return
        d = _os.environ.get("S2S_SESSIONS_DIR") or _os.path.join(_os.environ.get("S2S_LOGS", "/tmp/s2s/logs"),
                                                                  "sessions")
        os.makedirs(d, exist_ok=True)
        uin = np.concatenate(self._rec_in)
        uag = np.concatenate(self._rec_agc) if self._rec_agc else uin
        mo = np.concatenate(self._rec_out) if self._rec_out else np.zeros_like(uin)
        n = min(len(uin), len(mo))

        def w(path, arr, ch):
            a = np.clip(arr, -1, 1)
            with wave.open(path, "wb") as f:
                f.setnchannels(ch); f.setsampwidth(2); f.setframerate(SAMPLE_RATE)
                f.writeframes((a * 32767).astype("<i2").tobytes())
        w(f"{d}/{self.session_id}.wav", np.stack([mo[:n], uin[:n]], 1).reshape(-1), 2)
        w(f"{d}/{self.session_id}_user_agc.wav", uag, 1)
        if getattr(self, "_rec_min", None):              # D-FILL2: L = model out, R = model input incl. filler
            mi = np.concatenate(self._rec_min); m2 = min(len(mi), len(mo))
            w(f"{d}/{self.session_id}_model_in.wav", np.stack([mo[:m2], mi[:m2]], 1).reshape(-1), 2)
        if hasattr(self, "filler"):
            import json
            with open(f"{d}/{self.session_id}_fill.json", "w") as fh:
                json.dump({"metrics": self.filler.metrics(), "events": self.filler.events}, fh)
    except Exception as e:  # noqa: BLE001 - debugging aid must never break a session
        print("dump_recording failed:", e, flush=True)

EngineBase._dump_recording = _dump_recording_impl
