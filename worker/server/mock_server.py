"""S2S serverless worker: fork of DEP1 server/mock_server.py (CPU-only MockEngine; no torch). Path edits only.

Used by worker_server.py --mock (DESIGN.md 7 "Mock engine"); still runnable alone as in DEP1:
  <venv-pp>/bin/python worker/server/mock_server.py [--call food_07_g1] [--prompt-s 2]   (= worker_server --mock)

MockEngine replays <S2S_MOCK_REPLAY>/<call>_s1001.{json,wav} (DEP1: the hinglish tests/out/V3_A/V3 dir) frame by
frame (text events + Opus audio), paced by the client's incoming audio exactly like the real server.
<call> = "<record_id>_<pairing>" when such a replay exists, else --call. /internal/ring serves the matching
<S2S_MOCK_INPUTS>/<call>.wav (aligned to the replay frame), not the client's audio, unless --ring client.
inject / mute / play behave as in the real Engine (forced tokens appear one call later, like max_delay=1),
so Track 2/3/4 code can be exercised without the GPU. step_ms values are the mock's own (tiny) CPU times.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402
from core import EPAD, FRAME_SIZE, PAD, EngineBase  # noqa: E402

log = logging.getLogger("s2s.worker")
REPLAY_DIR = str(paths.MOCK_REPLAY)
INPUTS_DIR = str(paths.MOCK_INPUTS)
SPM = paths.MOCK_SPM
VOICE_FRAMES = 51    # NATF2.pt / NATM1.pt embeddings
SILENCE_FRAMES = 6   # int(0.5 * 12.5), twice


class MockEngine(EngineBase):
    def __init__(self, call="food_07_g1", replay_dir=REPLAY_DIR, inputs_dir=INPUTS_DIR, prompt_s=2.0, ring="input",
                 context=3000, spm_path=None, load_s=0.0):
        if load_s:
            time.sleep(float(load_s))   # S2S: simulated model load, so /ping 204 -> 200 is observable in tests
        spm = sentencepiece.SentencePieceProcessor(spm_path or SPM)
        super().__init__(spm, context=context, smi=False)
        self.default_call = call
        self.replay_dir, self.inputs_dir = Path(replay_dir), Path(inputs_dir)
        self.prompt_s = float(prompt_s)
        self.ring_src = ring
        self.load_s = 0.0
        self._kv0 = 0

    def _piece_id(self, p):
        if p == "PAD":
            return PAD
        if p == "EPAD":
            return EPAD
        return int(self.spm.piece_to_id(("▁" + p[1:]) if p.startswith(" ") else p))

    def _prompt_phase(self, cfg, should_abort):
        name = f"{cfg.get('record_id')}_{cfg.get('pairing')}"
        if not (self.replay_dir / f"{name}_s1001.json").exists():
            name = self.default_call
        self.call = name
        log.info("mock replay: %s (record_id=%s pairing=%s)", name, cfg.get("record_id"), cfg.get("pairing"))
        pieces = json.loads((self.replay_dir / f"{name}_s1001.json").read_text())
        self._toks = [self._piece_id(p) for p in pieces]
        a, sr = sphn.read(str(self.replay_dir / f"{name}_s1001.wav"))
        assert sr == 24000
        self._audio = a[0].astype(np.float32)
        a, sr = sphn.read(str(self.inputs_dir / f"{name}.wav"))
        self._input = a[0].astype(np.float32)
        text = cfg.get("role_prompt") or ""
        ids = self.spm.encode(_ascii_prompt(f"<system> {text.strip()} <system>")) if text.strip() else []
        self._kv0 = VOICE_FRAMES + 2 * SILENCE_FRAMES + len(ids)
        self._next_forced = None
        t_end = time.time() + self.prompt_s
        while time.time() < t_end:
            if should_abort is not None and should_abort():
                raise RuntimeError("aborted during mock prompt phase")
            time.sleep(0.05)

    def kv_used(self):
        return self._kv0 + self.frame

    def step_frame(self, pcm):
        if self.ring_src == "input":
            f = self.frame
            seg = self._input[f * FRAME_SIZE:(f + 1) * FRAME_SIZE]
            pcm = np.zeros(FRAME_SIZE, np.float32)
            pcm[:len(seg)] = seg
        return super().step_frame(pcm)

    def _model_step(self, pcm, forced_token):
        t0 = time.perf_counter()
        f = self.frame
        if self._next_forced is not None:
            tok = self._next_forced
        else:
            tok = self._toks[f] if f < len(self._toks) else PAD
        self._next_forced = forced_token
        seg = self._audio[f * FRAME_SIZE:(f + 1) * FRAME_SIZE]
        out = np.zeros(FRAME_SIZE, np.float32)
        out[:len(seg)] = seg
        ms = (time.perf_counter() - t0) * 1000.0
        return tok, out, ms, ms


def _ascii_prompt(t):
    paths.ensure_common_on_path()
    import session
    return session.ascii_prompt(t)


def main():
    import worker_server
    worker_server.main(["--mock"] + sys.argv[1:])


if __name__ == "__main__":
    main()
