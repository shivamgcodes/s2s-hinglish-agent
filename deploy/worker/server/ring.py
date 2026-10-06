"""DEP1 Track 1: rolling user-audio ring buffer (INTERFACE.md section 8). Thread safe, numpy only."""
import threading

import numpy as np

SAMPLE_RATE = 24000


class RingBuffer:
    """Rolling `seconds` of float32 mono PCM at 24 kHz. `end_sample` = total samples appended since reset."""

    def __init__(self, seconds: float = 30.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.seconds = float(seconds)
        self.capacity = int(round(seconds * sample_rate))
        self._buf = np.zeros(self.capacity, np.float32)
        self._lock = threading.Lock()
        self.end_sample = 0

    def reset(self):
        with self._lock:
            self._buf[:] = 0
            self.end_sample = 0

    def append(self, pcm: np.ndarray):
        x = np.asarray(pcm, np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                x = x[-self.capacity:]
                self.end_sample += n - self.capacity
                n = self.capacity
            pos = self.end_sample % self.capacity
            first = min(n, self.capacity - pos)
            self._buf[pos:pos + first] = x[:first]
            if first < n:
                self._buf[:n - first] = x[first:]
            self.end_sample += n

    def get_last_with_end(self, seconds: float):
        """(copy of the last min(seconds, 30, available) s, end_sample) taken atomically."""
        with self._lock:
            want = int(round(min(float(seconds), self.seconds) * self.sample_rate))
            n = max(0, min(want, self.end_sample, self.capacity))
            end = self.end_sample
            pos = end % self.capacity
            if n == 0:
                return np.zeros(0, np.float32), end
            start = (pos - n) % self.capacity
            if start < pos:
                out = self._buf[start:pos].copy()
            else:
                out = np.concatenate([self._buf[start:], self._buf[:pos]])
            return out, end

    def get_last(self, seconds: float) -> np.ndarray:
        return self.get_last_with_end(seconds)[0]
