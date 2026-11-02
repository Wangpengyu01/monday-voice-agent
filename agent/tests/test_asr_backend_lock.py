import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.asr_engine import _StreamingParaformerBackend  # noqa: E402


class ConcurrencyProbe:
    """Fake recognizer that records whether two threads were ever inside it at once."""

    def __init__(self) -> None:
        self.inside = 0
        self.max_inside = 0
        self._m = threading.Lock()

    def _enter(self) -> None:
        with self._m:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.01)
        with self._m:
            self.inside -= 1

    def reset(self) -> None:
        self._enter()

    def feed(self, samples, sample_rate=None) -> str:
        self._enter()
        return ""

    def finalize(self) -> str:
        self._enter()
        return "ok"


class AsrBackendLockTests(unittest.TestCase):
    def test_reset_never_overlaps_a_running_finalize(self) -> None:
        probe = ConcurrencyProbe()
        backend = _StreamingParaformerBackend(probe, 16000)  # type: ignore[arg-type]
        samples = [0.0] * 16000
        threads = [threading.Thread(target=backend.finalize, args=(samples, 16000)) for _ in range(3)]
        threads += [threading.Thread(target=backend.reset) for _ in range(5)]
        threads += [threading.Thread(target=backend.feed, args=([0.0] * 320, 16000)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(probe.max_inside, 1)


if __name__ == "__main__":
    unittest.main()
