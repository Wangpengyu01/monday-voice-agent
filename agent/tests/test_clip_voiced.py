"""Cached reply clips must not be near-silent (2026-11-07: the noon greeting clip played as silence)."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

import orchestrator.main as m  # noqa: E402

SR = 24000


def write(path: Path, loud_s: float, quiet_s: float) -> Path:
    tone = [int(6000 * math.sin(2 * math.pi * 220 * i / SR)) for i in range(int(SR * loud_s))]
    frames = tone + [0] * int(SR * quiet_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(int(x).to_bytes(2, "little", signed=True) for x in frames))
    return path


class VoicedShareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_speech_like_clip_passes(self) -> None:
        share = m._wav_voiced_share(write(self.dir / "ok.wav", 1.2, 0.4))
        self.assertGreaterEqual(share, m.MIN_VOICED_SHARE)
        self.assertAlmostEqual(share, 0.75, delta=0.05)

    def test_blip_in_silence_fails(self) -> None:   # the bad noon clip: 0.3 s of sound in 2.35 s
        share = m._wav_voiced_share(write(self.dir / "bad.wav", 0.3, 2.05))
        self.assertLess(share, m.MIN_VOICED_SHARE)

    def test_all_silence_is_zero(self) -> None:
        self.assertEqual(m._wav_voiced_share(write(self.dir / "zero.wav", 0, 1.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
