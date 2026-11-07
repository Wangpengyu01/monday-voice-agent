from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.volume_command import parse_volume_command  # noqa: E402
from tts.qwen3_tts import volume_scale  # noqa: E402


class VolumeCommandTests(unittest.TestCase):
    def target(self, text: str, current: int = 70):
        r = parse_volume_command(text, current)
        return None if r is None else r[0]

    def test_set_and_relative(self) -> None:
        self.assertEqual(self.target("音量调到三十"), 30)
        self.assertEqual(self.target("把音量调到百分之三十五"), 35)
        self.assertEqual(self.target("音量调到70%"), 70)
        self.assertEqual(self.target("调一下音量到二十"), 20)
        self.assertEqual(self.target("声音大一点"), 85)
        self.assertEqual(self.target("小点声"), 55)
        self.assertEqual(self.target("太吵了"), 55)
        self.assertEqual(self.target("音量开到一半"), 50)
        self.assertEqual(self.target("声音最大"), 100)
        self.assertEqual(self.target("小点声", current=10), 5)   # never fully silent by voice

    def test_query_and_not_a_command(self) -> None:
        self.assertEqual(parse_volume_command("现在音量多少", 40), (None, "现在音量是百分之40。"))
        self.assertIsNone(parse_volume_command("声音是怎么传播的", 70))
        self.assertIsNone(parse_volume_command("今天天气怎么样", 70))

    def test_volume_scale(self) -> None:
        self.assertEqual(volume_scale(100), 1.0)
        self.assertEqual(volume_scale(0), 0.0)
        self.assertLess(volume_scale(30), volume_scale(70))
        self.assertEqual(volume_scale("bad"), 1.0)


if __name__ == "__main__":
    unittest.main()
