from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import study_loop  # noqa: E402


class StudyLoopTests(unittest.TestCase):
    def test_commands_share_the_lock_and_count_practice_rounds(self) -> None:
        lesson, practice = study_loop.command("lesson", 7), study_loop.command("practice", 7)
        self.assertEqual(lesson[:4], ["/usr/bin/flock", "-w", "1800", study_loop.LOCK])
        self.assertEqual(lesson[-2:], ["-m", "learning.school"])
        self.assertEqual(practice[-4:], ["-m", "learning.practice", "--slot", "7"])


if __name__ == "__main__":
    unittest.main()
