from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import now  # noqa: E402


class NowTests(unittest.TestCase):
    def test_begin_step_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            with now.doing(d, "lesson", "Python 网络请求", "第 42 节 · 课本"):
                now.step("做第 1/3 题")
                cur = now.read(d)
                self.assertEqual((cur["title"], cur["step"]), ("Python 网络请求", "做第 1/3 题"))
            self.assertEqual(now.read(d), {})

    def test_dead_process_is_not_shown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / now.NAME).write_text('{"kind": "lesson", "title": "x", "pid": 999999999}', encoding="utf-8")
            self.assertEqual(now.read(d), {})


if __name__ == "__main__":
    unittest.main()
