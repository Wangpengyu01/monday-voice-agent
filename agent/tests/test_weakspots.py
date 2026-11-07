from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import drill, weakspots  # noqa: E402
from tests.test_drill import GOOD, AgentTeacher, CodeTeacher  # noqa: E402

PASSING_PLAN = (lambda ctx: '{"tool":"shell","cmd":"uptime -p"}' if "已执行" not in ctx
                else '{"tool":"done","say":"开机三小时了。"}')


class CountingAgentTeacher(AgentTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.task_requests = 0

    def complete(self, system, user, **kw):
        if system == drill.AGENT_TASK_SYSTEM:
            self.task_requests += 1
        return super().complete(system, user, **kw)


class WeakSpotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_dedupes_and_due_skips_items_tried_today(self) -> None:
        self.assertTrue(weakspots.add(self.dir, "agent", "板子开机多久了", "心算错", "eval", day="2026-11-07"))
        self.assertFalse(weakspots.add(self.dir, "agent", "板子开机多久了", "again", "drill"))
        self.assertFalse(weakspots.add(self.dir, "voice", "x", "y"))          # only modes with a real retest
        self.assertEqual([x["task"] for x in weakspots.due(self.dir, "agent", 3, day="2026-11-07")], ["板子开机多久了"])
        weakspots.record(self.dir, "agent", "板子开机多久了", False, "还是心算", day="2026-11-07")
        self.assertEqual(weakspots.due(self.dir, "agent", 3, day="2026-11-07"), [])   # once a day
        self.assertEqual(len(weakspots.due(self.dir, "agent", 3, day="2026-11-08")), 1)
        item = weakspots.load(self.dir)["open"][0]
        self.assertEqual((item["tries"], item["why"]), (1, "还是心算"))
        weakspots.record(self.dir, "agent", "板子开机多久了", True, day="2026-11-08")
        d = weakspots.load(self.dir)
        self.assertEqual((d["open"], d["closed"][0]["result"]), ([], "fixed"))

    def test_agent_drill_retests_a_weak_spot_first_and_closes_it_when_it_passes(self) -> None:
        weakspots.add(self.dir, "agent", "板子开机多久了", "心算错", "eval")
        t = CountingAgentTeacher()
        s = drill.agent_drill(teacher=t, learning_dir=self.dir, local_plan=PASSING_PLAN, idle=lambda: True,
                              execute=lambda st: (0, "up 3 hours"), n=1)
        self.assertEqual((s["passed"], s["results"][0]["retest"], t.task_requests), (1, True, 0))
        self.assertEqual(weakspots.load(self.dir)["open"], [])
        self.assertEqual(weakspots.load(self.dir)["closed"][0]["result"], "fixed")

    def test_agent_drill_failure_goes_on_the_list_only_if_the_teacher_can_do_it(self) -> None:
        s = drill.agent_drill(teacher=AgentTeacher(), learning_dir=self.dir, local_plan=lambda ctx: '{"tool":"shell","cmd":"sudo reboot"}',
                              idle=lambda: True, execute=lambda st: (0, "up 3 hours"))
        self.assertEqual(s["passed"], 0)
        item = weakspots.load(self.dir)["open"][0]
        self.assertEqual((item["mode"], item["task"], item["source"]), ("agent", "板子开机多久了", "drill"))

        class NobodyRight(AgentTeacher):   # the teacher's own run is judged wrong too: a bad task
            def complete(self, system, user, **kw):
                if system == drill.AGENT_JUDGE_SYSTEM:
                    return json.dumps({"ok": False, "why": "没答到"}, ensure_ascii=False)
                return super().complete(system, user, **kw)
        other = self.dir / "other"
        other.mkdir()
        drill.agent_drill(teacher=NobodyRight(), learning_dir=other, local_plan=lambda ctx: '{"tool":"shell","cmd":"sudo reboot"}',
                          idle=lambda: True, execute=lambda st: (0, "up 3 hours"))
        self.assertEqual(weakspots.load(other)["open"], [])
        # and a listed item whose retest nobody gets right is dropped, not kept forever
        weakspots.add(other, "agent", "板子开机多久了", "x", "eval")
        drill.agent_drill(teacher=NobodyRight(), learning_dir=other, local_plan=PASSING_PLAN, idle=lambda: True,
                          execute=lambda st: (0, "up 3 hours"), n=1)
        self.assertEqual((weakspots.load(other)["open"], weakspots.load(other)["closed"][0]["result"]), ([], "dropped"))

    def test_coding_failure_is_listed_with_its_tests_and_retested_next_time(self) -> None:
        wrong = {"写函数 add(a, b) 返回两数之和": f"```python\n{GOOD}\n```", "写函数 neg(x) 返回相反数": "def neg(x):\n    return x"}
        drill.coding_drill(teacher=CodeTeacher(), learning_dir=self.dir, student=wrong.get, idle=lambda: True)
        item = weakspots.load(self.dir)["open"][0]
        self.assertEqual((item["task"], item["tests"]), ("写函数 neg(x) 返回相反数", "assert neg(2) == -2"))

        class NoTeacher:
            def complete(self, *a, **kw):
                raise AssertionError("a retest needs no new task from the teacher")
        right = {"写函数 neg(x) 返回相反数": "def neg(x):\n    return -x"}
        s = drill.coding_drill(teacher=NoTeacher(), learning_dir=self.dir, student=right.get, idle=lambda: True, n=1)
        self.assertEqual((s["passed"], s["results"][0]["retest"]), (1, True))
        self.assertEqual(weakspots.load(self.dir)["open"], [])

    def test_summary_lines_for_the_self_review(self) -> None:
        weakspots.add(self.dir, "agent", "r1-tts-daemon 还活着吗", "说挂了，其实在跑", "eval")
        self.assertIn("真考没过（办事）：r1-tts-daemon 还活着吗", weakspots.summary(self.dir))
        self.assertEqual(json.loads(weakspots.path(self.dir).read_text(encoding="utf-8"))["open"][0]["source"], "eval")


if __name__ == "__main__":
    unittest.main()
