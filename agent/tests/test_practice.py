from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.memory import MemoryStore  # noqa: E402
from learning.lora_prep import SPOKEN_SYSTEM  # noqa: E402
from learning.practice import PRACTICE_SYSTEM, SLOT_S, SPOKEN_PRACTICE_SYSTEM, practice  # noqa: E402
from learning.reflect import GRADE_SYSTEM, VERIFY_SYSTEM  # noqa: E402

KEY = {"三月三十一日再过一天是几号": "四月一日。", "一斤等于多少克": "五百克。", "十二乘十五是多少": "一百八十。",
       "磁盘还剩多少怎么看": "用 df -h 看。"}


class Teacher:
    def __init__(self) -> None:
        self.quiz_prompts: list[str] = []

    def complete(self, system, user, **kw):
        if system == PRACTICE_SYSTEM:
            self.quiz_prompts.append(user)
            return json.dumps({"questions": list(KEY)[:3]}, ensure_ascii=False)
        if system == SPOKEN_PRACTICE_SYSTEM:
            self.quiz_prompts.append(user)
            return json.dumps({"questions": ["磁盘还剩多少怎么看", "一斤等于多少克"]}, ensure_ascii=False)
        if system == SPOKEN_SYSTEM:
            return json.dumps({"spoken": "用 df 命令加上 h 参数，就能看到还剩多少。"}, ensure_ascii=False)
        if system == GRADE_SYSTEM:
            qs = [line.split(". 问：", 1)[1] for line in user.splitlines() if ". 问：" in line]
            ans = [line.split("答：", 1)[1] for line in user.splitlines() if "答：" in line]
            return json.dumps({"items": [{"i": i, "verdict": "ok" if a == KEY[q] else "wrong", "right": KEY[q],
                                          "remember": True, "why": "t"} for i, (q, a) in enumerate(zip(qs, ans))]}, ensure_ascii=False)
        assert system == VERIFY_SYSTEM
        n = user.count("原回答：")
        return json.dumps({"items": [{"i": i, "really_wrong": True} for i in range(n)]})


class PracticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = MemoryStore(self.dir / "memory.db")
        self.local = {"三月三十一日再过一天是几号": "三月三十二日。", "一斤等于多少克": "五百克。", "十二乘十五是多少": "一百八十。",
                      "磁盘还剩多少怎么看": "用 df -h 看。"}

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()

    def run_once(self, teacher, idle=lambda: True, now=1_800_000_000):
        return practice(teacher=teacher, store=self.store, learning_dir=self.dir, local_answer=self.local.get,
                        system="s", idle=idle, n=3, now=now)

    def test_speaking_run_turns_clumsy_right_answers_into_spoken_data(self) -> None:
        s = self.run_once(Teacher(), now=1_800_000_000 + SLOT_S)  # odd slot -> speaking practice
        self.assertEqual(s["mode"], "speaking")
        self.assertEqual((s["wrong"], s["spoken"]), (0, 1))     # content right, but "df -h" reads badly
        self.assertIsNone(self.store.best_answer("磁盘还剩多少怎么看"))  # no lesson: the content was fine
        rows = [json.loads(x) for x in (self.dir / "lora_curated.jsonl").read_text(encoding="utf-8").splitlines()]
        reply = {r["messages"][1]["content"]: r["messages"][2]["content"] for r in rows}
        self.assertEqual(reply, {"磁盘还剩多少怎么看": "用 df 命令加上 h 参数，就能看到还剩多少。"})

    def test_learns_what_the_small_model_gets_wrong(self) -> None:
        s = self.run_once(Teacher())
        self.assertEqual(s["wrong"], 1)
        self.assertEqual(s["fixed"][0]["q"], "三月三十一日再过一天是几号")
        self.assertEqual(self.store.best_answer("三月三十一日再过一天是几号").answer, "四月一日。")
        self.assertTrue((self.dir / "lora_curated.jsonl").is_file())
        log = [json.loads(x) for x in (self.dir / "practice.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(log[0]["questions"]), 3)

    def test_stops_as_soon_as_the_owner_talks(self) -> None:
        calls = iter([True, False, False])
        s = self.run_once(Teacher(), idle=lambda: next(calls))
        self.assertTrue(s["stopped"])
        self.assertEqual(len(s["questions"]), 1)

    def test_does_not_repeat_recent_questions(self) -> None:
        self.run_once(Teacher())
        teacher = Teacher()
        s = self.run_once(teacher)
        self.assertEqual(s["questions"], [])               # all three were asked last time
        self.assertIn("十二乘十五是多少", teacher.quiz_prompts[0])  # and the teacher was told so


if __name__ == "__main__":
    unittest.main()
