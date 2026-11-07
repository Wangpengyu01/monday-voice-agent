from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.journal import log_turn, read_day  # noqa: E402
from learning.memory import MemoryStore  # noqa: E402
from learning.reflect import GRADE_SYSTEM, QUIZ_SYSTEM, VERIFY_SYSTEM, reflect  # noqa: E402


class FakeTeacher:
    """Grades by a fixed answer key; writes two quiz questions; a short diary with a focus list."""

    KEY = {"水的沸点是多少度": "一个标准大气压下是一百度。", "一公里等于多少米": "一千米。", "三加五等于几": "八。"}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, system: str, user: str, **kw) -> str:
        if system == GRADE_SYSTEM:
            self.calls.append("grade")
            items = []
            for line in user.splitlines():
                if "问：" in line:
                    i, q = line.split(". 问：", 1)
                    items.append({"i": int(i), "q": q})
            answers = [line.split("答：", 1)[1] for line in user.splitlines() if "答：" in line]
            out = []
            for it, a in zip(items, answers):
                right = self.KEY.get(it["q"])
                ok = right is None or a == right
                out.append({"i": it["i"], "verdict": "ok" if ok else "wrong", "right": "" if ok else right,
                            "remember": True, "days": None, "why": "测试"})
            return json.dumps({"items": out}, ensure_ascii=False)
        if system == VERIFY_SYSTEM:
            self.calls.append("verify")
            was = [line.split("原回答：", 1)[1] for line in user.splitlines() if "原回答：" in line]
            right = [line.split("老师的正确答案：", 1)[1] for line in user.splitlines() if "老师的正确答案：" in line]
            return json.dumps({"items": [{"i": i, "reason": "核对", "really_wrong": a.strip("。") not in r, "right": r}
                                         for i, (a, r) in enumerate(zip(was, right))]}, ensure_ascii=False)
        if system == QUIZ_SYSTEM:
            self.calls.append("quiz")
            return json.dumps({"questions": ["一公里等于多少米", "三加五等于几"]}, ensure_ascii=False)
        self.calls.append("diary")
        return '今天我把沸点说错了。\nFOCUS: {"focus": ["单位换算"]}'


class ReflectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.journal = self.dir / "turns"
        self.store = MemoryStore(self.dir / "memory.db")

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()

    def test_journal_roundtrip_skips_empty(self) -> None:
        log_turn(self.journal, "你好", "你好，主人。", who="local")
        log_turn(self.journal, "", "x", who="local")
        rows = read_day(self.journal, datetime.date.today())
        self.assertEqual([(r["q"], r["who"]) for r in rows], [("你好", "local")])

    def test_wrong_local_answers_become_lessons_memory_and_training_data(self) -> None:
        log_turn(self.journal, "水的沸点是多少度", "九十度。", who="local")
        log_turn(self.journal, "今天好累", "辛苦啦，早点休息。", who="local")
        log_turn(self.journal, "珠峰多高", "八千八百四十八米。", who="teacher")  # teacher answers are not re-graded
        teacher = FakeTeacher()
        local = {"一公里等于多少米": "一百米。", "三加五等于几": "八。"}
        rep = reflect(day=datetime.date.today(), teacher=teacher, store=self.store, learning_dir=self.dir,
                      journal_dir=self.journal, local_answer=local.get, system="系统提示", n_quiz=2)
        self.assertEqual(rep.errors, [])
        self.assertEqual(rep.reviewed, 2)
        self.assertEqual([w["q"] for w in rep.wrong], ["水的沸点是多少度"])
        self.assertEqual([w["q"] for w in rep.quiz_wrong], ["一公里等于多少米"])
        # corrections are live in memory right away
        self.assertEqual(self.store.best_answer("水的沸点是多少度").answer, "一个标准大气压下是一百度。")
        self.assertEqual(len(self.store.lessons(10)), 2)
        # diary + focus for tomorrow + curated training set
        diary = (self.dir / "diary" / f"{rep.day}.md").read_text(encoding="utf-8")
        self.assertIn("沸点", diary)
        self.assertNotIn("FOCUS", diary)
        self.assertEqual(json.loads((self.dir / "focus.json").read_text(encoding="utf-8"))["focus"], ["单位换算"])
        curated = [json.loads(x) for x in (self.dir / "lora_curated.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rep.dataset_size, 2)
        self.assertIn("一千米。", [r["messages"][2]["content"] for r in curated])

    def test_misjudged_answer_is_dropped_by_verification(self) -> None:
        # the first grader calls a right answer wrong (its "right" says the same thing); the check catches it
        log_turn(self.journal, "走路二十分钟，骑车快一半时间，骑车多久", "十分钟。", who="local")

        class Misjudging(FakeTeacher):
            KEY = {"走路二十分钟，骑车快一半时间，骑车多久": "骑车大约十分钟。"}

        teacher = Misjudging()
        rep = reflect(day=datetime.date.today(), teacher=teacher, store=self.store, learning_dir=self.dir,
                      journal_dir=self.journal, local_answer=lambda q: "x", system="s", n_quiz=0)
        self.assertIn("verify", teacher.calls)
        self.assertEqual(rep.wrong, [])
        self.assertEqual(self.store.stats()["lessons"], 0)

    def test_dry_run_writes_nothing(self) -> None:
        log_turn(self.journal, "水的沸点是多少度", "九十度。", who="local")
        rep = reflect(day=datetime.date.today(), teacher=FakeTeacher(), store=self.store, learning_dir=self.dir,
                      journal_dir=self.journal, local_answer=lambda q: "不知道", system="s", n_quiz=0, dry_run=True)
        self.assertEqual(len(rep.wrong), 1)
        self.assertEqual(self.store.stats(), {"knowledge": 0, "facts": 0, "lessons": 0})
        self.assertFalse((self.dir / "diary").exists())
        self.assertFalse((self.dir / "lora_curated.jsonl").exists())

    def test_teacher_failure_in_one_step_does_not_stop_the_rest(self) -> None:
        log_turn(self.journal, "水的沸点是多少度", "九十度。", who="local")

        class Broken(FakeTeacher):
            def complete(self, system, user, **kw):
                if system == GRADE_SYSTEM:
                    raise RuntimeError("network down")
                return super().complete(system, user, **kw)

        rep = reflect(day=datetime.date.today(), teacher=Broken(), store=self.store, learning_dir=self.dir,
                      journal_dir=self.journal, local_answer=lambda q: "x", system="s", n_quiz=2)
        self.assertTrue(any(e.startswith("review") for e in rep.errors))
        self.assertTrue((self.dir / "diary" / f"{rep.day}.md").is_file())


if __name__ == "__main__":
    unittest.main()
