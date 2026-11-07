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
from learning.school import (GRADE_SYSTEM, LECTURE_SYSTEM, SELF_SYSTEM, SKILL_SYSTEM, SYLLABUS, SchoolState,  # noqa: E402
                             check_code, draft_skill, run_sandboxed, school_day)

LECTURE = {
    "title": "Python 列表推导式", "explain": "列表推导式把循环和筛选写成一行。", "example": "[x * 2 for x in range(3)]",
    "key_points": ["[表达式 for 变量 in 序列 if 条件]", "比 for 循环 append 简洁"],
    "exercises": [
        {"kind": "read", "code": "print([x for x in range(5) if x % 2])", "question": "输出什么？", "answer": "[1, 3]"},
        {"kind": "read", "code": "print(len([c for c in 'abc']))", "question": "输出什么？", "answer": "3"},
        {"kind": "write", "question": "写函数 evens(n)，返回 0 到 n-1 里的偶数列表", "tests": "assert evens(5) == [0, 2, 4]",
         "answer": "def evens(n):\n    return [x for x in range(n) if x % 2 == 0]"},
    ],
}


class Teacher:
    def __init__(self, mastered=True, skill=None) -> None:
        self.mastered, self.skill = mastered, skill
        self.graded: list[str] = []

    def complete(self, system, user, **kw):
        if system == SELF_SYSTEM:
            return json.dumps({"who_am_i": "我是星期一", "purpose": "帮主人做事", "lesson_topic": "看懂 Python 报错",
                               "gaps": [{"gap": "看不懂异常堆栈", "why": "上次任务失败", "kind": "coding"}],
                               "ability_skill": self.skill}, ensure_ascii=False)
        if system == LECTURE_SYSTEM:
            return json.dumps(LECTURE, ensure_ascii=False)
        if system == GRADE_SYSTEM:
            self.graded.append(user)
            ok = "全部通过" in user or "通过：" in user and "没通过" not in user
            return json.dumps({"items": [{"i": 0, "pass": True, "comment": "对"}, {"i": 1, "pass": False, "comment": "数错了", "correct": "3"},
                                         {"i": 2, "pass": ok, "comment": "看运行结果", "correct": LECTURE["exercises"][2]["answer"]}],
                               "summary": "还行", "mastered": self.mastered}, ensure_ascii=False)
        assert system == SKILL_SYSTEM
        return json.dumps(self.skill_reply, ensure_ascii=False)


def student(q: str) -> str:
    if "evens" in q:
        return "```python\ndef evens(n):\n    return [x for x in range(n) if x % 2 == 0]\n```"
    return "[1, 3]" if "x % 2" in q else "2"


class SandboxTests(unittest.TestCase):
    def test_pass_fail_banned_timeout(self) -> None:
        self.assertEqual(run_sandboxed("def f():\n    return 2", "assert f() == 2")[0], True)
        ok, out = run_sandboxed("def f():\n    return 3", "assert f() == 2")
        self.assertFalse(ok)
        self.assertIn("AssertionError", out)
        self.assertIn("不允许", run_sandboxed("import os\nos.listdir('/')", "")[1])
        self.assertIn("不允许", check_code("open('/etc/passwd').read()"))
        self.assertIn("不允许", check_code("__import__('subprocess')"))
        self.assertFalse(run_sandboxed("while True:\n    pass", "", timeout=1)[0])


class SchoolDayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = MemoryStore(self.dir / "memory.db")
        self.trained: list[tuple[str, str]] = []

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()

    def day(self, teacher, state):
        return school_day(teacher=teacher, store=self.store, state=state, local_answer=student,
                          log_training=lambda q, a: self.trained.append((q, a)), identity="我是星期一",
                          abilities="会执行命令", troubles="", notes_dir=self.dir / "school", skills_dir=self.dir / "skills")

    def test_a_full_class(self) -> None:
        state = SchoolState(self.dir / "school" / "state.json")
        teacher = Teacher()
        r = self.day(teacher, state)
        self.assertEqual([x["kind"] for x in r["lessons"]], ["syllabus", "gap"])
        self.assertEqual(r["lessons"][0]["passed"], 2)       # read 1 right, read 2 wrong, code passed in the sandbox
        self.assertIn("运行结果：通过", teacher.graded[0])    # the teacher saw the real run
        self.assertEqual(state.index, 1)                       # mastered: the syllabus moves on
        self.assertTrue((self.dir / "school" / "state.json").is_file())
        notes = sorted((self.dir / "school").glob("*.md"))
        self.assertTrue(any("自省" in n.name for n in notes))
        text = next(n for n in notes if "自省" not in n.name).read_text(encoding="utf-8")
        self.assertIn("正确答案", text)
        self.assertTrue(any(q.startswith("给我讲讲") for q, _ in self.trained))   # lecture -> training data
        self.assertTrue(any("evens" in q for q, _ in self.trained))
        self.assertGreaterEqual(self.store.stats()["lessons"], 1)                  # the wrong answer is a lesson
        self.assertIsNotNone(self.store.best_answer("Python 列表推导式：要点1"))

    def test_not_mastered_repeats(self) -> None:
        state = SchoolState(self.dir / "s.json")
        self.day(Teacher(mastered=False), state)
        self.assertEqual(state.index, 0)
        self.assertEqual(state.next_topic(), SYLLABUS[0])

    def test_skill_drafts(self) -> None:
        teacher = Teacher()
        teacher.skill_reply = {"name": "file-push", "description": "把文件传到别的机器", "tags": "scp",
                               "skill_md": "用 scp", "script_name": "push.sh", "script": "#!/bin/bash\nscp \"$1\" \"$2\"\n"}
        self.assertEqual(draft_skill(teacher, {"name": "file-push"}, self.dir / "skills"), "file-push")
        md = (self.dir / "skills" / "file-push" / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("risk: ask", md)
        self.assertEqual(draft_skill(teacher, {"name": "file-push"}, self.dir / "skills"), "")   # never overwrites
        teacher.skill_reply = dict(teacher.skill_reply, name="wipe", script="rm -rf / --no-preserve-root")
        self.assertEqual(draft_skill(teacher, {"name": "wipe"}, self.dir / "skills"), "")
        teacher.skill_reply = dict(teacher.skill_reply, name="peek", script="cat /userdata/secrets/deepseek.env")
        self.assertEqual(draft_skill(teacher, {"name": "peek"}, self.dir / "skills"), "")


if __name__ == "__main__":
    unittest.main()


class SelfTrackTests(unittest.TestCase):
    """认识自己: lessons from her own code / hardware rotate in, the teacher gets the material."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = MemoryStore(self.dir / "memory.db")

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()

    def test_rotation_and_material(self) -> None:
        from learning.school import SELF_LECTURE_SYSTEM, school_session

        class SelfTeacher(Teacher):
            def __init__(self, mastered=True):
                super().__init__(mastered)
                self.materials: list[str] = []

            def complete(self, system, user, **kw):
                if system == SELF_LECTURE_SYSTEM:
                    self.materials.append(user)
                    return json.dumps(dict(LECTURE, title="你的引脚"), ensure_ascii=False)
                return super().complete(system, user, **kw)

        topics = [("认识自己 · 我的硬件：GPIO", "gpio-39 sysfs in hi"), ("认识自己 · 我的代码 a.py（f）", "def f(): pass")]
        state = SchoolState(self.dir / "school" / "state.json")
        kinds = []
        t = SelfTeacher()
        for _ in range(4):
            r = school_session(teacher=t, store=self.store, state=state, local_answer=student, log_training=lambda q, a: None,
                               identity="我是星期一", abilities="", troubles="", notes_dir=self.dir / "school",
                               skills_dir=self.dir / "skills", self_topics=lambda: topics)
            kinds.append(r["kind"])
        self.assertEqual(kinds, ["syllabus", "self", "gap", "syllabus"])
        self.assertIn("gpio-39", t.materials[0])                  # taught from the survey, not from memory
        self.assertEqual(state.self_index, 1)
        self.assertIsNotNone(self.store.best_answer("我的硬件：GPIO 是做什么的"))
        note = next(p for p in (self.dir / "school").glob("*.md") if "自省" not in p.name and "引脚" in p.name)
        self.assertIn("（认识自己）", note.read_text(encoding="utf-8"))
        # not mastered: retried once, then moves on
        t2 = SelfTeacher(mastered=False)
        state.sessions = 1
        for _ in range(2):
            school_session(teacher=t2, store=self.store, state=state, local_answer=student, log_training=lambda q, a: None,
                           identity="我是星期一", abilities="", troubles="", notes_dir=self.dir / "school",
                           skills_dir=self.dir / "skills", self_topics=lambda: topics)
            state.sessions = 1
        self.assertEqual(state.self_index, 2)


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = MemoryStore(self.dir / "memory.db")

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()

    def session(self, teacher, state):
        from learning.school import school_session
        return school_session(teacher=teacher, store=self.store, state=state, local_answer=student,
                              log_training=lambda q, a: None, identity="我是星期一", abilities="会执行命令", troubles="",
                              notes_dir=self.dir / "school", skills_dir=self.dir / "skills")

    def test_lessons_keep_coming(self) -> None:
        state = SchoolState(self.dir / "school" / "state.json")
        first = self.session(Teacher(), state)
        self.assertTrue(first["review"])                       # the day starts with the self-review
        self.assertEqual(first["kind"], "syllabus")
        self.assertEqual(len(state.pending), 2)                # lesson_topic + one per gap
        second = self.session(Teacher(mastered=False), state)
        self.assertFalse(second["review"])                     # once a day
        self.assertEqual(second["kind"], "gap")
        self.assertIn(second["topic"], state.pending)          # not mastered: queued again
        third = self.session(Teacher(), state)
        self.assertEqual(third["kind"], "syllabus")
        self.assertEqual(state.index, 2)
        diary = next((self.dir / "school").glob("*自省.md")).read_text(encoding="utf-8")
        self.assertEqual(diary.count("掌握了") + diary.count("还要再练"), 3)
        state.day = "2000-01-01"                                # a new day: review again
        self.assertTrue(self.session(Teacher(), state)["review"])
        reloaded = SchoolState.load(state.path)
        self.assertEqual((reloaded.index, reloaded.sessions), (state.index, state.sessions))
        self.assertEqual(reloaded.mastered, state.mastered)

    def test_mastered_topics_are_skipped_until_review(self) -> None:
        import time as _t
        from learning.school import REVIEW_DAYS
        today = _t.strftime("%Y-%m-%d")
        state = SchoolState(self.dir / "school" / "state.json", day=today)
        state.mastered = {t: today for t in SYLLABUS[:3]}
        r = self.session(Teacher(), state)
        self.assertEqual(r["topic"], SYLLABUS[3])                # the three mastered ones are skipped
        self.assertEqual(state.mastered[SYLLABUS[3]], today)      # and this one is now mastered too
        # everything mastered, nothing queued: no lesson and not a single teacher call
        state.mastered = {t: today for t in SYLLABUS}
        state.pending, state.sessions = [], 0

        class Silent:
            def complete(self, *a, **kw):
                raise AssertionError("the teacher must not be called")
        r = self.session(Silent(), state)
        self.assertTrue(r.get("nothing"))
        # REVIEW_DAYS later the oldest comes back for one review
        import datetime as _dt
        old = (_dt.date.today() - _dt.timedelta(days=REVIEW_DAYS)).isoformat()
        state.mastered[SYLLABUS[5]] = old
        state.sessions = 0
        self.assertEqual(self.session(Teacher(), state)["topic"], SYLLABUS[5])

    def test_old_state_learns_what_was_mastered_from_history(self) -> None:
        p = self.dir / "school" / "state.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"index": 3, "history": [{"day": "2026-11-05", "topic": "A", "mastered": True},
                                                          {"day": "2026-11-05", "topic": "B", "mastered": False}]}),
                     encoding="utf-8")
        self.assertEqual(SchoolState.load(p).mastered, {"A": "2026-11-05"})
