from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.intent import INTENT_SYSTEM, classify, fallback, recent_block  # noqa: E402


class Teacher:
    available = True

    def __init__(self, reply: dict | str | Exception) -> None:
        self.reply = reply
        self.seen: list[str] = []

    def complete(self, system, user, **kw):
        assert system == INTENT_SYSTEM
        self.seen.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


RECENT = [
    {"q": "sh到10.0.0.15", "a": "10.0.0.15 ping 不通，您确认一下地址", "who": "agent"},
]


class IntentTests(unittest.TestCase):
    def test_follow_up_becomes_a_complete_task(self) -> None:
        t = Teacher({"intent": "task", "rewrite": "SSH 到 10.0.0.15", "why": "补充地址"})
        it = classify(t, "是十五", RECENT)
        self.assertEqual((it.kind, it.rewrite, it.by), ("task", "SSH 到 10.0.0.15", "teacher"))
        self.assertIn("10.0.0.15 ping 不通", t.seen[0])  # the recent turns were given

    def test_ask_teacher(self) -> None:
        it = classify(Teacher({"intent": "ask_teacher", "rewrite": "怎么用密码 SSH 登录", "why": ""}), "你问你的老师", RECENT)
        self.assertEqual(it.kind, "ask_teacher")

    def test_bad_or_failed_reply_falls_back_to_rules(self) -> None:
        self.assertEqual(classify(Teacher("not json"), "帮我SH到196看磁盘", []).kind, "task")
        self.assertEqual(classify(Teacher(RuntimeError("down")), "去问depsek", []).kind, "ask_teacher")
        self.assertEqual(classify(Teacher({"intent": "weird"}), "今天好累", []).kind, "chat")

    def test_offline_rules(self) -> None:
        class Off:
            available = False
        self.assertEqual(classify(Off(), "扫描一下局域网", []).by, "rules")
        self.assertEqual(fallback("问一下你的老师").kind, "ask_teacher")

    def test_password_placeholder_kept(self) -> None:
        it = classify(Teacher({"intent": "task", "rewrite": "SSH 到 10.0.0.15，账号 pengyu 密码是那个", "why": ""}),
                      "账号pengyu密码是[密码已隐藏]", RECENT)
        self.assertIn("密码已提供", it.rewrite)

    def test_recent_block(self) -> None:
        self.assertIn("星期一（动手）", recent_block(RECENT))
        self.assertEqual(recent_block([]), "")


if __name__ == "__main__":
    unittest.main()
