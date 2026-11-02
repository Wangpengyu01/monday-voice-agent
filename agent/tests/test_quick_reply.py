import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.quick_reply import ack_for, quick_reply_for, strip_leading_ack  # noqa: E402

NOW = datetime.datetime(2026, 11, 2, 13, 53)  # Monday


class QuickReplyTests(unittest.TestCase):
    def test_trivial_questions_answered_locally(self) -> None:
        cases = {
            "你是谁": "我是小揽，您的语音管家，有事尽管吩咐。",
            "小揽，你叫什么名字？": "我是小揽，您的语音管家，有事尽管吩咐。",
            "现在几点了": "现在是下午一点五十三分。",
            "今天星期几呀": "今天是十一月二日，星期一。",
            "谢谢": "不客气，主人。",
            "你好": "下午好，主人。",
            "晚安": "晚安，主人，好梦。",
        }
        for q, want in cases.items():
            self.assertEqual(quick_reply_for(q, NOW), want, q)

    def test_clock_edge_cases(self) -> None:
        self.assertEqual(quick_reply_for("几点了", datetime.datetime(2026, 1, 1, 14, 5)), "现在是下午两点零五分。")
        self.assertEqual(quick_reply_for("几点了", datetime.datetime(2026, 1, 1, 20, 0)), "现在是晚上八点整。")

    def test_real_questions_still_go_to_the_llm(self) -> None:
        for q in ("你好，帮我查一下明天的天气", "天为什么是蓝色的", "你是谁发明的", "现在几点了北京和纽约差几个小时", "十五乘四是多少"):
            self.assertIsNone(quick_reply_for(q, NOW), q)


class AckTests(unittest.TestCase):
    def test_ack_choice(self) -> None:
        self.assertEqual(ack_for("推荐一部适合周末看的电影"), "好的，")
        self.assertEqual(ack_for("帮我想个名字"), "好的，")
        self.assertEqual(ack_for("天为什么是蓝色的"), "我想想，")
        self.assertEqual(ack_for("十五乘四是多少"), "嗯，")

    def test_strip_models_own_ack(self) -> None:
        self.assertEqual(strip_leading_ack("好的，主人。"), "")
        self.assertEqual(strip_leading_ack("好的主人，我为您推荐肖申克的救赎。"), "我为您推荐肖申克的救赎。")
        self.assertEqual(strip_leading_ack("嗯，这个问题很有意思。"), "这个问题很有意思。")
        self.assertEqual(strip_leading_ack("好莱坞电影很多。"), "好莱坞电影很多。")
        self.assertEqual(strip_leading_ack("主人，十五乘四等于六十。"), "主人，十五乘四等于六十。")


if __name__ == "__main__":
    unittest.main()
