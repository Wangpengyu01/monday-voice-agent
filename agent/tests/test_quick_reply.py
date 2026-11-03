import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random  # noqa: E402

from orchestrator.persona import INTRODUCTION  # noqa: E402

from orchestrator.quick_reply import ACK_GROUPS, ack_for, ack_kind, quick_reply_for, strip_leading_ack  # noqa: E402

NOW = datetime.datetime(2026, 11, 2, 13, 53)  # Monday


class QuickReplyTests(unittest.TestCase):
    def test_trivial_questions_answered_locally(self) -> None:
        cases = {
            "你是谁": INTRODUCTION,
            "小揽，你叫什么名字？": INTRODUCTION,
            "星期一，你叫什么名字？": INTRODUCTION,
            "谢谢星期一": "不客气，主人。",
            "现在几点了": "现在是下午一点五十三分。",
            "今天星期几呀": "今天是十一月二日，星期一。",
            "谢谢": "不客气，主人。",
            "你好": "下午好，主人。",
            "晚安": "晚安，主人，好梦。",
        }
        for q, want in cases.items():
            self.assertEqual(quick_reply_for(q, NOW), want, q)

    def test_collapsed_forms_from_asr_normalization_match(self) -> None:
        # normalize_user_text turns 谢谢 into 谢, 拜拜 into 拜, 谢谢你 into 谢你
        for q in ("谢", "谢你", "谢谢你"):
            self.assertEqual(quick_reply_for(q, NOW), "不客气，主人。", q)
        self.assertEqual(quick_reply_for("拜", NOW), "好的，主人，回头见。")

    def test_clock_edge_cases(self) -> None:
        self.assertEqual(quick_reply_for("几点了", datetime.datetime(2026, 1, 1, 14, 5)), "现在是下午两点零五分。")
        self.assertEqual(quick_reply_for("几点了", datetime.datetime(2026, 1, 1, 20, 0)), "现在是晚上八点整。")

    def test_real_questions_still_go_to_the_llm(self) -> None:
        for q in ("你好，帮我查一下明天的天气", "天为什么是蓝色的", "你是谁发明的", "现在几点了北京和纽约差几个小时", "十五乘四是多少"):
            self.assertIsNone(quick_reply_for(q, NOW), q)


class AckTests(unittest.TestCase):
    def test_ack_fits_what_was_said(self) -> None:
        cases = {
            "推荐一部适合周末看的电影": "request",
            "帮我把音量调大一点": "request",
            "给我讲个笑话吧": "request",
            "帮我放首开心的歌": "request",
            "从北京到上海有多远": "lookup",
            "明天天气怎么样": "lookup",
            "十五乘四是多少": "lookup",
            "天为什么是蓝色的": "explain",
            "你觉得我该换手机吗": "explain",
            "我今天好累啊": "comfort",
            "能不能安慰我一下，我好难过": "comfort",
            "我考上研究生了": "cheer",
            "今天过得还行": "chat",
        }
        for q, kind in cases.items():
            self.assertEqual(ack_kind(q), kind, q)
            self.assertIn(ack_for(q, random.Random(0)), ACK_GROUPS[kind], q)

    def test_ack_does_not_repeat_back_to_back(self) -> None:
        rng = random.Random(1)
        picks = [ack_for("帮我想个名字", rng) for _ in range(20)]
        self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])), picks)
        self.assertGreater(len(set(picks)), 1)

    def test_strip_models_own_ack(self) -> None:
        self.assertEqual(strip_leading_ack("好的，主人。"), "")
        self.assertEqual(strip_leading_ack("好的主人，我为您推荐肖申克的救赎。"), "我为您推荐肖申克的救赎。")
        self.assertEqual(strip_leading_ack("嗯，这个问题很有意思。"), "这个问题很有意思。")
        self.assertEqual(strip_leading_ack("好莱坞电影很多。"), "好莱坞电影很多。")
        self.assertEqual(strip_leading_ack("主人，十五乘四等于六十。"), "主人，十五乘四等于六十。")
        self.assertEqual(strip_leading_ack("让我看看，北京到上海大约一千二百公里。"), "北京到上海大约一千二百公里。")


if __name__ == "__main__":
    unittest.main()
