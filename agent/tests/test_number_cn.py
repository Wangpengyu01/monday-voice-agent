import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.number_cn import int_cn, verbalize_numbers  # noqa: E402
from tts.sentence_split import StreamingSentenceSplitter, sanitize_tts_text  # noqa: E402


class IntCnTests(unittest.TestCase):
    def test_quantities(self) -> None:
        cases = {
            0: "零", 2: "二", 10: "十", 11: "十一", 20: "二十", 32: "三十二", 100: "一百",
            105: "一百零五", 110: "一百一十", 200: "两百", 1005: "一千零五", 1200: "一千两百",
            2026: "两千零二十六", 10000: "一万", 10050: "一万零五十", 12000: "一万两千",
            100000: "十万", 1000001: "一百万零一", 230000000: "两亿三千万",
        }
        for n, want in cases.items():
            self.assertEqual(int_cn(n), want, n)


class VerbalizeTests(unittest.TestCase):
    def test_board_datetime_reply(self) -> None:
        self.assertEqual(
            verbalize_numbers("当前时间是2026年11月2日，星期一，下午1点32分。"),
            "当前时间是二零二六年十一月二日，星期一，下午一点三十二分。",
        )

    def test_rules(self) -> None:
        cases = {
            "十五乘四等于60": "十五乘四等于六十",
            "约3.5公里": "约三点五公里",
            "增长了12.5%": "增长了百分之十二点五",
            "成功率50%": "成功率百分之五十",
            "13:05开会": "十三点零五分开会",
            "8:00出发": "八点整出发",
            "电话13800138000": "电话一三八零零一三八零零零",
            "编号007": "编号零零七",
            "共1,234人": "共一千两百三十四人",
            "2个苹果": "两个苹果",
            "下午2点": "下午两点",
            "第2名": "第二名",
            "第2章": "第二章",
            "没有数字": "没有数字",
        }
        for src, want in cases.items():
            self.assertEqual(verbalize_numbers(src), want, src)


class TtsPipelineNumberTests(unittest.TestCase):
    def test_sanitize_reads_numbers_naturally(self) -> None:
        self.assertEqual(
            sanitize_tts_text("大约3.5公里，成功率50%，13:05开会。"),
            "大约三点五公里，成功率百分之五十，十三点零五分开会。",
        )
        self.assertEqual(sanitize_tts_text("3.5公里外就是了。"), "三点五公里外就是了。")
        self.assertEqual(sanitize_tts_text("1. 先打开设备"), "先打开设备")

    def test_splitter_keeps_numbers_split_across_tokens(self) -> None:
        sp = StreamingSentenceSplitter(min_clause_chars=7, max_clause_chars=26)
        out: list[str] = []
        for tok in ["会议定在下午", "13", ":", "05", "开始，地点在", "3.", "5公里外的会议室。"]:
            out += sp.feed(tok)
        out += sp.finish()
        self.assertEqual("".join(out), "会议定在下午十三点零五分开始，地点在三点五公里外的会议室。")


if __name__ == "__main__":
    unittest.main()
