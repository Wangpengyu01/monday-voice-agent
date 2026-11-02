import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.sentence_split import is_closing_filler  # noqa: E402

# Real endings from the board logs (2026-11-02) plus common variants.
FILLER = [
    "如果主人还有其他问题或需要进一步解释，",
    "请随时告诉我！",
    "需要我再解释一下计算过程吗？",
    "有什么我可以帮你的吗？",
    "请问有什么我可以帮忙的吗？",
    "如果你有任何需要，",
    "随时告诉我！",
    "还有什么需要了解的吗？",
    "希望这些信息对您有帮助。",
    "如有需要请随时吩咐。",
    "您还有其他问题吗？",
]
CONTENT = [
    "主人，十五乘四等于六十。",
    "我建议肖申克的救赎。",
    "你好，主人。",
    "这部电影非常适合周末观看。",
    "需要注意的是，黑洞并不是一个洞。",
    "如果明天下雨，记得带伞。",
    "有任何电器不要放在潮湿的地方。",
    "您需要先打开电源开关。",
    "您需要了解的是，这个功能还在测试。",
    "如果温度超过四十度，就需要停机检查。",
]


class ClosingFillerTests(unittest.TestCase):
    def test_filler_endings_are_detected(self) -> None:
        for t in FILLER:
            self.assertTrue(is_closing_filler(t), t)

    def test_content_sentences_are_kept(self) -> None:
        for t in CONTENT:
            self.assertFalse(is_closing_filler(t), t)


if __name__ == "__main__":
    unittest.main()
