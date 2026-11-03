import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.user_title import apply_title, parse_title_command  # noqa: E402


class TitleCommandTests(unittest.TestCase):
    def test_recognizes_requests(self) -> None:
        cases = {
            "以后叫我王哥": "王哥",
            "小揽，以后叫我老王吧": "老王",
            "你可以叫我小李就行": "小李",
            "从现在开始称呼我张总": "张总",
            "叫我阿杰": "阿杰",
            "管我叫大哥好了": "大哥",
            "别叫我主人了，叫我老板": "老板",
        }
        for q, want in cases.items():
            self.assertEqual(parse_title_command(q), want, q)

    def test_drop_address(self) -> None:
        self.assertEqual(parse_title_command("别叫我主人了"), "")
        self.assertEqual(parse_title_command("不要再喊我主人"), "")

    def test_not_a_title_request(self) -> None:
        for q in ("你叫什么名字", "他们都叫我什么来着", "叫我起床", "今天天气怎么样", "你可以叫我小揽吗"):
            self.assertIn(parse_title_command(q), (None,), q)


class ApplyTitleTests(unittest.TestCase):
    def test_swap(self) -> None:
        self.assertEqual(apply_title("好的，主人。", "王哥"), "好的，王哥。")
        self.assertEqual(apply_title("主人，现在是下午三点。", "王哥"), "王哥，现在是下午三点。")
        self.assertEqual(apply_title("这部小说的主人公很勇敢。", "王哥"), "这部小说的主人公很勇敢。")
        self.assertEqual(apply_title("好的，主人。", "主人"), "好的，主人。")
        self.assertEqual(apply_title("好的，主人。", None), "好的，主人。")

    def test_no_address(self) -> None:
        self.assertEqual(apply_title("好的，主人。", ""), "好的。")
        self.assertEqual(apply_title("主人，现在是下午三点。", ""), "现在是下午三点。")
        self.assertEqual(apply_title("随时待命，主人请吩咐。", ""), "随时待命，请吩咐。")
        self.assertEqual(apply_title("好的，主人，回头见。", ""), "好的，回头见。")

    def test_prompt(self) -> None:
        p = "你是小揽，称呼对方为主人。主人追问时再展开。"
        self.assertEqual(apply_title(p, "王哥", for_prompt=True), "你是小揽，称呼对方为王哥。王哥追问时再展开。")
        self.assertEqual(apply_title(p, "", for_prompt=True), "你是小揽，不用特别的称呼。对方追问时再展开。")


if __name__ == "__main__":
    unittest.main()
