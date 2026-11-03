import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.brain import Learner, MemoFilter, is_correction, parse_memo, route  # noqa: E402
from learning.memory import MemoryStore  # noqa: E402


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "m.db")

    def tearDown(self) -> None:
        self.store._db.close()
        self.tmp.cleanup()

    def test_remember_and_recall_similar_question(self) -> None:
        self.store.remember("澳大利亚的首都是哪里", "澳大利亚的首都是堪培拉。")
        hit = self.store.best_answer("澳大利亚首都是哪里呀？")
        self.assertIsNotNone(hit)
        self.assertIn("堪培拉", hit.answer)
        self.assertIsNone(self.store.best_answer("日本的首都是哪里"))

    def test_update_instead_of_duplicate(self) -> None:
        self.store.remember("北京到上海高铁要多久", "大约七个小时。")
        self.store.remember("北京到上海高铁要多久？", "最快四个半小时左右。")
        self.assertEqual(self.store.stats()["knowledge"], 1)
        self.assertIn("四个半", self.store.best_answer("北京到上海高铁要多久").answer)

    def test_expired_knowledge_is_ignored(self) -> None:
        self.store.remember("动车票价多少钱", "五百五十三元。", days=1)
        self.store._db.execute("UPDATE knowledge SET expires = ?", (time.time() - 10,))
        self.assertIsNone(self.store.best_answer("动车票价多少钱"))

    def test_facts_dedupe_and_context(self) -> None:
        self.assertTrue(self.store.add_fact("主人每天五点下课"))
        self.assertFalse(self.store.add_fact("主人每天五点下课。"))
        self.store.remember("天空为什么是蓝色的", "因为蓝光更容易被大气散射。")
        ctx = self.store.context_for("天空为啥是蓝色")
        self.assertIn("五点下课", ctx)
        self.assertIn("散射", ctx)


class RoutingTests(unittest.TestCase):
    def test_routes(self) -> None:
        self.assertEqual(route("你好呀", "chat", teacher_ok=True).who, "local")
        self.assertEqual(route("我今天好累", "comfort", teacher_ok=True).who, "local")
        self.assertEqual(route("澳大利亚首都是哪里", "lookup", teacher_ok=True).who, "teacher")
        self.assertEqual(route("为什么天是蓝的", "explain", teacher_ok=True).who, "teacher")
        self.assertEqual(route("明天会下雨吗", "chat", teacher_ok=True).who, "teacher")
        self.assertEqual(route("我每周三下午五点下课", "chat", teacher_ok=True).who, "teacher")
        self.assertEqual(route("澳大利亚首都是哪里", "lookup", teacher_ok=False).who, "local")
        r = route("澳大利亚首都是哪里", "lookup", teacher_ok=True, memory_answer="堪培拉。")
        self.assertEqual((r.who, r.memory_answer), ("memory", "堪培拉。"))
        # time-sensitive questions never answer from memory
        self.assertEqual(route("今天天气怎么样", "lookup", teacher_ok=True, memory_answer="晴").who, "teacher")
        self.assertEqual(route("不对", "chat", teacher_ok=True, correction=True).who, "teacher")

    def test_correction(self) -> None:
        for t in ("不对，应该是四个半小时", "你说错了", "错了"):
            self.assertTrue(is_correction(t), t)
        self.assertFalse(is_correction("对的"))


class MemoTests(unittest.TestCase):
    def test_marker_split_across_tokens_is_cut_from_speech(self) -> None:
        f = MemoFilter()
        pieces = ["澳大利亚的首都是", "堪培拉。\n【", "记", "忆】{\"remember\": true, ", "\"q\": \"澳大利亚首都\", \"a\": \"堪培拉\", \"days\": null}"]
        spoken = "".join(f.feed(p) for p in pieces) + f.finish()
        self.assertEqual(spoken.strip(), "澳大利亚的首都是堪培拉。")
        memo = parse_memo(f.memo)
        self.assertEqual((memo["q"], memo["a"], memo["days"]), ("澳大利亚首都", "堪培拉", None))

    def test_no_marker_passes_everything(self) -> None:
        f = MemoFilter()
        spoken = "".join(f.feed(p) for p in ["好的【", "注意】这里"]) + f.finish()
        self.assertEqual(spoken, "好的【注意】这里")
        self.assertEqual(parse_memo(f.memo), {})


class LearnerTests(unittest.TestCase):
    def test_learn_stores_and_logs_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(Path(d) / "m.db")
            log = Path(d) / "lora.jsonl"
            lr = Learner(store, log)
            lr.learn("我每周三五点下课", "好的，我记住了。",
                     {"remember": False, "facts": ["主人每周三五点下课"]}, system="你是星期一")
            lr.learn("北京到上海高铁要多久", "最快四个半小时左右。",
                     {"remember": True, "q": "北京到上海高铁要多久", "a": "最快四个半小时左右", "days": 365},
                     system="你是星期一", wrong="大约七个小时。")
            self.assertEqual(store.stats(), {"knowledge": 1, "facts": 1, "lessons": 1})
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([m["role"] for m in rows[1]["messages"]], ["system", "user", "assistant"])
            store._db.close()


if __name__ == "__main__":
    unittest.main()
