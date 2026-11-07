from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import distill, drill, usage  # noqa: E402
from learning.lora_prep import SPOKEN_SYSTEM, load_rows, route  # noqa: E402
from learning.school import CODING_SYSTEM  # noqa: E402

PERSONA = "你是星期一。"


class Teacher:
    def complete(self, system, user, **kw):
        if system == distill.VOICE_Q_SYSTEM:
            return json.dumps({"questions": ["晚上吃什么好", "怎么看磁盘还剩多少", "讲个笑话"]}, ensure_ascii=False)
        if system.startswith(PERSONA):
            return {"晚上吃什么好": "来碗番茄鸡蛋面吧，快又暖胃。", "怎么看磁盘还剩多少": "用 df -h 看",
                    "讲个笑话": "```code```"}[user]
        if system == SPOKEN_SYSTEM:
            return json.dumps({"spoken": "用 df 命令加 h 参数看。" if "磁盘" in user else "`x`"}, ensure_ascii=False)
        if system == drill.CODE_TASK_SYSTEM:
            return json.dumps({"tasks": [{"q": "写函数 add(a, b)", "tests": "assert add(1, 2) == 3", "solution": "def add(a, b):\n    return a + b"},
                                         {"q": "写函数 bad()", "tests": "assert bad() == 1", "solution": "def bad():\n    return 2"}]},
                              ensure_ascii=False)
        if system == distill.CLEAN_VOICE_SYSTEM:
            if "关火" in user:
                return json.dumps({"ok": False, "a": "这个我现在做不到，到点不会主动叫你，你用手机定个五分钟闹钟吧。"}, ensure_ascii=False)
            if "葱" in user:
                return json.dumps({"ok": False, "a": "好的，我五分钟后提醒你。"}, ensure_ascii=False)   # still a promise
            return json.dumps({"ok": True})
        raise AssertionError(system[:40])


class DistillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = distill.Store(self.dir / "distill")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def rows(self, kind):
        f = self.dir / "distill" / f"{kind}.jsonl"
        return [json.loads(x)["messages"] for x in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []

    def test_voice_keeps_speakable_rewrites_once_drops_the_rest(self) -> None:
        stats = {}
        distill.voice_unit(Teacher(), self.store, PERSONA, "吃饭", stats)
        got = {m[1]["content"]: m[2]["content"] for m in self.rows("voice")}
        self.assertEqual(got, {"晚上吃什么好": "来碗番茄鸡蛋面吧，快又暖胃。", "怎么看磁盘还剩多少": "用 df 命令加 h 参数看。"})
        self.assertEqual(stats["dropped"], 1)
        self.assertTrue(all(m[0]["content"] == PERSONA for m in self.rows("voice")))   # trained under her persona

    def test_coding_keeps_only_sandbox_verified_solutions(self) -> None:
        distill.coding_unit(Teacher(), self.store, {})
        rows = self.rows("coding")
        self.assertEqual([m[1]["content"] for m in rows], ["写函数 add(a, b)"])
        self.assertEqual(rows[0][0]["content"], CODING_SYSTEM)

    def test_store_resumes_counts_and_dedupes(self) -> None:
        self.assertTrue(self.store.add("voice", PERSONA, "你好呀", "你好。"))
        self.assertFalse(self.store.add("voice", PERSONA, "你好呀！", "你好。"))
        again = distill.Store(self.dir / "distill")
        self.assertEqual(again.count["voice"], 1)
        self.assertFalse(again.add("voice", PERSONA, "你好呀", "嗨。"))

    def test_lora_prep_reads_distilled_rows_in_their_modes(self) -> None:
        distill.voice_unit(Teacher(), self.store, PERSONA, "吃饭", {})
        distill.coding_unit(Teacher(), self.store, {})
        modes = sorted(route(r["messages"]) for r in load_rows(self.dir))
        self.assertEqual(modes, ["coding", "voice", "voice"])

    def test_run_stops_at_peak_and_at_budget(self) -> None:
        calls = []
        units = {"voice": lambda: calls.append(1)}
        # Budget reads the real usage log by default: on the board a night's spend passed this test's 10 yuan
        with mock.patch.object(usage, "is_peak", return_value=True), mock.patch.object(distill.Budget, "spent", return_value=0.0):
            res = distill.run(teacher=None, persona=PERSONA, learning_dir=self.dir, targets={"voice": 5}, max_yuan=10,
                              workers=1, any_time=False, units=units)
        self.assertEqual((calls, res["why"]), ([], "到了高峰时段，空闲时段再接着跑"))
        with mock.patch.object(distill.Budget, "spent", return_value=11.0):
            res = distill.run(teacher=None, persona=PERSONA, learning_dir=self.dir, targets={"voice": 5}, max_yuan=10,
                              workers=1, any_time=True, units=units)
        self.assertIn("预算", res["why"])
        state = json.loads((self.dir / "distill" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "stopped")

    def test_clean_fixes_promises_caps_functions_drops_repeats(self) -> None:
        st = self.store
        st.add("voice", PERSONA, "五分钟后提醒我关火", "好的主人，五分钟后我提醒您关火。")
        st.add("voice", PERSONA, "帮我记一下买葱", "记下了，买葱。")
        st.add("voice", PERSONA, "我会提醒自己的你别管", "好，我提醒一句：别太累。")
        st.add("voice", PERSONA, "晚上吃什么", "番茄鸡蛋面。")
        for i in range(7):
            st.add("coding", CODING_SYSTEM, f"写 add 第{i}版", "def add(a, b):\n    return a + b")
        st.add("coding", CODING_SYSTEM, "写 sub", "def sub(a, b):\n    return a - b")
        shell = lambda c: json.dumps({"tool": "shell", "cmd": c})  # noqa: E731
        st.add("agent", drill.AGENT_SYSTEM, "任务：内存\n下一步？", shell("free -m"))
        st.add("agent", drill.AGENT_SYSTEM, "任务：内存\n已执行：\n1. free -m -> 返回码 0\n下一步？", shell("free -m"), dedupe=False)
        st.add("agent", drill.AGENT_SYSTEM, "任务：内存\n已执行：\n1. free -m -> 返回码 0\n下一步？x",
               json.dumps({"tool": "done", "say": "够用。"}, ensure_ascii=False), dedupe=False)
        res = distill.clean(Teacher(), self.dir / "distill", workers=1)
        voice = {m[1]["content"]: m[2]["content"] for m in self.rows("voice")}
        self.assertEqual(voice["五分钟后提醒我关火"], "这个我现在做不到，到点不会主动叫你，你用手机定个五分钟闹钟吧。")
        self.assertNotIn("帮我记一下买葱", voice)                 # the rewrite still promised: dropped
        self.assertEqual(voice["我会提醒自己的你别管"], "好，我提醒一句：别太累。")   # teacher said it is fine
        self.assertEqual(voice["晚上吃什么"], "番茄鸡蛋面。")         # never sent to the teacher
        self.assertEqual(res["voice"], {"checked": 3, "rewritten": 1, "dropped": 1, "left": 3})
        self.assertEqual(res["coding"], {"before": 8, "left": distill.CODE_PER_FUNC + 1, "functions": 2})
        self.assertEqual(res["agent"], {"before": 3, "left": 2})

    def test_promises_reads_each_clause(self) -> None:
        self.assertTrue(distill.promises("好的主人，五分钟后我提醒您关火。"))
        self.assertTrue(distill.promises("我还做不到定时提醒，或者我帮您记下这件事？"))
        self.assertFalse(distill.promises("哎呀这个我做不到，我没法到点提醒你，你赶紧用手机定个闹钟吧。"))
        self.assertFalse(distill.promises("来碗番茄鸡蛋面吧。"))

    def test_generator_respects_the_function_cap(self) -> None:
        for i in range(distill.CODE_PER_FUNC):
            self.store.add("coding", CODING_SYSTEM, f"add {i}", "def add(a, b):\n    return a + b")
        again = distill.Store(self.dir / "distill")
        self.assertEqual(again.funcs["add"], distill.CODE_PER_FUNC)
        stats = {}
        distill.coding_unit(Teacher(), again, stats)          # teacher offers add() again: refused
        self.assertEqual(again.count["coding"], distill.CODE_PER_FUNC)
        self.assertEqual(stats["dropped"], 2)


class UsageTests(unittest.TestCase):
    def test_peak_window_and_cost(self) -> None:
        import datetime as dt
        bj = dt.timezone(dt.timedelta(hours=8))
        t = lambda *a: dt.datetime(*a, tzinfo=bj).timestamp()  # noqa: E731
        self.assertTrue(usage.is_peak(t(2026, 11, 5, 10, 0)))    # Thursday morning
        self.assertFalse(usage.is_peak(t(2026, 11, 5, 13, 0)))   # lunch
        self.assertFalse(usage.is_peak(t(2026, 11, 7, 10, 0)))   # Saturday
        self.assertAlmostEqual(usage.cost(1_000_000, 1_000_000, 1_000_000, peak=False), 0.02 + 1 + 4)
        self.assertAlmostEqual(usage.cost(0, 0, 1_000_000, peak=True), 8)

    def test_peak_hours_only_the_conversation_may_ask_the_teacher(self) -> None:
        from learning.teacher import DeepSeekTeacher, TeacherUnavailable
        t = DeepSeekTeacher()
        with mock.patch.object(usage, "is_peak", return_value=True),                 mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "x", "MONDAY_USAGE_TAG": "上课"}),                 mock.patch.object(t._session, "post", side_effect=AssertionError("must not reach the API")):
            with self.assertRaises(TeacherUnavailable):
                t.complete("s", "u")
            with self.assertRaises(TeacherUnavailable):
                t.stream("s", "u")
        with mock.patch.object(usage, "is_peak", return_value=True),                 mock.patch.dict("os.environ", {"MONDAY_USAGE_TAG": "对话"}):
            self.assertFalse(usage.peak_blocked())

    def test_record_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td, mock.patch.object(usage, "USAGE_DIR", Path(td)), \
                mock.patch.dict("os.environ", {"MONDAY_USAGE_TAG": "蒸馏"}):
            usage.record({"prompt_tokens": 1000, "prompt_cache_hit_tokens": 600, "completion_tokens": 200}, "m")
            day = sorted(Path(td).glob("*.jsonl"))[0].stem
            s = usage.summary(day, Path(td))
        self.assertEqual((s["total"]["calls"], s["total"]["hit"], s["total"]["miss"], s["total"]["out"]), (1, 600, 400, 200))
        self.assertIn("蒸馏", s["jobs"])


if __name__ == "__main__":
    unittest.main()
