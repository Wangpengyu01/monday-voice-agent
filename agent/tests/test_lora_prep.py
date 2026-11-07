from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.lora_prep import build, speakable  # noqa: E402


def row(q: str, a: str) -> str:
    return json.dumps({"messages": [{"role": "system", "content": "我是星期一"}, {"role": "user", "content": q},
                                    {"role": "assistant", "content": a}]}, ensure_ascii=False)


class Teacher:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies, self.calls = replies, 0

    def complete(self, system, user, **kw):
        self.calls += 1
        for k, v in self.replies.items():
            if k in user:
                return json.dumps({"spoken": v}, ensure_ascii=False)
        return "{}"


class LoraPrepTests(unittest.TestCase):
    def test_speakable(self) -> None:
        for bad in ["[3, 2, 1]", "def is_even(n):\n    return n % 2 == 0", "df -h 或 du -sh /path/to/dir", "['12', '3']",
                    "3 + 4 = 7", "用 `ls` 看", "ssh alice@host", "连接 10.0.0.5"]:
            self.assertFalse(speakable(bad), bad)
        for ok in ["一百七十五。", "主人，用 df 命令加上 h 参数就行。", "SSH 密钥登录是用一对公钥私钥代替密码。", "打印结果是 pet pet pet。"]:
            self.assertTrue(speakable(ok), ok)

    def test_build_keeps_at_most_two_copies_of_a_question(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "lora_data.jsonl").write_text("\n".join(row("讲讲 bash", f"第{i}版讲解。") for i in range(5)) + "\n",
                                               encoding="utf-8")
            stats = build(Teacher({}), p, workers=1)
            self.assertEqual((stats["written"], stats["repeats"]), (2, 3))

    def test_build_routes_voice_coding_rewrite(self) -> None:
        from learning.school import CODING_SYSTEM
        code = "def is_even(n):\n    return n % 2 == 0"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "lora_curated.jsonl").write_text("\n".join([
                row("一加一", "二。"),                                         # voice, speakable
                row("写函数 is_even(n)", code),                                 # class exercise under voice prompt
                row("这段代码输出什么：print([1])", "[1]"),                      # class exercise under voice prompt
                row("查看磁盘", "用了 61%，命令是 df -h"),                       # voice turn with a command
                row("扫描网络", "结果在 /tmp/scan.txt"),                         # voice turn, rewrite stays unspeakable
                row("账号", "账号是 bob，密码是 [密码已隐藏]"),
            ]) + "\n", encoding="utf-8")
            (p / "lora_data.jsonl").write_text(row("一加一", "二。") + "\n", encoding="utf-8")   # duplicate
            t = Teacher({"查看磁盘": "用了百分之六十一，用 df 命令看的。", "扫描网络": "放在 /tmp 里了"})
            stats = build(t, p, workers=1)
            self.assertEqual(stats, {"total": 5, "voice": 1, "coding": 2, "agent": 0, "rewritten": 1, "dropped": 1, "written": 4})
            self.assertEqual(t.calls, 2)                  # only the voice turns go to the teacher, never the code
            out = {json.loads(x)["messages"][-2]["content"]: json.loads(x)["messages"]
                   for x in (p / "lora_train.jsonl").read_text(encoding="utf-8").splitlines()}
            self.assertEqual(out["写函数 is_even(n)"][-1]["content"], code)            # real code kept...
            self.assertEqual(out["写函数 is_even(n)"][0]["content"], CODING_SYSTEM)    # ...in coding mode
            self.assertEqual(out["这段代码输出什么：print([1])"][0]["content"], CODING_SYSTEM)
            self.assertEqual(out["查看磁盘"][-1]["content"], "用了百分之六十一，用 df 命令看的。")
            self.assertEqual(out["查看磁盘"][0]["content"], "我是星期一")                 # voice prompt kept
            calls = t.calls
            build(t, p, workers=1)                        # cached: nothing asked again
            self.assertEqual(t.calls, calls)

    def test_old_agent_step_contexts_get_the_budget_line(self) -> None:
        from learning.drill import AGENT_SYSTEM
        old = "任务：开机多久了\n已执行：\n1. uptime -p -> 返回码 0\nup 3 hours\n下一步？"
        sample = {"messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": old},
                               {"role": "assistant", "content": '{"tool": "done", "say": "开机三小时了。"}'}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "lora_data.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(build(Teacher({}), p, workers=1)["agent"], 1)
            m = json.loads((p / "lora_train.jsonl").read_text(encoding="utf-8"))["messages"]
            self.assertTrue(m[1]["content"].endswith("还能执行 3 条命令。下一步？"))
            self.assertEqual(m[2]["content"], '{"tool": "done", "say": "开机三小时了。"}')

    def test_agent_samples_under_an_older_prompt_train_under_the_current_one(self) -> None:
        from learning.drill import AGENT_SYSTEM, AGENT_SYSTEM_OLD
        from learning.lora_prep import route
        msgs = [{"role": "system", "content": AGENT_SYSTEM_OLD[0]}, {"role": "user", "content": "任务：温度\n下一步？"},
                {"role": "assistant", "content": '{"tool": "shell", "cmd": "cat /sys/class/thermal/thermal_zone0/temp"}'}]
        self.assertEqual(route(msgs), "agent")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "lora_data.jsonl").write_text(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n", encoding="utf-8")
            build(Teacher({}), p, workers=1)
            m = json.loads((p / "lora_train.jsonl").read_text(encoding="utf-8"))["messages"]
            self.assertEqual(m[0]["content"], AGENT_SYSTEM)
            self.assertEqual(m[2]["content"], msgs[2]["content"])

    def test_coding_prompt_samples_stay(self) -> None:
        from learning.school import CODING_SYSTEM
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            rec = {"messages": [{"role": "system", "content": CODING_SYSTEM}, {"role": "user", "content": "求和"},
                                {"role": "assistant", "content": "sum(xs)"}]}
            (p / "lora_data.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            t = Teacher({})
            self.assertEqual(build(t, p, workers=1)["coding"], 1)
            self.assertEqual(t.calls, 0)


if __name__ == "__main__":
    unittest.main()
