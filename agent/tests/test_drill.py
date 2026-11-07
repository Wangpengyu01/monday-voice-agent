from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import drill  # noqa: E402
from learning.lora_prep import route  # noqa: E402
from learning.school import CODING_SYSTEM  # noqa: E402

GOOD = "def add(a, b):\n    return a + b"


class CodeTeacher:
    def complete(self, system, user, **kw):
        assert system == drill.CODE_TASK_SYSTEM
        return json.dumps({"tasks": [
            {"q": "写函数 add(a, b) 返回两数之和", "tests": "assert add(1, 2) == 3\nassert add(-1, 1) == 0", "solution": GOOD},
            {"q": "写函数 neg(x) 返回相反数", "tests": "assert neg(2) == -2", "solution": "def neg(x):\n    return -x"},
            {"q": "坏题", "tests": "assert bad() == 1", "solution": "def bad():\n    return 2"},  # teacher is wrong: dropped
        ]}, ensure_ascii=False)


class AgentTeacher:
    def __init__(self) -> None:
        self.plans = 0

    def complete(self, system, user, **kw):
        if system == drill.AGENT_TASK_SYSTEM:
            return json.dumps({"tasks": ["板子开机多久了"]}, ensure_ascii=False)
        if system == drill.AGENT_SYSTEM:  # the teacher's demonstration
            self.plans += 1
            if "已执行" not in user:
                return json.dumps({"tool": "shell", "cmd": "uptime -p"})
            return json.dumps({"tool": "done", "say": "开机三小时了。"}, ensure_ascii=False)
        assert system == drill.AGENT_JUDGE_SYSTEM
        return json.dumps({"ok": "三小时" in user and "uptime" in user, "why": "t"})


def rows(d: Path) -> list[dict]:
    f = d / "lora_data.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []


class DrillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_coding_keeps_own_passing_code_and_teacher_fix_for_failures(self) -> None:
        answers = {"写函数 add(a, b) 返回两数之和": f"```python\n{GOOD}\n```", "写函数 neg(x) 返回相反数": "def neg(x):\n    return x"}
        s = drill.coding_drill(teacher=CodeTeacher(), learning_dir=self.dir, student=answers.get, idle=lambda: True)
        self.assertEqual((s["passed"], s["total"]), (1, 2))   # broken teacher task never reaches the student
        data = {r["messages"][1]["content"]: r["messages"][2]["content"] for r in rows(self.dir)}
        self.assertEqual(data["写函数 add(a, b) 返回两数之和"], GOOD)
        self.assertEqual(data["写函数 neg(x) 返回相反数"], "def neg(x):\n    return -x")
        self.assertTrue(all(r["messages"][0]["content"] == CODING_SYSTEM for r in rows(self.dir)))
        self.assertTrue(all(route(r["messages"]) == "coding" for r in rows(self.dir)))

    def run_agent(self, local_plan, execute=lambda s: (0, "up 3 hours")):
        return drill.agent_drill(teacher=(t := AgentTeacher()), learning_dir=self.dir, local_plan=local_plan,
                                 idle=lambda: True, execute=execute), t

    def test_agent_pass_logs_every_step_of_its_own_run(self) -> None:
        def plan(ctx):
            return '{"tool":"shell","cmd":"uptime -p"}' if "已执行" not in ctx else '<think></think>{"tool":"done","say":"开机三小时了。"}'
        s, t = self.run_agent(plan)
        self.assertEqual((s["passed"], s["total"], t.plans), (1, 1, 0))
        data = rows(self.dir)
        self.assertEqual(len(data), 2)
        self.assertIn("up 3 hours", data[1]["messages"][1]["content"])  # the second step sees the first's output
        self.assertEqual(json.loads(data[1]["messages"][2]["content"])["tool"], "done")
        self.assertTrue(all(route(r["messages"]) == "agent" for r in data))

    def test_agent_unsafe_command_is_not_run_and_teacher_demonstrates(self) -> None:
        ran = []

        def execute(step):
            ran.append(step["cmd"])
            return 0, "up 3 hours"
        s, t = self.run_agent(lambda ctx: '{"tool":"shell","cmd":"sudo reboot"}', execute=execute)
        self.assertEqual(s["passed"], 0)
        self.assertTrue(s["results"][0]["learned"])
        self.assertEqual(ran, ["uptime -p"])                           # only the teacher's safe step ran
        self.assertEqual([json.loads(r["messages"][2]["content"])["tool"] for r in rows(self.dir)], ["shell", "done"])

    def test_agent_repeated_command_is_not_rerun_and_the_budget_is_shown(self) -> None:
        ran, seen = [], []

        def execute(step):
            ran.append(step["cmd"])
            return 0, "up 3 hours"

        def plan(ctx):
            seen.append(ctx)
            return '{"tool":"shell","cmd":"uptime -p"}' if "必须用 done" not in ctx else '{"tool":"done","say":"开机三小时了。"}'
        run = drill.run_agent("开机多久了", plan, execute=execute, classify=lambda s: ("safe", ""))
        self.assertEqual(ran, ["uptime -p"])                      # the repeats were not executed again
        self.assertIn(drill.REPEATED, seen[-1])
        self.assertIn("还能执行 3 条命令", seen[1])
        self.assertEqual((run["error"], run["say"]), ("", "开机三小时了。"))  # told to answer, it answers

    def test_old_agent_contexts_are_upgraded_to_the_budget_wording(self) -> None:
        old1 = "任务：开机多久了\n已执行：\n1. uptime -p -> 返回码 0\nup 3 hours\n下一步？"
        new1 = drill.upgrade_context(old1)
        self.assertTrue(new1.endswith("还能执行 3 条命令。下一步？"))
        self.assertEqual(drill.upgrade_context(new1), new1)              # idempotent
        hist = [{"cmd": f"c{i}", "code": 0, "output": "x"} for i in range(4)]
        old4 = drill.agent_context("t", hist).replace(drill.LAST_STEP, "\n下一步？")
        self.assertEqual(drill.upgrade_context(old4), drill.agent_context("t", hist))
        first = "任务：开机多久了\n下一步？"
        self.assertEqual(drill.upgrade_context(first), first)            # first step: same in both wordings

    def test_command_output_is_squeezed_and_the_newest_gets_more_room(self) -> None:
        self.assertEqual(drill.squeeze("PID   USER    %CPU\n\n  1   root    0.0  \n"), "PID USER %CPU\n1 root 0.0")
        hist = [{"cmd": "a", "code": 0, "output": "o" * 500}, {"cmd": "b", "code": 0, "output": "n" * 500}]
        ctx = drill.agent_context("t", hist)
        self.assertIn("o" * drill.AGENT_OLDER_CHARS + "…", ctx)
        self.assertNotIn("o" * (drill.AGENT_OLDER_CHARS + 1), ctx)
        self.assertIn("n" * drill.AGENT_LATEST_CHARS + "…", ctx)

    def test_task_requests_rotate_phrasing_styles(self) -> None:
        reqs = {drill.agent_task_request(2, ["帮我看看板子上现在磁盘"]) for _ in range(40)}
        self.assertGreater(len(reqs), 2)
        self.assertTrue(all(any(s in r for s in drill.AGENT_TASK_STYLES) for r in reqs))

    def test_run_agent_sends_ask_level_shell_steps_to_the_sandbox(self) -> None:
        from learning.agent import classify
        boxed, ran = [], []
        plans = iter(['{"tool":"shell","cmd":"ps aux | awk \'{print $1}\' | sort | uniq -c"}', '{"tool":"done","say":"三个用户在跑程序。"}'])
        run = drill.run_agent("谁在跑程序", lambda ctx: next(plans), classify=classify,
                              execute=lambda s: (ran.append(s), (0, "x"))[1], sandbox=lambda s: (boxed.append(s), (0, "3 root"))[1])
        self.assertEqual((run["error"], len(boxed), ran), ("", 1, []))
        no_box = drill.run_agent("谁在跑程序", lambda ctx: '{"tool":"shell","cmd":"ps aux | awk \'{print $1}\'"}',
                                 classify=classify, execute=lambda s: (0, "x"))
        self.assertIn("命令不在只读范围", no_box["error"])   # without a sandbox: as before

    def test_agent_unspeakable_answer_fails(self) -> None:
        def plan(ctx):
            return '{"tool":"shell","cmd":"uptime -p"}' if "已执行" not in ctx else '{"tool":"done","say":"用 uptime -p 看"}'
        s, _ = self.run_agent(plan)
        self.assertEqual(s["passed"], 0)
        self.assertIn("念了命令", s["results"][0]["why"])


if __name__ == "__main__":
    unittest.main()
