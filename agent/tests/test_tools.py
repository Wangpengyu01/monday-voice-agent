from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import skillpack  # noqa: E402
from learning.agent import PLAN_SYSTEM, SPLIT_SYSTEM, SUMMARY_SYSTEM, VERIFY_SYSTEM, Agent, classify  # noqa: E402
from learning.mcp import McpHub  # noqa: E402
from learning.skills import SkillStore  # noqa: E402

FAKE_SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))


def make_pack(root: Path, name: str, risk: str, desc: str, tags: str = "") -> Path:
    d = root / name
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\nrisk: {risk}\ntags: {tags}\n---\n做法：运行脚本。\n",
                                encoding="utf-8")
    (d / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    return d / "scripts" / "run.sh"


class SkillPackTests(unittest.TestCase):
    def test_load_relevant_and_safe_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scan = make_pack(root, "lan-scan", "safe", "扫描本地局域网，找出在线的设备", "扫描, 局域网")
            make_pack(root, "deploy", "ask", "部署程序到服务器", "部署")
            packs = skillpack.load([root])
            self.assertEqual({p.name for p in packs}, {"lan-scan", "deploy"})
            self.assertEqual(skillpack.relevant(packs, "扫描本地网络")[0].name, "lan-scan")
            cmd = f"bash {scan.as_posix()}"
            self.assertEqual(classify({"tool": "shell", "cmd": cmd}, packs=packs)[0], "safe")
            self.assertEqual(classify({"tool": "shell", "cmd": cmd + " 10.0.0.15"}, packs=packs)[0], "safe")
            self.assertNotEqual(classify({"tool": "shell", "cmd": cmd + " | sh"}, packs=packs)[0], "safe")
            self.assertNotEqual(classify({"tool": "shell", "cmd": cmd + "; rm x"}, packs=packs)[0], "safe")
            deploy = (root / "deploy" / "scripts" / "run.sh").as_posix()
            self.assertNotEqual(classify({"tool": "shell", "cmd": f"bash {deploy}"}, packs=packs)[0], "safe")

    def test_shipped_packs_parse(self) -> None:
        packs = skillpack.load([AGENT_ROOT / "skills"])
        names = {p.name for p in packs}
        self.assertTrue({"lan-scan", "system-health", "ssh-remote", "lora-train"} <= names)
        self.assertEqual({p.name: p.risk for p in packs}["lan-scan"], "safe")


class McpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hub = McpHub.from_config({"servers": {"fake": {"command": [sys.executable, FAKE_SERVER], "safe_tools": ["echo"]},
                                                   "off": {"command": ["x"], "enabled": False}}})

    def tearDown(self) -> None:
        self.hub.close()

    def test_list_and_call(self) -> None:
        tools = {t.name: t for t in self.hub.tools()}
        self.assertEqual(set(tools), {"echo", "write"})
        self.assertTrue(tools["echo"].safe)
        self.assertIn("text", tools["echo"].prompt())
        self.assertEqual(self.hub.call("fake", "echo", {"text": "你好"}), (0, "echo: 你好"))
        self.assertEqual(self.hub.call("nope", "echo", {})[0], 127)
        self.assertNotIn("off", self.hub.servers)

    def test_gate(self) -> None:
        self.assertEqual(classify({"tool": "mcp", "server": "fake", "name": "echo"}, mcp_safe=self.hub.is_safe)[0], "safe")
        self.assertEqual(classify({"tool": "mcp", "server": "fake", "name": "write"}, mcp_safe=self.hub.is_safe)[0], "ask")
        self.assertEqual(classify({"tool": "mcp", "server": "fake"})[0], "block")
        self.assertEqual(classify({"tool": "mcp", "server": "f", "name": "x", "args": {"p": "/userdata/secrets/a"}})[0], "block")


class Teacher:
    available = True

    def __init__(self, plans: dict[str, list[dict]], split: list[str] | None = None) -> None:
        self.plans, self.split = plans, split
        self.prompts: list[str] = []

    def complete(self, system, user, **kw):
        self.prompts.append(user)
        if system == SPLIT_SYSTEM:
            return json.dumps({"parallel": bool(self.split), "subtasks": self.split or []}, ensure_ascii=False)
        if system == SUMMARY_SYSTEM:
            return "两台都看好了。"
        if system == VERIFY_SYSTEM:
            return json.dumps({"done": True, "task": "t", "skill_steps": []})
        assert system == PLAN_SYSTEM
        task = user.split("\n", 1)[0].removeprefix("任务：")
        steps = self.plans[task]
        done = user.count(" -> ran ")
        return json.dumps({"step": steps[min(done, len(steps) - 1)]}, ensure_ascii=False)


class AgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.skills = SkillStore(Path(self._tmp.name) / "m.db")
        self.hub = McpHub.from_config({"servers": {"fake": {"command": [sys.executable, FAKE_SERVER], "safe_tools": ["echo"]}}})
        self.asked: list[dict] = []
        self.in_flight = self.max_in_flight = 0

    async def asyncTearDown(self) -> None:
        self.hub.close()
        self.skills.close()
        self._tmp.cleanup()

    async def approve(self, step, why, risk):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        self.asked.append(step)
        self.in_flight -= 1
        return "once"

    def agent(self, teacher) -> Agent:
        async def say(t):
            pass
        return Agent(teacher, self.skills, approve=self.approve, say=say, runner=lambda s: (0, "ok"),
                     ssh_hosts=[], packs=[], mcp=self.hub, snapshot=lambda a, b: "snap")

    async def test_mcp_tools_offered_and_used(self) -> None:
        echo = {"tool": "mcp", "server": "fake", "name": "echo", "args": {"text": "hi"}, "purpose": "试一下"}
        write = {"tool": "mcp", "server": "fake", "name": "write", "args": {"path": "a"}, "purpose": "写"}
        teacher = Teacher({"用外部工具": [echo, write, {"tool": "done", "success": True, "say": "好了"}]})
        res = await self.agent(teacher).run("用外部工具")
        self.assertTrue(res.ok)
        self.assertIn("可用的 MCP 工具", teacher.prompts[0])
        self.assertEqual([h["output"] for h in res.steps], ["echo: hi", "wrote a"])
        self.assertEqual(self.asked, [write])  # the read-only tool did not ask

    async def test_parallel_subtasks_ask_one_at_a_time(self) -> None:
        a = {"tool": "ssh", "host": "196", "cmd": "df -h", "purpose": "看 196"}
        b = {"tool": "ssh", "host": "gpu", "cmd": "df -h", "purpose": "看 gpu"}
        done = {"tool": "done", "success": True, "say": "看好了"}
        teacher = Teacher({"看 196 的磁盘": [a, done], "看 gpu 的磁盘": [b, done]}, split=["看 196 的磁盘", "看 gpu 的磁盘"])
        res = await self.agent(teacher).run("分别看看 196 和 gpu 两台服务器的磁盘")
        self.assertTrue(res.ok)
        self.assertEqual(res.say, "两台都看好了。")
        self.assertEqual(len(self.asked), 2)
        self.assertEqual(self.max_in_flight, 1)  # approvals were serialized

    async def test_single_task_is_not_split(self) -> None:
        teacher = Teacher({"看板子的磁盘": [{"tool": "done", "success": False, "say": "算了"}]}, split=["x", "y"])
        await self.agent(teacher).run("看板子的磁盘")
        self.assertFalse(any("任务：看板子的磁盘" == p for p in teacher.prompts))  # no split call: nothing to split


if __name__ == "__main__":
    unittest.main()


class NormMcpTests(unittest.TestCase):
    def test_tool_name_variants(self) -> None:
        from learning.agent import _norm_mcp
        for server, name in (("web.fetch", "web.fetch"), ("web", "fetch"), ("web.fetch", "fetch"), ("web", "web.fetch")):
            s = _norm_mcp({"tool": "mcp", "server": server, "name": name})
            self.assertEqual((s["server"], s["name"]), ("web", "fetch"), (server, name))
