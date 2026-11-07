from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.agent import PLAN_SYSTEM, VERIFY_SYSTEM, Agent, known_ssh_hosts  # noqa: E402
from learning.credentials import MASK, add_host, extract_credentials, install_log_redaction, mask_secrets  # noqa: E402
from learning.skills import SkillStore  # noqa: E402


class ExtractTests(unittest.TestCase):
    def test_forms(self) -> None:
        cases = {
            "sh到10.0.0.237账号和密码都是pi": ("10.0.0.237", "pi", "pi"),
            "SSH 到 10.0.0.237，用户名 pi 密码 raspberry1122": ("10.0.0.237", "pi", "raspberry1122"),
            "连到192.168.1.50 账号是root，密码是Abc#123": ("192.168.1.50", "root", "Abc#123"),
            "ssh到10点0点0点237 账号和密码都是pi": ("10.0.0.237", "pi", "pi"),
        }
        for text, (host, user, pw) in cases.items():
            masked, c = extract_credentials(text)
            self.assertEqual((c.host, c.user, c.password), (host, user, pw), text)
            self.assertIn(MASK, masked)
            self.assertNotIn(pw if pw != "pi" else "都是pi", masked)
            if pw != user:
                self.assertNotIn(pw, repr(c))

    def test_no_password_no_credentials(self) -> None:
        self.assertEqual(extract_credentials("帮我SSH到196看磁盘"), ("帮我SSH到196看磁盘", None))

    def test_repeated_characters_survive(self) -> None:
        # normalization collapses repeats ("1122" -> "12"); extraction runs before it
        _, c = extract_credentials("SSH到10.0.0.9 用户 pi 密码 aa1122")
        self.assertEqual(c.password, "aa1122")

    def test_mask_and_log_redaction(self) -> None:
        self.assertEqual(mask_secrets("账号和密码都是pi"), "账号和密码都是" + MASK)
        self.assertNotIn("hunter2", mask_secrets("[asr] partial: 密码是 hunter2"))
        install_log_redaction()
        buf = io.StringIO()
        log = logging.getLogger("test.redact")
        h = logging.StreamHandler(buf)
        log.addHandler(h)
        log.setLevel(logging.INFO)
        try:
            log.info("[asr] final: %s", "ssh到10.0.0.237 账号和密码都是pi")
        finally:
            log.removeHandler(h)
        self.assertNotIn("都是pi", buf.getvalue())
        self.assertIn(MASK, buf.getvalue())

    def test_add_host_writes_alias_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config"
            add_host("pi-237", "10.0.0.237", "pi", config=cfg)
            add_host("pi-237", "10.0.0.237", "pi", config=cfg)
            self.assertEqual(cfg.read_text(encoding="utf-8").count("Host pi-237"), 1)
            self.assertEqual(known_ssh_hosts(cfg), ["pi-237（pi@10.0.0.237）"])


class Teacher:
    available = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system, user, **kw):
        self.prompts.append(user)
        if system == PLAN_SYSTEM:
            if "已经执行的步骤" not in user or user.count("登录 pi-237 执行") == 0:
                return json.dumps({"step": {"tool": "ssh", "host": "pi-237", "cmd": "hostname", "purpose": "确认登录"}})
            return json.dumps({"step": {"tool": "done", "success": True, "say": "连上了。"}})
        assert system == VERIFY_SYSTEM
        return json.dumps({"done": True, "task": "SSH 登录主机", "skill_steps": [
            {"tool": "ssh_key_setup", "host": "pi@10.0.0.237"}, {"tool": "ssh", "host": "pi-237", "cmd": "hostname", "purpose": "确认登录"}]})


class AgentWithPasswordTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.skills = SkillStore(Path(self._tmp.name) / "m.db")
        self.said, self.asked, self.installed = [], [], []
        self.allow = True

    async def asyncTearDown(self) -> None:
        self.skills.close()
        self._tmp.cleanup()

    async def run_task(self, text: str, teacher: Teacher):
        masked, creds = extract_credentials(text)

        async def approve(step, why, risk="ask"):
            self.asked.append(step)
            return self.allow

        async def say(t):
            self.said.append(t)

        def install(c):
            self.installed.append((c.host, c.user, c.password))
            return True, "已经把公钥装好了"

        agent = Agent(teacher, self.skills, approve=approve, say=say, runner=lambda s: (0, "raspberrypi"),
                      ssh_hosts=[], install_key=install)
        agent.ssh_hosts = []
        import learning.agent as mod
        real_add, real_known = mod.add_host, mod.known_ssh_hosts
        mod.add_host = lambda *a, **k: None
        mod.known_ssh_hosts = lambda *a, **k: ["pi-237（pi@10.0.0.237）"]
        try:
            res = await agent.run(masked, creds)
        finally:
            mod.add_host, mod.known_ssh_hosts = real_add, real_known
        return res, creds

    async def test_password_used_once_never_shown(self) -> None:
        teacher = Teacher()
        res, creds = await self.run_task("sh到10.0.0.237账号和密码都是pi，看看主机名", teacher)
        self.assertTrue(res.ok)
        self.assertEqual(self.installed, [("10.0.0.237", "pi", "pi")])
        self.assertEqual(creds.password, "")                      # dropped after use
        self.assertEqual(self.asked[0]["tool"], "ssh_key_setup")  # key setup asks first
        everything = json.dumps([teacher.prompts, self.said, res.steps, [s.steps for s in self.skills.all()]], ensure_ascii=False)
        self.assertNotIn("都是pi", everything)
        self.assertNotIn('"password"', everything)
        # the one-off setup step is not part of the learned skill
        self.assertEqual([s["tool"] for s in self.skills.all()[0].steps], ["ssh"])

    async def test_denied_setup_never_uses_the_password(self) -> None:
        self.allow = False
        res, _ = await self.run_task("SSH 到 10.0.0.237，用户名 pi 密码 secret9", Teacher())
        self.assertFalse(res.ok)
        self.assertEqual(self.installed, [])


if __name__ == "__main__":
    unittest.main()
