"""Projects (learning.project), their sandbox (learning.projbox), the internet-only proxy (learning.egress_proxy)
and automatic delivery with rollback (learning.deliver)."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import deliver, egress_proxy, project, projbox  # noqa: E402


class ProxyTests(unittest.TestCase):
    def test_only_public_addresses(self) -> None:
        self.assertTrue(egress_proxy.public_ip("93.184.216.34"))
        self.assertTrue(egress_proxy.public_ip("198.18.0.7"))          # the LAN DNS's fake-ip for public names
        for ip in ("192.168.100.4", "10.0.0.177", "127.0.0.1", "169.254.1.1", "::1", "0.0.0.0"):
            self.assertFalse(egress_proxy.public_ip(ip), ip)

    def test_resolve_refuses_lan_and_odd_ports(self) -> None:
        def fake(host, port, type=0):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ({"pypi.org": "151.101.0.223", "gpu": "192.168.100.4"}[host], port))]
        with mock.patch.object(socket, "getaddrinfo", fake):
            self.assertEqual(egress_proxy.resolve_public("pypi.org", 443), ("151.101.0.223", ""))
            self.assertEqual(egress_proxy.resolve_public("gpu", 443)[0], "")
            self.assertEqual(egress_proxy.resolve_public("pypi.org", 22)[0], "")

    def test_request_heads(self) -> None:
        self.assertEqual(egress_proxy.parse_head(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org\r\n\r\n")[:3],
                         ("CONNECT", "pypi.org", 443))
        m, host, port, up = egress_proxy.parse_head(b"GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\n"
                                                    b"Proxy-Connection: keep-alive\r\n\r\n")
        self.assertEqual((m, host, port), ("GET", "example.com", 80))
        self.assertTrue(up.startswith(b"GET /a?b=1 HTTP/1.1\r\n"))
        self.assertNotIn(b"Proxy-Connection", up)
        with self.assertRaises(ValueError):
            egress_proxy.parse_head(b"GET /relative HTTP/1.1\r\n\r\n")


class SandboxTests(unittest.TestCase):
    def test_layout(self) -> None:
        a = projbox.argv(Path("/userdata/projects/p1"), "make test")
        s = " ".join(a)
        self.assertIn("--ro-bind / /", s)
        self.assertIn("--bind /userdata/projects/p1/work /home/work", s)
        self.assertIn("--unshare-all", a)                               # no network, own pids/users
        self.assertNotIn("--share-net", a)
        self.assertIn("https_proxy", a)
        self.assertEqual(a[-1], "make test")
        self.assertEqual(a[a.index("--chdir") + 1], "/home/work")


class FakeWorld:
    """A planner that follows a script; a sandbox that runs nothing."""

    def __init__(self, script: list[dict], test_code: int = 0) -> None:
        self.script, self.test_code, self.ran, self.delivered = list(script), test_code, [], []
        self.user_msgs: list[str] = []

    def ctx(self, spent: float = 0.0, peak=lambda: False) -> project.Ctx:
        def plan(system, user):
            self.user_msgs.append(user)
            return json.dumps(self.script.pop(0) if self.script else {"tool": "give_up", "why": "脚本走完了"}, ensure_ascii=False)

        def sandbox(pdir, cmd, timeout):
            self.ran.append(cmd)
            return (self.test_code, "tests ok") if cmd == "python3 test_hello.py" else (0, f"ran {cmd}")
        return project.Ctx(plan=plan, consult=lambda s, q: "用 argparse。", sandbox=sandbox,
                           search=lambda q: (True, "1. result"), fetch=lambda u, s: (True, "page"),
                           deliver=lambda files, why, slug: self.delivered.append(files) or {"ok": True, "stage": "done"},
                           spent=lambda slug: spent, peak=peak, sleep=lambda s: None)


class ProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.slug = project.create("写一个 hello 程序", root=self.root)
        self.pdir = self.root / self.slug

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_project_runs_to_done_only_after_its_test_passes(self) -> None:
        w = FakeWorld([
            {"tool": "note", "text": "计划：写 hello.py 和测试"},
            {"tool": "done", "summary": "做完了"},                                # no test yet: refused
            {"tool": "write", "path": "hello.py", "content": "print('hello')\n"},
            {"tool": "write", "path": "hello.py", "content": "print('hello, world')\n"},
            {"tool": "search", "query": "python argparse"},
            {"tool": "ask_ai", "question": "怎么写测试"},
            {"tool": "set_test", "cmd": "python3 test_hello.py"},
            {"tool": "done", "summary": "hello 程序写好了"},
        ])
        st = project.run(self.pdir, w.ctx())
        self.assertEqual((st["status"], st["steps"]), ("done", 8))
        self.assertEqual((self.pdir / "work" / "hello.py").read_text(encoding="utf-8"), "print('hello, world')\n")
        # versions/<n>/ keeps what a file was just before step n changed it
        self.assertEqual((self.pdir / "versions" / "4" / "hello.py").read_text(encoding="utf-8"), "print('hello')\n")
        log = project.steps(self.pdir)
        self.assertEqual(log[1]["status"], "fail")                             # done before set_test
        self.assertIn("计划", (self.pdir / "notes.md").read_text(encoding="utf-8"))
        self.assertIn("计划：写 hello.py", w.user_msgs[-1])                       # notes reach the planner
        self.assertIn("python3 test_hello.py", w.ran)

    def test_failed_acceptance_keeps_going(self) -> None:
        w = FakeWorld([{"tool": "set_test", "cmd": "python3 test_hello.py"}, {"tool": "done", "summary": "x"}], test_code=1)
        st = project.run(self.pdir, w.ctx())
        self.assertEqual(st["status"], "gave_up")                              # script ran out -> give_up
        self.assertIn("验收没通过", project.steps(self.pdir)[1]["result"])

    def test_writes_stay_inside_the_workspace(self) -> None:
        w = FakeWorld([{"tool": "write", "path": "../../etc/passwd", "content": "x"},
                       {"tool": "write", "path": "/home/work/ok.txt", "content": "y"}])
        project.run(self.pdir, w.ctx())
        self.assertEqual(project.steps(self.pdir)[0]["status"], "fail")
        self.assertTrue((self.pdir / "work" / "ok.txt").is_file())

    def test_budget_stops_it(self) -> None:
        st = project.run(self.pdir, FakeWorld([{"tool": "note", "text": "x"}]).ctx(spent=10.5))
        self.assertEqual((st["status"], st["steps"]), ("over_budget", 0))

    def test_peak_hours_wait_without_calling_the_teacher(self) -> None:
        calls = iter([True, True, False])
        w = FakeWorld([{"tool": "give_up", "why": "测试"}])
        st = project.run(self.pdir, w.ctx(peak=lambda: next(calls)))
        self.assertEqual((st["status"], len(w.user_msgs)), ("gave_up", 1))

    def test_deliver_goes_through_the_delivery_gate(self) -> None:
        (self.pdir / "work" / "skill.py").write_text("x = 1\n", encoding="utf-8")
        w = FakeWorld([{"tool": "deliver", "files": [{"src": "skill.py", "dst": "skills_extra/skill.py"}], "why": "新技能"},
                       {"tool": "deliver", "files": [{"src": "nope.py", "dst": "x.py"}], "why": "x"}])
        project.run(self.pdir, w.ctx())
        self.assertEqual(w.delivered[0][0]["dst"], "skills_extra/skill.py")
        self.assertEqual(project.steps(self.pdir)[1]["status"], "fail")

    def test_retrospective_and_notice(self) -> None:
        st = project.run(self.pdir, FakeWorld([{"tool": "give_up", "why": "没有 pip"}]).ctx())
        ctx = project.Ctx(plan=lambda s, u: json.dumps({"learned": ["先查板子上有什么工具"], "missing": ["不会装 pip"], "say": "没做成，缺 pip。"}, ensure_ascii=False),
                          consult=None, sandbox=None, search=None, fetch=None, deliver=None, spent=lambda s: 0)
        learning = self.root / "learning"
        out = project.retrospective(self.pdir, st, ctx, learning)
        self.assertEqual(out["missing"], ["不会装 pip"])
        self.assertIn("不会装 pip", project.review_lines(learning))
        project.notify("主人，项目没做成。", self.root / "notices")
        self.assertEqual(len(list((self.root / "notices").glob("*.txt"))), 1)


class VoiceCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.started: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def cmd(self, text, active=lambda slug: True):
        return project.project_command(text, root=self.root, start=lambda slug: self.started.append(slug) or "",
                                       active=active, is_peak=lambda: False)

    def test_start_status_and_one_at_a_time(self) -> None:
        self.assertIsNone(self.cmd("今天天气怎么样"))
        self.assertIsNone(self.cmd("开个项目"))                                   # no goal
        say = self.cmd("星期一，开个项目：写一个把日志压缩归档的小工具")()
        self.assertTrue(say.startswith("好，我开个项目：写一个把日志压缩归档的小工具"))
        self.assertEqual(project.load_state(self.root / self.started[0])["goal"], "写一个把日志压缩归档的小工具")
        self.assertIn("还有一个项目没做完", self.cmd("再开个项目：做个天气播报")())
        self.assertIn("正在做", self.cmd("项目做得怎么样了")())

    def test_a_dead_project_does_not_block_the_next(self) -> None:
        self.cmd("开个项目：第一个目标是这个")()
        say = self.cmd("开个项目：第二个目标是那个", active=lambda slug: False)()
        self.assertTrue(say.startswith("好，"))
        self.assertEqual(project.load_state(self.root / self.started[0])["status"], "stopped")


class DeliverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "agent"
        (self.root / "orchestrator").mkdir(parents=True)
        (self.root / "orchestrator" / "feature.py").write_text("OLD\n", encoding="utf-8")
        self.src = Path(self._tmp.name) / "work"
        self.src.mkdir()
        (self.src / "feature.py").write_text("NEW\n", encoding="utf-8")
        (self.src / "extra.py").write_text("EXTRA\n", encoding="utf-8")
        self.learning = Path(self._tmp.name) / "learning"
        self.learning.mkdir()
        self.saved: dict[str, str] = {}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def snapshot(self, label, reason):
        self.saved = {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8") for p in self.root.rglob("*.py")}
        return "snap-1"

    def restore(self, sid):
        for rel, text in self.saved.items():
            (self.root / rel).write_text(text, encoding="utf-8")

    def go(self, files, *, tests_after=frozenset(), exam=(30, 30), quiet=True):
        runs = iter([{"tests.old"}, set(tests_after) | {"tests.old"}])
        scores = iter(exam)
        restarted = []
        res = deliver.deliver(files, "新功能", project="p1", learning_dir=self.learning, root=self.root,
                              exam_score=lambda: next(scores), wait_quiet=lambda: quiet, tests=lambda: next(runs),
                              snapshot=self.snapshot, restore=self.restore,
                              restart_units=lambda units: restarted.append(list(units)) or "")
        return res, restarted

    def files(self):
        return [{"src": self.src / "feature.py", "dst": "orchestrator/feature.py"},
                {"src": self.src / "extra.py", "dst": "orchestrator/extra.py"}]

    def test_good_change_stays_and_restarts_what_it_touched(self) -> None:
        res, restarted = self.go(self.files())
        self.assertTrue(res["ok"])
        self.assertEqual(restarted, [["r1-orchestrator"]])
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "NEW\n")

    def test_new_test_failure_rolls_everything_back(self) -> None:
        res, _ = self.go(self.files(), tests_after={"tests.test_x.T.test_y"})
        self.assertFalse(res["ok"])
        self.assertTrue(res["rolled_back"])
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")
        self.assertFalse((self.root / "orchestrator" / "extra.py").exists())     # new files removed too

    def test_exam_drop_rolls_back(self) -> None:
        res, _ = self.go(self.files(), exam=(30, 25))
        self.assertEqual((res["ok"], res["stage"]), (False, "exam"))
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")

    def test_protected_files_are_refused_before_anything_happens(self) -> None:
        for dst in ("learning/agent.py", "learning/exam.py", "config/agent.yaml", "ops/snapshot.py", "../etc/x", "run/x"):
            res, _ = self.go([{"src": self.src / "feature.py", "dst": dst}])
            self.assertEqual((res["ok"], res["stage"]), (False, "check"), dst)

    def test_waits_for_the_owner_to_be_quiet(self) -> None:
        res, restarted = self.go(self.files(), quiet=False)
        self.assertEqual((res["ok"], res["stage"], restarted), (False, "wait", []))
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")

    def test_log(self) -> None:
        self.go(self.files())
        rows = [json.loads(x) for x in (self.learning / deliver.LOG_FILE).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["snapshot"], "snap-1")


if __name__ == "__main__":
    unittest.main()
