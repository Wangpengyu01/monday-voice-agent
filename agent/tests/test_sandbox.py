from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import sandbox  # noqa: E402


class SandboxRuleTests(unittest.TestCase):
    def test_programs(self) -> None:
        self.assertEqual(sandbox.programs("which a 2>&1; echo ---; systemctl is-active b 2>&1"), ["which", "echo", "systemctl"])
        self.assertEqual(sandbox.programs("LANG=C timeout 5 ss -tlnp | awk '{print $4}'"), ["ss", "awk"])
        self.assertIsNone(sandbox.programs("echo $(id)"))

    def test_network_tools_are_never_sandboxed(self) -> None:
        for cmd in ("curl -s http://x", "wget x", "ssh 196 df", "git clone x", "pip install y", "cat a | nc h 1"):
            self.assertFalse(sandbox.eligible({"tool": "shell", "cmd": cmd}), cmd)
        self.assertTrue(sandbox.eligible({"tool": "shell", "cmd": "ps aux | awk '{print $1}' | sort | uniq -c"}))
        self.assertFalse(sandbox.eligible({"tool": "ssh", "host": "196", "cmd": "df"}))

    def test_host_network_only_for_network_viewers(self) -> None:
        self.assertTrue(sandbox.needs_host_net("ss -tlnp | awk '{print $4}' | sort -u"))
        self.assertFalse(sandbox.needs_host_net("ss -tlnp | python3 -c 'import socket'"))
        self.assertFalse(sandbox.needs_host_net("ss -tlnp | awk '{system(\"id\")}'"))
        self.assertFalse(sandbox.needs_host_net("cat /proc/loadavg"))

    def test_dbus_only_for_viewers_without_changing_subcommands(self) -> None:
        self.assertTrue(sandbox.needs_dbus("systemctl status r1-tts-daemon --no-pager 2>/dev/null | head -5"))
        self.assertTrue(sandbox.needs_dbus("which avahi-browse; systemctl is-active avahi-daemon"))
        self.assertTrue(sandbox.needs_dbus("avahi-browse -rtp _airplay._tcp 2>/dev/null | grep ^= | head -n 1"))
        for cmd in ("systemctl restart r1-tts-daemon", "nmcli con down wifi", "busctl call org.x / a b",
                    "systemctl status x | python3 -c y", "systemctl status x | xargs systemctl stop"):
            self.assertFalse(sandbox.needs_dbus(cmd), cmd)

    def test_argv_hides_private_dirs_and_the_bus(self) -> None:
        a = " ".join(sandbox.argv("ls", net=False, dbus=False))
        self.assertIn("--unshare-net", a)
        self.assertIn("--unshare-pid", a)
        self.assertIn("--clearenv", a)
        if os.path.isdir("/run/dbus"):
            self.assertIn("--tmpfs /run/dbus", a)
        self.assertNotIn("--unshare-net", " ".join(sandbox.argv("ss", net=True)))


@unittest.skipUnless(sandbox.available(), "needs bwrap (the board)")
class SandboxRunTests(unittest.TestCase):
    def run_cmd(self, cmd: str) -> tuple[int, str]:
        return sandbox.run_step({"tool": "shell", "cmd": cmd})

    def test_reads_work_and_writes_fail(self) -> None:
        code, out = self.run_cmd("cat /proc/loadavg | wc -l")
        self.assertEqual((code, out), (0, "1"))
        code, out = self.run_cmd("touch /userdata/files/.sandbox_test")
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists("/userdata/files/.sandbox_test"))

    def test_private_dirs_are_empty_and_env_is_clean(self) -> None:
        self.assertEqual(self.run_cmd("ls -A /userdata/agent/run /userdata/secrets /home | grep -v ':$' | grep -c .")[1], "0")
        self.assertEqual(self.run_cmd("env | grep -ci -E 'key|token|secret'")[1], "0")

    def test_no_network_no_sudo_no_signals(self) -> None:
        self.assertNotEqual(self.run_cmd("python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 53), 2)'")[0], 0)
        self.assertNotEqual(self.run_cmd("sudo -n true")[0], 0)
        self.assertNotEqual(self.run_cmd(f"kill -0 {os.getpid()}")[0], 0)


if __name__ == "__main__":
    unittest.main()
