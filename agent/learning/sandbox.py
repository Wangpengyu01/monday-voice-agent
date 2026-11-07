"""A throw-away sandbox for 星期一's own shell commands on the board (bubblewrap, `bwrap`).

The owner's idea (2026-11-07): instead of asking about every command that is not on the read-only
whitelist, let her run it where it cannot touch anything. Inside the sandbox:
  - the whole filesystem is read-only; only /tmp is writable, and it is thrown away afterwards;
  - secrets and private data are hidden (empty): /userdata/secrets, /home (SSH keys), /root,
    /userdata/agent/run (conversations, diary, memory, the device vault) - command output goes to the
    teacher, so these must not be readable;
  - its own PID namespace: `ps` still sees every process (host /proc, read-only) but it cannot signal,
    kill or ptrace any of them (tested on the board: kill -9 of the orchestrator -> "No such process");
  - its own user namespace: sudo and every setuid program fail;
  - no network (own network namespace), except a command made only of network *viewers*
    (ss, ip, ping, ...) and plain text filters, which sees the host network;
  - no D-Bus (system bus socket and per-user buses hidden): writing files is not the only way to change
    things - asking systemd or NetworkManager over D-Bus is another. Only a command made of D-Bus
    *viewers* (systemctl status, avahi-browse, ...) with no changing sub-command keeps the system bus;
  - a clean environment (no keys inherited), a time limit and a memory limit.

What still goes through the owner (learning.agent.classify, unchanged): steps that change the real
system (danger: asked every time, snapshot first), anything that talks to the network (curl, wget,
ssh, ...: asked), other machines and MCP tools (asked), secrets / wiping / power (blocked).

    python3 -m learning.sandbox "ps aux | awk '{print $1}' | sort | uniq -c"     # try one command
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys

TIMEOUT_S = 20.0
MEMORY_BYTES = 2 * 1024 ** 3        # address space per process
HIDDEN = ("/userdata/secrets", "/home", "/root", "/userdata/agent/run", "/run/user")
DBUS_DIR = "/run/dbus"
NOTE = "（沙箱里运行：文件系统只读、不能联网）"
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# talk to the network: never sandboxed (they would just fail there); the owner is asked as before
NET_TOOLS = re.compile(r"(?<![\w./-])(curl|wget|nc|ncat|netcat|socat|ssh|scp|sftp|rsync|telnet|ftp|"
                       r"git\s+(clone|fetch|pull|push)|pip3?\s+(install|download)|apt(-get)?\s+(update|install|download))(?![\w-])")
# programs that only *look at* the host network
NET_VIEW = {"ss", "ip", "ping", "ifconfig", "hostname", "netstat", "arp", "iw", "iwconfig"}
# programs that only *ask* system services over D-Bus (as long as no changing sub-command is given)
DBUS_VIEW = {"systemctl", "avahi-browse", "avahi-resolve", "nmcli", "loginctl", "timedatectl", "hostnamectl",
             "resolvectl", "journalctl", "busctl"}
_DBUS_CHANGE = re.compile(
    r"(?<![\w-])(start|stop|restart|reload|try-restart|enable|disable|mask|unmask|kill|isolate|edit|set-[\w-]+|"
    r"daemon-reload|daemon-reexec|link|preset|revert|reset-failed|connect|disconnect|up|down|delete|modify|add|radio|"
    r"reapply|import|clone|terminate-\w+|lock-\w+|call|emit|power\w*|suspend|hibernate|reboot|halt|poweroff|"
    r"vacuum|rotate|flush)(?![\w-])")
# text filters that cannot start other programs (no sed: GNU sed "e" runs commands; no xargs / tee)
TEXT = {"grep", "egrep", "head", "tail", "wc", "sort", "uniq", "cut", "tr", "awk", "mawk", "column", "cat",
        "echo", "true", "printf", "tac", "nl", "fold",
        # plain read-only programs that cannot start others either (no find: -exec)
        "which", "ls", "date", "uname", "df", "free", "uptime", "ps", "id", "whoami", "pwd", "nproc", "lscpu",
        "lsblk", "du", "stat", "readlink", "basename", "dirname", "test", "["}
_AWK_RUNS = re.compile(r"system|getline|\|\s*\"|printf?\s*[|>]")   # awk calling out or writing
_REDIRECT = re.compile(r"\d*>&\d+|&>>?\s*\S+|\d*>>?\s*\S+|\d*<\s*\S+")


def available() -> bool:
    return sys.platform.startswith("linux") and shutil.which("bwrap") is not None


def programs(cmd: str) -> list[str] | None:
    """The programs a shell command runs (first word of every pipeline / list element, after
    VAR=x / env / timeout / nice prefixes). None if it cannot be parsed - treat as unknown."""
    if any(t in cmd for t in ("`", "$(", "<(", ">(")):
        return None
    out = []
    for seg in re.split(r"\|\||&&|[|;&\n]", _REDIRECT.sub(" ", cmd)):
        try:
            words = shlex.split(seg)
        except ValueError:
            return None
        while words and (("=" in words[0] and not words[0].startswith("=")) or words[0] in ("env", "nice", "nohup", "stdbuf")):
            words = words[1:]
        if words and words[0] == "timeout":
            words = [w for w in words[1:] if not w.startswith("-")][1:]
        if words:
            out.append(words[0].rsplit("/", 1)[-1])
    return out


def eligible(step: dict) -> bool:
    """A step the sandbox may run instead of asking: a shell command on the board that does not talk
    to the network. (The caller only offers ask-level steps; danger and block never get here.)"""
    cmd = str(step.get("cmd") or "")
    return step.get("tool") == "shell" and bool(cmd.strip()) and not NET_TOOLS.search(cmd)


def _only(cmd: str, view: set[str]) -> bool:
    """The command runs at least one program of `view` and otherwise only plain text filters."""
    progs = programs(cmd)
    if not progs or not any(p in view for p in progs) or not all(p in view or p in TEXT for p in progs):
        return False
    return not (any(p in ("awk", "mawk") for p in progs) and _AWK_RUNS.search(cmd))


def needs_host_net(cmd: str) -> bool:
    return _only(cmd, NET_VIEW)


def needs_dbus(cmd: str) -> bool:
    return _only(cmd, DBUS_VIEW) and not _DBUS_CHANGE.search(cmd)


def argv(cmd: str, *, net: bool = False, dbus: bool = False) -> list[str]:
    a = ["bwrap", "--ro-bind", "/", "/", "--ro-bind", "/proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for p in HIDDEN + (() if dbus else (DBUS_DIR,)):
        if os.path.isdir(p):
            a += ["--tmpfs", p]
    a += ["--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try"]
    if not net:
        a.append("--unshare-net")
    a += ["--die-with-parent", "--new-session", "--clearenv", "--setenv", "PATH", _PATH, "--setenv", "LANG", "C.UTF-8",
          "--setenv", "HOME", "/tmp", "--chdir", "/tmp", "/bin/bash", "-c", cmd]
    return a


def _limits() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    os.nice(10)


def run_step(step: dict, *, timeout: float = TIMEOUT_S) -> tuple[int, str]:
    """Run a shell step in the sandbox -> (code, output) like learning.agent.run_step."""
    cmd = str(step.get("cmd") or "")
    try:
        r = subprocess.run(argv(cmd, net=needs_host_net(cmd), dbus=needs_dbus(cmd)), capture_output=True, text=True,
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL, preexec_fn=_limits,
                           env={"PATH": _PATH})
    except subprocess.TimeoutExpired:
        return 124, f"超时（{timeout:.0f} 秒）"
    except OSError as exc:
        return 127, f"沙箱启动失败：{exc}"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
    return r.returncode, out.strip() or "（没有输出）"


if __name__ == "__main__":
    code, text = run_step({"tool": "shell", "cmd": " ".join(sys.argv[1:])})
    print(text)
    print(f"[exit {code}]")
