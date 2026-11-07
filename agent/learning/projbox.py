"""The project sandbox: one writable workspace per project, everything else read-only, no LAN.

Owner's choice (2026-11-07): projects run on the board, inside a sandbox where 星期一 may do anything
without asking, reach the internet but not the LAN, and can't break herself. Inside (bwrap):
  - /                    read-only (she can read her own code, the system, the docs, man pages)
  - /home/work           the project's files, writable: PROJECTS/<slug>/work
  - /home/proj           HOME, writable: PROJECTS/<slug>/home (pip --user, caches, downloads)
  - hidden               /userdata/secrets, /root, her memory and data (/userdata/agent/run), backups,
                         other projects, /run/user, D-Bus
  - network              none of its own; http(s)_proxy -> learning/projbox_fwd.py -> the egress proxy
                         (learning.egress_proxy), which only lets public addresses on 80/443 through
  - processes            own user/pid/ipc/uts namespaces: she can't see or signal the real services
  - limits               4 GB address space, 2 GB per file, nice 10, a timeout per command; the
                         runner stops a project whose folder passes DISK_MAX
Nothing in here can change her code, her prompts or the system; putting a result into her real self
is learning.deliver (snapshot, tests, exam, automatic rollback).

    python3 -m learning.projbox SLUG "command"      # run one command in a project's sandbox
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECTS = Path(os.environ.get("MONDAY_PROJECTS", "/userdata/projects"))
AGENT_ROOT = Path(__file__).resolve().parents[1]
FWD = AGENT_ROOT / "learning" / "projbox_fwd.py"
HIDDEN = ("/userdata/secrets", "/root", "/userdata/agent/run", "/userdata/backups", "/run/user", "/run/dbus",
          str(PROJECTS))
PROXY_PORT = 3128
TIMEOUT_S, TIMEOUT_MAX = 300, 1800
MEMORY_BYTES = 4 << 30
FILE_BYTES = 2 << 30
DISK_MAX = 4 << 30
OUT_MAX = 6000
_PATH = "/home/proj/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def available() -> bool:
    return sys.platform.startswith("linux") and any(
        os.access(os.path.join(p, "bwrap"), os.X_OK) for p in os.environ.get("PATH", "/usr/bin").split(":"))


def project_dir(slug: str) -> Path:
    return PROJECTS / slug


def prepare(pdir: Path) -> None:
    for sub in ("work", "home"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)


def argv(pdir: Path, cmd: str) -> list[str]:
    proxy = f"http://127.0.0.1:{PROXY_PORT}"
    a = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home"]
    for p in HIDDEN:
        if os.path.isdir(p):
            a += ["--tmpfs", p]
    a += ["--bind", (pdir / "work").as_posix(), "/home/work", "--bind", (pdir / "home").as_posix(), "/home/proj"]
    if (pdir / "egress.sock").exists():
        a += ["--bind", (pdir / "egress.sock").as_posix(), "/tmp/.egress.sock"]
    a += ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    env = {"PATH": _PATH, "HOME": "/home/proj", "LANG": "C.UTF-8", "TERM": "dumb", "USER": "monday",
           "http_proxy": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy,
           "no_proxy": "localhost,127.0.0.1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "GIT_TERMINAL_PROMPT": "0",
           "PYTHONUNBUFFERED": "1"}
    for k, v in env.items():
        a += ["--setenv", k, v]
    # the forwarder first (it is the sandbox's only way out), then the command itself
    boot = (f'python3 {FWD.as_posix()} /tmp/.egress.sock {PROXY_PORT} /tmp/.fwd_ready >/dev/null 2>&1 & '
            'for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do [ -e /tmp/.fwd_ready ] && break; sleep 0.1; done; '
            'exec /bin/bash -c "$0"')
    a += ["--chdir", "/home/work", "/bin/sh", "-c", boot, cmd]
    return a


def _limits() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    os.nice(10)


def clip(text: str, n: int = OUT_MAX) -> str:
    if len(text) <= n:
        return text
    return text[: n // 3] + f"\n…（中间省略 {len(text) - n} 字）…\n" + text[-(2 * n) // 3:]


def run(pdir: Path, cmd: str, timeout: float = TIMEOUT_S) -> tuple[int, str]:
    """(exit code, output) of one command in the project's sandbox; 124 = timed out."""
    timeout = max(5.0, min(float(timeout or TIMEOUT_S), TIMEOUT_MAX))
    try:
        p = subprocess.run(argv(pdir, cmd), capture_output=True, timeout=timeout, preexec_fn=_limits)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        return 124, clip(out.decode("utf-8", "replace")) + f"\n（超时：{timeout:.0f} 秒还没跑完，已停止）"
    out = (p.stdout or b"") + (b"\n" + p.stderr if p.stderr else b"")
    return p.returncode, clip(out.decode("utf-8", "replace").strip())


def disk_used(pdir: Path) -> int:
    total = 0
    for root, _, files in os.walk(pdir):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def main() -> None:
    from learning import egress_proxy
    slug, cmd = sys.argv[1], sys.argv[2]
    pdir = project_dir(slug)
    prepare(pdir)
    srv = egress_proxy.serve(pdir / "egress.sock", pdir / "egress.log")
    try:
        code, out = run(pdir, cmd)
    finally:
        srv.shutdown()
        (pdir / "egress.sock").unlink(missing_ok=True)
    print(out)
    print(f"[exit {code}]")


if __name__ == "__main__":
    main()
