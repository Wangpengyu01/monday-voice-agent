"""星期一 learns to do things the way a person would: look up how it did it before; if it never did it,
ask the teacher (DeepSeek) one step at a time, do the step, show the teacher what happened, go on;
when it is done, let the teacher confirm it really worked, then keep the steps as a skill
(learning.skills) and as LoRA training data, so next time it knows how.

Tools a step may use:
    {"tool": "shell", "cmd": "df -h"}                         on the board, as the service user
    {"tool": "ssh", "host": "gp@192.168.100.196", "cmd": "df -h"}   key-based only, never passwords
    {"tool": "note", "name": "x.md", "text": "..."}            /userdata/files/notes/
    {"tool": "done", "success": true, "say": "...", "notes": "..."}

Permission gate (the owner's rule: anything that can change or reach out asks first):
    BLOCK  secrets (/userdata/secrets, the device vault, private keys, /etc/shadow), disk wiping,
           power off, account changes, piping downloads into a shell. Never run, even if approved.
    DANGER can damage the system: sudo, installing/removing packages, starting/stopping services,
           rm/mv/cp/chmod/chown, writing files (redirects, sed -i, tee), kill, network changes, anything
           touching /etc, /boot, /usr or /userdata/agent. Asked every time - "remember" does not apply -
           and on the board a snapshot (ops.snapshot, "历史节点") is taken first; no snapshot, no run.
    ASK    everything else not known read-only (ssh, combined commands, scripts...). Asked once; the owner
           may answer "允许并记住" and the exact same step (tool + host + command) runs without asking
           from then on (learning.grants; revocable in the web page's memory panel). Either "允许并记住"
           or "这次都行" also lets the rest of *this task's* ask-level steps run without asking: the
           planner rewrites its command a little at every step, so an exact-match grant alone meant the
           owner was asked again and again. Danger-level steps are still asked every time.
    SAFE   a short list of read-only commands (df, free, uptime, ip addr, systemctl status ...).
The owner is asked by voice and on the web page; the step waits; no answer in time = no.
Command output is redacted (keys, tokens, passwords) and cut short before the teacher sees it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from learning.credentials import Credentials, add_host, alias_for, install_key
from learning import sandbox as sandbox_mod
from learning import skillpack
from learning.mcp import McpHub
from learning.skills import Skill, SkillStore

LOG = logging.getLogger(__name__)

MAX_STEPS = 10
KNOWN_SCORE = 0.6  # wording similarity at which a stored skill counts as "I've done this"
OUTPUT_LIMIT = 1500
NOTES_DIR = Path("/userdata/files/notes")

# ---------------------------------------------------------------- permission gate
_SECRET = re.compile(r"/userdata/secrets|vault|run/devices|deepseek\.env|\.ssh/id_|id_rsa|id_ed25519|id_ecdsa|"
                     r"/etc/shadow|/etc/sudoers|private[_ ]?key|authorized_keys", re.I)
_DESTRUCTIVE = re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+/(\s|$)|\bmkfs|\bdd\s+if=|>\s*/dev/(sd|mmc|nvme)|:\(\)\s*\{|"
                          r"\b(shutdown|poweroff|halt|reboot|init\s+0)\b|(^|[\s;&|])(sudo\s+)?(passwd|chpasswd|userdel|usermod|visudo)\b|"
                          r"(curl|wget)[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", re.I)
_SAFE_FIRST = {"df", "free", "uptime", "hostname", "date", "uname", "nproc", "lscpu", "lsblk", "whoami", "id",
               "ps", "pwd", "which", "echo", "ls", "du", "ping", "ip", "systemctl", "ss", "nmcli", "cat", "head",
               "tail", "grep", "wc", "sort", "uniq", "cut", "tr", "python3"}
_FILTERS = {"grep", "head", "tail", "wc", "sort", "uniq", "cut", "tr"}  # no awk/sed: they can run commands
_READ_PATHS = ("/proc/", "/sys/", "/etc/os-release", "/userdata/agent/logs/", "/userdata/files/")


_PKG = {"apt", "apt-get", "aptitude", "dpkg", "snap", "pip", "pip3", "npm", "conda"}
_PKG_CHANGE = {"install", "remove", "purge", "uninstall", "upgrade", "dist-upgrade", "full-upgrade", "autoremove",
               "update", "-i", "-r", "-P", "--install", "--remove", "--purge", "--configure"}
_WRITERS = {"rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp", "chattr", "ln", "dd", "truncate", "tee", "install",
            "shred", "kill", "pkill", "killall", "mount", "umount", "iptables", "ip6tables", "nft", "ufw", "route",
            "crontab", "useradd", "groupadd", "groupdel", "modprobe", "insmod", "rmmod", "sysctl", "hostnamectl",
            "timedatectl", "nmtui", "wget", "service", "update-rc.d", "fdisk", "parted", "mkswap", "swapon", "swapoff",
            "unzip", "patch", "rsync", "scp", "make", "su", "doas", "pkexec"}
_SYSTEMCTL_CHANGE = {"start", "stop", "restart", "reload", "try-restart", "reload-or-restart", "enable", "disable",
                     "mask", "unmask", "daemon-reload", "kill", "isolate", "set-property", "edit", "reset-failed", "link"}
_GIT_CHANGE = {"checkout", "reset", "clean", "pull", "merge", "rebase", "push", "rm", "mv", "commit", "stash", "apply",
               "restore", "switch"}
_PROTECTED = re.compile(r"""(^|[\s'"=:])(/etc|/boot|/usr|/lib|/bin|/sbin|/var/lib|/userdata/agent|/userdata/backups|"""
                        r"""~?/\.ssh|/home/[^/\s]+/\.ssh)(/|\s|$|['"])""")
_READERS = {"cat", "head", "tail", "grep", "ls", "stat", "du", "df", "wc", "file", "md5sum", "sha256sum", "diff",
            "less", "find", "readlink", "realpath", "test", "echo", "systemctl", "journalctl"}
_WRITE_REDIRECT = re.compile(r"(\d*)>>?\s*(&?)\s*([^\s;&|]*)")
_WRAPPERS = {"env", "nohup", "nice", "timeout", "xargs", "exec", "time", "stdbuf", "ionice", "bash", "sh"}


def _danger(cmd: str) -> str:
    """Why `cmd` can damage the system, or "" (deliberately coarse: when unsure, it is dangerous)."""
    for m in _WRITE_REDIRECT.finditer(cmd):
        if not m.group(2) and m.group(3) != "/dev/null":
            return "会写文件（重定向）"
    if re.search(r"\bsed\b[^|;&]*\s-[a-zA-Z]*i|\bperl\b[^|;&]*\s-[a-zA-Z]*i|\bfind\b[^|;&]*\s-(delete|exec|execdir|ok)\b", cmd):
        return "会改动或删除文件"
    reads_only = True
    for seg in re.split(r"\|\|?|&&|;|\$\(|`|\n", cmd):
        try:
            words = shlex.split(seg)
        except ValueError:
            words = seg.split()
        while words and re.match(r"^[A-Za-z_]\w*=", words[0]):
            words = words[1:]  # FOO=bar cmd
        while words and words[0].rsplit("/", 1)[-1] in _WRAPPERS:  # nohup / timeout 5 / env X=1 ... cmd
            words = words[1:]
            while words and (words[0].startswith("-") or re.match(r"^([\d.]+[smhd]?|[A-Za-z_]\w*=.*)$", words[0])):
                words = words[1:]
        if not words:
            continue
        first = words[0].rsplit("/", 1)[-1]
        if first.startswith("python") and words[1:2] == ["-m"] and len(words) > 2:
            first, words = words[2], words[2:]  # python3 -m pip install ...
        if first in ("sudo", "su", "doas", "pkexec"):
            return "要用 root 权限"
        if first in _WRITERS:
            return f"{first} 会改动系统或文件"
        if first in _PKG and any(w in _PKG_CHANGE for w in words[1:]):
            return "安装、卸载或升级软件"
        if first == "systemctl" and any(w in _SYSTEMCTL_CHANGE for w in words[1:]):
            return "会启动、停止或改动服务"
        if first == "git" and any(w in _GIT_CHANGE for w in words[1:]):
            return "会改动代码仓库"
        if first == "ip" and any(w in ("add", "del", "delete", "set", "flush", "change", "replace", "append") for w in words):
            return "会改网络"
        if first == "nmcli" and any(w in ("connect", "up", "down", "delete", "modify", "add", "radio", "reload") for w in words):
            return "会改网络"
        if first == "curl" and any(w in ("-O", "--output", "--remote-name", "-T", "--upload-file") or w.startswith("-o")
                                   for w in words[1:]):
            return "会下载或上传文件"
        if first == "tar" and any(w == "--extract" or re.match(r"^-?[a-wyzA-Z]*x", w) for w in words[1:2]):
            return "会解包写文件"
        if first not in _READERS:
            reads_only = False
    if not reads_only and _PROTECTED.search(cmd):
        return "涉及系统目录或星期一自己的代码"
    return ""


def classify(step: dict, *, mcp_safe=None, packs=None) -> tuple[str, str]:
    """-> ("safe" | "ask" | "danger" | "block", reason).
    mcp_safe(server, tool) -> bool marks read-only MCP tools; packs are the loaded skill packs (their
    risk: safe scripts run without asking)."""
    tool = step.get("tool")
    if tool == "done":
        return "safe", ""
    if tool == "mcp":
        server, name = str(step.get("server") or ""), str(step.get("name") or "")
        if not server or not name:
            return "block", "MCP 调用缺服务名或工具名"
        if _SECRET.search(json.dumps(step.get("args") or {}, ensure_ascii=False)):
            return "block", "涉及密钥或账号库，永远不碰"
        if mcp_safe is not None and mcp_safe(server, name):
            return "safe", ""
        return "ask", f"要调用外部工具 {server}.{name}"
    if tool == "ssh_key_setup":
        return "danger", "要用你给的密码登录另一台机器并装上我的公钥"
    if tool == "note":
        name = str(step.get("name") or "")
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return "block", "笔记文件名不合法"
        return "safe", ""
    if tool not in ("shell", "ssh"):
        return "block", f"没有这个工具：{tool}"
    cmd = str(step.get("cmd") or "").strip()
    if not cmd:
        return "block", "空命令"
    if _SECRET.search(cmd) or (tool == "ssh" and _SECRET.search(str(step.get("host") or ""))):
        return "block", "涉及密钥或账号库，永远不碰"
    if _DESTRUCTIVE.search(cmd):
        return "block", "危险操作（清盘、关机、改账号或下载即执行）"
    if tool == "shell" and packs and skillpack.script_is_safe(cmd, packs):
        return "safe", ""  # a read-only script of a risk: safe skill pack, called plainly
    if why := _danger(cmd):
        return "danger", why + ("（在另一台机器上）" if tool == "ssh" else "")
    if tool == "ssh":
        return "ask", "要连接另一台机器"
    if re.search(r"[;&`<>]|\$\(|\|\|", cmd) or "sudo" in cmd.split():
        return "ask", "组合命令、重定向或 sudo"
    for i, seg in enumerate(s.strip() for s in cmd.split("|")):
        try:
            words = shlex.split(seg)
        except ValueError:
            return "ask", "命令解析不了"
        if not words:
            return "ask", "空的管道段"
        first = words[0]
        if i > 0 and first not in _FILTERS:
            return "ask", f"管道里有 {first}"
        if first not in _SAFE_FIRST:
            return "ask", f"{first} 不在只读白名单里"
        if first == "systemctl" and (len(words) < 2 or words[1] not in ("status", "is-active", "is-enabled", "list-units", "list-timers")):
            return "ask", "systemctl 会改服务状态"
        if first == "ip" and any(w in ("add", "del", "set", "flush", "change", "replace") for w in words):
            return "ask", "会改网络"
        if first == "nmcli" and any(w in ("connect", "up", "down", "delete", "modify", "add", "radio") for w in words):
            return "ask", "会改网络"
        if first == "ping" and "-c" not in words:
            return "ask", "ping 没有限制次数"
        if first == "python3" and not (len(words) >= 2 and words[1] in ("--version", "-V")):
            return "ask", "运行 Python 代码"
        if first in ("cat", "head", "tail", "grep") and i == 0:
            args = [w for w in _drop_count_values(words[1:]) if not w.startswith("-")][(1 if first == "grep" else 0):]
            if not args or not all(a.startswith(_READ_PATHS) for a in args):
                return "ask", "读取的文件不在公开目录里"
    return "safe", ""


_COUNT_OPTS = {"-n", "-c", "-m", "-A", "-B", "-C"}  # head/tail -n 5, grep -m 3 / -A 2: a count, not a file


def _drop_count_values(args: list[str]) -> list[str]:
    """Skip the number after a count option ("tail -n 5 f" read "5" as a file to check and asked).
    Only a plain number is skipped; anything else after the option is still checked as a path."""
    out, skip = [], False
    for a in args:
        if skip and re.fullmatch(r"[+-]?\d+", a):
            skip = False
            continue
        skip = a in _COUNT_OPTS
        out.append(a)
    return out


_REDACT = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[私钥已隐藏]"),
    (re.compile(r"\b(sk|pk|ghp|gho|xox[abp])-?[A-Za-z0-9_\-]{16,}"), "[密钥已隐藏]"),
    (re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key|authorization)(\s*[=:]\s*|\s+)\S+"), r"\1=[已隐藏]"),
]


def redact(text: str, limit: int = OUTPUT_LIMIT) -> str:
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    if len(text) > limit:
        text = text[: limit // 2] + f"\n…（省略 {len(text) - limit} 字）…\n" + text[-limit // 2:]
    return text


# ---------------------------------------------------------------- running a step
def run_step(step: dict, *, timeout: float = 40.0, notes_dir: Path = NOTES_DIR) -> tuple[int, str]:
    tool = step["tool"]
    if tool == "note":
        notes_dir.mkdir(parents=True, exist_ok=True)
        p = notes_dir / str(step["name"])
        p.write_text(str(step.get("text") or ""), encoding="utf-8")
        return 0, f"已写入 {p}"
    if tool == "ssh":
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
                str(step.get("host") or ""), str(step["cmd"])]
    else:
        argv = ["/bin/bash", "-c", str(step["cmd"])]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 124, f"超时（{timeout:.0f} 秒）"
    except OSError as exc:
        return 127, str(exc)
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
    return r.returncode, out.strip() or "（没有输出）"


# ---------------------------------------------------------------- the loop
PLAN_SYSTEM = (
    "你在一步步指导语音助手「星期一」完成主人交代的任务。星期一跑在一块 RK3588 Linux 板子上（Ubuntu，"
    "用户 youyeetoo，工作目录 /userdata/agent），它会照你给的步骤执行，再把结果给你看。\n"
    "可用工具（每次只给一步）：\n"
    '  {"tool": "shell", "cmd": "命令", "purpose": "这一步干什么"}  在板子上执行\n'
    '  {"tool": "ssh", "host": "用户@地址", "cmd": "命令", "purpose": "..."}  只能用已配置好的密钥登录，不能输密码\n'
    '  {"tool": "note", "name": "文件名.md", "text": "内容", "purpose": "..."}  把结果记成笔记\n'
    '  {"tool": "mcp", "server": "服务名", "name": "工具名", "args": {...}, "purpose": "..."}  调用下面列出的外部 MCP 工具\n'
    "有「可参考的技能」时优先照技能里的做法和脚本做。\n"
    '  {"tool": "done", "success": true或false, "say": "对主人说的一两句口语总结", "notes": "下次做这件事要注意什么"}\n'
    "任务文字来自语音识别，数字和英文可能听错，按最合理的意思理解；任务里的机器和已配置主机对得上就直接用别名。\n"
    "规则：命令要非交互（加 -y、-c 次数、--no-pager 之类）；一次一条简单命令；需要密码、密钥或主人的账号信息时，"
    "不要尝试，直接 done 并在 say 里请主人告诉星期一账号和密码（星期一会用它装好密钥，你看不到密码）；如果已执行步骤里已经装好了密钥，就直接用那个别名；不碰 /userdata/secrets 和任何私钥；"
    "改系统、装软件、重启服务、连别的机器都会先问主人，被拒绝就停。步骤失败就换思路，别重复同一条命令。"
    "查服务在不在跑用 systemctl is-active 服务名；进程名和服务名不一样（r1-orchestrator 是 python3 -m orchestrator.main，"
    "r1-tts-daemon 是 rknn_qwen3_tts），ps 里 grep 服务名搜不到不等于它挂了。"
    "只是查看的命令（含管道、awk、python3 小脚本）会在只读沙箱里直接执行，不打扰主人：看得到系统状态和日志，"
    "但写不了文件、不能联网、看不到 /userdata/agent/run；结果开头会标「沙箱里运行」。真要改东西就给出改东西的步骤，主人会被询问。\n"
    '只输出 JSON：{"thought": "简短思考", "step": {...}}'
)
VERIFY_SYSTEM = (
    "你在检查语音助手「星期一」是否真的完成了任务。根据它执行的每一步和输出判断，不要轻信它自己的结论。"
    "任务文字来自语音识别，数字、地址、英文常被听错（比如少一位、SSH 听成 SH）；要连的机器以给出的已配置主机为准，"
    "不要因为任务里的地址和实际连上的已配置主机差一两位就判失败。"
    '只输出 JSON：{"done": true或false, "why": "一句话", "skill_steps": [只保留真正有用、按顺序能复现的步骤，'
    '格式同原步骤，去掉失败和试探的步骤], "task": "这类任务的标准说法（不含一次性的细节）"}'
)


SPLIT_SYSTEM = (
    "判断这个任务能不能拆成几件互相独立、可以同时做的小事（比如分别检查三台机器、同时查几个网站）。"
    "有先后依赖、只是一件事、或者拆了反而麻烦，就不拆。最多拆三件，每件写成一句完整的任务。"
    '只输出 JSON：{"parallel": true或false, "subtasks": ["...", "..."]}'
)
SUMMARY_SYSTEM = "把几件同时做完的小任务结果合成一两句口语汇报给主人，先说结论，数字用中文说法，不要列表。"
_MAYBE_SPLIT = re.compile(r"和|、|分别|都|每台|每个|几台|两台|三台|所有|全部|同时|一起|批量")


@dataclass
class AgentResult:
    ok: bool
    say: str
    steps: list[dict] = field(default_factory=list)
    skill_id: int = 0
    learned: bool = False


Approve = Callable[[dict, str, str], Awaitable[str]]  # (step, why, risk) -> "once" | "task" | "always" | "deny"
Say = Callable[[str], Awaitable[None]]
Progress = Callable[[dict], Awaitable[None]]

# a request to *do* something on a machine, or to go and learn how
_TASK = re.compile(
    r"(?i)(?<![a-z])s{1,2}h(?![a-z])|登录|登陆|远程|终端|命令行|"  # "SSH" arrives as "SH" (repeats are collapsed)
    r"(连到|连上|连接).{0,10}(服务器|电脑|机器|主机|\d+\.\d+)|"
    r"(帮我|你去|你来|去).{0,8}(查|看|检查|装|安装|卸载|配置|部署|修|重启|备份|清理|整理|测|跑).{0,14}"
    r"(板子|系统|服务|进程|磁盘|硬盘|内存|网络|网速|IP|端口|日志|文件|服务器|软件|程序|温度|CPU|NPU)|"
    r"(去|自己|你).{0,2}学(会|一下)?(怎么|如何)")
_YES = re.compile(r"^(确认|确定|可以|同意|允许|执行|好的?|行|没问题|做吧|继续|对)(的|吧|啊|呀)?$")
_ALWAYS = re.compile(r"^(确认|确定|可以|同意|允许|好的?|行)?(并|并且|而且|那就)?(记住?|记下来|以后都可以|以后都行|以后都允许|"
                     r"以后不用问|以后不用再问|以后别问|以后直接做|总是允许|一直允许)(了|吧|啊|呀)?$")
_TASK_OK = re.compile(r"^(确认|确定|可以|同意|允许|好的?|行)?(这次|这一次|这个任务|这件事|这回)?(都|全都|全部)(行|可以|允许|确认|同意|做吧|执行)"
                      r"(了|吧|啊|呀)?$")
_NO = re.compile(r"^(取消|不要|不行|不可以|别|拒绝|算了|停|不用|不同意|否)(了|吧|啊|做)?$")


def is_agent_task(text: str) -> bool:
    return bool(_TASK.search(text or ""))


def approval_answer(text: str) -> str | None:
    """Spoken answer to a pending approval: "once" | "task" | "always" | "deny"; None if it is none of them
    (short, exact phrases only)."""
    t = re.sub(r"[\s，,。.！!？?、]", "", text or "")
    if _NO.match(t):
        return "deny"
    if _ALWAYS.match(t):
        return "always"
    if _TASK_OK.match(t):
        return "task"
    if _YES.match(t):
        return "once"
    return None


def answer_of(value) -> str:
    """Normalize what an approve callback returned (older callbacks return bool)."""
    if value is True:
        return "once"
    if value in ("once", "task", "always"):
        return value
    return "deny"


def take_snapshot(label: str, reason: str) -> str:
    """A "历史节点" of the assistant before a dangerous step; -> snapshot id (raises if it could not be taken)."""
    from ops import snapshot

    return snapshot.create(label, reason).id


def _json(text: str) -> dict:
    i, j = (text or "").find("{"), (text or "").rfind("}")
    try:
        obj = json.loads(text[i:j + 1]) if 0 <= i < j else {}
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _step_of(plan: dict) -> dict:
    """The step from a plan reply, tolerating the usual variations ({"step": {...}}, the step at top level,
    "action" instead of "step", a list of steps)."""
    for key in ("step", "action", "next_step"):
        v = plan.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            v = v[0]
        if isinstance(v, dict) and v.get("tool"):
            return v
    if plan.get("tool"):
        return {k: v for k, v in plan.items() if k != "thought"}
    steps = plan.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], dict) and steps[0].get("tool"):
        return steps[0]
    return {}


def known_ssh_hosts(config: Path | None = None) -> list[str]:
    """Hosts set up for key login in ~/.ssh/config, as "alias（user@address）" (aliases only - never keys)."""
    config = config or Path.home() / ".ssh" / "config"
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    hosts, cur = [], None
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0].startswith("#"):
            continue
        key, val = parts[0].lower(), parts[1].strip()
        if key == "host":
            cur = {"alias": val.split()[0], "host": "", "user": ""}
            if "*" not in cur["alias"]:
                hosts.append(cur)
        elif cur is not None and key == "hostname":
            cur["host"] = val
        elif cur is not None and key == "user":
            cur["user"] = val
    return [f"{h['alias']}（{h['user'] + '@' if h['user'] else ''}{h['host'] or h['alias']}）" for h in hosts]


def _norm_mcp(step: dict) -> dict:
    """Planners write the same MCP tool several ways ("web"+"fetch", "web.fetch"+"fetch", "web.fetch"+"web.fetch")."""
    if step.get("tool") != "mcp":
        return step
    tokens = [t for t in f"{step.get('server') or ''}.{step.get('name') or ''}".split(".") if t]
    dedup = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]
    while len(dedup) >= 4 and len(dedup) % 2 == 0 and dedup[: len(dedup) // 2] == dedup[len(dedup) // 2:]:
        dedup = dedup[: len(dedup) // 2]
    if len(dedup) >= 2:
        step = {**step, "server": dedup[0], "name": dedup[-1]}
    return step


def describe_step(step: dict) -> str:
    if step.get("tool") == "mcp":
        return f"调用 {step.get('server')}.{step.get('name')} {json.dumps(step.get('args') or {}, ensure_ascii=False)[:120]}"
    if step.get("tool") == "ssh_key_setup":
        return f"用密码登录 {step.get('host')} 装公钥（别名 {step.get('alias')}）"
    if step.get("tool") == "ssh":
        return f"登录 {step.get('host')} 执行 {step.get('cmd')}"
    if step.get("tool") == "note":
        return f"写笔记 {step.get('name')}"
    return f"执行 {step.get('cmd')}"


class Agent:
    def __init__(self, teacher, skills: SkillStore, *, approve: Approve, say: Say,
                 log_training: Callable[[str, str], None] | None = None, progress: Progress | None = None,
                 runner: Callable[[dict], tuple[int, str]] = run_step, ssh_hosts: list[str] | None = None,
                 install_key: Callable[[Credentials], tuple[bool, str]] = install_key, grants=None,
                 snapshot: Callable[[str, str], str] | None = take_snapshot,
                 packs: list | None = None, mcp: McpHub | None = None, max_parallel: int = 3,
                 task_ok: dict | None = None, local_plan: Callable[[str], str] | None = None,
                 sandbox: Callable[[dict], tuple[int, str]] | None = None) -> None:
        self.teacher = teacher
        self.skills = skills
        self.approve = approve
        self.say = say
        self.log_training = log_training
        self.runner = runner
        self.progress = progress
        self.ssh_hosts = known_ssh_hosts() if ssh_hosts is None else ssh_hosts
        self.install_key = install_key
        self.grants = grants  # learning.grants.GrantStore, or None (then every ask-level step asks)
        self.snapshot = snapshot
        self.packs = skillpack.load() if packs is None else packs
        self.mcp = mcp
        self.max_parallel = max_parallel
        self._approve_lock = asyncio.Lock()  # parallel sub-agents ask one question at a time
        self._mcp_tools = None  # listed on first use
        # one Agent runs one task: once the owner says "这次都行" (or "记住"), the rest of its ask-level
        # steps go ahead; shared with parallel workers so one answer covers the whole task
        self.task_ok = {"ok": False} if task_ok is None else task_ok
        # local-first (config learning.agent_local_first): the local model tries a task with read-only
        # commands before the teacher is asked; None = teacher only
        self.local_plan = local_plan
        # learning.sandbox.run_step (config learning.agent_sandbox): ask-level shell steps on the board run
        # there - read-only, no network, nothing secret visible - instead of asking; None = ask as before
        self.sandbox = sandbox

    async def _gate_and_run(self, step: dict) -> tuple[str, int, str]:
        """-> (status, code, output); status: ran / blocked / denied."""
        risk, why = classify(step, mcp_safe=self.mcp.is_safe if self.mcp else None, packs=self.packs)
        if risk == "block":
            LOG.warning("[agent] blocked: %s (%s)", step, why)
            return "blocked", -1, f"被安全规则拒绝：{why}"
        sandboxed = False
        if risk == "ask" and self.grants is not None and self.grants.allowed(step):
            LOG.info("[agent] allowed earlier (grant): %s", describe_step(step))
        elif risk == "ask" and self.sandbox is not None and sandbox_mod.eligible(step):
            sandboxed = True
            LOG.info("[agent] sandboxed (%s): %s", why, describe_step(step))
        elif risk == "ask" and self.task_ok["ok"]:
            LOG.info("[agent] allowed for this task: %s", describe_step(step))
        elif risk in ("ask", "danger"):
            async with self._approve_lock:
                answer = answer_of(await self.approve(step, why, risk))
            if answer == "deny":
                return "denied", -1, "主人没有同意"
            if answer == "always" and risk == "ask" and self.grants is not None:
                self.grants.add(step)
                LOG.info("[agent] grant remembered: %s", describe_step(step))
            if answer in ("task", "always") and risk == "ask":
                self.task_ok["ok"] = True
        if risk == "danger" and step.get("tool") == "shell":
            first = (str(step.get("cmd") or "").split() or ["step"])[0].rsplit("/", 1)[-1]
            try:
                if self.snapshot is None:
                    raise RuntimeError("没有配置快照")
                sid = await asyncio.to_thread(self.snapshot, f"before-{first}", f"{describe_step(step)}（{why}）")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("[agent] snapshot before danger step failed: %s", exc)
                return "blocked", -1, f"执行前没能存好备份节点（{exc}），为了安全这一步不做"
            LOG.info("[agent] snapshot %s before: %s", sid, describe_step(step))
            await self.say(f"我先存了一个备份节点 {sid}，万一出问题可以回退。")
        if step.get("tool") == "mcp":
            if self.mcp is None:
                code, out = 127, "没有配置 MCP 服务"
            else:
                code, out = await asyncio.to_thread(self.mcp.call, str(step.get("server")), str(step.get("name")),
                                                    step.get("args") or {})
        elif sandboxed:
            code, out = await asyncio.to_thread(self.sandbox, step)
            out = f"{sandbox_mod.NOTE}\n{out}"
        else:
            code, out = await asyncio.to_thread(self.runner, step)
        LOG.info("[agent] ran%s %s -> %d", " (sandbox)" if sandboxed else "", describe_step(step), code)
        out = redact(out)
        if self.progress:
            await self.progress({"purpose": step.get("purpose", ""), "action": describe_step(step), "code": code, "output": out[:400]})
        return "ran", code, out

    async def _replay(self, skill: Skill) -> AgentResult | None:
        """No teacher: do it the way that worked before."""
        await self.say("这个我之前学过，我照着做。")
        done = []
        for step in skill.steps:
            status, code, out = await self._gate_and_run(step)
            if status == "denied":
                return AgentResult(False, "好的，那我先不做了。", done)
            if status != "ran" or code != 0:
                self.skills.record(skill.id, False)
                return AgentResult(False, f"照以前的做法做到「{step.get('purpose', '')}」这一步出错了。", done)
            done.append({**step, "code": code, "output": out})
        self.skills.record(skill.id, True)
        return AgentResult(True, f"做完了，最后一步的结果：{done[-1]['output'][:80] if done else ''}", done, skill.id)

    async def _setup_access(self, creds: Credentials) -> dict:
        """The owner gave a password: use it once to install our key, then the machine is a key-login alias."""
        alias = alias_for(creds.host, creds.user)
        step = {"tool": "ssh_key_setup", "host": f"{creds.user}@{creds.host}", "alias": alias,
                "purpose": "用你给的密码登录一次，装上我的公钥，以后用密钥登录，密码不保存"}
        if any(h.startswith(f"{alias}（") for h in self.ssh_hosts):
            return {**step, "status": "ran", "code": 0, "output": f"{alias} 已经配置过密钥登录"}
        risk, why = classify(step)
        if risk not in ("ask", "danger") or answer_of(await self.approve(step, why, risk)) == "deny":
            return {**step, "status": "denied", "code": -1, "output": "主人没有同意"}
        ok, msg = await asyncio.to_thread(self.install_key, creds)
        creds.password = ""  # used once; drop it now
        if ok:
            await asyncio.to_thread(add_host, alias, creds.host, creds.user)
            self.ssh_hosts = known_ssh_hosts()
            msg += f"，以后用别名 {alias} 登录"
        LOG.info("[agent] key setup %s@%s -> %s", creds.user, creds.host, ok)
        if self.progress:
            await self.progress({"purpose": step["purpose"], "action": describe_step(step), "code": 0 if ok else 1, "output": msg})
        return {**step, "status": "ran", "code": 0 if ok else 1, "output": msg}

    async def _tools_prompt(self, task: str) -> str:
        """Skill packs that fit the task and the MCP tools on offer, for the planner."""
        parts = []
        packs = skillpack.relevant(self.packs, task)
        if packs:
            parts.append("可参考的技能：\n" + "\n\n".join(p.prompt() for p in packs))
        if self.mcp is not None:
            if self._mcp_tools is None:
                self._mcp_tools = await asyncio.to_thread(self.mcp.tools)
            if self._mcp_tools:
                parts.append("可用的 MCP 工具：\n" + "\n".join(t.prompt() for t in self._mcp_tools[:30]))
        return "\n\n".join(parts)

    async def run(self, task: str, creds: Credentials | None = None) -> AgentResult:
        if creds is None and getattr(self.teacher, "available", False) and _MAYBE_SPLIT.search(task):
            subtasks = await self._split(task)
            if len(subtasks) >= 2:
                return await self._run_parallel(task, subtasks)
        return await self._run_single(task, creds)

    async def _split(self, task: str) -> list[str]:
        try:
            res = _json(await asyncio.to_thread(self.teacher.complete, SPLIT_SYSTEM, f"任务：{task}", json_mode=True,
                                                max_tokens=400))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[agent] split failed: %s", exc)
            return []
        subs = [str(t).strip() for t in res.get("subtasks") or [] if str(t).strip()] if res.get("parallel") else []
        return subs[: self.max_parallel] if len(subs) >= 2 else []

    async def _run_parallel(self, task: str, subtasks: list[str]) -> AgentResult:
        """A few independent pieces at once (at most max_parallel), then one summary."""
        await self.say(f"这件事我分成 {len(subtasks)} 件同时做。")
        LOG.info("[agent] parallel: %s", " | ".join(subtasks))
        workers = [Agent(self.teacher, self.skills, approve=self._serial_approve, say=self._quiet_say,
                         log_training=self.log_training, progress=self.progress, runner=self.runner,
                         ssh_hosts=self.ssh_hosts, install_key=self.install_key, grants=self.grants,
                         snapshot=self.snapshot, packs=self.packs, mcp=self.mcp, max_parallel=1,
                         task_ok=self.task_ok, sandbox=self.sandbox) for _ in subtasks]
        results = await asyncio.gather(*(w._run_single(t, None) for w, t in zip(workers, subtasks)), return_exceptions=True)
        lines, steps, ok_all = [], [], True
        for t, r in zip(subtasks, results):
            if isinstance(r, Exception):
                ok_all = False
                lines.append(f"{t}：出错了（{r}）")
                continue
            ok_all &= r.ok
            steps += r.steps
            lines.append(f"{t}：{'完成' if r.ok else '没做成'}，{r.say}")
        try:
            say = (await asyncio.to_thread(self.teacher.complete, SUMMARY_SYSTEM, f"总任务：{task}\n" + "\n".join(lines),
                                           max_tokens=300)).strip()
        except Exception:  # noqa: BLE001
            say = "；".join(lines)
        return AgentResult(ok_all, say or "；".join(lines), steps)

    async def _serial_approve(self, step: dict, why: str, risk: str):
        async with self._approve_lock:
            return await self.approve(step, why, risk)

    async def _quiet_say(self, text: str) -> None:
        if "备份节点" in text:  # the snapshot notice still matters when working in parallel
            await self.say(text)

    async def _try_local(self, task: str) -> AgentResult | None:
        """The local model does the task itself, the way it practises (learning.drill.run_agent): only
        commands classify() calls "safe" run, nothing is asked. -> a result, or None to go to the teacher
        (a step that would need approval, no conclusion, a repeated command, an unspeakable answer)."""
        from learning.drill import REPEATED, agent_execute, run_agent
        from learning.lora_prep import speakable

        execute = agent_execute(lambda s: self.runner(s))
        sandbox = agent_execute(self.sandbox) if self.sandbox is not None else None
        try:
            run = await asyncio.to_thread(run_agent, task, self.local_plan, execute=execute, classify=classify,
                                          sandbox=sandbox)
        except Exception as exc:  # noqa: BLE001 - the teacher path still works
            LOG.warning("[agent] local try crashed: %s", exc)
            return None
        hist = run["history"]
        why = (run["error"] or ("没执行命令就回答" if not hist else "")
               or ("重复了命令" if any(h["output"] == REPEATED for h in hist) else "")
               or ("" if speakable(run["say"]) else "结论念不出来"))
        LOG.info("[agent] local try %s: %s", "ok" if not why else f"failed ({why})", " | ".join(h["cmd"] for h in hist))
        if why:
            return None
        steps = [{"tool": "shell", "cmd": h["cmd"], "purpose": "本地自己查", "status": "ran", "code": h["code"],
                  "output": h["output"]} for h in hist]
        if self.progress:
            for s in steps:
                await self.progress({"purpose": s["purpose"], "action": describe_step(s), "code": s["code"], "output": s["output"]})
        return AgentResult(True, run["say"], steps)

    async def _run_single(self, task: str, creds: Credentials | None = None) -> AgentResult:
        if self.local_plan is not None and not creds:
            local = await self._try_local(task)
            if local is not None:
                return local
        known = [s for s in self.skills.search(task, k=3) if s.score >= 0.45]
        familiar = known[0] if known and known[0].score >= KNOWN_SCORE else None
        if not getattr(self.teacher, "available", False) and not creds:
            if familiar:
                return await self._replay(familiar)
            return AgentResult(False, "这个我还不会，老师现在也联系不上，等网络好了我再去学。")
        history: list[dict] = []
        if creds and creds.host:
            await self.say(f"好，我先用你给的账号登录 {creds.host}，把我的钥匙装上，以后就不用密码了。")
            setup = await self._setup_access(creds)
            history.append(setup)
            if setup["status"] == "denied":
                return AgentResult(False, "好的，那我先不连这台机器。", history)
            if setup["code"] != 0:
                return AgentResult(False, f"没连上：{setup['output']}。", history)
            if not getattr(self.teacher, "available", False):
                return AgentResult(True, f"连好了，以后用 {setup['alias']} 就能直接登录。", history)
        if familiar:
            await self.say("这个我学过，我按之前的做法来。")
        elif not history:
            await self.say("这个我还不太会，我去问问老师，一步步学着做。")
        reference = "\n\n".join(f"以前学会的做法「{s.task}」（成功 {s.successes} 次）：\n{s.describe()}" for s in known)
        tools_prompt = await self._tools_prompt(task)
        for _ in range(MAX_STEPS):
            ctx = f"任务：{task}\n"
            if self.ssh_hosts:
                ctx += f"已配置好密钥登录的主机（ssh 工具的 host 直接写别名，不用再查配置）：{'、'.join(self.ssh_hosts)}\n"
            if reference:
                ctx += f"\n{reference}\n（合适就直接照着做，不合适就按实际情况调整）\n"
            if tools_prompt:
                ctx += f"\n{tools_prompt}\n"
            if history:
                ctx += "\n已经执行的步骤：\n" + "\n".join(
                    f"{i + 1}. {describe_step(h)} -> {h['status']} 返回码 {h['code']}\n{h['output']}" for i, h in enumerate(history))
            step = {}
            for attempt in range(2):  # one retry when the reply is not the agreed JSON
                try:
                    raw = await asyncio.to_thread(self.teacher.complete, PLAN_SYSTEM, ctx, json_mode=True, max_tokens=1200)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("[agent] teacher failed: %s", exc)
                    return AgentResult(False, "问老师的时候网络断了，这次先没做成。", history)
                step = _norm_mcp(_step_of(_json(raw)))
                if step:
                    break
                LOG.warning("[agent] unusable plan (attempt %d): %s", attempt + 1, (raw or "")[:300])
            if not step:
                return AgentResult(False, "老师给的步骤我没看懂，这次先没做成。", history)
            if step.get("tool") == "done":
                return await self._finish(task, step, history, based_on=familiar)
            status, code, out = await self._gate_and_run(step)
            history.append({**step, "status": status, "code": code, "output": out})
            if status == "denied":
                return AgentResult(False, "好的，那这一步不做了，任务先停在这里。", history)
        return AgentResult(False, "试了好几步还没做成，我先停下来，免得越弄越乱。", history)

    async def _finish(self, task: str, step: dict, history: list[dict], based_on: Skill | None = None) -> AgentResult:
        say = str(step.get("say") or ("做完了。" if step.get("success") else "这次没做成。"))
        if not step.get("success") or not history:
            return AgentResult(False, say, history)
        # don't trust the claim: let the teacher check the evidence
        listing = "\n".join(f"{i + 1}. {json.dumps({k: h[k] for k in ('tool', 'cmd', 'host', 'name', 'purpose') if k in h}, ensure_ascii=False)}"
                            f" -> {h['status']} 返回码 {h['code']}\n{h['output']}" for i, h in enumerate(history))
        try:
            hosts = f"已配置主机：{'、'.join(self.ssh_hosts)}\n" if self.ssh_hosts else ""
            check = _json(await asyncio.to_thread(self.teacher.complete, VERIFY_SYSTEM, f"任务：{task}\n{hosts}步骤：\n{listing}",
                                                  json_mode=True, max_tokens=1200))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[agent] verify failed: %s", exc)
            return AgentResult(True, say, history)
        if not check.get("done"):
            return AgentResult(False, f"我以为做完了，但检查下来还没成：{check.get('why', '')}", history)
        steps = [s for s in check.get("skill_steps") or []
                 if isinstance(s, dict) and s.get("tool") != "ssh_key_setup" and classify(s)[0] != "block"]  # setup is one-off
        steps = [{k: s[k] for k in ("tool", "cmd", "host", "name", "text", "purpose") if k in s} for s in steps]
        skill_task = str(check.get("task") or task)
        # done the way a known skill does it: refresh that skill rather than keep a near-duplicate
        skill_id = self.skills.save(skill_task, steps, str(step.get("notes") or ""),
                                    update_id=based_on.id if based_on else 0) if steps else 0
        if skill_id and self.log_training:
            skill = Skill(skill_id, skill_task, steps, str(step.get("notes") or ""))
            self.log_training(skill_task, f"{say}\n我的做法是：\n{skill.describe()}")
        learned = bool(skill_id) and based_on is None
        return AgentResult(True, say + ("这个我学会了，下次我自己就能做。" if learned else ""), history, skill_id, learned)
