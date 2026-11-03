"""Boot self-test: check the pieces 星期一 needs and say one sentence about it.

Runs once per boot (keyed by the kernel boot id, so restarting the service during a deploy stays
quiet) and on request ("系统自检"). Each check is cheap and has its own timeout; the LLM and TTS
checks wait while the models are still loading right after power-on.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from tts.number_cn import int_cn

BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


@dataclass
class Check:
    name: str  # spoken name, e.g. "大模型"
    ok: bool
    problem: str = ""  # spoken when not ok, e.g. "大模型没有响应"
    detail: str = ""  # spoken in the full (manual) report


def current_boot_id() -> str:
    try:
        return BOOT_ID_FILE.read_text().strip()
    except OSError:
        return ""


def wait_llm(sock_path: str, ping, timeout: float) -> Check:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if Path(sock_path).exists() and ping(sock_path):
                return Check("大模型", True)
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return Check("大模型", False, "大模型还没有响应")
        time.sleep(3)


def wait_tts(sock_path: str | None, timeout: float) -> Check:
    if not sock_path:
        return Check("语音合成", True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(sock_path)
                s.sendall(b'{"type":"ping"}\n')
                if json.loads(s.makefile("r").readline()).get("type") == "pong":
                    return Check("语音合成", True)
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            return Check("语音合成", False, "语音合成服务没有响应")
        time.sleep(2)


def check_mic() -> Check:
    """The VAD keeps an arecord process open on the codec while listening."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if cmd.startswith(b"arecord"):
            return Check("麦克风", True)
    return Check("麦克风", False, "麦克风没有在收音")


def check_network(host: str = "api.anthropic.com", port: int = 443) -> list[Check]:
    lan = False
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 2 and fields[1] == "00000000":  # default route
                lan = True
                break
    except OSError:
        pass
    if not lan:
        return [Check("网络", False, "网络没有连接")]
    try:
        with socket.create_connection((host, port), timeout=4):
            pass
        return [Check("网络", True)]
    except OSError:
        return [Check("网络", True), Check("外网", False, "局域网正常，但外网暂时连不上")]


def _gb(n_bytes: float) -> str:
    v = round(n_bytes / 1024 ** 3, 1)
    whole, frac = int(v), int(round((v - int(v)) * 10))
    return f"{int_cn(whole)}点{int_cn(frac)}G" if frac else f"{int_cn(whole)}G"


def check_resources(data_dir: str = "/userdata") -> list[Check]:
    out: list[Check] = []
    try:
        free = shutil.disk_usage(data_dir).free
        out.append(Check("存储", free >= 2 * 1024 ** 3, f"存储空间只剩{_gb(free)}", f"存储剩余{_gb(free)}"))
    except OSError:
        pass
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        avail = int(info["MemAvailable"].split()[0]) * 1024
        out.append(Check("内存", avail >= 768 * 1024 ** 2, f"内存只剩{_gb(avail)}", f"内存剩余{_gb(avail)}"))
    except (OSError, KeyError, ValueError):
        pass
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    if temps:
        t = int(max(temps))
        out.append(Check("温度", t < 80, f"芯片温度偏高，有{int_cn(t)}度", f"芯片温度{int_cn(t)}度"))
    return out


def run_checks(*, llm_sock: str, llm_ping, tts_sock: str | None, wait_s: float = 120.0) -> list[Check]:
    checks = [wait_llm(llm_sock, llm_ping, wait_s), wait_tts(tts_sock, min(wait_s, 60.0)), check_mic()]
    checks += check_network()
    checks += check_resources()
    return checks


def announcement(checks: list[Check], *, full: bool = False, name: str = "星期一") -> str:
    bad = [c for c in checks if not c.ok]
    if not bad:
        head = f"{name}已就绪，系统自检全部正常。"
    else:
        problems = "；".join(c.problem for c in bad)
        head = f"{name}已启动。自检发现：{problems}。其余正常。"
    if full:
        details = "，".join(c.detail for c in checks if c.ok and c.detail)
        if details:
            head += details + "。"
    return head
