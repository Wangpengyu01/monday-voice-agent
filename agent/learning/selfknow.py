"""认识自己: what 星期一 is made of - her own code and the board she runs on - as class material.

    python3 -m learning.selfknow survey      # (re)write run/learning/self/hardware.md + code_map.json
    python3 -m learning.selfknow topics      # list the self lessons in order

The hardware survey is strictly read-only: it lists GPIO lines (who claimed them, in/out, level), PWM,
I2C, UART, audio, camera, USB, NPU, network and storage. Nothing is written to /sys or /dev; driving a
pin is an agent action and goes through the "ask" permission tier like any other device change.
The code map walks her own source (not tests, not run/, never /userdata/secrets) and cuts each file
at top-level def/class boundaries into pieces a teacher can read in one go.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["orchestrator", "llm", "asr", "tts", "learning", "ops", "devices", "ui", "daemon", "scripts", "mcp_servers"]
CODE_EXT = {".py", ".sh", ".cpp", ".cc", ".h"}
SKIP = re.compile(r"(^|/)(tests?|board_tests|__pycache__|thirdparty|third_party|internvl_sdk|build|bin)(/|$)|(^|/)_test_|_test\.py$|\.bak")
CHUNK = 3500
# the order she learns herself in: how a sentence travels through her first, the rest after
FIRST = ["orchestrator/main.py", "asr/normalize.py", "llm/client.py", "daemon/llm_daemon.cpp", "tts/sentence_split.py",
         "learning/intent.py", "learning/agent.py", "learning/memory.py", "learning/teacher.py", "learning/school.py",
         "learning/reflect.py", "ops/snapshot.py", "ui/server.py"]

RK_GPIO_RULE = ("瑞芯微 GPIO 命名规则：GPIO{组}_{A/B/C/D}{0-7}，Linux 编号 = 组×32 + 字母序号×8 + 数字"
                "（A=0、B=1、C=2、D=3）。例如 GPIO1_A7 = 1×32 + 0×8 + 7 = 39，GPIO4_C6 = 4×32 + 2×8 + 6 = 150。")


@dataclass
class Unit:
    path: str
    part: int
    names: list[str]
    text: str

    @property
    def topic(self) -> str:
        what = "、".join(self.names[:4]) or "开头部分"
        return f"认识自己 · 我的代码 {self.path}（{what}）"


def _run(cmd: str, root: bool = False) -> str:
    try:
        r = subprocess.run((["sudo", "-n"] if root else []) + ["bash", "-c", cmd], capture_output=True, text=True, errors="replace", timeout=10)
        return (r.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def survey() -> dict[str, str]:
    """Read-only facts about the board, section -> text."""
    s: dict[str, str] = {}
    s["整机"] = "\n".join(x for x in [
        "型号：" + _run("tr -d '\\0' < /proc/device-tree/model"),
        "内核：" + _run("uname -r"),
        "CPU：" + _run("lscpu | grep -E 'Model name|^CPU\\(s\\)' | sed 's/  */ /g' | tr '\\n' ' '"),
        "内存：" + _run("free -h | awk '/Mem/{print $2\" 总共，\"$7\" 可用\"}'"),
        "存储：" + _run("df -h / /userdata 2>/dev/null | awk 'NR>1{print $6\" \"$2\" 用了 \"$5}' | tr '\\n' ';'"),
        "NPU：RK3588 自带 NPU；另有 RK1828 协处理器跑大模型和语音合成（" + (_run("ls /dev | grep -iE 'rknpu|rk1820|rk182' | tr '\\n' ' '") or "设备节点名见驱动") + "）",
    ] if not x.endswith("："))
    gpio = _run("cat /sys/kernel/debug/gpio", root=True)
    s["GPIO"] = (RK_GPIO_RULE + "\n已被占用的引脚（编号、用途、方向、电平）：\n" + gpio) if gpio else (
        RK_GPIO_RULE + "\n控制器：" + _run("ls /dev | grep gpiochip | tr '\\n' ' '") + "（读占用情况需要 root）")
    s["PWM / I2C / 串口"] = "\n".join([
        "PWM：" + _run("for c in /sys/class/pwm/pwmchip*; do echo \"$(basename $c) $(readlink -f $c/device | sed 's#.*/##') 通道数 $(cat $c/npwm)\"; done | tr '\\n' ';'"),
        "I2C：" + _run("for b in /sys/class/i2c-adapter/i2c-*; do echo \"$(basename $b) $(cat $b/name)\"; done | tr '\\n' ';'"),
        "串口：" + _run("ls /dev/ttyS* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\\n' ' '"),
        "SPI：" + (_run("ls /dev/spidev* 2>/dev/null | tr '\\n' ' '") or "没有开启（设备树里没打开 spidev）"),
    ])
    s["声音 / 摄像头 / USB"] = "\n".join([
        "放音：" + _run("aplay -l 2>/dev/null | grep ^card | head -3 | tr '\\n' ' '"),
        "录音：" + _run("arecord -l 2>/dev/null | grep ^card | head -3 | tr '\\n' ' '"),
        "摄像头：" + _run("ls -l /dev/video-camera0 2>/dev/null | sed 's/.*-> //'; v4l2-ctl --list-devices 2>/dev/null | grep -v '^\\s' | head -6 | tr '\\n' ';'"),
        "USB：" + _run("lsusb | grep -v 'root hub' | sed 's/^Bus [0-9]* Device [0-9]*: ID [0-9a-f:]* //' | tr '\\n' ';'"),
    ])
    s["网络"] = _run("ip -br addr | grep -v '^lo' | awk '{print $1\" \"$2\" \"$3}' | sed 's#/[0-9]*##'")
    s["我自己的服务"] = _run("systemctl list-units 'r1-*' --all --no-legend --plain | awk '{print $1\" \"$3\"/\"$4}'")
    return {k: v for k, v in s.items() if v.strip()}


def hardware_md(sections: dict[str, str]) -> str:
    return "# 我的硬件（只读勘察）\n\n" + "\n\n".join(f"## {k}\n\n```\n{v}\n```" for k, v in sections.items()) + "\n"


def _chunks(text: str, lang: str) -> list[tuple[list[str], str]]:
    """Cut at top-level def/class (Python) or function definitions (C/C++/sh), keep pieces <= CHUNK chars."""
    head = re.compile(r"^(?:    )?(?:async\s+)?(?:def|class)\s+(\w+)" if lang == "py" else
                      r"^(?:static\s+)?[\w:<>*&\s]+?\b(\w+)\s*\([^;]*$|^(\w+)\s*\(\)\s*\{", re.M)
    starts = [(m.start(), next(g for g in m.groups() if g)) for m in head.finditer(text)]
    cuts = [0] + [p for p, _ in starts if p > 0] + [len(text)]
    names_at = dict(starts)
    out: list[tuple[list[str], str]] = []
    cur_names, cur = [], ""
    for a, b in zip(cuts, cuts[1:]):
        piece = text[a:b]
        nm = [names_at[a]] if a in names_at else []
        if cur and len(cur) + len(piece) > CHUNK:
            out.append((cur_names, cur))
            cur_names, cur = [], ""
        base = (nm or cur_names or ["开头"])[0]
        while len(piece) > CHUNK:            # one huge function: split it by lines
            cut = piece.rfind("\n", 0, CHUNK) or CHUNK
            out.append((nm or cur_names, piece[:cut]))
            piece, nm = piece[cut:], [f"{base}（续）"]
        cur_names += nm
        cur += piece
    if cur.strip():
        out.append((cur_names, cur))
    return out


def code_units(root: Path = AGENT_ROOT) -> list[Unit]:
    files = []
    for d in CODE_DIRS:
        base = root / d
        if base.is_dir():
            files += [p for p in base.rglob("*") if p.suffix in CODE_EXT and p.is_file()
                      and not SKIP.search(p.relative_to(root).as_posix())]
    rank = {p: i for i, p in enumerate(FIRST)}
    files.sort(key=lambda p: (rank.get(p.relative_to(root).as_posix(), len(FIRST)), p.relative_to(root).as_posix()))
    units: list[Unit] = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        text = f.read_text(encoding="utf-8", errors="replace")
        if "secrets" in rel or not text.strip():
            continue
        for i, (names, piece) in enumerate(_chunks(text, "py" if f.suffix == ".py" else "c")):
            units.append(Unit(rel, i, names, piece))
    return units


def self_topics(root: Path = AGENT_ROOT, learning_dir: Path | None = None) -> list[tuple[str, str]]:
    """(topic, material) in learning order: the hardware first, then her code."""
    learning_dir = learning_dir or root / "run" / "learning"
    hw_f = learning_dir / "self" / "hardware.json"
    try:
        hw = json.loads(hw_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        hw = {}
    topics = [(f"认识自己 · 我的硬件：{k}", v) for k, v in hw.items()]
    topics += [(u.topic, f"文件 {u.path}（第 {u.part + 1} 段）：\n{u.text}") for u in code_units(root)]
    return topics


def write_survey(learning_dir: Path, root: Path = AGENT_ROOT) -> dict:
    d = learning_dir / "self"
    d.mkdir(parents=True, exist_ok=True)
    sec = survey()
    (d / "hardware.json").write_text(json.dumps(sec, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "hardware.md").write_text(hardware_md(sec), encoding="utf-8")
    units = code_units(root)
    (d / "code_map.json").write_text(json.dumps([{k: v for k, v in asdict(u).items() if k != "text"} | {"chars": len(u.text)}
                                                 for u in units], ensure_ascii=False, indent=1), encoding="utf-8")
    return {"hardware_sections": list(sec), "code_files": len({u.path for u in units}), "code_units": len(units)}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "survey"
    learning_dir = AGENT_ROOT / "run" / "learning"
    if cmd == "survey":
        print(json.dumps(write_survey(learning_dir), ensure_ascii=False))
    elif cmd == "topics":
        for i, (t, m) in enumerate(self_topics(AGENT_ROOT, learning_dir)):
            print(f"{i:3d} {t} ({len(m)} chars)")


if __name__ == "__main__":
    main()
