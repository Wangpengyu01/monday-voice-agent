"""Voice side of the device agent: "扫描设备" / "家里有哪些设备" / "设备配对码"."""
from __future__ import annotations

import asyncio
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable

from devices import pairing
from devices.inventory import Inventory, Known
from devices.scanner import KIND_NAMES, scan
from devices.vault import Vault

PORT = 8767
_SCAN = re.compile(r"(?:扫描|扫一下|扫一扫|搜一下|搜索|查一下|找一下|看看).{0,6}(?:设备|局域网|网络里)")
_LIST = re.compile(r"(?:家里|局域网|网络里|现在)?(?:有哪些|有什么|有几台|多少台?)设备|设备(?:列表|清单)")
_PAIR = re.compile(r"(?:设备)?配对码|(?:填|录入|输入|添加)(?:设备的?)?(?:账号|密码)|设备管理")
_DIGITS = "零一二三四五六七八九"


def parse_command(text: str) -> str | None:
    t = re.sub(r"[\s，,。！!？?]", "", text or "")
    if len(t) > 20:
        return None  # a long sentence that only mentions devices is a question for the model
    if _PAIR.search(t):
        return "pair"
    if _SCAN.search(t):
        return "scan"
    if _LIST.search(t):
        return "list"
    return None


def spoken_digits(s: str) -> str:
    """'10.0.0.177' -> '一零点零点零点一七七' (how people read addresses and codes out loud)."""
    return "".join(_DIGITS[int(c)] if c.isdigit() else ("点" if c == "." else c) for c in s)


def board_ip() -> str:
    out = subprocess.run(["ip", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True).stdout
    m = re.search(r"\bsrc (\S+)", out)
    return m.group(1) if m else ""


def describe(devices: list[Known]) -> str:
    """'一个路由器、两台 Windows 电脑和一部手机'"""
    counts = Counter(d.kind for d in devices if d.kind != "unknown")
    unit = {"phone": "部", "router": "个", "ap": "个", "camera": "个", "speaker": "个", "printer": "台"}
    num = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    parts = [f"{num.get(n, str(n))}{unit.get(k, '台')}{KIND_NAMES[k]}" for k, n in counts.most_common()]
    text = "，".join(parts[:-1]) + "，还有" + parts[-1] if len(parts) > 1 else "".join(parts)
    unknown = sum(1 for d in devices if d.kind == "unknown")
    if unknown:
        text += ("，另外" if text else "") + f"{num.get(unknown, str(unknown))}台没认出来"
    return text


class DeviceAgent:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.inventory = Inventory(self.run_dir / "devices.db")
        self.vault = Vault(self.run_dir / "vault.db")
        self._scanning = asyncio.Lock()

    def pair_text(self) -> str:
        code = pairing.new_code(self.run_dir / "pair.json")
        ip = board_ip()
        return (f"配对码是{'，'.join(spoken_digits(c) for c in code)}，十分钟内有效。"
                f"用手机或电脑浏览器打开{spoken_digits(ip)}冒号{spoken_digits(str(PORT))}，输入配对码，就能给设备填账号密码了。")

    async def handle(self, cmd: str, say: Callable[[str], Awaitable[None]]) -> str:
        if cmd == "pair":
            return self.pair_text()
        if cmd == "list":
            devices = self.inventory.all(seen_within=7 * 86400)
            if not devices:
                cmd = "scan"
            else:
                return f"上次扫到{len(devices)}台设备：{describe(devices)}。" + self._cred_hint(devices)
        if self._scanning.locked():
            return "正在扫描呢，稍等一下。"
        async with self._scanning:
            await say("好嘞，我扫一下家里的网络，大概十几秒。")
            found = await scan()
            new = await asyncio.to_thread(self.inventory.update, found)
        known = self.inventory.all(seen_within=600)
        text = f"扫完了，一共{len(found)}台设备：{describe(known)}。"
        if new:
            names = "，".join(d.name for d in new[:3])
            text += f"其中{len(new)}台是新发现的，比如{names}。" if len(new) > 3 else f"新发现的有{names}。"
        return text + self._cred_hint(known)

    def _cred_hint(self, devices: list[Known]) -> str:
        stored = self.vault.describe()
        operable = [d for d in devices if d.kind in ("windows-pc", "mac", "linux", "nas", "router", "ap",
                                                     "camera", "home-assistant")]
        missing = [d for d in operable if d.key not in stored]
        if not missing:
            return ""
        return (f"有{len(missing)}台我还没有账号密码，以后想让我操作它们，就说“设备配对码”，"
                "在网页上填一下，密码只加密存在我这里。")
