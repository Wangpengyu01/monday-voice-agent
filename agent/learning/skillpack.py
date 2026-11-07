"""External skill packs: know-how written down as folders, the way Claude skills are.

    skills/<name>/SKILL.md        front matter (name, description, risk, tags) + instructions
    skills/<name>/scripts/*.sh    helpers the instructions refer to

Packs live in agent/skills/ (shipped with 星期一) and run/skills/ (installed later: uploaded on the
web page, or exported from what the agent learned). The agent's planner gets the 1-2 packs that fit
the task, so the teacher plans with that know-how instead of starting from nothing.

risk: safe  marks a pack whose scripts only read (e.g. a LAN scan): running exactly
            `bash <pack>/scripts/<file>.sh [plain args]` then needs no approval. Anything else a pack
            does goes through the normal permission gate (ask / danger / block).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from learning.memory import similarity

AGENT_ROOT = Path(os.environ.get("AGENT_ROOT", "/userdata/agent"))
DEFAULT_DIRS = (AGENT_ROOT / "skills", AGENT_ROOT / "run" / "skills")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
_SCRIPT_CALL = re.compile(r"^bash\s+(\S+/scripts/[\w\-.]+\.sh)((?:\s+[\w.\-/:@=,]+)*)\s*$")


@dataclass
class SkillPack:
    name: str
    description: str
    body: str
    dir: Path
    risk: str = "ask"
    tags: list[str] = field(default_factory=list)

    @property
    def scripts(self) -> list[str]:
        d = self.dir / "scripts"
        return sorted(str(p) for p in d.glob("*.sh")) if d.is_dir() else []

    def prompt(self, limit: int = 1800) -> str:
        body = self.body.strip()
        if len(body) > limit:
            body = body[:limit] + "\n…（后面省略）"
        scripts = "；".join(self.scripts) or "无"
        return f"### 技能「{self.name}」：{self.description}\n{body}\n脚本（用 shell 工具执行 bash 路径 参数）：{scripts}"


def _parse(path: Path) -> SkillPack | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict[str, str] = {}
    m = _FRONT.match(text)
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                meta[k.strip().lower()] = v.strip().strip("'\"")
        text = text[m.end():]
    name = meta.get("name") or path.parent.name
    desc = meta.get("description", "")
    if not desc:
        return None
    tags = [t.strip() for t in re.split(r"[,，\s]+", meta.get("tags", "").strip("[]")) if t.strip()]
    risk = meta.get("risk", "ask").lower()
    return SkillPack(name, desc, text, path.parent, risk if risk in ("safe", "ask") else "ask", tags)


def load(dirs=DEFAULT_DIRS) -> list[SkillPack]:
    packs: dict[str, SkillPack] = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*/SKILL.md")) if Path(d).is_dir() else []:
            p = _parse(f)
            if p:
                packs[p.name] = p  # later dirs (installed) override shipped ones of the same name
    return list(packs.values())


def relevant(packs: list[SkillPack], task: str, k: int = 2, threshold: float = 0.2) -> list[SkillPack]:
    scored = []
    for p in packs:
        score = max(similarity(task, p.description), similarity(task, p.name))
        score += 0.25 * sum(1 for t in p.tags if t and t.lower() in task.lower())
        if score >= threshold:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def script_is_safe(cmd: str, packs: list[SkillPack]) -> bool:
    """`bash <safe pack>/scripts/x.sh plain-args` - nothing else (no pipes, redirects, odd characters)."""
    m = _SCRIPT_CALL.match((cmd or "").strip())
    if not m:
        return False
    script = os.path.normpath(m.group(1))
    return any(p.risk == "safe" and script in {os.path.normpath(s) for s in p.scripts} for p in packs)
