"""Rules 星期一 writes for herself, added after her fixed base prompts.

The base prompts stay in code and config (learning.drill.AGENT_SYSTEM; the persona in config/agent.yaml).
What she learns about how to work goes into run/learning/self_prompt.json as short rules, one list per
prompt: "agent" (how she does board tasks) and "voice" (how she talks, who she is). learning.selftune
proposes changes and keeps one only if the fixed exam (learning.exam) says she got better. Every
version is appended to self_prompt_history.jsonl, so any step can be undone (rollback()).

Guards, checked on every save rather than trusted to whoever proposed the rules:
  - at most MAX_RULES rules per section, each at most MAX_RULE_CHARS, a section at most MAX_CHARS;
  - nothing about permissions, sudo, keys, passwords, confirmations, the network, deleting or
    restarting (FORBIDDEN): those are the owner's rules, not hers to rewrite;
  - no exam answer written into a rule (the caller passes learning.exam.FIXED_ANSWERS).
Training data never carries the rules: learning.lora_prep strips them (strip()). A rule that works shows
up in the samples it produced, those train the weights, and the rule can later be retired.

    cd /userdata/agent && python3 -m learning.selfprompt show | rollback
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("MONDAY_LEARNING_DIR", "/userdata/agent/run/learning"))
FILE = "self_prompt.json"
HISTORY = "self_prompt_history.jsonl"
MARKER = "\n\n我自己总结的经验："
SECTIONS = ("agent", "voice")
MAX_RULES = 8
MAX_RULE_CHARS = 80
MAX_CHARS = {"agent": 400, "voice": 300}
# "确认" alone is fine ("用 systemctl is-active 确认" = check); skipping the owner's confirmation is not.
# The first real round (2026-11-07) was refused over exactly that word.
FORBIDDEN = re.compile(r"sudo|密钥|密码|私钥|token|权限|(不用|无需|不必|别|免|省掉|跳过|绕过)[^，。；]{0,4}(问|确认|批准|同意|询问)|"
                       r"跳过|绕过|联网|上网|curl|wget|ssh|scp|rm\s|chmod|chown|删除|删掉|写入|覆盖|改系统|重启|关机|reboot|kill|"
                       r"systemctl\s+(start|stop|restart|enable|disable)",
                       re.I)


def _dir(learning_dir: Path | None) -> Path:
    return Path(learning_dir) if learning_dir is not None else DEFAULT_DIR


def clean(rules: dict) -> dict:
    """Section -> list of trimmed, non-empty, de-duplicated rules (other keys dropped)."""
    out = {}
    for s in SECTIONS:
        seen, items = set(), []
        for r in rules.get(s) or []:
            r = " ".join(str(r).split()).lstrip("-• ").strip()
            if r and r not in seen:
                seen.add(r)
                items.append(r)
        out[s] = items
    return out


def check(rules: dict, banned: tuple[str, ...] | list[str] = ()) -> list[str]:
    """What is wrong with these rules (empty list: they may be saved)."""
    problems = []
    for s, items in clean(rules).items():
        if len(items) > MAX_RULES:
            problems.append(f"{s}: {len(items)} 条，最多 {MAX_RULES} 条")
        if sum(len(r) for r in items) > MAX_CHARS[s]:
            problems.append(f"{s}: 一共超过 {MAX_CHARS[s]} 字")
        for r in items:
            if len(r) > MAX_RULE_CHARS:
                problems.append(f"{s}: 太长（{len(r)} 字）：{r[:30]}…")
            if m := FORBIDDEN.search(r):
                problems.append(f"{s}: 碰了主人的规矩（{m.group(0)}）：{r[:40]}")
            if hit := next((b for b in banned if b and b in r), ""):
                problems.append(f"{s}: 写进了考题答案（{hit}）：{r[:40]}")
    return problems


def load(learning_dir: Path | None = None) -> dict:
    try:
        d = json.loads((_dir(learning_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return {"version": int(d.get("version", 0)), "updated": str(d.get("updated", "")), **clean(d)}


def save(rules: dict, why: str, *, scores: dict | None = None, banned: tuple[str, ...] | list[str] = (),
         learning_dir: Path | None = None) -> int:
    """Store a new version (ValueError if check() objects). Returns the new version number."""
    problems = check(rules, banned)
    if problems:
        raise ValueError("；".join(problems))
    d = _dir(learning_dir)
    d.mkdir(parents=True, exist_ok=True)
    version = load(d)["version"] + 1
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    body = {"version": version, "updated": now, **clean(rules)}
    tmp = d / f".{FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, d / FILE)
    with (d / HISTORY).open("a", encoding="utf-8") as f:
        f.write(json.dumps({**body, "why": why, "scores": scores or {}}, ensure_ascii=False) + "\n")
    return version


def history(learning_dir: Path | None = None) -> list[dict]:
    f = _dir(learning_dir) / HISTORY
    out = []
    for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def rollback(learning_dir: Path | None = None) -> int:
    """Back to the rules before the current version (saved as a new version, so it can be undone too)."""
    cur = load(learning_dir)
    older = [h for h in history(learning_dir) if int(h.get("version", 0)) < cur["version"]]
    prev = older[-1] if older else {s: [] for s in SECTIONS}
    return save(prev, f"回滚：从第 {cur['version']} 版退回到第 {prev.get('version', 0)} 版", learning_dir=learning_dir)


def block(section: str, rules: dict | None = None, learning_dir: Path | None = None) -> str:
    items = (rules if rules is not None else load(learning_dir)).get(section) or []
    return (MARKER + "".join(f"\n- {r}" for r in items)) if items else ""


def apply(base: str, section: str, rules: dict | None = None, learning_dir: Path | None = None) -> str:
    """The base prompt with her rules for this section after it (rules=None: the stored ones)."""
    return strip(base) + block(section, clean(rules) if rules is not None else None, learning_dir)


def strip(text: str) -> str:
    """The prompt without her rules (what training data and prompt matching use)."""
    if MARKER not in (text or ""):
        return text
    head, _, tail = text.partition(MARKER)
    rest = tail.split("\n\n", 1)   # anything added after the rules block (e.g. memory context) stays
    return head + ("\n\n" + rest[1] if len(rest) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一's own prompt rules")
    ap.add_argument("cmd", choices=("show", "rollback"))
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    a = ap.parse_args()
    d = Path(a.dir)
    if a.cmd == "rollback":
        print("now version", rollback(d))
    cur = load(d)
    print(f"version {cur['version']} ({cur['updated'] or 'never'})")
    for s in SECTIONS:
        print(f"[{s}]")
        for r in cur[s]:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
