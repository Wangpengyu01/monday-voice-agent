"""上网查资料, like a person: search -> read the best pages -> the teacher writes it up with sources -> memory.

    python3 -m learning.research "RK3588 GPIO 编号怎么算"

Used by the daily owner reflection (the ideas she comes up with get researched) and available on its own.
Only public web pages are read (mcp_servers/web_fetch.py refuses LAN addresses). The write-up keeps the
source URLs, so a later answer can say where it came from.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))
from mcp_servers import web_fetch  # noqa: E402

LOG = logging.getLogger("research")

RESEARCH_SYSTEM = (
    "你在帮语音助手「星期一」上网查资料。下面是搜索结果和几篇网页的正文。只根据这些材料回答，"
    "材料互相矛盾时说明分歧，材料里没有的就说没查到，不要编。"
    '只输出 JSON：{"answer": "三到六句口语化的结论（会被朗读，不要符号和代码）", '
    '"facts": [{"q": "以后可能被问到的问题", "a": "一两句答案"}], "sources": ["用到的网址"], '
    '"confidence": "high|medium|low"}'
)


def gather(query: str, *, pages: int = 3, per_page: int = 3500,
           search=web_fetch.search, fetch=web_fetch.fetch) -> tuple[str, list[str]]:
    ok, listing = search({"query": query, "n": 6})
    if not ok:
        return "", []
    urls = re.findall(r"^https?://\S+", listing, re.M)
    texts, used = [listing], []
    for url in urls:
        if len(used) >= pages:
            break
        try:
            ok, body = fetch({"url": url, "max_length": per_page})
        except Exception:  # noqa: BLE001 - a dead link just gets skipped
            continue
        if ok and len(body) > 200:
            texts.append(f"【网页 {url}】\n{body}")
            used.append(url)
    return "\n\n".join(texts), used


def research(teacher, query: str, *, store=None, notes_dir: Path | None = None, **kw) -> dict:
    material, used = gather(query, **kw)
    if not material:
        return {"query": query, "answer": "", "facts": [], "sources": [], "confidence": "low", "error": "没搜到"}
    raw = teacher.complete(RESEARCH_SYSTEM, f"要查的问题：{query}\n\n{material[:14000]}", json_mode=True, max_tokens=1200)
    try:
        res = json.loads(raw)
    except ValueError:
        res = {}
    out = {"query": query, "answer": str(res.get("answer") or "").strip(),
           "facts": [f for f in res.get("facts") or [] if isinstance(f, dict) and f.get("q") and f.get("a")][:6],
           "sources": [s for s in res.get("sources") or used if str(s).startswith("http")][:6],
           "confidence": str(res.get("confidence") or "medium")}
    if store is not None and out["answer"] and out["confidence"] != "low":
        store.remember(query, out["answer"], source="web")
        for f in out["facts"]:
            store.remember(str(f["q"]), str(f["a"]), source="web")
    if notes_dir is not None and out["answer"]:
        notes_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w]+", "-", query)[:40].strip("-") or "research"
        md = [f"# 查资料：{query}", "", f"_{time.strftime('%Y-%m-%d %H:%M')} · 可信度 {out['confidence']}_", "", out["answer"], "",
              "## 记下的要点", *[f"- {f['q']}：{f['a']}" for f in out["facts"]], "", "## 来源", *[f"- {s}" for s in out["sources"]]]
        (notes_dir / f"{time.strftime('%Y-%m-%d')}-{slug}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    from learning.reflect import build_runtime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    query = " ".join(a for a in sys.argv[1:] if not a.startswith("--")).strip()
    if not query:
        raise SystemExit('usage: python3 -m learning.research "要查的问题"')
    rt = build_runtime(str(AGENT_ROOT / "config" / "agent.yaml"))
    out = research(rt.teacher, query, store=rt.store, notes_dir=rt.learning_dir / "research")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
