#!/usr/bin/env python3
"""MCP server (stdio, stdlib only): read web pages for 星期一.

Tools:
  fetch(url, max_length=4000, start=0)   the page as plain text (scripts/styles/tags removed)
  search(query, n=6)                     web search (DuckDuckGo HTML, cn.bing as fallback): title, url, snippet
Only http(s) URLs; private/loopback addresses are refused so a page can't be used to poke at the LAN.
"""
from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import sys
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux aarch64) MondayAgent/1.0"
SEARCH_UA = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
TOOLS = [{
    "name": "fetch",
    "description": "读取一个网页，返回去掉标签后的正文文字（长网页用 start 分段读）",
    "inputSchema": {"type": "object", "properties": {
        "url": {"type": "string", "description": "http 或 https 网址"},
        "max_length": {"type": "integer", "description": "最多返回多少字，默认 4000"},
        "start": {"type": "integer", "description": "从第几个字开始，默认 0"}},
        "required": ["url"]},
}, {
    "name": "search",
    "description": "上网搜索，返回若干条结果（标题、网址、摘要）；要看全文再用 fetch 读网址",
    "inputSchema": {"type": "object", "properties": {
        "query": {"type": "string", "description": "搜索词"},
        "n": {"type": "integer", "description": "最多几条，默认 6"}},
        "required": ["query"]},
}]


FAKE_IP = ipaddress.ip_network("198.18.0.0/15")  # the LAN's proxy answers DNS with these ("fake-ip"): really public


def _public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip in FAKE_IP:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def page_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def fetch(args: dict) -> tuple[bool, str]:
    url = str(args.get("url") or "").strip()
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False, "只能读取 http/https 网址"
    if not _public(parts.hostname):
        return False, "不读取局域网或本机地址"
    max_len = max(200, min(int(args.get("max_length") or 4000), 20000))
    start = max(0, int(args.get("start") or 0))
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6"})
    with urllib.request.urlopen(req, timeout=15) as r:
        ctype = r.headers.get("Content-Type", "")
        data = r.read(3_000_000)
        charset = r.headers.get_content_charset() or "utf-8"
    raw = data.decode(charset, "replace")
    text = page_text(raw) if "html" in ctype or raw.lstrip().startswith("<") else raw
    chunk = text[start:start + max_len]
    more = f"\n…（还有 {len(text) - start - max_len} 字，可用 start={start + max_len} 继续读）" if len(text) > start + max_len else ""
    return True, chunk + more


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": SEARCH_UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def parse_ddg(raw: str) -> list[dict]:
    out = []
    for m in re.finditer(r'(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', raw):
        href, title, rest = m.groups()
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(html.unescape(href)).query)
        url = q.get("uddg", [html.unescape(href)])[0]
        if url.startswith("//"):
            url = "https:" + url
        sn = re.search(r'(?s)class="result__snippet"[^>]*>(.*?)</a>', rest)
        if "duckduckgo.com/y.js" in url:   # ads
            continue
        out.append({"title": _clean(title), "url": url, "snippet": _clean(sn.group(1)) if sn else ""})
    return out


def parse_bing(raw: str) -> list[dict]:
    out = []
    for block in re.findall(r'(?s)<li class="b_algo"(.*?)</li>', raw):
        a = re.search(r'(?s)<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block)
        if not a:
            continue
        sn = re.search(r'(?s)<p[^>]*>(.*?)</p>', block)
        out.append({"title": _clean(a.group(2)), "url": html.unescape(a.group(1)), "snippet": _clean(sn.group(1)) if sn else ""})
    return out


def search(args: dict) -> tuple[bool, str]:
    query = str(args.get("query") or "").strip()
    if not query:
        return False, "搜索词为空"
    n = max(1, min(int(args.get("n") or 6), 10))
    results: list[dict] = []
    for url, parse in ((f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}", parse_ddg),
                       (f"https://cn.bing.com/search?q={urllib.parse.quote(query)}", parse_bing)):
        try:
            results = [r for r in parse(_get(url)) if r["url"].startswith("http")]
        except Exception:  # noqa: BLE001 - try the next engine
            results = []
        if results:
            break
    if not results:
        return False, "没搜到结果（搜索引擎可能暂时连不上）"
    return True, "\n\n".join(f"{i}. {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results[:n], 1))


def reply(rid, result=None, error=None) -> None:
    msg = {"jsonrpc": "2.0", "id": rid}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "id" not in msg:
            continue
        rid, method = msg["id"], msg.get("method")
        if method == "initialize":
            reply(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                        "serverInfo": {"name": "monday-web", "version": "1.0"}})
        elif method == "tools/list":
            reply(rid, {"tools": TOOLS})
        elif method == "tools/call":
            p = msg.get("params") or {}
            tool = {"fetch": fetch, "search": search}.get(p.get("name"))
            if tool is None:
                reply(rid, error={"code": -32602, "message": "unknown tool"})
                continue
            try:
                ok, text = tool(p.get("arguments") or {})
            except Exception as exc:  # noqa: BLE001 - report to the caller, keep serving
                ok, text = False, f"读取失败：{exc}"
            reply(rid, {"content": [{"type": "text", "text": text}], "isError": not ok})
        elif method == "ping":
            reply(rid, {})
        else:
            reply(rid, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
