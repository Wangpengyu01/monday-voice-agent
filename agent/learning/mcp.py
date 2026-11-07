"""Minimal MCP (Model Context Protocol) client: lets the agent use tools from external MCP servers.

Servers are configured in config/agent.yaml:

    mcp:
      servers:
        fetch:                                   # name the planner uses
          command: ["python3", "-m", "mcp_server_fetch"]
          env: {}                                # extra environment (never secrets in the repo)
          safe_tools: ["fetch"]                  # read-only tools: no approval needed
          enabled: true

Transport is stdio, JSON-RPC 2.0, one message per line. A server is started on first use and kept
running. Tools not listed in safe_tools go through the permission gate like any other step.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)
PROTOCOL = "2025-06-18"


class McpError(RuntimeError):
    pass


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    schema: dict = field(default_factory=dict)
    safe: bool = False

    def prompt(self) -> str:
        props = (self.schema or {}).get("properties") or {}
        required = set((self.schema or {}).get("required") or [])
        args = "，".join(f"{k}{'' if k in required else '?'}" for k in props) or "无参数"
        desc = (self.description or "").strip().replace("\n", " ")[:160]
        return f"{self.server}.{self.name}({args})：{desc}{'（只读）' if self.safe else ''}"


class McpServer:
    def __init__(self, name: str, command: list[str], env: dict | None = None, safe_tools: list[str] | None = None,
                 timeout: float = 30.0) -> None:
        self.name, self.command = name, command
        self.env = {**os.environ, "PYTHONIOENCODING": "utf-8", **{k: str(v) for k, v in (env or {}).items()}}
        self.safe_tools = set(safe_tools or [])
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.tools: list[McpTool] = []

    # -- transport
    def _send(self, msg: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + (timeout or self.timeout)
        assert self._proc and self._proc.stdout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise McpError(f"{self.name}: server closed the connection")
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # a log line on stdout
            if msg.get("id") != rid:
                continue  # a notification or a request from the server: not supported, ignored
            if "error" in msg:
                raise McpError(f"{self.name}.{method}: {msg['error'].get('message', msg['error'])}")
            return msg.get("result") or {}
        raise McpError(f"{self.name}.{method}: timed out")

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=self.env, bufsize=1)
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "monday-agent", "version": "1.0"}}, timeout=20)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = self._request("tools/list").get("tools") or []
        self.tools = [McpTool(self.name, t.get("name", ""), t.get("description", ""), t.get("inputSchema") or {},
                              t.get("name") in self.safe_tools) for t in tools if t.get("name")]
        LOG.info("[mcp] %s started: %d tools", self.name, len(self.tools))

    def list_tools(self) -> list[McpTool]:
        with self._lock:
            self.start()
            return list(self.tools)

    def call(self, tool: str, args: dict, timeout: float | None = None) -> tuple[bool, str]:
        with self._lock:
            self.start()
            res = self._request("tools/call", {"name": tool, "arguments": args or {}}, timeout=timeout)
        parts = []
        for item in res.get("content") or []:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type')} 内容]")
        return (not res.get("isError")), "\n".join(parts).strip() or "（没有输出）"

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _as_list(v) -> list[str]:
    """A YAML list, or the string the agent's small YAML loader leaves for "[a, b]" / "a b"."""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    text = str(v or "").strip()
    if text.startswith("[") and text.endswith("]"):
        return [x.strip().strip("'\"") for x in text[1:-1].split(",") if x.strip()]
    return text.split()


class McpHub:
    def __init__(self, servers: dict[str, McpServer]) -> None:
        self.servers = servers

    @classmethod
    def from_config(cls, cfg: dict | None) -> "McpHub":
        servers = {}
        for name, s in ((cfg or {}).get("servers") or {}).items():
            if not isinstance(s, dict) or not s.get("enabled", True) or not s.get("command"):
                continue
            enabled = str(s.get("enabled", True)).lower() not in ("false", "0", "no")
            if not enabled:
                continue
            servers[name] = McpServer(name, _as_list(s["command"]), s.get("env") if isinstance(s.get("env"), dict) else None,
                                      _as_list(s.get("safe_tools")), float(s.get("timeout", 30)))
        return cls(servers)

    def tools(self) -> list[McpTool]:
        out = []
        for s in self.servers.values():
            try:
                out += s.list_tools()
            except Exception as exc:  # noqa: BLE001 - a broken server must not take the agent down
                LOG.warning("[mcp] %s unavailable: %s", s.name, exc)
        return out

    def is_safe(self, server: str, tool: str) -> bool:
        s = self.servers.get(server)
        return bool(s and tool in s.safe_tools)

    def call(self, server: str, tool: str, args: dict) -> tuple[int, str]:
        s = self.servers.get(server)
        if s is None:
            return 127, f"没有这个 MCP 服务：{server}"
        try:
            ok, text = s.call(tool, args)
        except Exception as exc:  # noqa: BLE001
            return 1, f"MCP 调用失败：{exc}"
        return (0 if ok else 1), text

    def close(self) -> None:
        for s in self.servers.values():
            s.close()
