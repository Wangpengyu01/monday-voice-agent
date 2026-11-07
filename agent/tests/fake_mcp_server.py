"""A tiny stdio MCP server for tests: tools `echo` (read-only) and `write` (changes things)."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo the text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "write", "description": "Pretend to write a file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
]

for line in sys.stdin:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if "id" not in msg:
        continue  # notification
    method, rid = msg.get("method"), msg["id"]
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        text = f"echo: {args.get('text', '')}" if name == "echo" else f"wrote {args.get('path')}"
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no such method"}}), flush=True)
        continue
    print("a log line that is not JSON", flush=True)  # servers sometimes do this; the client must cope
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)
