"""Internet-only HTTP(S) proxy for the project sandbox (learning.projbox).

The sandbox has no network of its own (bwrap --unshare-net). Programs in it use
http(s)_proxy=http://127.0.0.1:3128; a tiny forwarder inside the sandbox (learning/projbox_fwd.py) passes
each connection to this proxy's Unix socket, which is bind-mounted in. Here every target is resolved once
and refused unless all its addresses are public, and the connection goes to exactly the address that was
checked (no second lookup to rebind). The owner's LAN - the PC, the GPU server, 100.3, the board itself -
stays out of reach (owner's choice, 2026-11-07: "外网可以，内网不行"). Only ports 80 and 443.

    serve(socket_path, log_path) -> server; server.shutdown() stops it.
"""
from __future__ import annotations

import ipaddress
import json
import select
import socket
import socketserver
import threading
import time
import urllib.parse
from pathlib import Path

PORTS = (80, 443)
FAKE_IP = ipaddress.ip_network("198.18.0.0/15")  # the LAN's DNS answers public names with these ("fake-ip")
HEAD_MAX = 16384
IDLE_S = 300
CONNECT_S = 15


def public_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if a in FAKE_IP:
        return True
    return not (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast or a.is_unspecified)


def resolve_public(host: str, port: int) -> tuple[str, str]:
    """-> (ip to connect to, "") or ("", why refused)."""
    if port not in PORTS:
        return "", f"port {port} not allowed"
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return "", f"cannot resolve: {exc}"
    ips = [i[4][0] for i in infos]
    if not ips or not all(public_ip(ip) for ip in ips):
        return "", "LAN or local address"
    v4 = [ip for ip in ips if ":" not in ip]
    return (v4 or ips)[0], ""


def parse_head(head: bytes) -> tuple[str, str, int, bytes]:
    """-> (method, host, port, request to send upstream; b"" for CONNECT)."""
    lines = head.split(b"\r\n")
    method, target, version = lines[0].decode("latin-1").split(" ", 2)
    if method.upper() == "CONNECT":
        host, _, port = target.rpartition(":")
        return "CONNECT", host.strip("[]"), int(port or 443), b""
    u = urllib.parse.urlsplit(target)
    if u.scheme != "http" or not u.hostname:
        raise ValueError("only absolute http:// URLs or CONNECT")
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    rest = [ln for ln in lines[1:] if not ln.lower().startswith(b"proxy-")]
    upstream = f"{method} {path} {version}".encode("latin-1") + b"\r\n" + b"\r\n".join(rest)
    return method, u.hostname, u.port or 80, upstream


def _pipe(a: socket.socket, b: socket.socket) -> None:
    socks = [a, b]
    while True:
        r, _, _ = select.select(socks, [], [], IDLE_S)
        if not r:
            return
        for s in r:
            data = s.recv(65536)
            if not data:
                return
            (b if s is a else a).sendall(data)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        c: socket.socket = self.request
        c.settimeout(30)
        head = b""
        try:
            while b"\r\n\r\n" not in head and len(head) < HEAD_MAX:
                chunk = c.recv(4096)
                if not chunk:
                    return
                head += chunk
            head, _, body = head.partition(b"\r\n\r\n")
            method, host, port, upstream = parse_head(head + b"\r\n\r\n")
        except (ValueError, OSError, IndexError):
            c.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        ip, why = resolve_public(host, port)
        self.server.log(host, port, why)   # type: ignore[attr-defined]
        if not ip:
            c.sendall(f"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nblocked: {why}\n".encode())
            return
        try:
            up = socket.create_connection((ip, port), timeout=CONNECT_S)
        except OSError as exc:
            c.sendall(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{exc}\n".encode())
            return
        with up:
            up.settimeout(None)
            c.settimeout(None)
            if method == "CONNECT":
                c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if body:
                    up.sendall(body)
            else:
                up.sendall(upstream + body)
            _pipe(c, up)


_UNIX = getattr(socketserver, "UnixStreamServer", socketserver.TCPServer)   # (Windows has none: tests only)


class _Server(socketserver.ThreadingMixIn, _UNIX):
    daemon_threads = True

    def __init__(self, path: str, log_path: Path | None) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        super().__init__(path, _Handler)

    def log(self, host: str, port: int, why: str) -> None:
        if self.log_path is None:
            return
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "host": host, "port": port,
                                "ok": not why, "why": why}, ensure_ascii=False) + "\n")


def serve(socket_path: Path, log_path: Path | None = None) -> _Server:
    socket_path.unlink(missing_ok=True)
    srv = _Server(str(socket_path), log_path)
    threading.Thread(target=srv.serve_forever, name="egress-proxy", daemon=True).start()
    return srv
