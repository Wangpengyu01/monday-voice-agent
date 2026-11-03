"""Device page (HTTP :8767, LAN only): see what 星期一 found, name devices, enter their accounts.

Entering a password goes browser -> board only; it is encrypted into the vault and never shown again.
Access: private-network clients only, after typing the pairing code 星期一 says ("设备配对码").
"""
from __future__ import annotations

import asyncio
import html
import http.server
import ipaddress
import logging
import os
import secrets
import socketserver
import threading
import time
import urllib.parse
from pathlib import Path

from devices import pairing
from devices.inventory import Inventory
from devices.scanner import scan
from devices.vault import SERVICES, Vault, VaultLocked

LOG = logging.getLogger("devices.web")
RUN = Path(os.environ.get("AGENT_ROOT", "/userdata/agent")) / "run" / "devices"
SESSION_TTL_S = 12 * 3600
MAX_FAILS, FAIL_WINDOW_S = 5, 600

CSS = """body{font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;padding:16px;
background:#0f1115;color:#e8e8e8;max-width:900px;margin:auto}h1{font-size:20px}a{color:#7cb7ff}
.card{background:#1a1d24;border-radius:10px;padding:12px 14px;margin:10px 0}.muted{color:#8a8f98;font-size:13px}
input,select,button{font-size:15px;padding:7px 9px;border-radius:7px;border:1px solid #333;background:#11141a;
color:#eee;margin:3px 0}button{background:#2d6cdf;border:none;cursor:pointer}.del{background:#5a2a2a}
.tag{display:inline-block;background:#24324a;border-radius:5px;padding:1px 7px;margin-right:4px;font-size:13px}
.ok{background:#1f4a2e}form.inline{display:inline}"""


class State:
    def __init__(self) -> None:
        self.inventory = Inventory(RUN / "devices.db")
        self.vault = Vault(RUN / "vault.db")
        self.sessions: dict[str, float] = {}
        self.fails: list[float] = []
        self.scan_lock = threading.Lock()


STATE: State | None = None


def page(body: str) -> bytes:
    return (f"<!doctype html><html lang=zh><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>星期一 · 家里的设备</title>"
            f"<style>{CSS}</style><body>{body}</body></html>").encode("utf-8")


def esc(v) -> str:
    return html.escape(str(v or ""), quote=True)


def pair_page(msg: str = "") -> bytes:
    return page(f"""<h1>星期一 · 家里的设备</h1><div class=card>
<p>对星期一说 <b>“设备配对码”</b>，把它念的四位数字填在这里。</p>
<form method=post action=/pair><input name=code inputmode=numeric maxlength=4 autofocus
placeholder=配对码> <button>进入</button></form><p class=muted>{esc(msg)}</p></div>""")


def devices_page(st: State, msg: str = "") -> bytes:
    stored = st.vault.describe() if st.vault.unlocked else {}
    svc_opts = "".join(f"<option value={k}>{esc(v)}</option>" for k, v in SERVICES.items())
    rows = []
    for d in st.inventory.all():
        have = "".join(
            f"<span class='tag ok'>{esc(SERVICES.get(s, s))} ✓"
            f"<form class=inline method=post action=/cred/delete><input type=hidden name=device value='{esc(d.key)}'>"
            f"<input type=hidden name=service value={esc(s)}><button class=del title=删除>×</button></form></span>"
            for s in stored.get(d.key, []))
        ago = int((time.time() - d.last_seen) / 60)
        rows.append(f"""<div class=card><b>{esc(d.name)}</b> <span class=tag>{esc(d.kind_name)}</span>
<span class=muted>{esc(d.ip)} · {esc(d.mac or '无MAC')} · 端口 {esc(','.join(map(str, d.ports)) or '无')}
· {'刚刚' if ago < 2 else f'{ago} 分钟前'}在线</span>
<form method=post action=/name><input type=hidden name=device value='{esc(d.key)}'>
<input name=nickname value='{esc(d.nickname)}' placeholder='起个名字，比如“我的电脑”' maxlength=20>
<button>保存名字</button></form>
<div>{have}</div>
<details><summary>添加账号密码</summary><form method=post action=/cred autocomplete=off>
<input type=hidden name=device value='{esc(d.key)}'><select name=service>{svc_opts}</select><br>
<input name=username placeholder=用户名 autocomplete=off><br>
<input name=secret type=password placeholder=密码 autocomplete=new-password><br>
<input name=note placeholder='备注（可选，比如端口、用途）' maxlength=100><br><button>加密保存</button>
</form></details></div>""")
    lock_note = "" if st.vault.unlocked else "<p class=card>⚠ 密码库没有密钥，暂时不能保存密码。</p>"
    return page(f"""<h1>星期一 · 家里的设备</h1>{lock_note}
<p class=muted>{esc(msg)}</p>
<form method=post action=/scan><button>重新扫描（约 15 秒）</button></form>
{''.join(rows) or '<p>还没扫描过。</p>'}
<p class=muted>密码只加密保存在星期一本地，不会发给任何云端模型，保存后网页上也不再显示。</p>""")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "monday-devices"

    def _client_ok(self) -> bool:
        ip = ipaddress.ip_address(self.client_address[0])
        return ip.is_private or ip.is_loopback

    def _authed(self) -> bool:
        st = STATE
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "sid" and st.sessions.get(v, 0) > time.time():
                return True
        return False

    def _send(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, msg: str = "", headers: dict | None = None) -> None:
        loc = "/" + (f"?msg={urllib.parse.quote(msg)}" if msg else "")
        self._send(b"", 303, {"Location": loc, **(headers or {})})

    def _form(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 8192:
            return {}
        data = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
        return {k: v[0] for k, v in data.items()}

    def do_GET(self) -> None:  # noqa: N802
        if not self._client_ok():
            return self._send(b"forbidden", 403)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        msg = q.get("msg", [""])[0]
        self._send(devices_page(STATE, msg) if self._authed() else pair_page(msg))

    def do_POST(self) -> None:  # noqa: N802
        st = STATE
        if not self._client_ok():
            return self._send(b"forbidden", 403)
        path = urllib.parse.urlparse(self.path).path
        form = self._form()
        if path == "/pair":
            now = time.time()
            st.fails = [t for t in st.fails if now - t < FAIL_WINDOW_S]
            if len(st.fails) >= MAX_FAILS:
                return self._send(pair_page("错误次数太多，十分钟后再试。"), 429)
            if not pairing.check_code(RUN / "pair.json", form.get("code", "")):
                st.fails.append(now)
                return self._send(pair_page("配对码不对或已过期，请让星期一重新念一个。"))
            sid = secrets.token_urlsafe(24)
            st.sessions[sid] = now + SESSION_TTL_S
            LOG.info("paired from %s", self.client_address[0])
            return self._redirect(headers={"Set-Cookie": f"sid={sid}; HttpOnly; SameSite=Strict; Path=/; "
                                                         f"Max-Age={SESSION_TTL_S}"})
        if not self._authed():
            return self._send(pair_page("请先输入配对码。"), 401)
        device = form.get("device", "")
        if path == "/name":
            st.inventory.set_nickname(device, form.get("nickname", ""))
            return self._redirect("名字已保存")
        if path == "/cred":
            if not st.inventory.get(device) or not form.get("secret"):
                return self._redirect("没保存：缺少设备或密码")
            try:
                st.vault.put(device, form.get("service", "other"), form.get("username", ""), form["secret"],
                             form.get("note", ""))
            except (VaultLocked, ValueError) as exc:
                return self._redirect(f"没保存：{exc}")
            LOG.info("credential stored: device=%s service=%s", device, form.get("service"))
            return self._redirect("已加密保存")
        if path == "/cred/delete":
            st.vault.delete(device, form.get("service", ""))
            return self._redirect("已删除")
        if path == "/scan":
            if not st.scan_lock.acquire(blocking=False):
                return self._redirect("正在扫描，稍等")
            try:
                found = asyncio.run(scan())
                new = st.inventory.update(found)
            finally:
                st.scan_lock.release()
            return self._redirect(f"扫描完成：{len(found)} 台设备，新发现 {len(new)} 台")
        self._send(b"not found", 404)

    def log_message(self, fmt: str, *args) -> None:
        LOG.info("%s %s", self.client_address[0], fmt % args)  # request line only, never the form body


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    global STATE
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    STATE = State()
    port = int(os.environ.get("DEVICES_PORT", 8767))
    LOG.info("device page on :%d (vault %s)", port, "unlocked" if STATE.vault.unlocked else "LOCKED")
    Server(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
