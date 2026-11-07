"""Passwords the owner says or types ("SSH 到 10.0.0.237，账号和密码都是 pi") are used once and never kept.

extract_credentials() takes them out of the raw text *before* anything else sees it (normalization
would also mangle them: it collapses repeated characters). The rest of the pipeline - logs, the
journal, the web page, the teacher - only ever sees "[密码已隐藏]". The password is used once to put
星期一's public key on that machine (install_key), and from then on the machine is reached by key.
A logging filter masks anything that still looks like "密码是 xxx" (e.g. raw ASR partials).
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MASK = "[密码已隐藏]"
_TOKEN = r"[A-Za-z0-9_.@#$%^&*!+=\-]+"
_IP = re.compile(r"(?<![\d.])(\d{1,3}(?:\s*[.点]\s*\d{1,3}){3})(?![\d.])")
_BOTH = re.compile(rf"(?:账号|帐号|用户名|用户)\s*(?:和|跟|与)\s*密码\s*(?:都是|都为|都|是|为|[:：])\s*({_TOKEN})")
_USER = re.compile(rf"(?:账号|帐号|用户名|用户|user(?:name)?)\s*(?:是|为|叫|[:：=])?\s*({_TOKEN})", re.I)
_PASS = re.compile(rf"(?:(?:密码|口令)\s*(?:是|为|[:：=])?|(?<![A-Za-z])(?:password|passwd|pwd)(?:\s*[:：=]|\s+(?:is\s+)?))\s*({_TOKEN})", re.I)
_LOG_MASK = re.compile(rf"((?:(?:密码|口令)\s*(?:都是|都为|是|为|[:：=])?|(?<![A-Za-z])(?:password|passwd|pwd)(?:\s*[:：=]|\s+(?:is\s+)?))\s*)({_TOKEN})", re.I)
_LOG_BOTH = re.compile(rf"((?:账号|帐号|用户名|用户)\s*(?:和|跟|与)\s*密码\s*(?:都是|都为|都|是|为|[:：])\s*)({_TOKEN})")


@dataclass
class Credentials:
    host: str
    user: str
    password: str
    at: float  # when it was said (stale ones are dropped)

    def __repr__(self) -> str:  # never print the password, even by accident
        return f"Credentials(host={self.host!r}, user={self.user!r}, password=***)"


def extract_credentials(text: str) -> tuple[str, Credentials | None]:
    """-> (text with the password masked, credentials or None)."""
    text = text or ""
    password = user = ""
    m = _BOTH.search(text)
    if m:
        user = password = m.group(1)
        text = text[:m.start(1)] + MASK + text[m.end(1):]
    else:
        mp = _PASS.search(text)
        if mp:
            password = mp.group(1)
            text = text[:mp.start(1)] + MASK + text[mp.end(1):]
        mu = _USER.search(text)
        if mu and MASK not in mu.group(1):
            user = mu.group(1)
    if not password:
        return text, None
    mh = _IP.search(text)
    host = re.sub(r"\s|点", lambda x: "." if x.group(0) == "点" else "", mh.group(1)) if mh else ""
    if not host:
        mh2 = re.search(r"@\s*([A-Za-z0-9.\-]+)", text)
        host = mh2.group(1) if mh2 else ""
    return text, Credentials(host=host, user=user or "root", password=password, at=time.monotonic())


def mask_secrets(text: str) -> str:
    if not text:
        return text
    text = _LOG_BOTH.sub(lambda m: m.group(1) + MASK, text)
    return _LOG_MASK.sub(lambda m: m.group(1) + (m.group(2) if m.group(2) == MASK or m.group(2).startswith("[") else MASK), text)


def install_log_redaction() -> None:
    """Mask passwords in every log record from here on (all loggers, all handlers)."""
    old = logging.getLogRecordFactory()
    if getattr(old, "_masks_secrets", False):
        return

    def factory(*args, **kwargs):
        record = old(*args, **kwargs)
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string is the caller's bug, not ours
            return record
        masked = mask_secrets(msg)
        if masked != msg:
            record.msg, record.args = masked, None
        return record

    factory._masks_secrets = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


# ---------------------------------------------------------------- key setup
SSH_DIR = Path.home() / ".ssh"
KEY = SSH_DIR / "id_ed25519_monday"


def ensure_key(key: Path = KEY) -> str:
    if not key.is_file():
        key.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "monday-agent@r1", "-f", str(key)], check=True)
    return Path(str(key) + ".pub").read_text().strip()


def install_key(creds: Credentials, *, port: int = 22, key: Path = KEY, timeout: float = 10.0) -> tuple[bool, str]:
    """Log in once with the password and add 星期一's public key ("restrict": commands only)."""
    import paramiko

    pub = ensure_key(key)
    body = pub.split()[1]
    known = SSH_DIR / "known_hosts"
    cli = paramiko.SSHClient()
    if known.is_file():
        cli.load_host_keys(str(known))
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # first contact, like accept-new
    try:
        cli.connect(creds.host, port=port, username=creds.user, password=creds.password, look_for_keys=False,
                    allow_agent=False, timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
        cmd = ("umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && "
               f"(grep -qF '{body}' ~/.ssh/authorized_keys || echo 'restrict {pub}' >> ~/.ssh/authorized_keys) && echo KEY_OK")
        _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace") + stderr.read().decode("utf-8", "replace")
        known.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        cli.save_host_keys(str(known))
    except paramiko.AuthenticationException:
        return False, "账号或密码不对，登录被拒绝了"
    except Exception as exc:  # noqa: BLE001 - network, host key, shell problems: report, don't crash
        return False, f"连接失败：{type(exc).__name__}: {exc}"
    finally:
        cli.close()
    return ("KEY_OK" in out), ("已经把公钥装好了" if "KEY_OK" in out else f"公钥没装上：{out.strip()[:200]}")


def add_host(alias: str, host: str, user: str, *, config: Path | None = None, key: Path = KEY) -> None:
    config = config or SSH_DIR / "config"
    config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = config.read_text(encoding="utf-8") if config.is_file() else ""
    if re.search(rf"(?m)^Host\s+{re.escape(alias)}(\s|$)", existing):
        return
    block = (f"\n# 星期一 agent: key-only (added {time.strftime('%Y-%m-%d %H:%M')})\nHost {alias}\n    HostName {host}\n"
             f"    User {user}\n    IdentityFile {key}\n    IdentitiesOnly yes\n    BatchMode yes\n    ConnectTimeout 8\n")
    with config.open("a", encoding="utf-8") as f:
        f.write(block)
    config.chmod(0o600)


def alias_for(host: str, user: str) -> str:
    tail = host.split(".")[-1] if re.fullmatch(r"[\d.]+", host) else host.split(".")[0]
    return re.sub(r"[^\w\-]", "-", f"{user}-{tail}")
