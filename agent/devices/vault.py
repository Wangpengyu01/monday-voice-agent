"""Device accounts and passwords, encrypted on the board (Fernet: AES-128-CBC + HMAC-SHA256).

- The key is R1_VAULT_KEY, loaded by systemd from /userdata/secrets/vault.env (root-only on disk);
  without it nothing can be stored or read.
- The encrypted database lives in run/devices/vault.db.
- Secrets are only ever handed to the local tools that operate a device. Nothing here may go into a
  model prompt: describe() returns what is stored (which device, which account type), never the values.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

SERVICES = {
    "ssh": "SSH 登录",
    "windows": "Windows 账号（远程关机/远程桌面）",
    "web": "网页后台（路由器、NAS、摄像头）",
    "app": "App / 云账号（米家、涂鸦等）",
    "other": "其他",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS creds (
    device TEXT NOT NULL, service TEXT NOT NULL, username BLOB, secret BLOB, note TEXT DEFAULT '',
    updated REAL, PRIMARY KEY (device, service));
"""


class VaultLocked(RuntimeError):
    """No key: the vault can't be used."""


@dataclass
class Credential:
    device: str
    service: str
    username: str
    secret: str
    note: str


class Vault:
    def __init__(self, path: str | Path, key: str | bytes | None = None) -> None:
        key = key if key is not None else os.environ.get("R1_VAULT_KEY", "")
        self._fernet = Fernet(key) if key else None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._db.executescript(SCHEMA)
        self._db.commit()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @property
    def unlocked(self) -> bool:
        return self._fernet is not None

    def _f(self) -> Fernet:
        if self._fernet is None:
            raise VaultLocked("R1_VAULT_KEY is not set")
        return self._fernet

    def put(self, device: str, service: str, username: str, secret: str, note: str = "") -> None:
        if service not in SERVICES:
            raise ValueError(f"unknown service {service!r}")
        f = self._f()
        with self._lock:
            self._db.execute(
                "INSERT INTO creds(device, service, username, secret, note, updated) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(device, service) DO UPDATE SET username=excluded.username, secret=excluded.secret, "
                "note=excluded.note, updated=excluded.updated",
                (device, service, f.encrypt(username.encode()), f.encrypt(secret.encode()), note.strip()[:100],
                 time.time()))
            self._db.commit()

    def get(self, device: str, service: str) -> Credential | None:
        """For the local tools that operate the device - never for a prompt or a log line."""
        f = self._f()
        with self._lock:
            row = self._db.execute("SELECT username, secret, note FROM creds WHERE device=? AND service=?",
                                   (device, service)).fetchone()
        if not row:
            return None
        try:
            return Credential(device, service, f.decrypt(row[0]).decode(), f.decrypt(row[1]).decode(), row[2])
        except InvalidToken:
            raise VaultLocked("R1_VAULT_KEY does not match the stored data") from None

    def delete(self, device: str, service: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM creds WHERE device=? AND service=?", (device, service))
            self._db.commit()
        return cur.rowcount > 0

    def describe(self) -> dict[str, list[str]]:
        """device -> account types stored. Safe to show or say; contains no secret."""
        with self._lock:
            rows = self._db.execute("SELECT device, service FROM creds ORDER BY device, service").fetchall()
        out: dict[str, list[str]] = {}
        for device, service in rows:
            out.setdefault(device, []).append(service)
        return out
