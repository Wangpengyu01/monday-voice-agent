"""What 星期一 knows about the devices at home: the last scan, plus the names the user gave them."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from devices.scanner import KIND_NAMES, Device

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    key TEXT PRIMARY KEY,           -- MAC address, or the IP when the MAC is unknown
    ip TEXT, mac TEXT, hostname TEXT, label TEXT, kind TEXT, ports TEXT, services TEXT, vendor TEXT,
    nickname TEXT DEFAULT '',       -- what the user calls it ("我的电脑", "宿舍门")
    first_seen REAL, last_seen REAL);
"""


@dataclass
class Known:
    key: str
    ip: str
    mac: str
    hostname: str
    label: str
    kind: str
    ports: list[int]
    nickname: str
    first_seen: float
    last_seen: float

    @property
    def name(self) -> str:
        return self.nickname or self.label or self.ip

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, self.kind)


class Inventory:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def update(self, scanned: list[Device], now: float | None = None) -> list[Known]:
        """Store a scan; returns the devices never seen before."""
        now = now or time.time()
        new = []
        with self._lock:
            for d in scanned:
                row = self._db.execute("SELECT key FROM devices WHERE key = ?", (d.key,)).fetchone()
                vals = (d.ip, d.mac, d.hostname, d.label, d.kind, json.dumps(d.ports),
                        json.dumps(d.services, ensure_ascii=False), d.vendor, now, d.key)
                if row:
                    self._db.execute("UPDATE devices SET ip=?, mac=?, hostname=?, label=?, kind=?, ports=?, "
                                     "services=?, vendor=?, last_seen=? WHERE key=?", vals)
                else:
                    self._db.execute("INSERT INTO devices(ip, mac, hostname, label, kind, ports, services, vendor, "
                                     "last_seen, key, first_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?)", vals + (now,))
                    new.append(d.key)
            self._db.commit()
        return [k for k in self.all() if k.key in new]

    def all(self, *, seen_within: float | None = None) -> list[Known]:
        q = "SELECT key, ip, mac, hostname, label, kind, ports, nickname, first_seen, last_seen FROM devices"
        args: tuple = ()
        if seen_within:
            q += " WHERE last_seen >= ?"
            args = (time.time() - seen_within,)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY last_seen DESC", args).fetchall()
        out = [Known(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6] or "[]"), r[7] or "", r[8], r[9])
               for r in rows]
        return sorted(out, key=lambda k: tuple(int(x) for x in k.ip.split(".")) if k.ip else (999,))

    def get(self, key: str) -> Known | None:
        return next((k for k in self.all() if k.key == key), None)

    def set_nickname(self, key: str, nickname: str) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE devices SET nickname = ? WHERE key = ?", (nickname.strip()[:20], key))
            self._db.commit()
        return cur.rowcount > 0

    def find(self, words: str) -> Known | None:
        """The device the user means: by nickname, label or kind name ("打开我的电脑")."""
        best = None
        for k in self.all():
            for cand in (k.nickname, k.label, k.hostname, k.kind_name):
                if cand and cand in words and (best is None or len(cand) > best[0]):
                    best = (len(cand), k)
        return best[1] if best else None
