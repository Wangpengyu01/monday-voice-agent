"""Daily off-board backup of 星期一 (r1-backup.timer, 04:10, as the service user).

1. take a "daily" snapshot (ops.snapshot) and prune local snapshots (newest N + newest GOOD ones);
2. rsync the snapshot directory to <host>:<remote_dir>/snapshots/ (never --delete: the server keeps history);
3. copy run/learning (memory, diary, training sets; the database via SQLite's backup API) to
   <remote_dir>/learning/<YYYYMMDD>/, hard-linked against the previous day so unchanged files cost nothing;
4. remote cleanup by age: only date-named entries inside <remote_dir>/snapshots and <remote_dir>/learning,
   names checked against a strict pattern before anything is deleted, and never below remote_min_keep snapshots.

Login is key-only through the ~/.ssh/config alias (config `backup.host`); without a working key it does
nothing but say so in the log. Config: the `backup:` section of config/agent.yaml.

    python3 -m ops.backup [--dry-run] [--config /userdata/agent/config/agent.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ops import snapshot

LOG = logging.getLogger("backup")

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
_REMOTE_DIR = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*(/[A-Za-z0-9_][A-Za-z0-9_.\-]*)*$")
_SNAP_NAME = re.compile(r"^(\d{8})-\d{6}-[\w\-]+x*\.(tar\.gz|json)$")
_DAY_NAME = re.compile(r"^\d{8}$")


@dataclass
class Config:
    enabled: bool = False
    host: str = "backup"
    remote_dir: str = "monday-backups/r1"
    keep_days: int = 60
    remote_min_keep: int = 10
    local_keep: int = 20
    local_keep_good: int = 3

    @classmethod
    def from_yaml(cls, cfg: dict) -> "Config":
        b = cfg.get("backup") or {}
        c = cls()
        c.enabled = str(b.get("enabled", "false")).lower() in ("true", "yes", "1", "on")
        c.host = str(b.get("host", c.host))
        c.remote_dir = str(b.get("remote_dir", c.remote_dir)).strip().rstrip("/")
        for k in ("keep_days", "remote_min_keep", "local_keep", "local_keep_good"):
            try:
                setattr(c, k, int(b.get(k, getattr(c, k))))
            except (TypeError, ValueError):
                pass
        return c


_NEVER_INSIDE = ("nc_data", "nextcloud", "db_data", "docker", "lost+found", "etc", "home", "root", "usr", "var", "boot")


def check_remote_dir(d: str) -> str:
    """Plain names only (no "..", no shell characters). Relative = under the remote home, at least two levels
    (monday-backups/r1); absolute = at least three levels (/srv/<disk>/monday-backups/r1) and never inside
    another service's data (Nextcloud's nc_data, databases, docker) or a system directory."""
    parts = d.lstrip("/").split("/") if d else []
    absolute = d.startswith("/")
    ok = (bool(d) and _REMOTE_DIR.match(d.lstrip("/")) is not None and not any(p in (".", "..") for p in parts)
          and len(parts) >= (3 if absolute else 2)
          and not (absolute and (parts[0] in _NEVER_INSIDE or any(p in _NEVER_INSIDE for p in parts[1:-1]))))
    if not ok:
        raise ValueError(f"remote_dir 不安全：{d!r}（要求 家目录下至少两级的普通路径，如 monday-backups/r1，"
                         "或独立的绝对路径，如 /srv/nextcloud-data/monday-backups/r1）")
    return d


def old_entries(names: list[str], pattern: re.Pattern, keep_days: int, now: float, min_keep: int = 0) -> list[str]:
    """Date-named entries older than keep_days (the date is the first 8 digits); the newest min_keep are kept."""
    cutoff = time.strftime("%Y%m%d", time.localtime(now - keep_days * 86400))
    dated = sorted((m.group(1) if m.groups() else m.group(0), n) for n in names if (m := pattern.match(n)))
    keep = {n for _, n in dated[-min_keep:]} if min_keep else set()
    if pattern is _SNAP_NAME and min_keep:  # count snapshots, not files: keep both .tar.gz and .json of a kept one
        ids = sorted({n.split(".")[0] for _, n in dated})[-min_keep:]
        keep = {n for _, n in dated if n.split(".")[0] in ids}
    return [n for day, n in dated if day < cutoff and n not in keep]


class Backup:
    def __init__(self, cfg: Config, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry = dry_run
        self.remote = check_remote_dir(cfg.remote_dir)

    def ssh(self, cmd: str, timeout: float = 120) -> tuple[int, str]:
        r = subprocess.run(["ssh", *SSH_OPTS, self.cfg.host, cmd], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout + r.stderr).strip()

    def rsync(self, src: list[str], dst: str, extra: list[str] | None = None) -> None:
        argv = ["rsync", "-a", "--partial", "-e", "ssh " + " ".join(SSH_OPTS), *(extra or []), *src,
                f"{self.cfg.host}:{dst}"]
        if self.dry:
            argv.insert(1, "--dry-run")
        r = subprocess.run(argv, capture_output=True, text=True, timeout=3000, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"rsync {' '.join(src)} 失败：{(r.stderr or r.stdout).strip()[-400:]}")

    def run(self) -> dict:
        rc, out = self.ssh("true", timeout=30)
        if rc != 0:
            raise ConnectionError(f"连不上备份服务器（{self.cfg.host}）：{out[-300:]}。"
                                  "请先在备份服务器上装好星期一的公钥（tools/monday_keys.pub），并在 ~/.ssh/config 配好别名。")
        q = shlex.quote
        snaps_dir, learn_dir = f"{self.remote}/snapshots", f"{self.remote}/learning"
        self.ssh(f"mkdir -p {q(snaps_dir)} {q(learn_dir)}")

        snap = None if self.dry else snapshot.create("daily", "每日备份")
        removed_local = [] if self.dry else snapshot.prune(self.cfg.local_keep, self.cfg.local_keep_good)
        self.rsync([str(snapshot.SNAP_DIR) + "/"], snaps_dir + "/", ["--exclude", "*.part", "--exclude", "*.tmp"])

        day = time.strftime("%Y%m%d")
        _, listing = self.ssh(f"ls -1 {q(learn_dir)}")
        prev = sorted(n for n in listing.splitlines() if _DAY_NAME.match(n) and n != day)
        link = ["--link-dest", f"../{prev[-1]}"] if prev else []
        learning = snapshot.AGENT_ROOT / "run" / "learning"
        if learning.is_dir():
            with tempfile.TemporaryDirectory() as td:
                stage = Path(td) / "memory.db"
                db = learning / "memory.db"
                if db.is_file():
                    src, dst = sqlite3.connect(str(db)), sqlite3.connect(str(stage))
                    try:
                        src.backup(dst)
                    finally:
                        src.close()
                        dst.close()
                self.rsync([str(learning) + "/"], f"{learn_dir}/{day}/", [*link, "--exclude", "memory.db*"])
                if stage.is_file():
                    self.rsync([str(stage)], f"{learn_dir}/{day}/", link)

        removed_remote = self.cleanup(time.time())
        _, du = self.ssh(f"du -sh {q(self.remote)} | cut -f1; df -h {q(self.remote)} | tail -1")
        return {"snapshot": snap.id if snap else None, "local_pruned": removed_local, "remote_removed": removed_remote,
                "remote_usage": du.replace("\n", " | ")}

    def cleanup(self, now: float) -> list[str]:
        q = shlex.quote
        out = []
        for sub, pat, min_keep, flag in (("snapshots", _SNAP_NAME, self.cfg.remote_min_keep, "-f"),
                                         ("learning", _DAY_NAME, 7, "-rf")):
            base = f"{self.remote}/{sub}"
            rc, listing = self.ssh(f"ls -1 {q(base)}")
            if rc != 0:
                continue
            old = old_entries(listing.splitlines(), pat, self.cfg.keep_days, now, min_keep)
            for i in range(0, len(old), 50):
                batch = old[i:i + 50]
                paths = " ".join(q(f"{base}/{n}") for n in batch)  # names matched a strict pattern: no "/", no ".."
                if not self.dry:
                    self.ssh(f"rm {flag} -- {paths}")
                out += [f"{sub}/{n}" for n in batch]
        return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [backup] %(message)s")
    ap = argparse.ArgumentParser(description="星期一 daily off-board backup")
    ap.add_argument("--config", default=str(snapshot.AGENT_ROOT / "config" / "agent.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from orchestrator.config_loader import load_yaml

    cfg = Config.from_yaml(load_yaml(a.config))
    status_path = snapshot.AGENT_ROOT / "run" / "backup_status.json"
    if not cfg.enabled:
        LOG.info("远程备份没开（config/agent.yaml 的 backup.enabled）；只清理本地快照")
        LOG.info("local pruned: %s", snapshot.prune(cfg.local_keep, cfg.local_keep_good))
        return 0
    t0 = time.time()
    try:
        res = Backup(cfg, dry_run=a.dry_run).run()
    except Exception as exc:  # noqa: BLE001
        LOG.error("备份失败：%s", exc)
        status = {"ok": False, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(exc)[:500]}
        rc = 1
    else:
        LOG.info("备份完成 %.0fs %s", time.time() - t0, json.dumps(res, ensure_ascii=False))
        status = {"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **res}
        rc = 0
    if not a.dry_run:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
