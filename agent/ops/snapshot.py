"""Snapshots ("历史节点") of 星期一: code, config, its systemd units and its memory, in one small tar.gz.

    python3 -m ops.snapshot create --label before-upgrade [--reason "..."]
    python3 -m ops.snapshot list
    python3 -m ops.snapshot restore <id> [--with-data]     (root: also puts the systemd units back)
    python3 -m ops.snapshot prune [--keep 20]

What is in a snapshot: /userdata/agent minus caches/logs (browser profile, TTS clips, logs, camera
frames, build leftovers), plus /etc/systemd/system/r1-* (units and drop-ins). The memory database is
copied with SQLite's backup API, so it is consistent even while the assistant is writing to it.
Secrets live in /userdata/secrets and are never part of a snapshot.

Restore puts the code/config/units back; learned data (run/) only with --with-data, so rolling back a
broken change does not throw away what was learned since. Every restore first snapshots the current
state ("pre-restore"), so a restore can itself be undone. Files added after the snapshot are left in
place (they are harmless to the restored code, and deleting is not reversible).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

AGENT_ROOT = Path(os.environ.get("AGENT_ROOT", "/userdata/agent"))
SNAP_DIR = Path(os.environ.get("SNAPSHOT_DIR", "/userdata/backups/snapshots"))
SYSTEMD_DIR = Path(os.environ.get("SNAPSHOT_SYSTEMD_DIR", "/etc/systemd/system"))
EXCLUDE = (
    "run/chromium-kiosk", "run/chromium-kiosk/*", "run/ack_cache", "run/ack_cache/*", "run/tts", "run/tts/*",
    "logs", "logs/*", ".codex-backups", ".codex-backups/*", ".codex-build", ".codex-build/*",
    "*/__pycache__", "*/__pycache__/*", "*.pyc", "run/*.jpg", "ui/latest_frame.jpg", "run/learning/memory.db-*",
    "*.part", "*.tmp", "*.restore-part", "__pycache__", "backup_stable_*", "backup_stable_*/*",
)
CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".html", ".js", ".css", ".json", ".service", ".timer", ".conf", ".desktop"}
MEMORY_DB = "run/learning/memory.db"


@dataclass
class Snapshot:
    id: str
    created: float
    label: str
    reason: str
    code_hash: str
    files: int
    size: int
    good: bool = False

    @property
    def path(self) -> Path:
        return SNAP_DIR / f"{self.id}.tar.gz"


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE)


def _agent_files(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        dirnames[:] = [d for d in dirnames if not _excluded(d if rel_dir == "." else f"{rel_dir}/{d}")]
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if not _excluded(rel):
                out.append(Path(dirpath) / name)
    return sorted(out)


def _unit_files(systemd_dir: Path) -> list[Path]:
    if not systemd_dir.is_dir():
        return []
    out = []
    for p in sorted(systemd_dir.glob("r1-*")):
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file())
    return out


def code_hash(root: Path | None = None, systemd_dir: Path | None = None) -> str:
    """Fingerprint of everything that decides how the assistant behaves (code, config, units) - not data."""
    root, systemd_dir = root or AGENT_ROOT, systemd_dir or SYSTEMD_DIR
    h = hashlib.sha256()
    for p in _agent_files(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix in CODE_SUFFIXES and not rel.startswith("run/"):
            h.update(rel.encode())
            h.update(_read(p))
    for p in _unit_files(systemd_dir):
        h.update(p.relative_to(systemd_dir).as_posix().encode())
        h.update(_read(p))
    return h.hexdigest()[:16]


def _read(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except OSError:
        return b"?"


def _as_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _own_like(path: Path, ref: Path) -> None:
    """Run as root (the self-heal watchdog), files we write get the owner of `ref`, so the assistant's own
    user can still read, prune and overwrite them."""
    if not _as_root():
        return
    try:
        st = ref.stat()
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


def _meta_path(sid: str) -> Path:
    return SNAP_DIR / f"{sid}.json"


def _save_meta(s: Snapshot) -> None:
    tmp = _meta_path(s.id).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_meta_path(s.id))
    _own_like(_meta_path(s.id), SNAP_DIR)


def create(label: str = "manual", reason: str = "", *, root: Path | None = None, systemd_dir: Path | None = None) -> Snapshot:
    root, systemd_dir = root or AGENT_ROOT, systemd_dir or SYSTEMD_DIR
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SNAP_DIR.glob("*.part"):  # left by a snapshot that was interrupted
        if time.time() - stale.stat().st_mtime > 3600:
            stale.unlink(missing_ok=True)
    slug = re.sub(r"[^\w\-]+", "-", label, flags=re.UNICODE).strip("-")[:40] or "snap"
    sid = time.strftime("%Y%m%d-%H%M%S") + f"-{slug}"
    while (SNAP_DIR / f"{sid}.tar.gz").exists():
        sid += "x"
    files = _agent_files(root)
    units = _unit_files(systemd_dir)
    tmp = SNAP_DIR / f"{sid}.tar.gz.part"
    with tarfile.open(tmp, "w:gz") as tar:
        for p in files:
            rel = p.relative_to(root).as_posix()
            if rel == MEMORY_DB:
                continue  # added below from a consistent copy
            try:
                tar.add(p, arcname=f"agent/{rel}", recursive=False)
            except OSError:
                continue  # unreadable file (e.g. root-only): skip rather than fail the snapshot
        db = root / MEMORY_DB
        if db.is_file():
            with tempfile.TemporaryDirectory() as td:
                copy = Path(td) / "memory.db"
                src, dst = sqlite3.connect(str(db)), sqlite3.connect(str(copy))
                try:
                    src.backup(dst)
                finally:
                    src.close()
                    dst.close()
                tar.add(copy, arcname=f"agent/{MEMORY_DB}")
        for p in units:
            try:
                tar.add(p, arcname=f"systemd/{p.relative_to(systemd_dir).as_posix()}", recursive=False)
            except OSError:
                continue
        snap = Snapshot(sid, time.time(), label, reason, code_hash(root, systemd_dir), len(files) + len(units), 0)
        info = tarfile.TarInfo("snapshot.json")
        blob = json.dumps(asdict(snap), ensure_ascii=False).encode()
        info.size, info.mtime = len(blob), int(snap.created)
        tar.addfile(info, io.BytesIO(blob))
    tmp.replace(snap.path)
    _own_like(snap.path, SNAP_DIR)
    snap.size = snap.path.stat().st_size
    _save_meta(snap)
    return snap


def list_snapshots() -> list[Snapshot]:
    out = []
    for meta in SNAP_DIR.glob("*.json"):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
            s = Snapshot(**d)
        except (ValueError, TypeError):
            continue
        if s.path.is_file():
            out.append(s)
    return sorted(out, key=lambda s: s.created)


def get(sid: str) -> Snapshot:
    for s in list_snapshots():
        if s.id == sid or s.id.startswith(sid):
            return s
    raise FileNotFoundError(f"no snapshot {sid}")


def mark_good(sid: str) -> None:
    s = get(sid)
    s.good = True
    _save_meta(s)


def latest_good(before_hash_change: str | None = None) -> Snapshot | None:
    good = [s for s in list_snapshots() if s.good and (before_hash_change is None or s.code_hash != before_hash_change)]
    return good[-1] if good else None


def restore(sid: str, *, with_data: bool = False, root: Path | None = None, systemd_dir: Path | None = None,
            write_units: bool = True) -> tuple[Snapshot, Snapshot]:
    """-> (restored snapshot, the pre-restore snapshot of what was there before)."""
    root, systemd_dir = root or AGENT_ROOT, systemd_dir or SYSTEMD_DIR
    snap = get(sid)
    before = create("pre-restore", f"before restoring {snap.id}", root=root, systemd_dir=systemd_dir)
    with tarfile.open(snap.path, "r:gz") as tar, tempfile.TemporaryDirectory() as td:
        members = [m for m in tar.getmembers() if m.isfile() and ".." not in Path(m.name).parts and not m.name.startswith("/")]
        tar.extractall(td, members=members)  # names checked above: no absolute paths, no ".."
        base = Path(td)
        for m in members:
            src = base / m.name
            if m.name.startswith("agent/"):
                rel = m.name[len("agent/"):]
                if rel.startswith("run/") and not with_data:
                    continue
                dst = root / rel
                owner = dst if dst.exists() else root
            elif m.name.startswith("systemd/") and write_units:
                dst = systemd_dir / m.name[len("systemd/"):]
                owner = systemd_dir
            else:
                continue
            if dst.is_symlink():
                continue  # never write through a link that appeared since
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".restore-part")
            shutil.copy2(src, part)
            _own_like(part, owner)
            part.replace(dst)
            if rel_is_db(m.name):
                for side in ("-wal", "-shm"):  # a stale WAL next to the restored database would corrupt it
                    dst.with_name(dst.name + side).unlink(missing_ok=True)
    return snap, before


def rel_is_db(name: str) -> bool:
    return name == f"agent/{MEMORY_DB}"


def prune(keep: int = 20, keep_good: int = 3) -> list[str]:
    """Keep the newest `keep` snapshots and always the newest `keep_good` good ones; delete the rest."""
    snaps = list_snapshots()
    keep_ids = {s.id for s in snaps[-keep:]} | {s.id for s in [s for s in snaps if s.good][-keep_good:]}
    removed = []
    for s in snaps:
        if s.id not in keep_ids:
            s.path.unlink(missing_ok=True)
            _meta_path(s.id).unlink(missing_ok=True)
            removed.append(s.id)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一 snapshots (历史节点)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--label", default="manual")
    c.add_argument("--reason", default="")
    sub.add_parser("list")
    r = sub.add_parser("restore")
    r.add_argument("id")
    r.add_argument("--with-data", action="store_true", help="also roll back memory/learning data (run/)")
    r.add_argument("--no-units", action="store_true", help="leave /etc/systemd/system alone")
    m = sub.add_parser("mark-good")
    m.add_argument("id")
    p = sub.add_parser("prune")
    p.add_argument("--keep", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "create":
        s = create(a.label, a.reason)
        print(f"{s.id}  {s.size / 1024:.0f} KB  hash={s.code_hash}")
    elif a.cmd == "list":
        for s in list_snapshots():
            print(f"{s.id:44s} {'GOOD' if s.good else '    '} {s.size / 1024:7.0f} KB  hash={s.code_hash}  {s.reason}")
    elif a.cmd == "restore":
        snap, before = restore(a.id, with_data=a.with_data, write_units=not a.no_units)
        print(f"restored {snap.id} (previous state saved as {before.id}); now: systemctl daemon-reload and restart r1-*")
    elif a.cmd == "mark-good":
        mark_good(a.id)
    elif a.cmd == "prune":
        print("removed:", ", ".join(prune(a.keep)) or "none")


if __name__ == "__main__":
    main()
