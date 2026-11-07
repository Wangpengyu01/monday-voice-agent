"""Snapshots (ops.snapshot), the self-heal watchdog (ops.selfheal) and the off-board backup's safety checks
(ops.backup), all in temporary directories - nothing here touches systemd or the network."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from ops import backup, selfheal, snapshot  # noqa: E402


class TempAgent(unittest.TestCase):
    """A fake /userdata/agent + /etc/systemd/system + snapshot dir."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root, self.units, self.snaps = base / "agent", base / "systemd", base / "snapshots"
        for d in (self.root / "orchestrator", self.root / "config", self.root / "run" / "learning", self.root / "logs",
                  self.root / "run" / "tts", self.root / "orchestrator" / "__pycache__", self.units / "r1-x.service.d"):
            d.mkdir(parents=True)
        (self.root / "orchestrator" / "main.py").write_text("print('v1')\n")
        (self.root / "config" / "agent.yaml").write_text("a: 1\n")
        (self.root / "logs" / "orchestrator.log").write_text("noise\n")
        (self.root / "run" / "tts" / "clip.wav").write_bytes(b"RIFF")
        (self.root / "orchestrator" / "__pycache__" / "main.cpython-310.pyc").write_bytes(b"x")
        (self.root / "run" / "learning" / "diary.md").write_text("day 1\n")
        db = sqlite3.connect(str(self.root / "run" / "learning" / "memory.db"))
        db.execute("CREATE TABLE facts (fact TEXT)")
        db.execute("INSERT INTO facts VALUES ('likes tea')")
        db.commit()
        db.close()
        (self.units / "r1-x.service").write_text("[Service]\nExecStart=/bin/true\n")
        (self.units / "r1-x.service.d" / "override.conf").write_text("[Service]\nNice=1\n")
        (self.units / "other.service").write_text("not ours\n")
        self._saved = (snapshot.AGENT_ROOT, snapshot.SNAP_DIR, snapshot.SYSTEMD_DIR)
        snapshot.AGENT_ROOT, snapshot.SNAP_DIR, snapshot.SYSTEMD_DIR = self.root, self.snaps, self.units

    def tearDown(self) -> None:
        snapshot.AGENT_ROOT, snapshot.SNAP_DIR, snapshot.SYSTEMD_DIR = self._saved
        self._tmp.cleanup()


class SnapshotTests(TempAgent):
    def test_create_contains_code_units_and_memory_but_not_caches(self) -> None:
        import tarfile

        s = snapshot.create("before-test", "unit test")
        self.assertTrue(s.path.is_file())
        with tarfile.open(s.path) as tar:
            names = set(tar.getnames())
        self.assertIn("agent/orchestrator/main.py", names)
        self.assertIn("agent/run/learning/memory.db", names)
        self.assertIn("systemd/r1-x.service", names)
        self.assertIn("systemd/r1-x.service.d/override.conf", names)
        self.assertNotIn("systemd/other.service", names)
        for junk in ("agent/logs/orchestrator.log", "agent/run/tts/clip.wav", "agent/orchestrator/__pycache__/main.cpython-310.pyc"):
            self.assertNotIn(junk, names)
        self.assertEqual([x.id for x in snapshot.list_snapshots()], [s.id])
        self.assertEqual(snapshot.get(s.id[:15]).id, s.id)

    def test_restore_rolls_code_back_keeps_data_and_can_be_undone(self) -> None:
        good = snapshot.create("v1")
        h1 = snapshot.code_hash()
        (self.root / "orchestrator" / "main.py").write_text("raise SystemExit('broken')\n")
        (self.units / "r1-x.service").write_text("[Service]\nExecStart=/bin/false\n")
        (self.root / "run" / "learning" / "diary.md").write_text("day 2 - learned since\n")
        self.assertNotEqual(snapshot.code_hash(), h1)

        snap, before = snapshot.restore(good.id, write_units=False)
        self.assertEqual(snap.id, good.id)
        self.assertEqual((self.root / "orchestrator" / "main.py").read_text(encoding="utf-8"), "print('v1')\n")
        self.assertIn("false", (self.units / "r1-x.service").read_text(encoding="utf-8"))  # --no-units: units untouched
        self.assertIn("day 2", (self.root / "run" / "learning" / "diary.md").read_text(encoding="utf-8"))  # data kept
        self.assertEqual(before.label, "pre-restore")

        snapshot.restore(good.id)  # with units this time
        self.assertEqual(snapshot.code_hash(), h1)

        snapshot.restore(before.id)  # undo: the broken version is back
        self.assertIn("broken", (self.root / "orchestrator" / "main.py").read_text(encoding="utf-8"))

    def test_restore_with_data_brings_memory_back(self) -> None:
        good = snapshot.create("v1")
        db = sqlite3.connect(str(self.root / "run" / "learning" / "memory.db"))
        db.execute("DELETE FROM facts")
        db.commit()
        db.close()
        snapshot.restore(good.id, with_data=True)
        db = sqlite3.connect(str(self.root / "run" / "learning" / "memory.db"))
        self.assertEqual(db.execute("SELECT fact FROM facts").fetchall(), [("likes tea",)])
        db.close()

    def test_code_hash_ignores_data(self) -> None:
        h = snapshot.code_hash()
        (self.root / "run" / "learning" / "diary.md").write_text("changed\n")
        (self.root / "logs" / "orchestrator.log").write_text("more\n")
        self.assertEqual(snapshot.code_hash(), h)
        (self.root / "config" / "agent.yaml").write_text("a: 2\n")
        self.assertNotEqual(snapshot.code_hash(), h)

    def test_prune_keeps_newest_and_good(self) -> None:
        ids = []
        for i in range(6):
            s = snapshot.create(f"n{i}")
            ids.append(s.id)
            time.sleep(0.01)
        snapshot.mark_good(ids[0])
        removed = snapshot.prune(keep=2, keep_good=1)
        left = [s.id for s in snapshot.list_snapshots()]
        self.assertEqual(left, [ids[0], ids[4], ids[5]])
        self.assertEqual(sorted(removed), sorted(ids[1:4]))
        self.assertEqual(snapshot.latest_good().id, ids[0])


class FakeWatchdog(selfheal.Watchdog):
    def __init__(self) -> None:
        self.states = {n: {"enabled": True, "active": "active", "restarts": 0} for n in selfheal.SERVICES}
        self.restarted: list[str] = []
        self.reloads = 0

    def status(self, name):  # type: ignore[override]
        return dict(self.states[name])

    def restart(self, name):  # type: ignore[override]
        self.restarted.append(name)
        return True

    def daemons(self):  # type: ignore[override]
        return ["r1-devices", "r1-orchestrator"]

    def daemon_reload(self) -> None:
        self.reloads += 1


class SelfHealTests(TempAgent):
    def events(self) -> list[str]:
        p = self.root / "run" / "selfheal_events.jsonl"
        return [json.loads(x)["event"] for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []

    def test_healthy_for_ten_minutes_takes_a_good_snapshot_once(self) -> None:
        w, t = FakeWatchdog(), 1_000_000.0
        for i in range(7):  # every 2 minutes, 12 minutes
            self.assertEqual(w.check(now=t + i * 120)["action"], "ok")
        good = [s for s in snapshot.list_snapshots() if s.good]
        self.assertEqual(len(good), 1)
        self.assertEqual(good[0].code_hash, snapshot.code_hash())
        self.assertEqual(self.events().count("good_snapshot"), 1)
        self.assertEqual(w.restarted, [])

    def test_down_service_is_restarted(self) -> None:
        w = FakeWatchdog()
        w.states["r1-tts-daemon"]["active"] = "failed"
        res = w.check(now=1000.0)
        self.assertEqual(res["action"], "unhealthy")
        self.assertEqual(w.restarted, ["r1-tts-daemon"])
        self.assertIn("restart", self.events())

    def test_disabled_service_is_left_alone(self) -> None:
        w = FakeWatchdog()
        w.states["r1-agent-ui"].update(enabled=False, active="inactive")
        w.check(now=1000.0)
        self.assertEqual(w.restarted, [])

    def test_crash_loop_after_a_code_change_rolls_back_once(self) -> None:
        good = snapshot.create("v1")
        snapshot.mark_good(good.id)
        (self.root / "orchestrator" / "main.py").write_text("raise SystemExit('broken')\n")
        w, t = FakeWatchdog(), 50_000.0
        w.check(now=t)
        for i in range(1, 4):  # systemd keeps restarting it on its own
            w.states["r1-orchestrator"]["restarts"] = i
            res = w.check(now=t + i * 120)
        self.assertEqual(res["action"], "rollback")
        self.assertEqual((self.root / "orchestrator" / "main.py").read_text(encoding="utf-8"), "print('v1')\n")
        self.assertEqual(w.reloads, 1)
        self.assertEqual(set(w.restarted), {"r1-devices", *selfheal.SERVICES})
        notice = (self.root / "run" / "selfheal_notice.txt").read_text(encoding="utf-8")
        self.assertIn(good.id, notice)
        self.assertTrue(any(s.label == "pre-restore" for s in snapshot.list_snapshots()))
        # still crashing right after: no second rollback inside the cooldown
        for i in range(4, 8):
            w.states["r1-orchestrator"]["restarts"] = i
            res = w.check(now=t + i * 120)
        self.assertNotEqual(res["action"], "rollback")
        self.assertEqual(self.events().count("rollback"), 1)

    def test_crash_loop_on_good_code_does_not_roll_back(self) -> None:
        good = snapshot.create("v1")
        snapshot.mark_good(good.id)
        w, t = FakeWatchdog(), 50_000.0
        w.states["r1-llm-daemon"]["active"] = "failed"
        for i in range(3):
            res = w.check(now=t + i * 120)
        self.assertEqual(res["action"], "crash_loop")
        self.assertIn("crash_loop_no_rollback", self.events())
        self.assertIn("没有回退", (self.root / "run" / "selfheal_notice.txt").read_text(encoding="utf-8"))

    def test_paused_means_hands_off(self) -> None:
        w = FakeWatchdog()
        selfheal.save_state({"paused_until": 10_000.0})
        w.states["r1-tts-daemon"]["active"] = "failed"
        w.check(now=1000.0)
        self.assertEqual(w.restarted, [])
        self.assertIn("down_paused", self.events())


class BackupSafetyTests(unittest.TestCase):
    def test_remote_dir_must_be_a_plain_own_path(self) -> None:
        for good in ("monday-backups/r1", "/srv/nextcloud-data/monday-backups/r1"):
            self.assertEqual(backup.check_remote_dir(good), good)
        for bad in ("", "/", "/srv/x", "monday-backups/../..", "a", "a/b; rm -rf ~", "~/x/y", "a/./b", "a/b c",
                    "/srv/nextcloud-data/nc_data/r1", "/etc/monday/r1", "/home/ikun/x", "/srv/../etc/x", "//srv/x"):
            with self.assertRaises(ValueError, msg=bad):
                backup.check_remote_dir(bad)

    def test_old_entries_by_date_name_only(self) -> None:
        now = time.mktime(time.strptime("20261104", "%Y%m%d"))
        days = ["20260801", "20261001", "20261103", "notes.txt", "..", "2026", "20260801x"]
        self.assertEqual(backup.old_entries(days, backup._DAY_NAME, 60, now), ["20260801"])
        snaps = ["20260801-120000-daily.tar.gz", "20260801-120000-daily.json", "20261101-120000-daily.tar.gz",
                 "20261101-120000-daily.json", "evil.tar.gz", "20260801-120000-../x.tar.gz"]
        self.assertEqual(backup.old_entries(snaps, backup._SNAP_NAME, 30, now),
                         ["20260801-120000-daily.json", "20260801-120000-daily.tar.gz"])
        self.assertEqual(backup.old_entries(snaps, backup._SNAP_NAME, 30, now, min_keep=2), [])  # never below min_keep

    def test_config(self) -> None:
        c = backup.Config.from_yaml({"backup": {"enabled": "true", "keep_days": 30, "remote_dir": "x/y/"}})
        self.assertTrue(c.enabled)
        self.assertEqual((c.keep_days, c.remote_dir), (30, "x/y"))
        self.assertFalse(backup.Config.from_yaml({}).enabled)


if __name__ == "__main__":
    unittest.main()
